"""
XRAG+ Reranker module (reworked to use centralized config)
File: src/reranker.py

This version reads model names and hyperparameters from `src.config.settings`.
It keeps the same behavior: prefer CrossEncoder (if available) otherwise fall
back to an embedding-based cosine similarity approach.

Key features
- lazy model loading (so imports are cheap at module import time)
- respects settings for batch sizes, model IDs and normalization
- returns documents with attached `score` field
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
import math

from src.chunker.chunkers import TokenChunker, SlidingWindowChunker, SentenceChunker, ParagraphChunker, context_aware_chunking

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



class Reranker:
    """
    Re-ranker that prefers a CrossEncoder but falls back to embedding-based
    cosine similarity. Configurable via `src.config.settings`.
    """
    from .config import Settings
    settings = Settings()

    # preferred approach: CrossEncoder from sentence_transformers
    try:
        from sentence_transformers import CrossEncoder
        _HAS_CROSS_ENCODER = True
    except Exception:
        _HAS_CROSS_ENCODER = False

    # Fallback: sentence-transformers (for embeddings) + sklearn cosine
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        _HAS_SENTENCE_TRANSFORMER = True
    except Exception:
        _HAS_SENTENCE_TRANSFORMER = False

    try:
        _HAS_TRANSFORMERS = True
    except Exception:
        _HAS_TRANSFORMERS = False

    def __init__(self, _use_cross=False, use_jina=False) -> None:
        self._use_cross = _use_cross
        self.cross_encoder = None
        self.mono_encoder = None
        self._use_jina = use_jina
        self.jina_model = None
        self._loaded = False


    def _lazy_load(self) -> None:
        """
        Load CrossEncoder if available and desired, else MonoEncoder Model will be loaded.
        """
        if self._loaded:
            return

        # Cross-encoder (pairwise)
        if self._HAS_CROSS_ENCODER and self._use_cross:
            try:
                logger.info("Loading CrossEncoder: %s", self.settings.CROSS_ENCODER_MODEL)
                self.cross_encoder = self.CrossEncoder(model_name_or_path=self.settings.CROSS_ENCODER_MODEL, trust_remote_code=True, device="cuda")
                logger.info("Loaded CrossEncoder successfully.")
            except Exception as e:
                logger.warning("Failed to load CrossEncoder (%s). Falling back. Error: %s",
                            self.settings.CROSS_ENCODER_MODEL, e)
                self.cross_encoder = None
                self._use_cross = False
        elif self._use_cross is False:
            logger.info("CrossEncoder has been disabled; using embedding fallback or Jina as configured.")

        # Jina listwise reranker (special handling)
        if self._use_jina:
            logger.info("JinaAI has been enabled for Reranking.")
            if not self._HAS_TRANSFORMERS:
                raise RuntimeError("transformers is required to load Jina reranker via trust_remote_code=True")
            try:
                jina_mid = getattr(self.settings, "JINA_RERANKER_MODEL", "jinaai/jina-reranker-v3")
                logger.info("Loading Jina listwise reranker (trust_remote_code): %s", jina_mid)
                # This will execute model-specific code from the model repository

                from transformers import AutoModel
                self.jina_model = AutoModel.from_pretrained(jina_mid, trust_remote_code=True)
                logger.info("Loaded Jina reranker model.")
            except Exception as e:
                logger.exception("Failed to load Jina reranker %s: %s", jina_mid, e)
                self.jina_model = None
                self._use_jina = False

        # Mono encoder fallback (only if we need it)
        if not self._use_cross and not self._use_jina:
            if not self._HAS_SENTENCE_TRANSFORMER:
                raise RuntimeError("No reranker backend available. Install 'sentence-transformers' and 'scikit-learn'.")
            logger.info("Loading SentenceTransformer for embedding fallback: %s", self.settings.MONO_ENCODER_MODEL)
            self.mono_encoder = self.SentenceTransformer(model_name_or_path=self.settings.MONO_ENCODER_MODEL)

        self._loaded = True


    def _batched_chunks(self, iterable, size: int):
        for i in range(0, len(iterable), size):
            yield iterable[i : i + size]


    @staticmethod
    def _minmax_normalize(values: List[float], eps: float = 1e-12) -> List[float]:
        if not values:
            return []
        vmin = min(values)
        vmax = max(values)
        if math.isclose(vmin, vmax, rel_tol=1e-12):
            return [0.5 for _ in values]
        return [(v - vmin) / (vmax - vmin + eps) for v in values]


    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.settings.TOP_K

        if not documents:
            return []

        docs: List[Dict[str, Any]] = []
        for d in documents:
            if isinstance(d, str):
                docs.append({"text": d})
            elif isinstance(d, dict) and "text" in d:
                docs.append(d.copy())
            else:
                raise ValueError("Each document must be either a string or a dict containing the key 'text'")

        self._lazy_load()           # lazy load models

        # if self._use_cross and self.cross_encoder is not None:
        #     scores = self._rerank_with_cross_encoder(query, docs, self.settings.BATCH_SIZE)
        # else:
        #     scores = self._rerank_with_embeddings(query, docs, self.settings.BATCH_SIZE)
        # choose path: jina (listwise) > cross-encoder > mono-encoder
        if self._use_jina and self.jina_model is not None:
            scores = self._rerank_with_jina_listwise(query, docs, batch_size=self.settings.JINA_LISTWISE_MAX_DOCS)
        elif self._use_cross and self.cross_encoder is not None:
            scores = self._rerank_with_cross_encoder(query, docs, self.settings.BATCH_SIZE)
        else:
            scores = self._rerank_with_embeddings(query, docs, self.settings.BATCH_SIZE)

        for doc, score in zip(docs, scores):            # attach scores and sort
            doc["score"] = float(score)

        docs_sorted = sorted(docs, key=lambda x: x["score"], reverse=True)
        return docs_sorted[: min(top_k, len(docs_sorted))]      # Goes out as Top results


    def _rerank_with_cross_encoder(self, query: str, docs: List[Dict[str, Any]], batch_size: Optional[int] = None) -> List[float]:
        assert self.cross_encoder is not None
        if batch_size is None:
            batch_size = self.settings.BATCH_SIZE

        pairs = [(query, d["text"]) for d in docs]
        scores: List[float] = []

        import spacy
        chunker = TokenChunker(
            tokenizer=spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"]),
            chunk_size=20, 
            stride=5
        )
        chunks = [chunker.chunk(d) for d in docs]
    
        # pair_chunks = [(query, chunk_text) for chunk_text in chunks]
        pair_chunks = [(query, chunk_text[0][0]) for chunk_text in chunks]

        for chunk in self._batched_chunks(pair_chunks, batch_size):
            try:
                chunk_scores = self.cross_encoder.predict(chunk)
            except Exception as e:
                logger.exception("CrossEncoder.predict failed on a chunk. Error: %s", e)
                chunk_scores = [0.0] * len(chunk)
            scores.extend([float(x) for x in chunk_scores])

        if self.settings.NORMALIZE_SCORES:
            scores = self._minmax_normalize(scores)
        return scores


    def _rerank_with_embeddings(self, query: str, docs: List[Dict[str, Any]], batch_size: Optional[int] = None) -> List[float]:
        if batch_size is None:
            batch_size = self.settings.BATCH_SIZE
        assert self.mono_encoder is not None

        # encode query
        try:
            q_emb = self.mono_encoder.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        except TypeError:
            q_emb = self.mono_encoder.encode(query, convert_to_numpy=True, show_progress_bar=False)
        except Exception as e:
            logger.exception("Failed to encode query with SentenceTransformer: %s", e)
            raise

        # encode docs
        # doc_texts = [d["text"] for d in docs]
        doc_embs = []

        import spacy
        chunker = TokenChunker(
            tokenizer=spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"]),
            chunk_size=20, 
            stride=5
        )
        chunks = [chunker.chunk(d) for d in docs]
    
        # pair_chunks = [(query, chunk_text) for chunk_text in chunks]
        pair_chunks = [(query, chunk_text) for chunk_text in chunks]

        for chunk in self._batched_chunks(pair_chunks, batch_size):
            enc = self.mono_encoder.encode(chunk, convert_to_numpy=True, show_progress_bar=False)
            doc_embs.extend(enc)

        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity([q_emb], doc_embs)[0].tolist()
        # normalize from [-1,1] to [0,1]
        sims = [(s + 1.0) / 2.0 for s in sims]
        return sims



    def _rerank_with_jina_listwise(self, query: str, docs: List[Dict[str, Any]], batch_size: Optional[int] = None) -> List[float]:
        """
        Use a Jina listwise reranker via model.rerank(query, documents, top_n=...).
        This collects raw relevance_score for each doc (across chunks if needed),
        then applies a single global normalization (if enabled).
        """
        assert self.jina_model is not None
        if batch_size is None:
            batch_size = getattr(self.settings, "JINA_LISTWISE_MAX_DOCS", 64)

        texts = [d["text"] for d in docs]
        n = len(texts)
        raw_scores = [0.0] * n

        # A helper to map reranker output to original indices safely (handles duplicate texts)
        def map_rerank_results_to_indices(chunk_texts, reranked_list, base_idx):
            # build text -> queue of indices for this chunk
            mapping = {}
            for offset, t in enumerate(chunk_texts):
                mapping.setdefault(t, []).append(base_idx + offset)
            # iterate reranked items (should be ordered by relevance)
            for item in reranked_list:
                doc_text = item.get("document", item.get("text", None))
                score = float(item.get("relevance_score", item.get("score", 0.0)))
                if doc_text is None:
                    continue
                idx_list = mapping.get(doc_text)
                if idx_list:
                    idx = idx_list.pop(0)  # take first unused index for duplicate-safe mapping
                    raw_scores[idx] = score
                else:
                    # best-effort fallback: if exact text not found, skip (or you could match by fuzzy)
                    pass

        # chunk and call listwise reranker
        for i, chunk in enumerate(self._batched_chunks(texts, batch_size)):
            base_idx = i * batch_size
            try:
                reranked = self.jina_model.rerank(query, list(chunk), top_n=len(chunk))
                # reranked is expected to be a list of dicts with 'document' and 'relevance_score'
                map_rerank_results_to_indices(list(chunk), reranked, base_idx)
            except Exception as e:
                logger.exception("Jina listwise reranker failed on chunk starting %d: %s", base_idx, e)
                # fallback: set zeros for this chunk (or you could fall back to mono)
                for offset in range(len(chunk)):
                    raw_scores[base_idx + offset] = 0.0

        # finally, normalize globally if requested
        if self.settings.NORMALIZE_SCORES:
            normed = self._minmax_normalize(raw_scores)
            return normed
        return raw_scores