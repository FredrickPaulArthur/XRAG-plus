### File: src/retrieval/retriever.py
"""
Retrieve from chroma collection using embedding similarity, keyword search, and hybrid methods.
"""
from typing import List, Dict, Optional, Tuple
import logging
from .chroma_client import ChromaManager
from .config import Settings

logger = logging.getLogger(__name__)




class Retriever:
    """Retriever that is specific to a Language, uses LANG_EMBEDDING_MAP from Retriever Config to decide Embedding Model
    according to the Language of the retriever."""
    def __init__(self, settings: Settings = None, chroma_manager: ChromaManager = None, language: str = "en"):
        self.settings = settings or Settings()
        self.chroma_manager = chroma_manager or ChromaManager(persist_directory=self.settings.CHROMA_PERSIST_DIR)
        self.language = language

        lang_map = self.settings.LANG_EMBEDDING_MAP.get(self.language, None)
        self.semantic_provider = None
        self.semantic_model_name = None
        if isinstance(lang_map, dict):
            self.semantic_provider = lang_map.get("provider")
            self.semantic_model_name = lang_map.get("model")
        else:
            # fallback to defaults
            self.semantic_provider = getattr(self.settings, "DEFAULT_EMBEDDING_PROVIDER", None)
            self.semantic_model_name = getattr(self.settings, "DEFAULT_EMBEDDING_MODEL", None)

    def _merge_and_rank(self, results_list: List[Dict], prefer_source: Optional[str] = None,
                        boost: float = 0.18, top_k: int = 5) -> Dict:
        """
        Merge a list of retrieval outputs (each with keys: ids, distances, metadatas, documents)
        Normalize scores across collections, dedupe, apply optional source boost, and return top_k.
        Returns dict with keys: ids, distances, metadatas, documents (length <= top_k)
        """
        items = []
        for res in results_list:
            ids = res.get("ids", []) or []
            dists = res.get("distances", []) or []
            metas = res.get("metadatas", []) or []
            docs = res.get("documents", []) or []

            # If number of distances differs, pad with None
            while len(dists) < len(docs):
                dists.append(None)

            for i, doc in enumerate(docs):
                raw_dist = dists[i] if i < len(dists) else None
                # convert distance -> similarity (higher better)
                try:
                    if raw_dist is None:
                        sim = 0.0
                    else:
                        rd = float(raw_dist)
                        # treat zero as perfect
                        sim = 1.0 if rd == 0 else 1.0 / (1.0 + rd)
                except Exception:
                    sim = 0.0

                items.append({
                    "id": ids[i] if i < len(ids) else None,
                    "doc": doc,
                    "meta": metas[i] if i < len(metas) else {},
                    "sim": sim,
                    "raw_dist": raw_dist
                })

        if not items:
            return {"ids": [], "distances": [], "metadatas": [], "documents": []}

        # min-max normalize sims across entire pool
        sims = [it["sim"] for it in items]
        mn, mx = min(sims), max(sims)
        if mx - mn > 1e-12:
            for it in items:
                it["norm_sim"] = (it["sim"] - mn) / (mx - mn)
        else:
            for it in items:
                it["norm_sim"] = it["sim"]

        # apply source boost and compute final score
        for it in items:
            boost_val = boost if (prefer_source and it["meta"].get("source") == prefer_source) else 0.0
            it["score"] = min(1.0, it["norm_sim"] + boost_val)

        # dedupe: prefer (doc_id, chunk_index) then checksum then text snippet
        seen = set()
        uniq = []
        for it in sorted(items, key=lambda x: x["score"], reverse=True):
            meta = it["meta"] or {}
            if meta.get("doc_id") and meta.get("chunk_index") is not None:
                key = (meta.get("doc_id"), meta.get("chunk_index"))
            elif meta.get("checksum"):
                key = ("checksum", meta.get("checksum"))
            else:
                key = ("text_snip", it["doc"].strip()[:200])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(it)
            if len(uniq) >= top_k:
                break

        # produce return format (convert score back to approximate distance)
        out_ids, out_docs, out_metas, out_dists = [], [], [], []
        for it in uniq:
            out_ids.append(it["id"])
            out_docs.append(it["doc"])
            out_metas.append(it["meta"])
            # approximate distance from score (avoid division by zero)
            s = it.get("score", 0.0)
            if s > 0:
                try:
                    approx_dist = (1.0 / s) - 1.0
                except Exception:
                    approx_dist = None
            else:
                approx_dist = None
            out_dists.append(approx_dist)

        return {
            "ids": out_ids,
            "distances": out_dists,
            "metadatas": out_metas,
            "documents": out_docs,
        }


    def retrieve_semantic(self, collection_name: str, query: str, k: int = 5, where: Optional[Dict] = None) -> Dict:
        """
        Embedding-based retrieval. Returns dict with keys: ids, distances, metadatas, documents (each a list length <= k).
        """
        col = self.chroma_manager.get_collection(collection_name)

        # Embedding for the query
        if self.semantic_provider == "sentence_transformers":
            from src.indexing.embeddings import SentenceTransformersProvider
            q_embs = SentenceTransformersProvider(self.semantic_model_name).embed_documents([query])
        elif self.semantic_provider == "cohere":
            from src.indexing.embeddings import CohereAIEmbeddingProvider
            q_embs = CohereAIEmbeddingProvider(self.semantic_model_name).embed_documents([query])
        elif self.semantic_provider == "openai":
            from src.indexing.embeddings import OpenAIEmbeddingProvider
            q_embs = OpenAIEmbeddingProvider(model=self.semantic_model_name).embed_documents([query])

        if not isinstance(q_embs, list) or len(q_embs) == 0:
            raise RuntimeError("Embedding function did not return embeddings for the query.")
        q_emb = q_embs[0]

        # query chroma
        results = col.query(query_embeddings=[q_emb], n_results=k, where=where)

        # normalize results (collection.query returns lists per query)
        ids = results.get("ids", [[]])[0] if results.get("ids") else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        documents = results.get("documents", [[]])[0] if results.get("documents") else []

        return {
            "ids": ids,
            "distances": distances,
            "metadatas": metadatas,
            "documents": documents,
        }


    def retrieve_keyword(self, collection_name: str, query: str, k: int = 5, where: Optional[Dict] = None,
                         require_all_terms: bool = False) -> Dict:
        """
        Keyword retrieval
        - simple substring/token matching + ranking
        - Fetches stored documents from Chroma (respecting `where` if provided via collection.get).
        - Scores documents by simple token/substring match count.
        - Returns top-k documents by score.

        Parameters:
            require_all_terms: if True, only documents containing ALL query terms are considered.
        """
        col = self.chroma_manager.get_collection(collection_name)

        # Fetch stored docs (ids, documents, metadatas). `where` is honored if provided.
        # Note: this pulls all rows that match `where`. For very large collections, you should page.
        try:
            all_items = col.get(include=["documents", "metadatas"], where=where)
        except TypeError:
            # Some chroma versions expect parameters differently; try without where
            all_items = col.get(include=["documents", "metadatas"])

        ids: List[str] = all_items.get("ids", [])
        docs: List[str] = all_items.get("documents", [])
        metas: List[Dict] = all_items.get("metadatas", [])

        query_lower = query.lower().strip()
        # basic tokenization: split on whitespace and punctuation
        import re
        tokens = [t for t in re.split(r"\W+", query_lower) if t]

        scores: List[Tuple[int, int]] = []  # (score, idx)
        for idx, doc in enumerate(docs):
            doc_text = (doc or "").lower()
            # count occurrences of each token
            counts = [doc_text.count(tok) for tok in tokens] if tokens else [0]
            # score is sum of counts (higher better)
            score = sum(counts)
            if require_all_terms and any(c == 0 for c in counts):
                # skip if not all tokens present
                continue
            # small heuristic: boost if exact phrase appears
            if tokens and query_lower in doc_text:
                score += 2
            if score > 0:
                scores.append((score, idx))

        # sort scores descending
        scores.sort(reverse=True, key=lambda x: x[0])
        top = scores[:k]

        out_ids = [ids[idx] for (_, idx) in top]
        out_docs = [docs[idx] for (_, idx) in top]
        out_metas = [metas[idx] for (_, idx) in top]
        # For keyword retrieval we don't have real distances; instead provide inverse of normalized score
        if scores:
            max_score = max(s[0] for s in scores)
        else:
            max_score = 1
        out_distances = [1.0 - (s / max_score) if max_score > 0 else 1.0 for (s, _) in top]

        return {
            "ids": out_ids,
            "distances": out_distances,
            "metadatas": out_metas,
            "documents": out_docs,
        }


    def retrieve_hybrid(self, collection_name: str, query: str, k: int = 5, where: Optional[Dict] = None,
                        alpha: float = 0.7) -> Dict:
        """
        Hybrid retrieval:
        - Runs semantic retrieval to get semantic candidates and distances.
        - Runs keyword scoring across candidate set (or entire collection if needed).
        - Combines semantic similarity and keyword score using `alpha`:
            final_score = alpha * semantic_score + (1 - alpha) * keyword_score
        - Returns top-k items sorted by final_score.

        alpha: weight of semantic score in [0,1]. 1.0 => pure semantic, 0.0 => pure keyword.
        """
        # 1) get semantic results (we ask for more than k to allow keyword boost to reorder)
        sem_k = max(k * 3, k)  # fetch up to 3x to give keyword boosting room
        sem_res = self.retrieve_semantic(collection_name, query, k=sem_k, where=where)
        sem_ids = sem_res.get("ids", [])
        sem_docs = sem_res.get("documents", [])
        sem_metas = sem_res.get("metadatas", [])
        sem_dists = sem_res.get("distances", [])

        # Convert distances -> semantic similarity in [0,1] (higher better).
        # handle missing distances gracefully
        sem_sims = []
        if sem_dists:
            # distances are non-negative; smaller is better. convert to sim = 1/(1+dist)
            for d in sem_dists:
                try:
                    sim = 1.0 / (1.0 + float(d))
                except Exception:
                    sim = 0.0
                sem_sims.append(sim)
        else:
            # fallback: assign descending scores by rank
            for rank in range(len(sem_ids)):
                sem_sims.append(1.0 - (rank / max(1, len(sem_ids) - 1)))

        # 2) compute keyword scores for the semantic candidate set (faster than whole DB)
        # Build a map id -> doc text / meta
        id_to_doc = {i: d for i, d in zip(sem_ids, sem_docs)}
        id_to_meta = {i: m for i, m in zip(sem_ids, sem_metas)}

        # keyword scoring same as in retrieve_keyword but restricted to sem candidates
        import re
        q_lower = query.lower().strip()
        tokens = [t for t in re.split(r"\W+", q_lower) if t]

        keyword_scores = {}
        max_count = 0
        for idx, doc in id_to_doc.items():
            doc_text = (doc or "").lower()
            counts = [doc_text.count(tok) for tok in tokens] if tokens else [0]
            score = sum(counts)
            if tokens and q_lower in doc_text:
                score += 2
            keyword_scores[idx] = score
            if score > max_count:
                max_count = score

        # Normalize keyword scores to [0,1]
        if max_count <= 0:
            # none matched; all zero
            norm_keyword = {i: 0.0 for i in id_to_doc.keys()}
        else:
            norm_keyword = {i: (keyword_scores.get(i, 0) / max_count) for i in id_to_doc.keys()}

        # 3) combine scores and rank
        combined_list = []
        for pos, doc_id in enumerate(sem_ids):
            sem_score = sem_sims[pos] if pos < len(sem_sims) else 0.0
            kw_score = norm_keyword.get(doc_id, 0.0)
            combined = alpha * sem_score + (1.0 - alpha) * kw_score
            combined_list.append((combined, doc_id, sem_score, kw_score, id_to_doc.get(doc_id), id_to_meta.get(doc_id)))

        # Sort by combined score descending and take top-k
        combined_list.sort(reverse=True, key=lambda x: x[0])
        top = combined_list[:k]

        out_ids = [item[1] for item in top]
        out_documents = [item[4] for item in top]
        out_metadatas = [item[5] for item in top]
        # we provide semantic distance where possible (converted back from sim approx): d = (1/sim) - 1
        out_distances = []
        for item in top:
            sem_score = item[2]
            if sem_score > 0:
                try:
                    d = (1.0 / sem_score) - 1.0
                except Exception:
                    d = None
            else:
                d = None
            out_distances.append(d)

        return {
            "ids": out_ids,
            "distances": out_distances,
            "metadatas": out_metadatas,
            "documents": out_documents,
        }


    def retrieve(self, collection_name: str, query: str, k: int = 5, where: Optional[Dict] = None,
                 method: str = "semantic", **kwargs) -> Dict:
        """
        Wrapper entrypoint. method can be: "semantic" (default), "keyword", or "hybrid".
        Extra kwargs are forwarded to the specific method (e.g., alpha for hybrid).
        """

        collection_startswith = f"xragg_collection__{self.language}"

        method = (method or "semantic").lower()
        if method == "semantic":
            return self.retrieve_semantic(collection_name, query, k=k, where=where)
        elif method == "keyword":
            return self.retrieve_keyword(collection_name, query, k=k, where=where, **kwargs)
        elif method == "hybrid":
            alpha = kwargs.get("alpha", 0.7)
            return self.retrieve_hybrid(collection_name, query, k=k, where=where, alpha=alpha)
        else:
            raise ValueError(f"Unknown retrieval method: {method}. Supported: semantic, keyword, hybrid.")


    # def retrieve_lang_specific(self, query: str, k: int = 5, language: Optional[str] = None, method: str = "hybrid", 
    #         where: Optional[Dict] = None, prefer_source: Optional[str] = None, boost: float = 0.18, 
    #         alpha: float = 0.7, top_k: Optional[int] = None) -> Dict:
    #     """
    #     Search across collections that match the specified language.
    #     - language: e.g. "en", "de". If None, uses self.language.
    #     - method: "semantic" | "keyword" | "hybrid"
    #     - prefer_source: optional metadata source to boost (e.g. 'wiki')
    #     - alpha: hybrid weight forwarded to hybrid retrieval
    #     - top_k: final number of merged results to return (defaults to k)
    #     """
    #     lang = language
    #     collection_prefix = f"xragg_collection__{lang}"
    #     try:
    #         collections = self.chroma_manager.list_collections(name_startswith=collection_prefix)
    #     except Exception:   # fallback: list everything
    #         collections = [c for c in self.chroma_manager.list_collections() if c.startswith(collection_prefix)]

    #     if not collections:
    #         return {"ids": [], "distances": [], "metadatas": [], "documents": []}

    #     results_pool = []
    #     per_col_k = max(3, k)       # pick per-collection retrieval k (small) to reduce API load

    #     for coll in collections:
    #         try:
    #             if method == "semantic":
    #                 r = self.retrieve_semantic(coll, query, k=per_col_k, where=where)
    #             elif method == "keyword":
    #                 r = self.retrieve_keyword(coll, query, k=per_col_k, where=where)
    #             elif method == "hybrid":
    #                 r = self.retrieve_hybrid(coll, query, k=per_col_k, where=where, alpha=alpha)
    #             else:
    #                 r = self.retrieve_semantic(coll, query, k=per_col_k, where=where)
    #         except Exception as e:
    #             logger.warning("Retrieval failed for collection %s: %s", coll, e)
    #             continue
    #         results_pool.append(r)

    #     final_top_k = top_k or k
    #     return self._merge_and_rank(results_pool, prefer_source=prefer_source, boost=boost, top_k=final_top_k)


    def retrieve_embedding_specific(self, query: str, k: int = 5, embedding: Optional[str] = None, method: str = "hybrid", 
            where: Optional[Dict] = None, prefer_source: Optional[str] = None, boost: float = 0.18,
            alpha: float = 0.7, top_k: Optional[int] = None) -> Dict:
        """
        Search across collections that contain the given embedding-model-name substring.
        - embedding: substring to match inside the collection name (e.g. "embed-multilingual-v3.0" or "all-MiniLM")
        - method, alpha same as above
        - top_k final returned size
        """
        # search among all collections but filter by embedding substring
        try:
            all_colls = self.chroma_manager.list_collections()
        except Exception:
            all_colls = []

        if embedding:
            candidate_colls = [c for c in all_colls if embedding in c]
        else:
            # if no embedding provided, default to all language-specific collections
            candidate_colls = [c for c in all_colls if c.startswith(f"xragg_collection__{self.language}")]

        if not candidate_colls:
            return {"ids": [], "distances": [], "metadatas": [], "documents": []}

        results_pool = []
        per_col_k = max(3, k)
        for coll in candidate_colls:
            try:
                if method == "semantic":
                    r = self.retrieve_semantic(coll, query, k=per_col_k, where=where)
                elif method == "keyword":
                    r = self.retrieve_keyword(coll, query, k=per_col_k, where=where)
                elif method == "hybrid":
                    r = self.retrieve_hybrid(coll, query, k=per_col_k, where=where, alpha=alpha)
                else:
                    r = self.retrieve_semantic(coll, query, k=per_col_k, where=where)
            except Exception as e:
                logger.warning("Retrieval failed for collection %s: %s", coll, e)
                continue
            results_pool.append(r)

        final_top_k = top_k or k
        return self._merge_and_rank(results_pool, prefer_source=prefer_source, boost=boost, top_k=final_top_k)