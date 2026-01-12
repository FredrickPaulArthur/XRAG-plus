"""
Main Summarizer class used by XRAG+.

Goals:
- Single entrypoint for summarizing either single doc or list of docs
- Support abstractive HF models, OpenAI, and extractive fallback
- Safe optional dependencies with helpful error messages
- Chunking for long documents
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
import logging
import os
from dotenv import load_dotenv
load_dotenv()

from src.summarizer.config import Settings
from src.summarizer.utils import clean_text, extractive_textrank, _normalize
from src.summarizer.model_wrapper import HFWrapper, OpenAIWrapper, CohereAIWrapper

logger = logging.getLogger(__name__)




class Summarizer:
    def __init__(self, settings: Settings, chunking_method: str= None):
        self.settings = settings if settings is not None else Settings()
        self.chunking_method = chunking_method

        if settings.MODEL_PROVIDER == "sentence_transformers":
            self.model_wrapper = HFWrapper(
                model=self.settings.MODEL,
                device=self.settings.DEVICE,
                temperature=self.settings.TEMPERATURE
            )
        elif settings.MODEL_PROVIDER == "openai":
            self.model_wrapper = OpenAIWrapper(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=self.settings.MODEL,
                temperature=self.settings.TEMPERATURE
            )
        elif settings.MODEL_PROVIDER == "cohere":
            self.model_wrapper = CohereAIWrapper(
                api_key=os.getenv("COHERE_KEY"),
                model=self.settings.MODEL,
                temperature=self.settings.TEMPERATURE,
                max_tokens = self.settings.MAX_SUMMARY_TOKENS
            )


    def summarize_docs(self, docs: List[Dict[str, Any]] | Dict[str, Any]) -> List[Dict[str, Any]] | Dict[str, Any]:
        """
        Accept documents in `List[Dict[str, Any]]` and `Dict[str, Any]` format and return after adding "summary" from the "text"
        """
        if type(docs) is dict:
            docs = [docs]       # To bring single doc to List[Dict[str, Any]] format

        summarized_docs: List[Dict[str, Any]] = []

        # TODO: To implement multiple document passing on one API call for Cohere and HuggingFace, instead of Iteration.
        for doc in docs:
            summarized_docs.append(self._summarize_doc(doc))
        return summarized_docs


    def _summarize_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize a single Document in `Dict[str, Any]` format and return the summary string.
        """
        provider = self.settings.MODEL_PROVIDER
        doc = clean_text(doc)
        if not doc:
            return ""

        # TODO: to give option for Chunking Method.
        from src.chunker.chunkers import context_aware_chunking
        chunks = context_aware_chunking(
            doc, max_chars=self.settings.MAX_CHUNK_CHARS, overlap_sentences=self.settings.OVERLAP_SENTENCES
        )

        print(chunks)
        # exit()

        summaries = []
        for chunk_text, metadata in chunks:        # (tuple[list, list] | list[tuple])
            if provider == "sentence_transformers":
                out = self.model_wrapper.summarize(
                    chunk_text, max_length=self.settings.MAX_SUMMARY_TOKENS,
                    min_length=self.settings.MIN_SUMMARY_TOKENS
                )
                if isinstance(out, list):   summaries.append(out[0]["summary_text"].strip())
                else:                       summaries.append(str(out).strip())

            elif provider == "openai":
                out = self.model_wrapper.summarize(chunk_text, max_tokens=self.settings.MAX_SUMMARY_TOKENS)
                summaries.append(out[0]["summary_text"].strip())

            elif provider == "cohere":
                try:
                    out = self.model_wrapper.summarize(chunk_text)
                    summaries.append(out[0]["summary_text"].strip())
                except Exception as e:
                    logger.warning(f"Cohere summarization failed: {e}")

        # Join chunk summaries and optionally run a short final summarization step
        joined = "\n\n".join([s for s in summaries if s])
        # If we have multiple chunk summaries we can compress them using extractive_textrank
        if len(summaries) > 1:
            try:
                doc["summary"] = extractive_textrank(joined, top_k=self.settings.TOP_K)
                return doc
            except Exception:
                doc["summary"] = joined
                return doc
        doc["summary"] = joined
        return doc



    # def _summarize_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Summarize a single Document (dict with at least 'text' and optionally 'title').
    #     Chunking strategy is selected by `self.settings.CHUNKING_METHOD`.

    #     Outputs written to `doc`:
    #     - doc["chunks"] = list of {"text": ..., "meta": {...}}
    #     - doc["chunk_summaries"] = list of summaries (strings) in same order as chunks
    #     - doc["summary"] = final (possibly compressed) summary string

    #     Supported CHUNKING_METHOD values:
    #     - "context_aware_chunking" (paragraph-aware, with sentence overlap)
    #     - "sentence"       (SentenceChunker)
    #     - "paragraph"      (ParagraphChunker)
    #     - "sliding"        (SlidingWindowChunker)
    #     - "token"          (TokenChunker)  -- requires a tokenizer dependency
    #     - "none"           (no chunking: whole doc as single chunk)
    #     """
    #     provider = getattr(self.settings, "MODEL_PROVIDER", "cohere")
    #     doc = clean_text(doc)
    #     if not doc or not doc.get("text"):
    #         doc["summary"] = ""
    #         doc["chunks"] = []
    #         doc["chunk_summaries"] = []
    #         return doc

    #     # --- CHUNKER SELECTION ---
    #     method = getattr(self.settings, "CHUNKING_METHOD")
    #     max_chunk_chars = getattr(self.settings, "MAX_CHUNK_CHARS")
    #     chunk_items = []  # will hold tuples (text, meta)

    #     try:
    #         from src.chunker.chunkers import (
    #             TokenChunker,
    #             SlidingWindowChunker,
    #             SentenceChunker,
    #             ParagraphChunker,
    #             context_aware_chunking,
    #         )
    #     except Exception:
    #         # If chunker module missing, fallback to single chunk
    #         context_aware_chunking = None


    #     # Choose chunker
    #     try:
    #         if method == "context_aware_chunking" and context_aware_chunking is not None:
    #             raw = context_aware_chunking(doc, max_chars=max_chunk_chars, overlap_sentences=1)
    #             chunk_items = _normalize(raw)
    #         elif method == "sentence_chunking":
    #             sc = SentenceChunker(min_tokens=5)
    #             raw = sc.chunk(doc)
    #             chunk_items = _normalize(raw)
    #         elif method == "paragraph_chunking":
    #             pc = ParagraphChunker(min_chars=200)
    #             raw = pc.chunk(doc)
    #             chunk_items = _normalize(raw)
    #         elif method == "sliding_window_chunking":
    #             # SlidingWindowChunker may expect chunk_size/overlap in chars or tokens depending on your implementation.
    #             # We pass a chars-based window and overlap of ~15%.
    #             overlap = max(1, int(max_chunk_chars * 0.15))
    #             sw = SlidingWindowChunker(chunk_size=max_chunk_chars, overlap=overlap)
    #             raw = sw.chunk(doc)
    #             chunk_items = _normalize(raw)
    #         elif method == "token_chunking":
    #             # TokenChunker typically needs a tokenizer (e.g., spacy). If absent, fall back.
    #             try:
    #                 # If a tokenizer is available in settings, use it; else attempt spacy if installed.
    #                 tokenizer = getattr(self.settings, "TOKENIZER", None)
    #                 if tokenizer is None:
    #                     try:
    #                         import spacy
    #                         tokenizer = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"])
    #                     except Exception:
    #                         tokenizer = None
    #                 tc = TokenChunker(tokenizer=tokenizer, chunk_size=getattr(self.settings, "TOKEN_CHUNK_SIZE", 1024), stride=getattr(self.settings, "TOKEN_CHUNK_STRIDE", 128))
    #                 raw = tc.chunk(doc)
    #                 chunk_items = _normalize(raw)
    #             except Exception:
    #                 # token chunker unavailable - fallback to sentence chunker
    #                 sc = SentenceChunker(min_tokens=5)
    #                 raw = sc.chunk(doc)
    #                 chunk_items = _normalize(raw)
    #         elif method in ("none", "single", "nochunk"):
    #             chunk_items = [(doc.get("text", "").strip(), {"context_title": doc.get("title")})]
    #         else:
    #             # Unknown method: fallback to context_aware_chunking if available otherwise single chunk
    #             if context_aware_chunking is not None:
    #                 raw = context_aware_chunking(doc, max_chars=max_chunk_chars, overlap_sentences=1)
    #                 chunk_items = _normalize(raw)
    #             else:
    #                 chunk_items = [(doc.get("text", "").strip(), {"context_title": doc.get("title")})]
    #     except Exception as e:
    #         # any chunker error -> fallback to whole doc chunk
    #         logger.warning("Chunking (%s) failed, falling back to whole-document chunk: %s", method, e)
    #         chunk_items = [(doc.get("text", "").strip(), {"context_title": doc.get("title")})]

    #     # Ensure we have something
    #     if not chunk_items:
    #         chunk_items = [(doc.get("text", "").strip(), {"context_title": doc.get("title")})]

    #     # Summarize each chunk (per-provider) with fallback
    #     chunk_texts = [t for t, _ in chunk_items]
    #     chunk_meta = [m for _, m in chunk_items]

    #     summaries: List[str] = []
    #     chunk_summaries_meta: List[Dict[str, Any]] = []

    #     for idx, text in enumerate(chunk_texts):
    #         # protect empty chunk
    #         if not text:
    #             summaries.append("")
    #             chunk_summaries_meta.append({"idx": idx})
    #             continue

    #         try:
    #             if provider == "sentence_transformers":
    #                 out = self.model_wrapper.summarize(
    #                     text,
    #                     max_length=getattr(self.settings, "MAX_SUMMARY_TOKENS", 256),
    #                     min_length=getattr(self.settings, "MIN_SUMMARY_TOKENS", 32),
    #                 )
    #                 # normalize HF pipeline-like response
    #                 if isinstance(out, list) and out:
    #                     s = out[0].get("summary_text") if isinstance(out[0], dict) else str(out[0])
    #                 elif isinstance(out, dict):
    #                     s = out.get("summary_text", str(out))
    #                 else:
    #                     s = str(out)
    #                 s = (s or "").strip()
    #                 if not s:
    #                     raise ValueError("empty HF summary")
    #                 summaries.append(s)
    #                 chunk_summaries_meta.append({"idx": idx, "source": provider})
    #             elif provider == "openai":
    #                 out = self.model_wrapper.summarize(text, max_tokens=getattr(self.settings, "MAX_SUMMARY_TOKENS", 150))
    #                 # Expect wrapper returns list[{"summary_text":...}] or a string
    #                 if isinstance(out, list) and out and isinstance(out[0], dict):
    #                     s = out[0].get("summary_text", "")
    #                 elif isinstance(out, str):
    #                     s = out
    #                 else:
    #                     s = str(out)
    #                 s = (s or "").strip()
    #                 if not s:
    #                     raise ValueError("empty OpenAI summary")
    #                 summaries.append(s)
    #                 chunk_summaries_meta.append({"idx": idx, "source": provider})
    #             elif provider == "cohere":
    #                 out = self.model_wrapper.summarize(text)
    #                 if isinstance(out, list) and out and isinstance(out[0], dict):
    #                     s = out[0].get("summary_text", "")
    #                 elif isinstance(out, dict):
    #                     s = out.get("summary_text", "")
    #                 else:
    #                     s = str(out)
    #                 s = (s or "").strip()
    #                 if not s:
    #                     raise ValueError("empty Cohere summary")
    #                 summaries.append(s)
    #                 chunk_summaries_meta.append({"idx": idx, "source": provider})
    #             else:
    #                 # unknown provider -> deterministic extractive fallback
    #                 s = extractive_textrank(text, top_k=getattr(self.settings, "TOP_K", 3))
    #                 summaries.append(s)
    #                 chunk_summaries_meta.append({"idx": idx, "source": "extractive"})
    #         except Exception as e:
    #             logger.warning("Summarization failed for provider=%s chunk=%d: %s", provider, idx, e)
    #             # fallback to extractive textrank for this chunk
    #             try:
    #                 s = extractive_textrank(text, top_k=getattr(self.settings, "TOP_K", 3))
    #             except Exception:
    #                 s = text[:getattr(self.settings, "MAX_SUMMARY_TOKENS", 200)]
    #             summaries.append(s)
    #             chunk_summaries_meta.append({"idx": idx, "source": "fallback_extractive", "error": str(e)})

    #     # Final compression/merge step
    #     doc["chunks"] = [{"text": chunk_texts[i], "meta": chunk_meta[i]} for i in range(len(chunk_texts))]
    #     doc["chunk_summaries"] = summaries

    #     # If more than 1 chunk summary, compress deterministically with extractive_textrank
    #     if len(summaries) > 1:
    #         joined = "\n\n".join([s for s in summaries if s])
    #         try:
    #             # use TOP_K to determine how many sentences to keep in the compressed summary
    #             doc["summary"] = extractive_textrank(joined, top_k=getattr(self.settings, "TOP_K", 3))
    #         except Exception:
    #             doc["summary"] = joined
    #     else:
    #         doc["summary"] = summaries[0] if summaries else ""

    #     return doc