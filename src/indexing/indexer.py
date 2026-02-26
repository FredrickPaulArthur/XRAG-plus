"""
Chroma-based Indexer for XRAG+

Features:
- One collection per (language, source) combination
- Per-language embedding provider configurable
- Context-aware chunking with overlap
- Batched upserts, deduplication by checksum
- Helpful metadata stored per chunk

Dependencies:
- chromadb (pip install chromadb)
- (optional) sentence-transformers, openai (see embeddings.py)
"""

from typing import List, Dict, Any, Optional, Tuple
import chromadb
import hashlib
import time
import math

from src.chunker.chunkers import context_aware_chunking, ParagraphChunker, SentenceChunker, SlidingWindowChunker, TokenChunker
from src.indexing.embeddings import EmbeddingProvider, SentenceTransformersProvider, CohereAIEmbeddingProvider, OpenAIEmbeddingProvider
import logging
from .config import Settings
from .chroma_client import ChromaManager

logger = logging.getLogger("xr.indexer")
logger.setLevel(logging.INFO)



class ChromaIndexer:
    def __init__(self, settings: Settings):
        """
        Create a Chroma client and a mapping for per-language providers.

        - persist_directory: path to persist Chroma DB (if None, uses in-memory)
        - collection_prefix: prefix for collection names
        """
        self.settings = settings
        persist_directory = self.settings.CHROMA_PERSIST_DIRECTORY
        self.collection_prefix = self.settings.COLLECTION_PREFIX
        self.LANG_EMBEDDING_MAP = self.settings.LANG_EMBEDDING_MAP
        self.client = ChromaManager(persist_directory)
        self._embedding_providers = {}  # key: (provider_name, lang, model_name)


    def get_provider_for_lang(self, lang, device="cuda"):
        """
        Return a cached provider instance for the language/provider/model combo.
        Creates it lazily on first request.
        """
        provider = self.LANG_EMBEDDING_MAP.get(lang)["provider"]
        model_name = self.LANG_EMBEDDING_MAP.get(lang)["model"]

        # create provider once
        if provider == "sentence_transformers":
            key = (provider, lang, model_name)
            if key in self._embedding_providers:
                return self._embedding_providers[key]
            provider_obj = SentenceTransformersProvider(model_name=model_name, device=device, batch_size=self.settings.EMBEDDING_BATCH_SIZE)

        elif provider == "cohere":  provider_obj = CohereAIEmbeddingProvider(model=model_name)
        elif provider == "openai":  provider_obj = OpenAIEmbeddingProvider(model=model_name)

        if provider =="sentence_transformers":    self._embedding_providers[key] = provider_obj
        return provider_obj


    def _collection_name(self, language: str, source: str, provider: str, model_name: str, chunking_method) -> str:
        """Returns a normalized Collection name"""
        lang = language.lower()
        src = source.lower().replace(" ", "_")
        model_name = model_name.replace("/", "_")
        return f"{self.collection_prefix}__{lang}__{src}__{model_name}__{chunking_method}"


    def ensure_collection(self, language: str, source: str, emb_provider_obj: Optional[EmbeddingProvider] = None, chunking_method: str = "default"):
        """
        Ensure a Chroma collection exists for language+source. If embedding_provider is provided,
        it will be stored in the collection's metadata for record-keeping (note: chroma collection cannot
        store complex objects — we store provider name in metadata only).
        """
        provider_name = getattr(emb_provider_obj, "provider", None)
        model_name = getattr(emb_provider_obj, "model_name", None) or getattr(emb_provider_obj, "model", None)

        name = self._collection_name(language, source, provider_name, model_name, chunking_method)
        collections = [c.name for c in self.client.list_collections()]

        if name in collections:
            logger.info("Collection already exists: %s", name)
            coll = self.client.get_collection(name)
        else:
            logger.info("Creating collection: %s", name)
            coll = self.client.create_collection(name)
            # collection created with embedding function in its name
        try:
            if provider_name:
                try:
                    coll.metadata = {
                        "embedding_provider": str(provider_name), "model": str(model_name),
                        "language": language, "source": source
                    }
                except Exception:
                    pass    # ignore if client version doesn't support
        except Exception:
            pass
        return coll


    # Deduplication
    @staticmethod
    def _checksum(string: str) -> str:
        return hashlib.sha1(string.encode("utf8")).hexdigest()


    def index_documents(
        self,
        docs: List[Dict[str, Any]], language: Optional[str] = None,
        language_field: str = "language",
        source_field: str = "source",
        text_field: str = "text",
        id_field: str = "id",
        title_field: str = "title",
        date_field: str = "date_publish",
        chunking_method: str = "context_aware_chunking",
        rebuild: bool = False,
        persist: bool = True,
        upsert_on_conflict: bool = True,
    ) -> Dict[str, Any]:
        """
        Streaming, memory-safe indexing into Chroma.

        Key points:
        - Streams documents group-by (language, source).
        - Chunks each document, computes checksum, filters duplicates (using a single existing_checksums set per collection).
        - Embeds small batches (self.settings.EMBEDDING_BATCH_SIZE) and upserts them immediately.
        - Frees memory after each batch (del + gc.collect()).
        - Optionally deletes collection when `rebuild=True`.
        """

        import gc
        import psutil
        from chromadb.errors import DuplicateIDError

        indexed = 0
        skipped = 0
        upserted_ids = []

        # Group by (language, source)
        groups = {}
        for d in docs:
            lang = d.get(language_field, "") or language
            src = d.get(source_field, "") or ""
            groups.setdefault((lang, src), []).append(d)

        batch_size = getattr(self.settings, "EMBEDDING_BATCH_SIZE", 8) or 8
        chunk_max_chars = getattr(self.settings, "CHUNK_MAX_CHARS", 500)
        overlap_sentences = getattr(self.settings, "CHUNK_OVERLAP_SENTENCES", 1)

        # helper to log memory
        def _log_mem(stage: str):
            p = psutil.Process()
            mem_mb = p.memory_info().rss / (1024 * 1024)
            logger.info(f"[MEM] {stage}: {mem_mb:.1f} MB")

        for (lang, src), group_docs in groups.items():
            logger.info(f"\nIndexing group: language={lang}, source={src} (#docs={len(group_docs)})")

            # get embedding provider for language (try cuda then cpu)
            device = "cuda"
            try:
                # check if cuda actually available on provider side — providers may ignore this arg
                provider = self.get_provider_for_lang(lang, device=device)
            except Exception:
                device = "cpu"
                provider = self.get_provider_for_lang(lang, device=device)

            # ensure collection exists
            col = self.ensure_collection(lang, src, provider, chunking_method)

            # handle rebuild: delete collection then recreate
            if rebuild:
                try:
                    name = col.name
                    logger.info(f"Rebuild requested: deleting collection {name}")
                    try:
                        self.client.delete_collection(name)
                    except Exception:
                        # older chroma clients might use client.delete_collection
                        try:
                            self.client.delete_collection(name)
                        except Exception:
                            pass
                    col = self.ensure_collection(lang, src, provider, chunking_method)
                except Exception as e:
                    logger.warning(f"Could not delete/recreate collection: {e}")

            # Fetch existing checksums once (fast de-dupe)
            existing_checksums = set()

            # include metadatas only to reduce bandwidth
            existing = col.get(include=["metadatas"])
            metadatas = existing.get("metadatas", []) if isinstance(existing, dict) else []
            # Some clients return nested lists: normalize
            if metadatas and isinstance(metadatas[0], list):
                flat_mds = metadatas[0]
            else:
                flat_mds = metadatas
            for m in flat_mds:
                if isinstance(m, dict) and "checksum" in m:
                    existing_checksums.add(m["checksum"])

            logger.info(f"Existing checksums loaded: {len(existing_checksums)}")
            _log_mem("after-load-checksums")

            # streaming buffers
            batch_texts = []
            batch_metadatas = []
            batch_ids = []
            batch_checksums = []

            def _flush_batch():
                nonlocal indexed, skipped, upserted_ids, batch_texts, batch_metadatas, batch_ids, batch_checksums
                if not batch_texts:
                    return
                try:
                    # embed
                    logger.info(f"Embedding batch size = {len(batch_texts)} [Language={lang}, Source={src}]")
                    embeddings = provider.embed_documents(batch_texts)

                    # use upsert if available (idempotent and avoids duplicate id errors)
                    if hasattr(col, "upsert"):
                        col.upsert(documents=batch_texts, metadatas=batch_metadatas, ids=batch_ids, embeddings=embeddings)
                    else:
                        try:
                            col.add(documents=batch_texts, metadatas=batch_metadatas, ids=batch_ids, embeddings=embeddings)
                        except DuplicateIDError:
                            # fallback: attempt add per-item, skipping duplicates
                            for i in range(len(batch_texts)):
                                try:
                                    col.add(documents=[batch_texts[i]], metadatas=[batch_metadatas[i]], ids=[batch_ids[i]], embeddings=[embeddings[i]])
                                except Exception:
                                    # skip duplicates or failures
                                    continue

                    # update counters & checksum set
                    indexed += len(batch_texts)
                    upserted_ids.extend(batch_ids)
                    existing_checksums.update(batch_checksums)

                    logger.info(f"Indexed batch: +{len(batch_texts)} (total indexed={indexed})")
                except Exception as e:
                    logger.warning(f"Failed to upsert/add batch to Chroma: {e}")
                finally:
                    # free memory
                    del batch_texts, batch_metadatas, batch_ids, batch_checksums
                    gc.collect()
                    # recreate empty buffers
                    batch_texts = []
                    batch_metadatas = []
                    batch_ids = []
                    batch_checksums = []
                    _log_mem("after-flush")

            # choose chunking method
            try:
                if chunking_method == "paragraph_chunking":
                    chunker = ParagraphChunker(min_chars=200)
                elif chunking_method == "token_chunking":
                    chunker = TokenChunker(chunk_size=512, stride=128)
                elif chunking_method == "sliding_window_chunking":
                    chunker = SlidingWindowChunker(chunk_size=512, overlap=128)
                elif chunking_method == "sentence_chunking":
                    chunker = SentenceChunker(min_tokens=5)

            except Exception as e:
                logger.warning(f"Chunking failed for doc {doc_idx} (lang={lang}): {e}")
                # fallback single chunk
                chunks = [(raw_text, {})]

            # iterate docs and create chunks
            for doc_idx, doc in enumerate(group_docs):
                raw_text = doc.get(text_field, "") or doc.get("context", "")
                if not raw_text:
                    skipped += 1
                    continue

                title = doc.get(title_field, "")
                url = doc.get("url", "")
                date = doc.get(date_field, "")

                # deterministic doc_id
                raw_id = f"{raw_text[:30]}|{lang}|{doc.get(id_field, '')}"
                doc_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()

                # choose chunking method
                try:
                    if chunking_method == "context_aware_chunking":
                        chunks = context_aware_chunking(
                            doc,
                            max_chars=self.settings.CHUNK_MAX_CHARS,
                            overlap_sentences=self.settings.CHUNK_OVERLAP_SENTENCES,
                        )
                    elif chunking_method == "paragraph_chunking":
                        chunks = chunker.chunk(doc)
                    elif chunking_method == "token_chunking":
                        chunks = chunker.chunk(doc)
                    elif chunking_method == "sliding_window_chunking":
                        chunks = chunker.chunk(doc)
                    elif chunking_method == "sentence_chunking":
                        chunks = chunker.chunk(doc)

                except Exception as e:
                    logger.warning(f"Chunking failed for doc {doc_idx} (lang={lang}): {e}")
                    # fallback single chunk
                    chunks = [(raw_text, {})]


                for cidx, (chunk_text, meta_partial) in enumerate(chunks):
                    checksum = self._checksum(chunk_text)
                    if checksum in existing_checksums:
                        skipped += 1
                        continue

                    uid = f"{doc_id}__chunk_{cidx}__{checksum[:12]}"
                    meta = {
                        "doc_id": doc_id,
                        "chunk_index": cidx,
                        "title": title,
                        "url": url,
                        "language": lang,
                        "source": src,
                        "checksum": checksum,
                        "date": date,
                    }
                    meta.update(meta_partial or {})

                    # append to batch
                    batch_texts.append(chunk_text)
                    batch_metadatas.append(meta)
                    batch_ids.append(uid)
                    batch_checksums.append(checksum)

                    # flush if batch full
                    if len(batch_texts) >= batch_size:
                        _flush_batch()

                # after finishing doc, flush leftover small batch if it grew big (optional optimization)
                # but we keep it to EMBEDDING_BATCH_SIZE threshold

            # flush remaining for this group
            _flush_batch()

            # persist client if requested (helps durability)
            if persist:
                try:
                    if hasattr(self.client, "persist"):
                        try:
                            self.client.persist()
                        except Exception:
                            # some versions accept client.persist() differently
                            pass
                    logger.info("Chroma client persisted (if supported).")
                except Exception as e:
                    logger.warning(f"Failed to persist Chroma client: {e}")

        summary = {"indexed_chunks": int(indexed), "skipped": int(skipped), "upserted_ids_count": len(upserted_ids)}
        return summary


    def list_collections(self) -> List[str]:
        return [c.name for c in self.client.list_collections()]

    def delete_collection(self, language: str, source: str, emb_provider_obj: Optional[EmbeddingProvider] = None):
        provider_name = getattr(emb_provider_obj, "provider", None)
        model_name = getattr(emb_provider_obj, "model_name", None) or getattr(emb_provider_obj, "model", None)

        name = self._collection_name(language, source, provider_name, model_name)
        logger.info("\n🔴 Deleting collection %s", name)

        return self.client.delete_collection(name)