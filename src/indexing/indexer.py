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
from chromadb.config import Settings
import hashlib
import time
import math

from src.chunker.chunkers import context_aware_chunking, llm_based_chunking, ParagraphChunker, SentenceChunker, SlidingWindowChunker, TokenChunker
from src.indexing.embeddings import EmbeddingProvider, SentenceTransformersProvider, CohereAIEmbeddingProvider, OpenAIEmbeddingProvider
import logging
from .config import Settings
from .chroma_client import ChromaManager

logger = logging.getLogger("xr.indexer")
logger.setLevel(logging.INFO)


# -----------------------------
# ChromaIndexer
# -----------------------------
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
        self.lang_provider_map = self.settings.LANG_EMBEDDING_MAP
        self.client = ChromaManager(persist_directory)

    def get_provider_for_lang(self, lang: str) -> EmbeddingProvider:
        provider = self.lang_provider_map.get(lang)["provider"]
        model_name = self.lang_provider_map.get(lang)["model"]

        if provider == "sentence_transformers": provider_obj = SentenceTransformersProvider(model_name=model_name)
        elif provider == "cohere":  provider_obj = CohereAIEmbeddingProvider(model=model_name)
        elif provider == "openai":  provider_obj = OpenAIEmbeddingProvider(model=model_name)

        if provider_obj:    return provider_obj

        raise ValueError(f"No embedding provider registered for language '{lang}', and no default provided.")


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
    def _checksum(text: str) -> str:
        return hashlib.sha1(text.encode("utf8")).hexdigest()


    # Main Indexing Method
    def index_documents(
        self, docs: List[Dict[str, Any]], *,
        language_field: str = "language",
        source_field: str = "source",
        text_field: str = "text",
        id_field: str = "id",
        title_field: str = "title",
        chunking_method = "context_aware_chunking"
    ) -> Dict[str, Any]:
        """
        Indexing a list of documents. Each doc must contain:
            - id
            - text
            - language
            - source (e.g. 'wiki', 'ccnews')

        Returns summary dict with counts.
        """
        indexed = 0
        skipped = 0
        upserted_ids = []

        # Group by language+source to create separate collection per group
        groups = {}
        for d in docs:
            lang = d.get(language_field)
            src = d.get(source_field)
            key = (lang, src)
            groups.setdefault(key, []).append(d)

        for (lang, src), group_docs in groups.items():
            print(f"Document language and source : ({lang}, {src})")

            emb_provider_obj = self.get_provider_for_lang(lang)
            col = self.ensure_collection(lang, src, emb_provider_obj, chunking_method)

            # prepare all chunks for this group
            chunk_texts = []
            chunk_metadatas = []
            chunk_ids = []

            for doc in group_docs:
                doc_id = str(doc.get(id_field, hashlib.sha1((doc.get(text_field, "") + str(time.time())).encode()).hexdigest()))
                text = doc.get(text_field, "")
                title = doc.get(title_field, "") or ""
                url = doc.get("url", "")

                if chunking_method == "context_aware_chunking":
                    chunks = context_aware_chunking(
                        doc, max_chars=self.settings.CHUNK_MAX_CHARS,
                        overlap_sentences=self.settings.CHUNK_OVERLAP_SENTENCES,
                    )

                elif chunking_method == "paragraph_chunking":
                    paragraph_chunker = ParagraphChunker(min_chars=200)
                    chunks = paragraph_chunker.chunk(doc)

                elif chunking_method == "token_chunking":
                    import spacy
                    token_chunker = TokenChunker(
                        tokenizer=spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"]),
                        chunk_size=10,
                        stride=1
                    )
                    chunks = token_chunker.chunk(doc)

                elif chunking_method == "sliding_window_chunking":
                    sliding_chunker = SlidingWindowChunker(chunk_size=12, overlap=4)
                    chunks = sliding_chunker.chunk(doc)

                elif chunking_method == "sentence_chunking":
                    sentence_chunker = SentenceChunker(min_tokens=5)
                    chunks = sentence_chunker.chunk(doc)


                for idx, (chunk_text, meta_partial) in enumerate(chunks):
                    checksum = self._checksum(chunk_text)
                    uid = f"{doc_id}__chunk_{idx}__{checksum[:12]}"
                    meta = {
                        "doc_id": doc_id,
                        "chunk_index": idx,
                        "title": title,
                        "url": url,
                        "language": lang,
                        "source": src,
                        "checksum": checksum,
                    }
                    meta.update(meta_partial)

                    chunk_texts.append(chunk_text)
                    chunk_metadatas.append(meta)
                    chunk_ids.append(uid)

            if self.settings.DEDUP_ENABLED and len(chunk_metadatas) > 0:
                existing_checksums = set()

                existing = col.get(include=["metadatas"])
                for m in existing.get("metadatas", []):
                    if isinstance(m, dict) and "checksum" in m:
                        existing_checksums.add(m["checksum"])

                # Filtering out duplicates
                filtered_texts, filtered_metadatas, filtered_ids = [], [], []
                for t, m, i_ in zip(chunk_texts, chunk_metadatas, chunk_ids):
                    if m.get("checksum") in existing_checksums:
                        skipped += 1
                        continue
                    filtered_texts.append(t)
                    filtered_metadatas.append(m)
                    filtered_ids.append(i_)
                chunk_texts, chunk_metadatas, chunk_ids = filtered_texts, filtered_metadatas, filtered_ids

            # Embedding and adding in batches
            for i in range(0, len(chunk_texts), self.settings.EMBEDDING_BATCH_SIZE):
                index = i + self.settings.EMBEDDING_BATCH_SIZE
                batch_texts = chunk_texts[i:index]
                batch_meta = chunk_metadatas[i:index]
                batch_ids = chunk_ids[i:index]

                if not batch_texts:
                    continue

                logger.info(f"Embedding batch (lang={lang}, source={src}) size={batch_texts}")

                embeddings = emb_provider_obj.embed_documents(batch_texts)

                col.add(        # Add/upsert to collection
                    documents=batch_texts,
                    metadatas=batch_meta,
                    ids=batch_ids,
                    embeddings=embeddings,
                )
                indexed += len(batch_texts)
                upserted_ids.extend(batch_ids)

        return {"indexed_chunks": indexed, "skipped": skipped, "upserted_ids_count": len(upserted_ids)}

    def list_collections(self) -> List[str]:
        return [c.name for c in self.client.list_collections()]

    def delete_collection(self, language: str, source: str, emb_provider_obj: Optional[EmbeddingProvider] = None):
        provider_name = getattr(emb_provider_obj, "provider", None)
        model_name = getattr(emb_provider_obj, "model_name", None) or getattr(emb_provider_obj, "model", None)

        name = self._collection_name(language, source, provider_name, model_name)
        logger.info("\n⚠️ Deleting collection %s", name)

        return self.client.delete_collection(name)