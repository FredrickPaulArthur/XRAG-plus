### File: src/retrieval/indexer.py
"""
Indexing utilities. Given raw documents, chunk them, embed them, and add them to Chroma.
"""
from typing import List, Dict, Optional
import uuid
import logging
from .utils import clean_text
from .chunking import fixed_width_chunking, semantic_chunking
from .embeddings import embed
from .chroma_client import ChromaManager
from .config import Settings

logger = logging.getLogger(__name__)



def index_documents(
    collection_name: str,
    chunk_method,
    documents: List[str],
    metadatas: Optional[List[Dict]] = None,
    ids: Optional[List[str]] = None,
    settings: Settings = None,
    chroma_manager: Optional[ChromaManager] = None,
    language: str = "default",
):
    """
    Index documents into Chroma. Documents will be chunked and each chunk added as a separate item.

    - documents: list of raw text strings
    - metadatas: optional list of metadata dicts, aligned with documents (per *document*, not per chunk)
    - ids: optional list of base ids per document
    - settings: Settings instance for chunk sizes
    - embedding_client: EmbeddingClient instance
    - chroma_manager: ChromaManager instance
    - language: language code to pick embedding model
    """
    if settings is None:
        settings = Settings()

    emb_model = settings.LANG_EMBEDDING_MAP.get(language)["model"] or settings.DEFAULT_EMBEDDING_MODEL
    provider = settings.LANG_EMBEDDING_MAP.get(language)["provider"] or settings.DEFAULT_EMBEDDING_PROVIDER

    if chroma_manager is None:
        chroma_manager = ChromaManager(persist_directory=settings.CHROMA_PERSIST_DIR)

    collection = chroma_manager.get_or_create_collection(collection_name)

    to_add_ids = []
    to_add_docs = []
    to_add_metadatas = []
    to_add_embs = []

    for i, doc in enumerate(documents):
        base_meta = metadatas[i] if metadatas and i < len(metadatas) else {}
        base_id = ids[i] if ids and i < len(ids) else str(uuid.uuid4())
        
        if chunk_method == fixed_width_chunking:
            chunks = fixed_width_chunking(doc, chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)

        elif chunk_method == semantic_chunking:
            chunks = semantic_chunking(
                text=doc,
                model=settings.SEMANTIC_CHUNKING_MODEL,
                provider=settings.SEMANTIC_CHUNKING_PROVIDER,
                threshold=0.7
            )

        # Build per-chunk ids and metadata
        if not chunks:
            continue

        # embed all chunks in batch
        print("Chunks: ", chunks)
        emb_chunks = embed(
            chunks,
            provider=provider,
            model_name=emb_model
        )
        for j, chunk in enumerate(chunks):
            chunk_id = f"{base_id}:chunk:{j}"
            meta = {**base_meta, "source_doc_id": base_id, "chunk_index": j, "text_length": len(chunk)}
            to_add_ids.append(chunk_id)
            to_add_docs.append(chunk)
            to_add_metadatas.append(meta)
            to_add_embs.append(emb_chunks[j])

    # Add in batch
    if to_add_ids:
        collection.add(ids=to_add_ids, documents=to_add_docs, metadatas=to_add_metadatas, embeddings=to_add_embs)

    return {
        "n_indexed": len(to_add_ids),
        "collection": collection_name,
    }
