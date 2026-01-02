"""
Configuration for XRAG+ indexing (src/indexing).

Place runtime configuration here so `src/indexing` modules can import it.

This file exposes a single `settings` object (an instance of Settings) with
easy-to-edit fields. You can replace this with environment-driven loading
(e.g., pydantic, dynaconf) if you prefer.

Notes
-----
- Keep language keys short (e.g. "en", "hi", "ru"). Use the "default" entry
  to fall back when a language-specific entry is missing.
- Providers are treated as identifiers used by src/indexing.embeddings to
  instantiate the correct provider class (e.g. "sentence_transformers",
  "openai", "huggingface").
- Many values can be overridden via environment variables if you prefer
  runtime config (see `from_env()` below).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class Settings:
    # Chroma database persistence
    CHROMA_PERSIST_DIRECTORY: Optional[str] = field(default="./.chroma_db")

    # Prefix applied to all collection names to keep your project's collections isolated.
    COLLECTION_PREFIX: str = "xrag_collection"

    DEFAULT_EMBEDDING_PROVIDER: str = "sentence_transformers"

    DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    LANG_EMBEDDING_MAP = {
        "en": {
            "provider": "sentence_transformers",
            "model": "all-MiniLM-L6-v2"
        },
        # "en": {
        #     "provider": "cohere",
        #     "model": "embed-multilingual-v3.0"
        # },
        "hi": {
            "provider": "sentence_transformers",
            "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        },
        "ru": {
            "provider": "sentence_transformers", 
            "model": "paraphrase-multilingual-mpnet-base-v2"
        },
        "de": {
            "provider": "sentence_transformers",
            "model": "all-MiniLM-L6-v2"
        },
        "es": {
            "provider": "sentence_transformers",
            "model": "all-MiniLM-L6-v2"
        },
        # Using "default" value for any other languages
        "default": {
            "provider": "openai",
            "model": "text-embedding-3-small"
        }
    }


    # Max tokens per chunk (approx). The indexer converts to chars by APPROX_CHARS_PER_TOKEN.
    CHUNK_MAX_CHARS = 512
    CHUNK_OVERLAP_SENTENCES= 1

    # -------------------------
    # Indexing runtime parameters
    # -------------------------
    # Textual items to send to the embedding provider per batch, for OPENAI API safe.
    EMBEDDING_BATCH_SIZE: int = 32

    # INDEX_BATCH_SIZE: int = 64  # How many chunks to add/upsert to Chroma in a single operation (should be tuned with memory)

    DEDUP_ENABLED: bool = True  # Whether to attempt deduplication before indexing (true recommended)

    # Deduplication strategy:
    #  - "checksum" : compute sha1 of chunk text and compare with collection metadata (simple)
    #  - "external_db" : expects DEDUP_EXTERNAL_DB_URL configured and external store used
    DEDUP_METHOD: str = "checksum"

    # If using external DB for dedup, set connection/DSN here (optional)
    DEDUP_EXTERNAL_DB_URL: Optional[str] = None

    # Enable a simple on-disk cache of embeddings keyed by checksum to avoid re-embedding identical chunks
    EMBEDDING_CACHE_ENABLED: bool = True
    EMBEDDING_CACHE_DIR: str = "./chromadb_persist/embedding_cache"

    # Upsert behavior: try add(), if fails fallback to upsert()
    UPSERT_ON_CONFLICT: bool = True

    # Verbosity / logging
    VERBOSE: bool = True
    LOG_LEVEL: str = "INFO"

    REQUIRED_METADATA_FIELDS: tuple = ("doc_id", "chunk_index", "title", "url", "language", "source", "checksum")


    def to_dict(self) -> Dict[str, Any]:
        pass