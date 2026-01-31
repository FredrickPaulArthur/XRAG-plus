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


from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class Settings:
    CHROMA_PERSIST_DIRECTORY: Optional[str] = field(default="./.chroma_db")
    COLLECTION_PREFIX: str = "xrag_collection"

    DEFAULT_EMBEDDING_PROVIDER: str = "sentence_transformers"
    DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    LANG_EMBEDDING_MAP: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: {
            "en": {
                "provider": "sentence_transformers",
                "model": "all-MiniLM-L6-v2",
            },
            # "en": {
            #     "provider": "cohere",
            #     "model": "embed-multilingual-v3.0"
            # },
            "hi": {
                "provider": "sentence_transformers",
                "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            },
            "ru": {
                "provider": "sentence_transformers",
                "model": "paraphrase-multilingual-mpnet-base-v2",
            },
            "de": {
                "provider": "sentence_transformers",
                "model": "all-MiniLM-L6-v2",
            },
            "es": {
                "provider": "sentence_transformers",
                "model": "all-MiniLM-L6-v2",
            },
            "default": {
                "provider": "openai",
                "model": "text-embedding-3-small",
            },
        }
    )


    # Max tokens per chunk (approx).
    CHUNK_MAX_CHARS: int = 1000
    CHUNK_OVERLAP_SENTENCES: int = 2

    # -------------------------
    # Indexing runtime parameters
    # -------------------------
    # Textual items to send to the embedding provider per batch, for OPENAI API safe.
    # How many chunks to add/upsert to Chroma in a single operation (should be tuned with memory)
    EMBEDDING_BATCH_SIZE: int = 64

    DEDUP_ENABLED: bool = True  # Attempts deduplication before indexing (true recommended)

    # Deduplication strategy - "checksum" : compute sha1 of chunk text and compare with collection metadata (simple)
    DEDUP_METHOD: str = "checksum"

    # TODO: To implement caching inside the same Directory.
    # # Enable a simple on-disk cache of embeddings keyed by checksum to avoid re-embedding identical chunks
    # EMBEDDING_CACHE_ENABLED: bool = True
    # EMBEDDING_CACHE_DIR: str = "./chromadb_persist/embedding_cache"

    # # Upsert behavior: try add(), if fails fallback to upsert()
    # UPSERT_ON_CONFLICT: bool = True

    # Verbosity / logging
    VERBOSE: bool = True
    LOG_LEVEL: str = "INFO"

    REQUIRED_METADATA_FIELDS: tuple = ("doc_id", "chunk_index", "title", "url", "language", "source", "checksum")


    def to_dict(self) -> Dict[str, Any]:
        return {
            "CHROMA_PERSIST_DIRECTORY": self.CHROMA_PERSIST_DIRECTORY,
            "COLLECTION_PREFIX": self.COLLECTION_PREFIX,
            "DEFAULT_EMBEDDING_PROVIDER": self.DEFAULT_EMBEDDING_PROVIDER,
            "DEFAULT_EMBEDDING_MODEL": self.DEFAULT_EMBEDDING_MODEL,
            "LANG_EMBEDDING_MAP": self.LANG_EMBEDDING_MAP,
            "CHUNK_MAX_CHARS": self.CHUNK_MAX_CHARS,
            "CHUNK_OVERLAP_SENTENCES": self.CHUNK_OVERLAP_SENTENCES,
            "EMBEDDING_BATCH_SIZE": self.EMBEDDING_BATCH_SIZE,
            "DEDUP_ENABLED": self.DEDUP_ENABLED,
            "DEDUP_METHOD": self.DEDUP_METHOD,
            # "EMBEDDING_CACHE_ENABLED": self.EMBEDDING_CACHE_ENABLED,
            # "EMBEDDING_CACHE_DIR": self.EMBEDDING_CACHE_DIR,
            # "UPSERT_ON_CONFLICT": self.UPSERT_ON_CONFLICT,
            "VERBOSE": self.VERBOSE,
            "LOG_LEVEL": self.LOG_LEVEL,
            "REQUIRED_METADATA_FIELDS": self.REQUIRED_METADATA_FIELDS,
        }