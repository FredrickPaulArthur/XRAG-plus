### File: src/retrieval/config.py
"""
Configuration and defaults for retrieval.
"""
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Settings:
    RETRIEVAL_METHODS = ["semantic", "keyword", "hybrid"]

    CHROMA_PERSIST_DIR: str = "./.chroma_db"

    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_EMBEDDING_PROVIDER = "sentence_transformers"

    SEMANTIC_CHUNKING_MODEL = "all-MiniLM-L6-v2"
    SEMANTIC_CHUNKING_PROVIDER = "sentence_transformers"
    SEMANTIC_CHUNKING_THRESHOLD = 0.7

    LANG_EMBEDDING_MAP: Dict[str, str] = None

    CHUNK_SIZE: int = 1000      # simple char-based chunking
    CHUNK_OVERLAP: int = 200


    def __post_init__(self):
        if self.LANG_EMBEDDING_MAP is None:     # sensible defaults — can be overridden in runtime
            self.LANG_EMBEDDING_MAP = {
                # # Best for English
                # "en": {
                #     "provider": "sentence_transformers",
                #     "model": "BAAI/bge-large-en-v1.5"
                # },
                # Best for multilingual - OpenAI
                "en": {
                    "provider": "sentence_transformers",
                    "model": "all-MiniLM-L6-v2"
                },
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "CHROMA_PERSIST_DIR": self.CHROMA_PERSIST_DIR,
            "DEFAULT_EMBEDDING_MODEL": self.DEFAULT_EMBEDDING_MODEL,
            "DEFAULT_EMBEDDING_PROVIDER": self.DEFAULT_EMBEDDING_PROVIDER,
            "SEMANTIC_CHUNKING_MODEL": self.SEMANTIC_CHUNKING_MODEL,
            "SEMANTIC_CHUNKING_PROVIDER": self.SEMANTIC_CHUNKING_PROVIDER,
            "LANG_EMBEDDING_MAP": self.LANG_EMBEDDING_MAP,
            "CHUNK_SIZE": self.CHUNK_SIZE,
            "CHUNK_OVERLAP": self.CHUNK_OVERLAP
        }