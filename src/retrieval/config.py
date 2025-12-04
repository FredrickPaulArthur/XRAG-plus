### File: src/retrieval/config.py
"""
Configuration and defaults for retrieval.
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class Settings:
    CHROMA_PERSIST_DIR: str = "./.chroma_db"

    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_EMBEDDING_PROVIDER = "sentence_transformers"

    SEMANTIC_CHUNKING_MODEL = "all-MiniLM-L6-v2"
    SEMANTIC_CHUNKING_PROVIDER = "sentence_transformers"

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
                    "provider": "cohere",
                    "model": "embed-multilingual-v3.0"
                },
                "es": {
                    "provider": "cohere",
                    "model": "embed-multilingual-v3.0"
                },
                "de": {
                    "provider": "cohere",
                    "model": "embed-multilingual-v3.0"
                },
                "hi": {
                    "provider": "cohere",
                    "model": "embed-multilingual-v3.0"
                },
                "ru": {
                    "provider": "cohere",
                    "model": "embed-multilingual-v3.0"
                }
            }