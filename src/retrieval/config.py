### File: src/retrieval/config.py
"""
Configuration and defaults for retrieval.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List



@dataclass
class Settings:
    RETRIEVAL_METHODS: List[str] = field(
        default_factory=lambda: ["semantic", "keyword", "hybrid"]
    )
    CHROMA_PERSIST_DIR: str = "./.chroma_db"

    DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    DEFAULT_EMBEDDING_PROVIDER: str = "sentence_transformers"

    SEMANTIC_CHUNKING_MODEL: str = "all-MiniLM-L6-v2"
    SEMANTIC_CHUNKING_PROVIDER: str = "sentence_transformers"
    SEMANTIC_CHUNKING_THRESHOLD: float = 0.7

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    LANG_EMBEDDING_MAP = {
        # Best for English
        "en": {
            "provider": "sentence_transformers",
            "model": "BAAI/bge-large-en-v1.5"
        },
        # Best for multilingual - OpenAI
        'de': {
            'model': 'all-MiniLM-L6-v2',
            'provider': 'sentence_transformers'
        },
        'en': {
            'model': 'all-MiniLM-L6-v2',
            'provider': 'sentence_transformers'
        },
        'es': {
            'model': 'all-MiniLM-L6-v2',
            'provider': 'sentence_transformers'
        },
        'hi': {
            'model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'provider': 'sentence_transformers'
        },
        'ru': {
            'model': 'paraphrase-multilingual-mpnet-base-v2',
            'provider': 'sentence_transformers'
        },
        'default': {
            'model': 'text-embedding-3-small',
            'provider': 'openai'
        }
    }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "RETRIEVAL_METHODS": self.RETRIEVAL_METHODS,
            "CHROMA_PERSIST_DIR": self.CHROMA_PERSIST_DIR,
            "DEFAULT_EMBEDDING_MODEL": self.DEFAULT_EMBEDDING_MODEL,
            "DEFAULT_EMBEDDING_PROVIDER": self.DEFAULT_EMBEDDING_PROVIDER,
            "SEMANTIC_CHUNKING_MODEL": self.SEMANTIC_CHUNKING_MODEL,
            "SEMANTIC_CHUNKING_PROVIDER": self.SEMANTIC_CHUNKING_PROVIDER,
            "SEMANTIC_CHUNKING_THRESHOLD": self.SEMANTIC_CHUNKING_THRESHOLD,
            "LANG_EMBEDDING_MAP": self.LANG_EMBEDDING_MAP,
            "CHUNK_SIZE": self.CHUNK_SIZE,
            "CHUNK_OVERLAP": self.CHUNK_OVERLAP,
        }