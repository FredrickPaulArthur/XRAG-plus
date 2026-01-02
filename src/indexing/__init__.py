"""
XRAG+ indexing module (package init)

Exports:
- ChromaIndexer: primary class to create collections and index documents
- EmbeddingProvider: abstract base for pluggable embedding providers
- SentenceTransformersProvider, OpenAIEmbeddingProvider: example providers
"""

from .indexer import ChromaIndexer
from .embeddings import (
    EmbeddingProvider,
    SentenceTransformersProvider,
    OpenAIEmbeddingProvider,
    CohereAIEmbeddingProvider
)

__all__ = [
    "ChromaIndexer",
    "EmbeddingProvider",
    "SentenceTransformersProvider",
    "OpenAIEmbeddingProvider",
    "CohereAIEmbeddingProvider"
]