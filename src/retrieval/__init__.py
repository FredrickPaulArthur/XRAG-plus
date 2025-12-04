# XRAG+ - src/retrieval
# This single file contains a multi-file layout for the `src/retrieval` package.
# Each file is separated by a header line beginning with "### File: <path>".
# Copy each file into your repository under the indicated path.


### File: src/retrieval/__init__.py
"""
XRAG+ retrieval package (Chroma-backed)
"""
from .config import Settings
from .chroma_client import ChromaManager
from .embeddings import embed
from .indexer import index_documents
from .retriever import Retriever


__all__ = [
    "Settings",
    "ChromaManager",
    "embed",
    "index_documents",
    "Retriever",
]