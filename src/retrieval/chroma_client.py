### File: src/retrieval/chroma_client.py
"""
ChromaManager: helper around chromadb Client and collections.
It handles creating collections, persisting, and simple queries.
"""
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings as ChromaSettings
import os


@dataclass
class ChromaManager:
    persist_directory: str = "./.chroma_db"
    _client: Optional[chromadb.PersistentClient] = None

    def _ensure_client(self):
        if self._client is None:
            os.makedirs(self.persist_directory, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_directory)
        return self._client

    def create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None, embedding_function=None):
        client = self._ensure_client()
        return client.create_collection(name=name, metadata=metadata, embedding_function=embedding_function)

    def get_or_create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None, embedding_function=None):
        client = self._ensure_client()
        return client.get_or_create_collection(name=name, metadata=metadata, embedding_function=embedding_function)

    def get_collection(self, name: str):
        client = self._ensure_client()
        return client.get_collection(name)

    def list_collections(self):
        client = self._ensure_client()
        return [c["name"] for c in client.list_collections()]