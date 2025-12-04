### File: src/retrieval/retriever.py
"""
Retrieve from chroma collection using embedding similarity.
"""
from typing import List, Dict, Optional
from .chroma_client import ChromaManager
from .embeddings import embed
from .config import Settings


class Retriever:
    def __init__(self, settings: Settings = None, chroma_manager: ChromaManager = None, language = "default"):
        self.settings = settings or Settings()
        self.chroma_manager = chroma_manager or ChromaManager(persist_directory=self.settings.CHROMA_PERSIST_DIR)
        self.language = language
        # self.embedding_client = embedding_client or EmbeddingClient(model_name=self.settings.DEFAULT_EMBEDDING_MODEL)

    def retrieve(self, collection_name: str, query: str, k: int = 5, where: Optional[Dict] = None) -> Dict:
        query_emb = embed(
            texts=[query],
            provider=self.settings.LANG_EMBEDDING_MAP.get(self.language)["provider"],
            model_name=self.settings.LANG_EMBEDDING_MAP.get(self.language)["model"]
        )
        collection = self.chroma_manager.get_collection(collection_name)

        results = collection.query(query_embeddings=query_emb, n_results=k, where=where)

        # results is a dict with keys: ids, distances, metadatas, documents (lists per query)
        final_result = {
            "ids": results.get("ids", [])[0] if results.get("ids") else [],
            "distances": results.get("distances", [])[0] if results.get("distances") else [],
            "metadatas": results.get("metadatas", [])[0] if results.get("metadatas") else [],
            "documents": results.get("documents", [])[0] if results.get("documents") else [],
        }

        return final_result