"""
Embedding providers for XRAG+ indexing.

- EmbeddingProvider: abstract class
- SentenceTransformersProvider: uses sentence-transformers (local)
- OpenAIEmbeddingProvider: uses OpenAI embeddings (optional)

Each provider implements embed_documents(List[str]) -> List[List[float]].

Make sure optional deps are installed in your environment when using the provider:
- sentence-transformers: pip install sentence-transformers
- openai: pip install openai
"""

from abc import ABC, abstractmethod
from typing import List, Iterable, Optional
import os
import hashlib
from dotenv import load_dotenv
load_dotenv()





class EmbeddingProvider(ABC):
    """Abstract base: implement embed_documents(list[str]) -> list[list[float]]"""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError



class SentenceTransformersProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise ImportError("sentence-transformers is required for SentenceTransformersProvider") from e

        self.model_name = model_name
        self.provider = "sentence_transformer"
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()



class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Uses api_key from .env if not explicitly specified"""
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        try:
            import openai
        except Exception as e:
            raise ImportError("openai package is required for OpenAIEmbeddingProvider") from e

        self.model = model
        self.provider = "openai"
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment or passed to provider")
        openai.api_key = api_key
        self._openai = openai

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Batching for safety — OpenAI API has rate/size limits.
        batch_size = 32
        res = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            r = self._openai.Embeddings.create(model=self.model, input=batch)
            vectors = [item["embedding"] for item in r["data"]]
            res.extend(vectors)
        return res



class CohereAIEmbeddingProvider(EmbeddingProvider):
    """Uses api_key from .env if not explicitly specified"""
    def __init__(self, model: str = "embed-multilingual-v3.0", api_key: str = None):
        try:
            import cohere
            self.co_client = cohere.Client(api_key or os.getenv("COHERE_KEY"))
        except Exception as e:
            raise ImportError("cohere package is required for CohereAIEmbeddingProvider") from e

        self.model = model
        self.provider = "cohere"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        emb = self.co_client.embed(
            model=self.model,
            texts=texts,
            input_type="search_document"   # default for RAG document storage
        ).embeddings     # /embed-multilingual-v3.0
        return emb