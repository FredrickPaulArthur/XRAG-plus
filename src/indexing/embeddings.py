"""
Embedding providers for XRAG+ indexing.

- EmbeddingProvider: abstract class
- SentenceTransformersProvider: uses sentence-transformers (local)
- OpenAIEmbeddingProvider: uses OpenAI embeddings (optional)
- CohereAIEmbeddingProvider: uses Cohere embeddings (optional)

Each provider implements embed_documents(List[str]) -> List[List[float]].

Make sure optional deps are installed in your environment when using the provider:
- sentence-transformers: pip install sentence-transformers
- openai: pip install openai
"""

from abc import ABC, abstractmethod
from typing import List, Iterable, Optional
import os
import torch, gc
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
load_dotenv()





class EmbeddingProvider(ABC):
    """Abstract base: implement embed_documents(list[str]) -> list[list[float]]"""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError



class SentenceTransformersProvider(EmbeddingProvider):
    def __init__(self, model_name: str, device: Optional[str] = None, batch_size: int = 16):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.provider = "sentence_transformer"
        self.device = device
        self.batch_size = batch_size

        # load explicitly on CPU first to avoid OOM during initialization
        self.model = SentenceTransformer(model_name, device="cpu")

        if self.device != "cpu":
            # try to reduce GPU memory pressure before moving
            gc.collect()
            torch.cuda.empty_cache()
            try:
                self.model.to(self.device)
            except Exception:           # fallback to CPU if move fails
                self.model.to("cpu")
                self.device = "cpu"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Encodes in small batches and avoid passing device to encode() - model already loaded on device in __init__
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            emb = self.model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            embeddings.append(emb)
        import numpy as np
        embeddings = np.vstack(embeddings)
        return embeddings.tolist()



class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Uses api_key from .env if not explicitly specified.
    Batch size is specified inside the embed method for safety - OpenAI API has rate/size limits."""
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        try:
            import openai
        except Exception as e:
            raise ImportError("openai package is required for OpenAIEmbeddingProvider") from e

        self.model = model
        self.provider = "openai"
        self.batch_size = 32
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment or passed to provider")
        openai.api_key = api_key
        self._openai = openai

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Batching for safety — OpenAI API has rate/size limits.
        res = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            r = self._openai.Embeddings.create(model=self.model, input=batch)
            vectors = [item["embedding"] for item in r["data"]]
            res.extend(vectors)
        return res



class CohereAIEmbeddingProvider(EmbeddingProvider):
    """
    Uses api_key from .env if not explicitly specified
    TODO:
        To implement usage of input_type="search_document" for indexing and input_type="search_query" when embedding user queries.
    """
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
            input_type="search_document",   # default for RAG document storage
            batching=True
        ).embeddings     # /embed-multilingual-v3.0
        return emb