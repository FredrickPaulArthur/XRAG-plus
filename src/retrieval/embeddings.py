"""
EmbeddingClient: wrapper to produce numeric embeddings.
Supports sentence-transformers (local) or OpenAI (optional).
"""
from typing import List, Sequence, Optional
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Lazy import to avoid heavy imports unless used


def embed(texts: Sequence[str], provider, model_name):
    """Return list of embeddings for given texts.
    Uses sentence-transformers by default.
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    if provider == "sentence_transformers":
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformers model %s...", model_name)
        emb_client = SentenceTransformer(model_name)

        # model.encode returns numpy array
        arr = emb_client.encode(texts, show_progress_bar=False)
        return arr

    elif provider == "cohere":    
        import cohere
        co = cohere.Client(os.getenv("COHERE_KEY"))
        emb = co.embed(
            model=model_name,
            texts=texts,
            input_type="search_document"   # default for RAG document storage
        ).embeddings     # /embed-multilingual-v3.0
        return emb

    elif provider == "openai":
        import openai
        # model_name expected to be an OpenAI embedding model like 'text-embedding-3-small'
        embeddings = []
        for t in texts:
            resp = openai.Embedding.create(input=t, model=model_name)
            embeddings.append(resp["data"][0]["embedding"])  # type: ignore
        return embeddings

    else:
        raise ValueError(f"Unknown provider: {provider}")