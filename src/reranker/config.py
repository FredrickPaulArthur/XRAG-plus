"""
Configuration for XRAG+ reranker and related retrieval components.
Place runtime configuration here so `src/reranker.py` can import it.

This file exposes a single `settings` object with easy-to-edit fields.
You can replace this with environment-driven loading (e.g., pydantic, dynaconf)
if you prefer.
"""
from dataclasses import dataclass
from typing import Dict, Any




@dataclass
class Settings:
    CROSS_ENCODER_MODEL_PROVIDER = "sentence_transformers"
    CROSS_ENCODER_MODEL= "jinaai/jina-reranker-v2-base-multilingual"    # cross-encoder/ms-marco-MiniLM-L-6-v2
    MONO_ENCODER_MODEL_PROVIDER = "sentence_transformers"
    MONO_ENCODER_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    JINA_RERANKER_MODEL: str = "jinaai/jina-reranker-v3"
    JINA_LISTWISE_MAX_DOCS: int = 4   # adjust by memory/throughput

    BATCH_SIZE: int = 64    # Batch size for encoding/prediction

    TOP_K: int = 10    # Default top-k to return

    # Whether to min-max normalize cross-encoder raw scores
    NORMALIZE_SCORES: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "CROSS_ENCODER_MODEL_PROVIDER": self.CROSS_ENCODER_MODEL_PROVIDER,
            "CROSS_ENCODER_MODEL": self.CROSS_ENCODER_MODEL,
            "MONO_ENCODER_MODEL_PROVIDER": self.MONO_ENCODER_MODEL_PROVIDER,
            "MONO_ENCODER_MODEL": self.MONO_ENCODER_MODEL,
            "JINA_RERANKER_MODEL": self.JINA_RERANKER_MODEL,
            "JINA_LISTWISE_MAX_DOCS": self.JINA_LISTWISE_MAX_DOCS,
            "BATCH_SIZE": self.BATCH_SIZE,
            "TOP_K": self.TOP_K,
            "NORMALIZE_SCORES": self.NORMALIZE_SCORES,
            "VERBOSE": self.VERBOSE,
            "LANG_RERANKER_MAP": self.LANG_RERANKER_MAP
        }