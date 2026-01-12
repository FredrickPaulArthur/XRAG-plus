"""
Configuration for XRAG+ Summarizer (src/summarizer).

No environment-variable loading is used. All runtime behavior is configured
directly in this file. This keeps the module deterministic, reproducible, 
and version-controlled, without relying on `.env` or external overrides.

This file exposes a single `settings` object (an instance of Settings)
that all summarizer modules import & use.

Notes
-----
- MODEL_TYPE determines which backend is used: 'hf', 'openai', etc.
- Chunking parameters and compression settings tune long-document summarization.
- Caching can be optionally enabled for repeated summarization tasks.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Settings:
    MODEL_PROVIDER: str = "sentence_transformers"

    if MODEL_PROVIDER == "sentence_transformers":   MODEL = "facebook/bart-large-cnn"  # HuggingFace model
    elif MODEL_PROVIDER == "openai":                MODEL = "gpt-4o-mini"
    elif MODEL_PROVIDER == "cohere":                MODEL = "command-r-plus-08-2024"

    MAX_SUMMARY_TOKENS: int = 90
    MIN_SUMMARY_TOKENS: int = 50

    MAX_CHUNK_CHARS: int = 2500   # Split docs into smaller slices for Summarization limit
    TOP_K: int = 3          # Extractive compression when merging chunk summaries

    DEVICE: str = "cpu"

    SENTENCE_SPLITTER: str = "nltk"  # "nltk" or "simple"
    BATCH_SIZE: int = 8
    TEMPERATURE = 0.0

    OVERLAP_SENTENCES = 1

    # from src.chunker.chunkers import TokenChunker, SlidingWindowChunker, SentenceChunker, ParagraphChunker, context_aware_chunking
    CHUNKING_METHOD = "context_aware_chunking"
    # TODO: How to say this when it comes to Chunking for Summarizer?

    def to_dict(self) -> Dict[str, Any]:
        return {
            "MODEL_PROVIDER": self.MODEL_PROVIDER,
            "MODEL": self.MODEL,
            "MAX_SUMMARY_LENGTH": self.MAX_SUMMARY_TOKENS,
            "MIN_SUMMARY_LENGTH": self.MIN_SUMMARY_TOKENS,
            "MAX_CHUNK_CHARS": self.MAX_CHUNK_CHARS,
            "FINAL_TOP_K": self.TOP_K,
            "DEVICE": self.DEVICE,
            "SENTENCE_SPLITTER": self.SENTENCE_SPLITTER,
            "BATCH_SIZE": self.BATCH_SIZE,
        }

__all__ = ["Settings"]