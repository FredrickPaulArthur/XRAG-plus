from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import re


@dataclass
class Settings:
    DEFAULT_TOKEN_CHUNK_SIZE: int = 512        # balanced preset
    DEFAULT_SLIDING_OVERLAP: int = 128
    DEFAULT_MIN_TOKENS_SENTENCE: int = 12
    DEFAULT_MIN_PARAGRAPH_CHARS: int = 150
    DEFAULT_CHUNKER_LANGUAGE: str = "en"