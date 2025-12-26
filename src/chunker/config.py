from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import re


@dataclass
class Settings:
    DEFAULT_TOKEN_CHUNK_SIZE = 256
    DEFAULT_SLIDING_OVERLAP = 128
    DEFAULT_MIN_TOKENS_SENTENCE = 8
    DEFAULT_MIN_PARAGRAPH_CHARS = 50
    DEFAULT_CHUNKER_LANGUAGE = "en"