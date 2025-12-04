### File: src/retrieval/utils.py
"""
Utility functions: text cleaning and chunking.
"""
import re
from typing import List



def clean_text(text: str) -> str:
    """Basic text cleaning: normalize whitespace and remove control characters."""
    if not text:
        return ""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Optionally remove weird control characters
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
    return text



def rewrite_query(query, model_name):
    pass