"""
Utility helpers for summarizer: sentence-splitting, cleaning, extractive TextRank implementation.
"""

from __future__ import annotations
from typing import List, Dict, Any
import re



def clean_text(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and sanitize the textual content of a document in-place.

    This function cleans the `text` field of a document by:
    - Replacing newline characters with spaces
    - Collapsing multiple consecutive whitespace characters into a single space
    - Stripping leading and trailing whitespace
    - Does not perform linguistic normalization (e.g., lowercasing, stemming, or punctuation removal).

    Parameters
    ----------
    doc : Dict[str, Any]
        A structured document dictionary containing a `"text"` field.
        Example keys: `"id"`, `"language"`, `"source"`, `"title"`, `"text"`.

    Returns
    -------
        The same document object with its `"text"` field normalized.

    Notes
    -----
    - This function assumes `doc["text"]` exists and is a string.
    - Safe to call multiple times (idempotent).
    """
    text = doc["text"].replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    doc["text"] = text
    return doc



def extractive_textrank(text: str, top_k: int = 3) -> str:
    """
    Simple extractive summarizer using TF-IDF + PageRank on sentence similarity graph.

    This avoids heavy model dependencies and is useful as a reliable fallback.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import networkx as nx

    sents = split_sentences(text)
    if not sents:
        return ""
    if len(sents) <= top_k:
        return " ".join(sents)

    tf = TfidfVectorizer(stop_words="english")
    try:
        mat = tf.fit_transform(sents)
    except Exception:
        # in case of tiny input or strange tokens
        mat = tf.fit_transform([s for s in sents])

    sim = cosine_similarity(mat)
    nx_graph = nx.from_numpy_array(sim)
    scores = nx.pagerank(nx_graph)
    ranked = sorted(((scores[i], s) for i, s in enumerate(sents)), reverse=True)
    selected = [s for _, s in ranked[:top_k]]

    # Preserve original order
    selected_sorted = sorted(selected, key=lambda s: sents.index(s))
    return " ".join(selected_sorted)



def split_sentences(text: str) -> List[str]:
    # Prefer nltk if available, otherwise simple punctuation split
    try:
        import nltk
        nltk.data.find("tokenizers/punkt")
    except Exception:
        # fallback naive split
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    else:
        from nltk.tokenize import sent_tokenize

        return sent_tokenize(text)



# Helper: normalize chunker outputs into list[(text, meta)]
def _normalize(raw):
    out = []
    if raw is None:
        return out
    # If chunker returned a list of (text,meta) tuples already
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, tuple) and len(item) >= 1:
                text = item[0]
                meta = item[1] if len(item) > 1 else {}
                out.append((str(text).strip(), dict(meta or {})))
                continue
            if isinstance(item, dict):
                # accept shapes like {'text':..., 'metadata':...}
                if "text" in item:
                    meta = item.get("metadata") or item.get("meta") or {}
                    out.append((str(item["text"]).strip(), dict(meta)))
                    continue
                # fallback: serialize dict to text
                out.append((str(item), {}))
                continue
            # plain string
            out.append((str(item).strip(), {}))
        return out

    # single string
    if isinstance(raw, str):
        return [(raw.strip(), {})]
    # dict-like single chunk
    if isinstance(raw, dict):
        if "text" in raw:
            return [(str(raw["text"]).strip(), dict(raw.get("metadata") or raw.get("meta") or {}))]
        return [(str(raw), {})]
    return []