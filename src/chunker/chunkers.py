"""
Chunker implementations for XRAG+.

Includes:
- BaseChunker
- TokenChunker
- SlidingWindowChunker
- SentenceChunker
- ParagraphChunker
"""
from __future__ import annotations

import re
import uuid
from typing import List, Dict, Any, Optional, Tuple, Callable


from .utils import (
    make_chunk_id,
    char_spans_for_whitespace_tokens,
    whitespace_tokens,
)

from .config import Settings
settings = Settings()




class BaseChunker:
    """
    Abstract base class for all chunkers.

    - Input: a `doc` dict with at least:
        - 'doc_id' (optional): str
        - 'text': str
        - 'meta' (optional): dict (may include 'language', 'source', etc.)
    - Output: a list of chunk dicts. Each chunk should include:
        - 'doc_id', 'chunk_id', 'text', 'start_char', 'end_char',
          'token_count', 'chunk_type', 'language', 'meta'
      If a chunk is composed of multiple non-contiguous spans, start_char/end_char
      may be -1 and precise provenance should be supplied in a `spans` list.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def chunk(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError




class TokenChunker(BaseChunker):
    """
    Token-based fixed-size chunker.

    - Splits text into tokens (using a provided tokenizer or the fallback whitespace tokenizer).
    - Produces contiguous token chunks of `chunk_size` tokens.
    - Supports token overlap via `stride` (sliding-window style) when `stride > 0`.
    - Ensures a minimum number of tokens per chunk via `min_tokens`.
    - If token-to-character spans are available from the tokenizer, `start_char` and `end_char` are set to the span edges. Otherwise they default to -1.

    Parameters
    ----------
    tokenizer : optional
        Any tokenizer providing `tokenize()` or `encode()` (+ optional `convert_ids_to_tokens()`).
        If omitted a whitespace-based tokenizer is used.
    chunk_size : int
        Number of tokens per chunk.
    stride : int
        If 0 (default) chunks are non-overlapping. If >0, used to compute overlap.
    min_tokens : int
        Minimum tokens a chunk must contain; may cause the final chunk to be larger.
    chunk_type : str
        A descriptive label stored on each chunk (default "token").

    Outputs (per chunk)
    -------------------------
    - 'chunk_id': deterministic id (using make_chunk_id)
    - 'text': original substring for contiguous chunks; a best-effort reconstruction if tokenizer cannot produce character spans
    - 'start_char'/'end_char': character offsets when available, else -1
    - 'token_count': integer
    - 'spans': optional (list) for provenance if token->char mapping exists

    Notes
    -----
    - This chunker is best when you need low-level, fixed-sized units for indexing.
    - Token counts are approximate unless you pass a model tokenizer.
    """
    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        chunk_size: int = settings.DEFAULT_TOKEN_CHUNK_SIZE,
        stride: int = 0,
        min_tokens: int = 1,
        chunk_type: str = "token_chunks",
        **kwargs,
    ):
        super().__init__(kwargs)
        self.tokenizer = tokenizer
        self.chunk_size = int(chunk_size)
        self.stride = int(stride)
        self.min_tokens = int(min_tokens)
        self.chunk_type = chunk_type

    def _tokenize(self, text: str) -> Tuple[List[str], Optional[List[Tuple[int, int]]]]:
        if self.tokenizer is None:
            tokens = whitespace_tokens(text)
            spans = char_spans_for_whitespace_tokens(text, tokens)
            return tokens, spans

        elif hasattr(self.tokenizer, "tokenizer"):  # spaCy, NLTK, SentencePiece-style wrappers
            # print("✅Entered the Spacy Tokenizer")
            doc = self.tokenizer.tokenizer(text)
            tokens = [tok.text for tok in doc]
            spans = [(tok.idx, tok.idx + len(tok.text)) for tok in doc]
            return tokens, spans

        elif hasattr(self.tokenizer, "encode"):     # HuggingFace tokenizers
            enc = self.tokenizer.encode(text)
            if hasattr(self.tokenizer, "convert_ids_to_tokens"):
                toks = self.tokenizer.convert_ids_to_tokens(enc)
                spans = char_spans_for_whitespace_tokens(text, toks)
                return toks, spans
            else:
                return [str(x) for x in enc], None

        else:
            tokens = whitespace_tokens(text)
            spans = char_spans_for_whitespace_tokens(text, tokens)
            return tokens, spans

    def chunk(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc_id = doc.get("doc_id") or str(uuid.uuid4())
        text = doc.get("text", "")
        meta = doc.get("meta", {})

        lang = meta.get("language")

        tokens, spans = self._tokenize(text)
        total = len(tokens)
        if total == 0:
            return []

        chunks: List[Dict[str, Any]] = []
        index = 0
        while index < total:
            end = min(index + self.chunk_size, total)
            if end - index < self.min_tokens and end < total:
                end = min(index + max(self.min_tokens, self.chunk_size), total)
            chunk_tokens = tokens[index:end]

            if spans is not None and len(spans) == total:
                start_char = spans[index][0]
                end_char = spans[end - 1][1]
                chunk_text = text[start_char:end_char]
            else:
                chunk_text = " ".join(chunk_tokens)
                start_char = -1
                end_char = -1

            chunk = {
                "doc_id": doc_id,
                "chunk_id": make_chunk_id(doc_id, len(chunks), lang),
                "text": chunk_text,
                "start_char": start_char,
                "end_char": end_char,
                "token_count": len(chunk_tokens),
                "chunk_type": self.chunk_type,
                "language": meta.get("language"),
                "meta": meta,
            }
            chunks.append(chunk)

            if self.stride <= 0:
                index = end
            else:
                index = index + self.chunk_size - self.stride

        return chunks




class SlidingWindowChunker(TokenChunker):
    """
    Sliding-window token chunker (thin wrapper around TokenChunker).

    Behavior
    --------
    - Configures TokenChunker to produce token windows with overlap.
    - `overlap` argument controls how many tokens are kept between adjacent windows.
      Internally this is exposed as `stride` passed to TokenChunker.

    Use cases
    ---------
    - Useful when you want dense local redundancy to improve recall at the cost of more
      chunks/embeddings (e.g., retrieval for short-span QA).
    - Adjust `chunk_size` and `overlap` (or `stride`) to trade off coverage vs cost.

    Notes
    -----
    - This class inherits the same output schema as TokenChunker.
    - Keep overlap moderate to avoid excessive duplication.
    """
    def __init__(self, tokenizer: Optional[Any] = None, chunk_size: int = settings.DEFAULT_TOKEN_CHUNK_SIZE, overlap: int = settings.DEFAULT_SLIDING_OVERLAP, **kwargs):
        stride = int(overlap)
        super().__init__(tokenizer=tokenizer, chunk_size=chunk_size, stride=stride, chunk_type="sliding_window_chunks", **kwargs)







_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|(?<=\n)\s*")

class SentenceChunker(BaseChunker):
    """
    Sentence-level chunker with short-sentence merging.

    Behavior
    --------
    - Uses a sentence-splitting regex to find sentence boundaries.
    - Optionally merges adjacent sentences until a minimum token count (`min_tokens`)
      is achieved. This prevents many very-short single-sentence chunks.
    - Attempts to compute `start_char`/`end_char` by searching the document text
      for the merged sentence text. If not found, offsets are -1.

    Parameters
    ----------
    tokenizer : optional
        Tokenizer used only for token counting when computing `min_tokens`. If omitted,
        a whitespace-based token counter is used.
    min_tokens : int
        Minimum tokens per returned sentence chunk; short sentences are merged forward.
    chunk_type : str
        Stored on the chunk (default 'sentence').

    Output fields
    -------------
    - 'text' contains the sentence or merged-sentence block
    - 'start_char'/'end_char' assigned when the substring can be located
    - 'token_count' is derived from the tokenizer or whitespace tokenization

    Caveats
    -------
    - The regex-based splitter is language-agnostic for many scripts but may miss tricky
      abbreviations or newline-only punctuation cases; consider a proper sentence tokenizer
      for high-precision needs.
    - Using `.find()` to compute offsets is simple but may produce the first occurrence
      in repeated text. For robust provenance, prefer tokenizers that provide token->char spans.
    """
    def __init__(
        self, tokenizer: Optional[Any] = None, min_tokens: int = settings.DEFAULT_MIN_TOKENS_SENTENCE, 
        chunk_type: str = "sentence_chunks", **kwargs
    ):
        super().__init__(kwargs)
        self.tokenizer = tokenizer
        self.min_tokens = int(min_tokens)
        self.chunk_type = chunk_type

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            return len(whitespace_tokens(text))
        if hasattr(self.tokenizer, "encode"):
            return len(self.tokenizer.encode(text))
        if hasattr(self.tokenizer, "tokenize"):
            return len(self.tokenizer.tokenize(text))
        return len(whitespace_tokens(text))

    def chunk(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc_id = doc.get("doc_id") or str(uuid.uuid4())
        text = doc.get("text", "")
        meta = doc.get("meta", {})

        lang = meta.get("language")

        parts = [p.strip() for p in re.split(_SENTENCE_SPLIT_RE, text) if p and p.strip()]
        chunks: List[Dict[str, Any]] = []

        i = 0
        merged_idx = 0
        while i < len(parts):
            curr = parts[i]
            token_count = self._count_tokens(curr)
            j = i + 1
            while token_count < self.min_tokens and j < len(parts):
                curr = curr + " " + parts[j]
                token_count = self._count_tokens(curr)
                j += 1

            start = text.find(curr)
            if start == -1:
                start = -1
                end = -1
            else:
                end = start + len(curr)

            chunk = {
                "doc_id": doc_id,
                "chunk_id": make_chunk_id(doc_id, merged_idx, lang),
                "text": curr,
                "start_char": start,
                "end_char": end,
                "token_count": token_count,
                "chunk_type": self.chunk_type,
                "language": lang,
                "meta": meta,
            }
            chunks.append(chunk)
            merged_idx += 1
            i = j

        return chunks






_PARAGRAPH_SPLIT_RE = re.compile(r"(?:\r?\n){2,}|</p>|<br\s*/?>", flags=re.IGNORECASE)

class ParagraphChunker(BaseChunker):
    """
    Paragraph-level chunker
    -----------------------
    - Splits text on double newlines and common HTML paragraph/linebreak tags
      (configured by `_PARAGRAPH_SPLIT_RE`) to extract paragraph candidates.
    - If a paragraph is shorter than `min_chars`, it will be merged with the next paragraph
      so chunks are not trivially tiny.
    - Returns contiguous paragraph chunks with valid `start_char`/`end_char` where possible.

    Parameters
    ----------
    min_chars : int
        Minimum number of characters a paragraph must have; otherwise it is merged
        with subsequent paragraphs until the threshold is reached.
    chunk_type : str
        Stored on the chunk (default 'paragraph').

    Output fields
    -------------
    - 'text' contains the paragraph string
    - 'start_char'/'end_char' contain the exact offsets into the original document
    - 'token_count' is computed with the whitespace tokenizer for quick heuristics
    """
    def __init__(self, min_chars: int = settings.DEFAULT_MIN_PARAGRAPH_CHARS, chunk_type: str = "paragraph_chunks", **kwargs):
        super().__init__(kwargs)
        self.min_chars = int(min_chars)
        self.chunk_type = chunk_type

    def chunk(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc_id = doc.get("doc_id") or str(uuid.uuid4())
        text = doc.get("text", "")
        meta = doc.get("meta", {})

        lang = meta.get("language")

        parts = [p.strip() for p in re.split(_PARAGRAPH_SPLIT_RE, text) if p and p.strip()]
        chunks: List[Dict[str, Any]] = []
        i = 0
        idx = 0
        while i < len(parts):
            curr = parts[i]
            j = i + 1
            while len(curr) < self.min_chars and j < len(parts):
                curr = curr + "\n\n" + parts[j]
                j += 1

            start = text.find(curr)
            if start == -1:
                start = -1
                end = -1
            else:
                end = start + len(curr)

            chunk = {
                "doc_id": doc_id,
                "chunk_id": make_chunk_id(doc_id, idx, lang),
                "text": curr,
                "start_char": start,
                "end_char": end,
                "token_count": len(whitespace_tokens(curr)),
                "chunk_type": self.chunk_type,
                "language": lang,
                "meta": meta,
            }
            chunks.append(chunk)
            idx += 1
            i = j

        return chunks