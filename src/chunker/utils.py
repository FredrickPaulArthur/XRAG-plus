from __future__ import annotations


import re
from typing import List, Tuple, Dict, Any




def make_chunk_id(doc_id: str, index: int, lang: str) -> str:
    return f"{doc_id}::{lang}::chunk::{index}"



def char_spans_for_whitespace_tokens(text: str, tokens: List[str]) -> List[Tuple[int, int]]:
    """Best-effort mapping of whitespace tokens to character spans.

    Scans forward from the last known position and match the token using str.find.
    This will fail for subword tokenizers (where tokens don't match substrings) — callers
    should pass a proper tokenizer in that case.
    """
    spans: List[Tuple[int, int]] = []
    pos = 0
    for tok in tokens:
        while pos < len(text) and text[pos].isspace():      # skip contiguous whitespace
            pos += 1
        if pos >= len(text):
            spans.append((len(text), len(text)))
            continue
        idx = text.find(tok, pos)       # find token at or after pos; prefer exact match
        if idx == -1:
            m = re.search(r"\S+", text[pos:])       # fallback: find next whitespace-delimited word
            if m:
                start = pos + m.start()
                end = pos + m.end()
                spans.append((start, end))
                pos = end
            else:
                spans.append((len(text), len(text)))
        else:
            spans.append((idx, idx + len(tok)))
            pos = idx + len(tok)
    return spans



_WHITESPACE_TOKEN_RE = re.compile(r"\S+")

def whitespace_tokens(text: str) -> List[str]:
    return _WHITESPACE_TOKEN_RE.findall(text)



def print_chunks(title: str, chunks: List[Dict[str, Any]]):
    """Pretty Print Chunks"""
    print(f"\n\n\n=== {title} ===")
    for c in chunks:
        print(
            f"\n{c['chunk_id']} | tokens={c['token_count']} | "
            f"chars=({c['start_char']},{c['end_char']})\n"
            f"  TEXT: {c['text'][:]}"
        )