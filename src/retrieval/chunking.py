import re
import numpy as np
from typing import List
# from sklearn.metrics.pairwise import cosine_similarity

from .utils import clean_text
from .embeddings import embed



def fixed_width_chunking(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Simple character-based chunker. Returns list of chunks of approx chunk_size
    with given overlap.
    """
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_len:
            break
        start = end - overlap
    return chunks



def recursive_char_splitting():
    pass


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_chunking(text, model, provider, threshold=0.8):
    from sklearn.metrics.pairwise import cosine_similarity

    """Minimum unit is a Sentence. More than 1 sentences can be added in the same chunk if they have similar meaning (semantic)"""
    sentences = re.split(r'(?<=[.!?]) +', text)    # Split by sentence

    embeddings = [embed(texts=[sentence], model_name=model, provider=provider) for sentence in sentences]

    chunks, current_chunk, current_embedding = [], sentences[0], embeddings[0]
    for i in range(1, len(sentences)):
        print("current embedding: ", type(current_embedding), len(current_embedding))
        print("embeddings: ", type(embeddings[i]), len(embeddings[i]))
        print()
        similarity = cosine_similarity(
            current_embedding,
            embeddings[i]
        )[0][0]
        if similarity < threshold:
            chunks.append(current_chunk)
            current_chunk = sentences[i]
            current_embedding = embeddings[i]
        else:
            current_chunk += " " + sentences[i]
            current_embedding = (current_embedding + embeddings[i]) / 2
    if current_chunk:
      chunks.append(current_chunk)

    return chunks



def context_aware_chunking(
    text: str,
    max_chars: int = 1000,
    overlap_sentences: int = 1,
):
    """
    Context-aware chunking.
    Returns:
        chunks: list[str]
        metadatas: list[{ 'context_title': str }]
    """
    import re

    if not text:
        return [], []

    # --- 1. Paragraph split ---
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]

    # --- 2. Sentence split (simple heuristic) ---
    def split_sentences(p):
        pattern = r'(?<=[\.\!\?])\s+(?=[A-Z0-9"])'
        sents = re.split(pattern, p)
        return [s.strip() for s in sents if s.strip()]

    sentences = []
    for p in paragraphs:
        s = split_sentences(p)
        if not s:
            s = [p]  # fallback
        sentences.extend(s)

    chunks = []
    # metadatas = []

    current = []
    i = 0
    n = len(sentences)

    def join_current():
        return " ".join(current).strip()

    while i < n:
        sent = sentences[i]
        candidate = (join_current() + " " + sent).strip() if current else sent

        # If adding this sentence exceeds budget → finalize current chunk
        if len(candidate) > max_chars:
            if current:
                chunks.append(join_current())
                # metadatas.append({"context_title": title})
                # prepare next chunk with overlap
                current = current[-overlap_sentences:] if overlap_sentences > 0 else []
                continue
            else:
                # sentence itself too large → force split
                chunks.append(sent[:max_chars])
                # metadatas.append({"context_title": title})
                sentences[i] = sent[max_chars:]
                continue

        # Add normally
        current.append(sent)
        i += 1

    # Add last chunk
    if current:
        chunks.append(join_current())
        # metadatas.append({"context_title": title})

    return chunks   # , metadatas



from transformers import AutoModelForCausalLM, AutoTokenizer

def llm_based_chunking(text, model_name):
    # Load the Mistral model from Hugging Face or another source
    model_name = "mistral"  # Substitute this with the actual name if different
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    sentences = re.split(r'(?<=[.!?]) +', text)  # Split by sentence
    chunks = []
    
    # Feed sentences or chunks to the Mistral model for dynamic chunking
    for i in range(0, len(sentences), 5):  # Process in small batches for better results
        batch_sentences = " ".join(sentences[i:i + 5])
        input_ids = tokenizer.encode(batch_sentences, return_tensors='pt')
        
        # Get model response - you can modify the prompt according to the task
        output = model.generate(input_ids, max_length=500, num_return_sequences=1)
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Use model's response to chunk (e.g., summarized chunk)
        chunks.append(decoded_output.strip())

    return chunks