"""
Example usage of XRAG+ chunkers.

Run using - python -m src.chunker.cli_example

Demonstrates how to apply different chunking strategies (Token, Sliding Window, Sentence, Paragraph) to the same document,
provided documents be in a certain format.
"""

import spacy

from src.chunker import (
    TokenChunker,
    SlidingWindowChunker,
    SentenceChunker,
    ParagraphChunker,
    context_aware_chunking
)
from src.chunker.config import Settings
from src.chunker.utils import print_chunks


DOCUMENT = {
    "doc_id": "example-doc-002",
    "title": "Example Doc 002",
    "language": "en", 
    "source": "example",
    "text": (
        "Artificial intelligence (AI) is the simulation of human intelligence processes by machines. "
        "It enables computers to perform tasks that normally require human cognition, such as visual perception, speech recognition, and decision-making.\n\n"

        "Machine learning is a subset of AI that uses statistical techniques to give machines the ability to 'learn' from data. "
        "Supervised, unsupervised, and reinforcement learning are the main paradigms in machine learning.\n\n"

        "Quantum computing leverages quantum mechanics to perform computations far more efficiently than classical computers for certain problems. "
        "Qubits, entanglement, and superposition are the core concepts behind quantum computation.\n\n"

        "Renewable energy technologies, such as solar panels and wind turbines, are vital for reducing greenhouse gas emissions. "
        "Advances in energy storage and smart grids are accelerating the adoption of renewables.\n\n"

        "Nutrition science studies the effects of food and diet on health. "
        "A balanced diet with adequate vitamins, minerals, and macronutrients is essential for optimal physical and cognitive function.\n\n"

        "Climate change impacts ecosystems, human health, and global economies. "
        "Mitigation strategies include carbon capture, reforestation, and transitioning to low-carbon energy sources."
    )
}





if __name__ == "__main__":
    settings = Settings()

    # Token-based chunking
    token_chunker = TokenChunker(
        tokenizer=spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"]),
        chunk_size=10,
        stride=1
    )
    token_chunks = token_chunker.chunk(DOCUMENT)
    print_chunks("\nTokenChunker", token_chunks)

    # Sliding window chunking
    sliding_chunker = SlidingWindowChunker(chunk_size=12, overlap=4)
    sliding_chunks = sliding_chunker.chunk(DOCUMENT)
    print_chunks("\nSlidingWindowChunker", sliding_chunks)

    # Sentence-based chunking
    sentence_chunker = SentenceChunker(min_tokens=5)
    sentence_chunks = sentence_chunker.chunk(DOCUMENT)
    print_chunks("\nSentenceChunker", sentence_chunks)

    # Paragraph-based chunking
    paragraph_chunker = ParagraphChunker(min_chars=200)
    paragraph_chunks = paragraph_chunker.chunk(DOCUMENT)
    print_chunks("\nParagraphChunker", paragraph_chunks)

    # Context-aware chunking
    max_chars = 900
    ctx = context_aware_chunking(DOCUMENT, max_chars=max_chars, overlap_sentences=1)
    if not ctx:
        print("No context-aware chunks returned.")
    else:
        for idx, (chunk_text, meta) in enumerate(ctx):
            print(f"\ncontext-chunk-{idx} | tokens≈{len(chunk_text.split()):d} | meta={meta}")
            print("  ", chunk_text[:max_chars].replace("\n", " "))
            print()


    # Next step would be to embed the Chunks and save it to the VectorDB. Carried out in src.Indexing module.
    # Semantic Chunking Implementation