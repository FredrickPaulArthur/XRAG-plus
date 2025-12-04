from .config import Settings
from .indexer import index_documents
from .retriever import Retriever
from pprint import pprint
from .chunking import fixed_width_chunking, semantic_chunking


settings = Settings()

docs = [
    "ChromaDB is a lightweight embedding database.",
    "ChromaDB is a Vector database",
    "ChromaDB stores vectors and associated documents.",
    "RAG systems combine a retriever and a generator to answer questions using external knowledge.",
    "Vector embeddings represent text as numerical arrays that capture semantic meaning.",
    "Semantic chunking groups sentences based on meaning to improve retrieval accuracy.",
    "Context-aware chunking considers document structure and relationships between sentences.",
    "FAISS is a popular similarity search library optimized for high-dimensional vectors.",
    "Embedding models convert text or images into vectors suitable for similarity search.",
    "Retrieval latency can be reduced by using approximate nearest neighbor indexing.",
    "Metadata filtering helps retrieve relevant documents based on structured attributes.",
    "Reranking models reorder retrieved results to improve precision before generation."
]

metas = [
    {"title": "chroma_intro"},
    {"title": "chroma_intro"},
    {"title": "chroma_intro"},
    {"title": "rag_overview"},
    {"title": "vector_embeddings"},
    {"title": "semantic_chunking"},
    {"title": "context_aware_chunking"},
    {"title": "faiss_intro"},
    {"title": "embedding_models"},
    {"title": "retrieval_latency"},
    {"title": "metadata_filtering"},
    {"title": "reranking_models"}
]



language = "en"
collection_name = f"xragg_collection_{language}"
query = "Why should I reorder retrieved documents?"

# res = index_documents(
#     collection_name=collection_name,
#     chunk_method=semantic_chunking,
#     language=language,
#     documents=docs, metadatas=metas, settings=settings
# )
# print("Indexed:", res)

r = Retriever(settings=settings, language="en")
out = r.retrieve(collection_name, query, k=5)

print(f"Retrieve for Query : {query}")
pprint(out, width=150)