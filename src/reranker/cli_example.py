# Model used - jinaai/jina-reranker-v3



# A hybrid search (semantic + keyword) is first performed on the VectorDB to retrieve the most relevant candidate documents. 
# These retrieved documents are then reranked using a cross-encoder reranker to refine their order based on deeper semantic relevance to the query. 
# The top reranked documents are finally inserted into the prompt template and passed to the LLM in the LangChain RAG pipeline.

# User given Query
#       |
# Hybrid Search (Semantic+Keyword)
#       |
# Top-k Retrieved Documents
#       |
# Cross-Encoder Reranker (Jina v2/v3)
#       |
# Sorted & Filtered Documents
#       |
# Prompt Template (LangChain)
#       |
# LLM Generates Final Answer



from transformers import AutoModel
from sentence_transformers import CrossEncoder
texts = [
    "Follow the white rabbit.",  # English
    "Sigue al conejo blanco.",  # Spanish
    "Suis le lapin blanc.",  # French
    "跟着白兔走。",  # Chinese
    "اتبع الأرنب الأبيض.",  # Arabic
    "Folge dem weißen Kaninchen.",  # German
]


reranker = CrossEncoder("jinaai/jina-reranker-v2-base-multilingual", trust_remote_code=True)

query = "Follow the white rabbit."
pairs = [(query, doc) for doc in texts]

scores = reranker.predict(pairs)

print("\nReranker scores - v2:")
for text, score in zip(texts, scores):
    print(f"{score:.4f} | {text}")




from transformers import AutoModel

model = AutoModel.from_pretrained('jinaai/jina-reranker-v3', dtype="auto", trust_remote_code=True)

print("\nReranker scores - v3:")

# Rerank documents
results = model.rerank(query, texts)

# Results are sorted by relevance score (highest first)
for result in results:
    print(f"{result['relevance_score']:.4f} | {result['document']}")

# Output:
# Reranker scores - v3:
# 0.3781 | Follow the white rabbit.
# 0.2982 | 跟着白兔走。
# 0.2071 | Folge dem weißen Kaninchen.
# 0.1980 | اتبع الأرنب الأبيض.
# 0.1728 | Sigue al conejo blanco.
# 0.1154 | Suis le lapin blanc.


query = "What are the health benefits of green tea?"
documents = [
    "Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells from damage.",
    "El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro.",
    "Studies show that drinking green tea regularly can improve brain function and boost metabolism.",
    "Basketball is one of the most popular sports in the United States.",
    "绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。",
    "Le thé vert est riche en antioxydants et peut améliorer la fonction cérébrale.",
]

# Rerank documents
results = model.rerank(query, documents)

print(f"\n for Query: {query}")
# Results are sorted by relevance score (highest first)
for result in results:
    print(f"Score: {result['relevance_score']:.4f}")
    print(f"Document: {result['document'][:100]}...")
    print()

# Output:
# Score: 0.2976
# Document: Green tea contains antioxidants called catechins that may help reduce inflammation and protect ce...
#
# Score: 0.2258
# Document: 绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。
#
# Score: 0.1911
# Document: Studies show that drinking green tea regularly can improve brain function and boost metabolism.
#
# Score: 0.1640
# Document: Le thé vert est riche en antioxydants et peut améliorer la fonction cérébrale.



from .reranker import Reranker

# # small demo using settings defaults
# logging.getLogger().setLevel(logging.INFO)

formatted_docs = [
    {
        "id": "news_en_iphone",
        "language": "en",
        "source": "news",
        "title": "Apple Releases New iPhone",
        "text": "Apple releases new iPhone 17 with improved battery and camera.",
        "url": "https://example.com/news/apple-new-iphone",
    },
    {
        "id": "local_en_bakery",
        "language": "en",
        "source": "local_news",
        "title": "Bakery Introduces New Sourdough Recipe",
        "text": "Local bakery introduces a new sourdough loaf recipe.",
        "url": "https://example.com/local/bakery-sourdough",
    },
    {
        "id": "research_en_battery",
        "language": "en",
        "source": "research",
        "title": "Battery Chemistry Improvements Enable Faster Charging",
        "text": "Research paper: battery chemistry improvements enable faster charging.",
        "url": "https://example.com/research/battery-chemistry",
    },
    {
        "id": "health_en_green_tea_antioxidants",
        "language": "en",
        "source": "health",
        "title": "Green Tea Antioxidants and Inflammation",
        "text": (
            "Green tea contains antioxidants called catechins that may help reduce "
            "inflammation and protect cells from damage."
        ),
        "url": "https://example.com/health/green-tea-antioxidants",
    },
    {
        "id": "economy_es_coffee_price",
        "language": "es",
        "source": "economy",
        "title": "Aumento del Precio del Café",
        "text": (
            "El precio del café ha aumentado un 20% este año debido a problemas "
            "en la cadena de suministro."
        ),
        "url": "https://example.com/economia/precio-cafe",
    },
    {
        "id": "health_en_green_tea_benefits",
        "language": "en",
        "source": "health",
        "title": "Health Benefits of Drinking Green Tea",
        "text": (
            "Studies show that drinking green tea regularly can improve brain "
            "function and boost metabolism."
        ),
        "url": "https://example.com/health/green-tea-benefits",
    },
    {
        "id": "sports_en_basketball",
        "language": "en",
        "source": "sports",
        "title": "Popularity of Basketball in the United States",
        "text": "Basketball is one of the most popular sports in the United States.",
        "url": "https://example.com/sports/basketball-usa",
    },
    {
        "id": "health_zh_green_tea",
        "language": "zh",
        "source": "health",
        "title": "绿茶的健康益处",
        "text": (
            "绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重"
        ),
        "url": "https://example.com/health/green-tea-cn",
    },
    {
        "id": "health_fr_green_tea",
        "language": "fr",
        "source": "health",
        "title": "Les Bienfaits du Thé Vert",
        "text": (
            "Le thé vert est riche en antioxydants et peut améliorer la fonction cérébrale."
        ),
        "url": "https://example.com/sante/the-vert",
    },
    {
        "id": "history_ru_coffee_legend",
        "language": "ru",
        "source": "history",
        "title": "Легенда об открытии кофе в Эфиопии",
        "text": (
            "В Эфиопии пастухи заметили, что козы становятся необычайно энергичными "
            "после поедания ягод кофейного дерева — так зародилась легенда об "
            "открытии кофе."
        ),
        "url": "https://example.com/history/coffee-ethiopia",
    }
]




query = "Fact about Coffee and Tea"

# To implement - Rerank the retrieved documents
from .reranker import Reranker
reranker = Reranker(_use_cross=False)
top_ranked = reranker.rerank(query, formatted_docs, top_k=5)

print(f"\nTop multilingual Re-ranked results for query - {query}\n")
for reranked_doc in top_ranked:
    print(f"id={reranked_doc.get('id')} \ntext={reranked_doc.get('text')[:80]}... \nscore={reranked_doc.get('score'):.4f}\n")


# To implement - Rerank the retrieved documents
from .reranker import Reranker
reranker = Reranker(_use_cross=True)
top_ranked = reranker.rerank(query, formatted_docs, top_k=5)

print(f"\nTop multilingual Re-ranked results for query - {query}\n")
for reranked_doc in top_ranked:
    print(f"id={reranked_doc.get('id')} \ntext={reranked_doc.get('text')[:80]}... \nscore={reranked_doc.get('score'):.4f}\n")



# if you want to use Jina reranker:
r = Reranker(_use_cross=False, use_jina=True)
top = r.rerank(query, formatted_docs, top_k=5)

print(f"\nTop multilingual Re-ranked results for query - {query}\n")
for reranked_doc in top:
    print(f"id={reranked_doc.get('id')} \ntext={reranked_doc.get('text')[:80]}... \nscore={reranked_doc.get('score'):.4f}\n")