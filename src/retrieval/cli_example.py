from .retriever import Retriever
from pprint import pprint
from src.indexing.indexer import ChromaIndexer


# multilingual_test_docs.py

docs = [
    # ========== ENGLISH — DEFINITIVE (RELEVANT) ==========
    {
        "id": "wiki_en_energy_main",
        "language": "en",
        "source": "wiki",
        "title": "Renewable energy (overview)",
        "text": (
            "Renewable energy comes from natural sources that are replenished "
            "constantly — sunlight, wind, water (hydro) and geothermal heat. "
            "These sources reduce greenhouse gas emissions compared to fossil fuels. "
            "Solar panels and wind turbines convert natural flows into electricity. "
            "Storage and grid integration remain key challenges.\n\n"
            "Renewable energy systems play a critical role in climate change mitigation. "
            "Unlike coal or oil, renewable resources do not deplete over time and can be "
            "used sustainably across generations. Governments worldwide support renewables "
            "through policy incentives, subsidies, and research funding.\n\n"
        ),
        "url": "https://en.wikipedia.org/wiki/Renewable_energy",
    },

    # ========== ENGLISH — DISTRACTOR (TECH) ==========
    {
        "id": "news_en_tech_startups",
        "language": "en",
        "source": "news",
        "title": "Tech funding boom",
        "text": (
            "Startups raised record funding this quarter, especially in AI and cloud tooling. "
            "Investors favored companies with strong product-market fit and recurring revenue.\n\n"
            "Venture capital firms focused on software platforms, automation tools, and data-driven services. "
            "Despite economic uncertainty, technology innovation continues to attract global capital.\n\n"
        ),
        "url": "https://news.example.com/tech-funding",
    },

    # ========== ENGLISH — RELATED (INVESTMENT ANGLE) ==========
    {
        "id": "news_en_energy_invest",
        "language": "en",
        "source": "news",
        "title": "Renewables attract investment",
        "text": (
            "Global investments in renewable energy have increased. Large funds allocated capital "
            "to solar farms and offshore wind projects. Returns depend on policy stability and grid access.\n\n"
            "Energy investors increasingly prioritize low-carbon assets as governments tighten emission regulations. "
            "Long-term power purchase agreements help stabilize revenue for renewable energy developers.\n\n"
        ),
        "url": "https://news.example.com/energy-invest",
    },

    # ========== SPANISH — SAME TOPIC (PARAPHRASE) ==========
    {
        "id": "wiki_es_energia",
        "language": "es",
        "source": "wiki",
        "title": "Energía renovable",
        "text": (
            "La energía renovable proviene de fuentes naturales como el sol, el viento y el agua. "
            "Estas fuentes se reponen de forma natural y ayudan a disminuir las emisiones de CO2. "
            "La energía solar y eólica son tecnologías en rápido crecimiento.\n\n"
            "El uso de energías renovables reduce la dependencia de combustibles fósiles "
            "y mejora la seguridad energética. Muchos países están invirtiendo en infraestructuras "
            "renovables para cumplir objetivos climáticos.\n\n"
        ),
        "url": "https://es.wikipedia.org/wiki/Energía_renovable",
    },

    # ========== RUSSIAN — SAME TOPIC (PARAPHRASE) ==========
    {
        "id": "wiki_ru_vozobnovlyaemaya",
        "language": "ru",
        "source": "wiki",
        "title": "Возобновляемая энергия",
        "text": (
            "Возобновляемая энергия получается из природных источников, таких как солнце, ветер и вода. "
            "Она помогает сократить выбросы парниковых газов. Солнечные и ветровые установки широко применяются.\n\n"
            "Развитие возобновляемой энергетики способствует устойчивому развитию экономики. "
            "Современные технологии повышают эффективность производства и хранения энергии.\n\n"
        ),
        "url": "https://ru.wikipedia.org/wiki/Возобновляемая_энергия",
    },

    # ========== HINDI — DIFFERENT TOPIC (DISTRACTOR) ==========
    {
        "id": "wiki_hi_ai",
        "language": "hi",
        "source": "wiki",
        "title": "कृत्रिम बुद्धिमत्ता",
        "text": (
            "कृत्रिम बुद्धिमत्ता (AI) मशीनों को मानव जैसी सोच और निर्णय क्षमता प्रदान करने वाली "
            "कम्प्यूटर विज्ञान की शाखा है। इसमें मशीन लर्निंग और न्यूरल नेटवर्क महत्वपूर्ण भूमिकाएँ निभाते हैं।\n\n"
            "AI का उपयोग स्वास्थ्य, वित्त, और स्वचालन जैसे क्षेत्रों में व्यापक रूप से किया जा रहा है। "
            "डेटा और एल्गोरिदम इसकी सफलता के प्रमुख घटक हैं।\n\n"
        ),
        "url": "https://hi.wikipedia.org/wiki/कृत्रिम_बुद्धिमत्ता",
    },

    # ========== GERMAN — RELATED (ENERGY + POLICY) ==========
    {
        "id": "news_de_energiewende",
        "language": "de",
        "source": "news",
        "title": "Energiewende in Deutschland",
        "text": (
            "Die Energiewende bezeichnet den Umbau des Energiesystems in Deutschland hin zu erneuerbaren "
            "Energiequellen wie Wind und Solar. Netzstabilität und Speicher sind zentrale Themen der Debatte.\n\n"
            "Politische Maßnahmen und Förderprogramme unterstützen den Ausbau erneuerbarer Energien. "
            "Gleichzeitig stellen steigende Strompreise eine gesellschaftliche Herausforderung dar.\n\n"
        ),
        "url": "https://news.example.de/energiewende",
    },

    # ========== RUSSIAN — AI IN ENERGY (CROSS-TOPIC) ==========
    {
        "id": "wiki_ru_ai_energy",
        "language": "ru",
        "source": "wiki",
        "title": "ИИ в энергетике",
        "text": (
            "Искусственный интеллект используется для прогнозирования потребления энергии, оптимизации распределения "
            "и управления микросетями. Это повышает эффективность интеграции возобновляемых источников.\n\n"
            "Модели машинного обучения помогают операторам сетей принимать решения в реальном времени. "
            "ИИ снижает потери энергии и повышает устойчивость энергосистем.\n\n"
        ),
        "url": "https://ru.wikipedia.org/wiki/ИИ_в_энергетике",
    },

    # ========== ENGLISH — MIXED/TRICK (AI+ENERGY) ==========
    {
        "id": "blog_en_ai_energy",
        "language": "en",
        "source": "blog",
        "title": "AI, energy forecasting, and renewables",
        "text": (
            "AI improves renewable energy forecasting by predicting solar irradiance and wind patterns. "
            "Better forecasts reduce curtailment and improve grid planning.\n\n"
            "Advanced forecasting models allow utilities to balance supply and demand more effectively. "
            "This leads to lower costs and increased adoption of renewable power.\n\n"
        ),
        "url": "https://blog.example.com/ai-energy",
    },

    # ========== SPANISH — MIXED ==========
    {
        "id": "news_es_mixed",
        "language": "es",
        "source": "news",
        "title": "Inversión y tecnología",
        "text": (
            "La inversión en 'renewables' creció en 2024. Nuevas tecnologías como 'AI forecasting' "
            "ayudan a mejorar la producción.\n\n"
            "Las empresas energéticas adoptan soluciones digitales para optimizar operaciones "
            "y reducir costos operativos.\n\n"
        ),
        "url": "https://news.example.es/inversiones",
    },

    # ========== GERMAN — DISTRACTOR (NON-ENERGY) ==========
    {
        "id": "books_de_ai_intro",
        "language": "de",
        "source": "books",
        "title": "Einführung in KI",
        "text": (
            "Künstliche Intelligenz umfasst Methoden wie maschinelles Lernen, Statistik und Optimierung. "
            "Anwendungen finden sich in Bildverarbeitung, Sprache und autonomen Systemen.\n\n"
            "Grundlagen der KI werden in vielen Studiengängen vermittelt. "
            "Mathematik und Programmierung bilden das Fundament moderner KI-Systeme.\n\n"
        ),
        "url": "https://books.example.de/ki-einfuhrung",
    },
]

metas = [
    {"doc_id": d["id"], "title": d["title"], "language": d["language"], "source": d["source"], "url": d["url"]}
    for d in docs
]

single_multilingual_collection_docs = docs.copy()




query = "What is renewable energy and why is it important?"
# Was ist erneuerbare Energie?
# How is renewable energy attracting global investment?
# How is AI used in renewable energy systems?
# How does AI improve grid stability in renewable energy systems?
# How is artificial intelligence helping the transition to renewable energy?


# Indexing
from src.indexing.indexer import ChromaIndexer
from src.indexing.config import Settings
idx = ChromaIndexer(settings=Settings())
res = idx.index_documents(docs, chunking_method="sentence_chunking")     # chunking method specified inside

# print("Indexed:", res)


# Retrieval in GERMAN

from src.retrieval.retriever import Retriever
from src.retrieval.config import Settings
de_retriever = Retriever(settings=Settings(), language="de")

collection_name = f"xrag_collection__de"
print(f"Collections with name - {collection_name}:\n", de_retriever.chroma_manager.list_collections(name_startswith=collection_name))


# search all German collections (wiki, news, blog) using hybrid strategy, prefer wiki
res = de_retriever.retrieve_lang_specific(
    query="What is renewable energy?",
    language="de",
    method="hybrid",
    k=6,
    prefer_source="wiki",
    alpha=0.75
)
pprint(res)
print("\n\n")

# # search all the collections that use the 'all-MiniLM-L6-v2' model irrespective of the language
# print(f"\nPerforming Retrieval for Query: '{query}'")
# res2 = de_retriever.retrieve_embedding_specific(
#     query=query,
#     embedding="all-MiniLM-L6-v2",
#     method="semantic",
#     k=5
# )
# pprint(res2)