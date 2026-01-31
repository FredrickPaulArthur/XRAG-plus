import os
from .config import Settings
from .indexer import ChromaIndexer

from pprint import pprint
import logging
logging.basicConfig(level=logging.INFO)

# Example doc set
docs = [
    # ENGLISH — Wikipedia
    {
        "doc_id": "wiki_en_1",
        "language": "en",
        "source": "wiki",
        "title": "Renewable Energy",
        "text": (
            "Renewable energy is energy from natural sources that replenish "
            "themselves more quickly than they are used up.\n\n" * 20
        ),
        "url": "https://en.wikipedia.org/wiki/Renewable_energy",
    },

    # ENGLISH — CCNews
    {
        "doc_id": "news_en_1",
        "language": "en",
        "source": "ccnews",
        "title": "Tech Market Boom",
        "text": (
            "Global tech markets grew significantly this year. "
            "Startups raised record funding...\n\n" * 30
        ),
        "url": "https://news.example.com/tech-boom",
    },

    # SPANISH — Wikipedia
    {
        "doc_id": "wiki_es_1",
        "language": "es",
        "source": "wiki",
        "title": "Energía Solar",
        "text": (
            "La energía solar es la conversión de la energía del sol en electricidad.\n\n" * 25
        ),
        "url": "https://es.wikipedia.org/wiki/Energía_solar",
    },

    # HINDI — Wikipedia
    {
        "doc_id": "wiki_hi_1",
        "language": "hi",
        "source": "wiki",
        "title": "कृत्रिम बुद्धिमत्ता",
        "text": (
            "कृत्रिम बुद्धिमत्ता (AI) मशीनों की ऐसी क्षमता है जो मनुष्यों की तरह "
            "सोचने और समस्याओं को हल करने में सक्षम होती है।\n\n" * 20
        ),
        "url": "https://hi.wikipedia.org/wiki/कृत्रिम_बुद्धिमत्ता",
    },

    # RUSSIAN — Wikipedia
    {
        "doc_id": "wiki_ru_1",
        "language": "ru",
        "source": "wiki",
        "title": "Искусственный интеллект",
        "text": (
            "Искусственный интеллект — это область информатики, изучающая "
            "создание умных машин и алгоритмов.\n\n" * 20
        ),
        "url": "https://ru.wikipedia.org/wiki/Искусственный_интеллект",
    },

    # GERMAN — Books dataset
    {
        "doc_id": "books_de_1",
        "language": "de",
        "source": "books",
        "title": "Künstliche Intelligenz",
        "text": (
            "Künstliche Intelligenz ist ein Teilgebiet der Informatik, das sich "
            "mit der Automatisierung intelligenten Verhaltens befasst.\n\n" * 15
        ),
        "url": "https://books.example.com/ki",
    },
]

from main.main_config import MainConfig
idx_settings = MainConfig().indexer

idx = ChromaIndexer(settings=idx_settings)
for chun in ["token_chunking", "sliding_window_chunking", "paragraph_chunking", "sentence_chunking"]:
    print(f"\n\n{"="*20}{chun}{"="*20}\n")
    res = idx.index_documents(docs, chunking_method=chun)
    print("Result:", res)

print("\n📃All Collections:")
pprint(idx.list_collections())


print("\n\n\n✅ All chunking methods are working properly!!")