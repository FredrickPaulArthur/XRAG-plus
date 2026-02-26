import os
from .config import Settings
from .indexer import ChromaIndexer

from pprint import pprint
import logging
logging.basicConfig(level=logging.INFO)


docs = [
    # ENGLISH — Wikipedia
    {
        "doc_id": "wiki_en_1",
        "language": "en",
        "source": "wiki",
        "title": "Renewable Energy",
        "text": (
            "Renewable energy refers to energy derived from natural processes that are continuously replenished on a human timescale. "
            "These sources include sunlight, wind, flowing water, geothermal heat, and biological materials. Unlike fossil fuels, "
            "which take millions of years to form and release significant greenhouse gases when burned, renewable energy systems "
            "offer a cleaner and more sustainable alternative for meeting global energy demands.\n\n"

            "Solar energy harnesses radiation from the sun using photovoltaic panels or concentrated solar power systems. "
            "Wind energy captures kinetic energy from atmospheric motion through turbines placed on land or offshore. "
            "Hydropower converts the movement of water in rivers or dams into electricity, while geothermal systems tap into "
            "heat stored beneath the Earth's surface. Biomass energy utilizes organic materials such as plant waste or biofuels "
            "to generate heat, electricity, or transportation fuels.\n\n"

            "The transition toward renewable energy has accelerated due to climate change concerns, declining technology costs, "
            "and international policy commitments. Governments worldwide are investing in grid modernization, battery storage, "
            "and smart distribution systems to accommodate variable energy sources. Technological innovation continues to improve "
            "efficiency rates, reduce material waste, and enhance energy storage capacity.\n\n"

            "Beyond environmental benefits, renewable energy contributes to economic growth by creating jobs in manufacturing, "
            "installation, maintenance, and research sectors. It also increases energy security by reducing dependence on imported fuels. "
            "As global demand for electricity rises with digitalization and population growth, renewable infrastructure plays a critical "
            "role in building resilient and decentralized energy systems for the future."
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
            "Global technology markets experienced remarkable expansion this year, driven by advancements in artificial intelligence, "
            "cloud computing, and semiconductor innovation. Investors demonstrated renewed confidence as startups secured record-breaking "
            "funding rounds across fintech, health tech, and green technology sectors.\n\n"

            "Major public companies reported strong quarterly earnings, supported by increased enterprise spending on digital "
            "transformation initiatives. Demand for cybersecurity solutions, remote collaboration platforms, and data analytics tools "
            "rose sharply as organizations adapted to hybrid work environments.\n\n"

            "Venture capital activity reached historic highs, with early-stage firms benefiting from favorable capital conditions and "
            "cross-border investment flows. Emerging markets also saw increased participation, reflecting the global nature of technological "
            "entrepreneurship. Analysts attribute this momentum to rapid innovation cycles and consumer demand for smarter, connected devices.\n\n"

            "Despite concerns about inflation and regulatory scrutiny, the sector remains resilient. Policymakers are examining competition "
            "frameworks and digital privacy standards while companies continue expanding research and development efforts. "
            "Overall, the tech industry's growth signals structural shifts in how economies generate value in an increasingly digital world."
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
            "La energía solar es la transformación de la radiación proveniente del sol en formas útiles de energía, como electricidad o calor. "
            "Este proceso puede realizarse mediante paneles fotovoltaicos, que convierten la luz en corriente eléctrica, o mediante sistemas "
            "termosolares que concentran el calor para producir vapor y mover turbinas.\n\n"

            "La tecnología solar ha evolucionado rápidamente en las últimas décadas, reduciendo costos y aumentando la eficiencia de los módulos. "
            "Gracias a estos avances, la energía solar se ha convertido en una de las fuentes renovables más competitivas a nivel mundial. "
            "Muchos países han implementado incentivos fiscales y programas de subsidios para fomentar su adopción en hogares e industrias.\n\n"

            "Además de su contribución a la reducción de emisiones de gases de efecto invernadero, la energía solar permite la electrificación "
            "de zonas rurales y comunidades aisladas donde el acceso a redes tradicionales es limitado. Los sistemas autónomos con baterías "
            "facilitan el almacenamiento de energía para su uso nocturno o en días nublados.\n\n"

            "El desarrollo continuo en materiales, como las células solares de perovskita y las tecnologías bifaciales, promete aumentar aún más "
            "la eficiencia y versatilidad de las instalaciones solares. En el contexto de la transición energética global, la energía solar "
            "representa un pilar fundamental para lograr sostenibilidad y seguridad energética."
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
            "कृत्रिम बुद्धिमत्ता (AI) कंप्यूटर विज्ञान की एक महत्वपूर्ण शाखा है, जिसका उद्देश्य ऐसी मशीनों और प्रणालियों का निर्माण करना है "
            "जो मानव जैसी सोच, तर्क और निर्णय लेने की क्षमता प्रदर्शित कर सकें। इसमें मशीन लर्निंग, डीप लर्निंग, प्राकृतिक भाषा संसाधन "
            "और कंप्यूटर विज़न जैसी तकनीकों का उपयोग किया जाता है।\n\n"

            "AI का उपयोग स्वास्थ्य सेवा, वित्त, शिक्षा, परिवहन और मनोरंजन सहित अनेक क्षेत्रों में तेजी से बढ़ रहा है। "
            "स्वचालित निदान प्रणाली, स्मार्ट चैटबॉट, सिफारिश एल्गोरिद्म और स्वायत्त वाहन इसके प्रमुख उदाहरण हैं। "
            "डेटा की उपलब्धता और उच्च कंप्यूटिंग क्षमता ने इस क्षेत्र में नवाचार को और गति दी है।\n\n"

            "हालांकि कृत्रिम बुद्धिमत्ता के लाभ व्यापक हैं, इसके साथ नैतिक और सामाजिक चुनौतियाँ भी जुड़ी हुई हैं। "
            "डेटा गोपनीयता, एल्गोरिद्मिक पक्षपात और रोजगार पर प्रभाव जैसे मुद्दे नीति-निर्माताओं और शोधकर्ताओं के लिए महत्वपूर्ण विषय हैं। "
            "जिम्मेदार AI विकास के लिए पारदर्शिता और जवाबदेही आवश्यक मानी जाती है।\n\n"

            "भविष्य में, AI प्रणालियाँ मानव क्षमताओं को बढ़ाने और जटिल समस्याओं को हल करने में महत्वपूर्ण भूमिका निभा सकती हैं। "
            "सतत अनुसंधान और वैश्विक सहयोग इस क्षेत्र की दिशा और प्रभाव को निर्धारित करेंगे।"
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
            "Искусственный интеллект — это область информатики, направленная на создание систем, способных выполнять задачи, "
            "требующие человеческого мышления, обучения и анализа. К таким задачам относятся распознавание речи, обработка изображений, "
            "принятие решений и автоматический перевод текстов.\n\n"

            "Современные методы ИИ основаны на машинном обучении и нейронных сетях, которые позволяют системам обучаться на больших "
            "объёмах данных и улучшать свои результаты без явного программирования каждого шага. Развитие вычислительных мощностей "
            "и облачных технологий значительно ускорило прогресс в этой области.\n\n"

            "ИИ активно применяется в промышленности, медицине, банковском секторе и транспорте. Автоматизированные системы анализа данных "
            "помогают компаниям оптимизировать процессы и повышать эффективность. В медицине алгоритмы используются для диагностики заболеваний "
            "и разработки персонализированных методов лечения.\n\n"

            "Одновременно с развитием технологий возникают вопросы безопасности, этики и регулирования. "
            "Общество обсуждает влияние ИИ на рынок труда, конфиденциальность информации и ответственность за решения, принимаемые алгоритмами."
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
            "Künstliche Intelligenz bezeichnet ein Fachgebiet der Informatik, das sich mit der Entwicklung von Systemen beschäftigt, "
            "die eigenständig lernen, planen und Probleme lösen können. Ziel ist es, menschliche Denkprozesse teilweise nachzubilden "
            "oder in bestimmten Bereichen sogar zu übertreffen.\n\n"

            "Zu den zentralen Methoden gehören maschinelles Lernen, neuronale Netze und wissensbasierte Systeme. "
            "Durch die Analyse großer Datenmengen erkennen Algorithmen Muster und treffen Vorhersagen, die in Wirtschaft, "
            "Wissenschaft und Alltag vielfältige Anwendung finden.\n\n"

            "In der Industrie ermöglichen KI-Systeme eine präzisere Qualitätskontrolle und vorausschauende Wartung von Maschinen. "
            "Im Bereich der Mobilität werden autonome Fahrzeuge entwickelt, die mithilfe von Sensoren und Echtzeitdaten sicher navigieren. "
            "Auch im Bildungswesen kommen adaptive Lernplattformen zum Einsatz.\n\n"

            "Die gesellschaftliche Debatte über Chancen und Risiken der KI ist intensiv. Themen wie Transparenz, Datenschutz "
            "und ethische Leitlinien stehen im Mittelpunkt politischer und wissenschaftlicher Diskussionen. "
            "Langfristig wird erwartet, dass KI-Technologien zentrale Innovationstreiber der digitalen Transformation bleiben."
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
    print("Indexing Result:", res)

print("\n📃All Collections:")
pprint(idx.list_collections())


print("\n\n\n✅ All chunking methods are working properly!!")



"""
Sample Output:

====================sentence_chunking====================

INFO:xr.indexer:
Indexing group: language=en, source=wiki (#docs=1)
INFO:xr.indexer:Collection already exists: xrag_collection__en__wiki__all-MiniLM-L6-v2__sentence_chunking
INFO:xr.indexer:Existing checksums loaded: 13
INFO:xr.indexer:[MEM] after-load-checksums: 753.0 MB
INFO:xr.indexer:Chroma client persisted (if supported).
INFO:xr.indexer:
Indexing group: language=en, source=ccnews (#docs=1)
INFO:xr.indexer:Collection already exists: xrag_collection__en__ccnews__all-MiniLM-L6-v2__sentence_chunking
INFO:xr.indexer:Existing checksums loaded: 10
INFO:xr.indexer:[MEM] after-load-checksums: 758.3 MB
INFO:xr.indexer:Chroma client persisted (if supported).
INFO:xr.indexer:
Indexing group: language=es, source=wiki (#docs=1)
INFO:xr.indexer:Collection already exists: xrag_collection__es__wiki__all-MiniLM-L6-v2__sentence_chunking
INFO:xr.indexer:Existing checksums loaded: 9
INFO:xr.indexer:[MEM] after-load-checksums: 763.5 MB
INFO:xr.indexer:Chroma client persisted (if supported).
INFO:xr.indexer:
Indexing group: language=hi, source=wiki (#docs=1)
INFO:xr.indexer:Collection already exists: xrag_collection__hi__wiki__sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2__sentence_chunking
INFO:xr.indexer:Existing checksums loaded: 4
INFO:xr.indexer:[MEM] after-load-checksums: 768.7 MB
INFO:xr.indexer:Chroma client persisted (if supported).
INFO:xr.indexer:
Indexing group: language=ru, source=wiki (#docs=1)
INFO:xr.indexer:Collection already exists: xrag_collection__ru__wiki__paraphrase-multilingual-mpnet-base-v2__sentence_chunking
INFO:xr.indexer:Existing checksums loaded: 9
INFO:xr.indexer:[MEM] after-load-checksums: 773.7 MB
INFO:xr.indexer:Chroma client persisted (if supported).
INFO:xr.indexer:
Indexing group: language=de, source=books (#docs=1)
INFO:xr.indexer:Collection already exists: xrag_collection__de__books__all-MiniLM-L6-v2__sentence_chunking
INFO:xr.indexer:Existing checksums loaded: 10
INFO:xr.indexer:[MEM] after-load-checksums: 773.8 MB
INFO:xr.indexer:Chroma client persisted (if supported).
Indexing Result: {'indexed_chunks': 0, 'skipped': 55, 'upserted_ids_count': 0}
"""