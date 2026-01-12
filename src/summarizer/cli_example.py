"""
Small CLI for quick testing of summarizer from command-line.
"""

from __future__ import annotations
from src.summarizer import Summarizer
from .config import Settings
from typing import Dict, Any, List


# article = "The Soviet Union (USSR) emerged in 1922 after years of civil war and political upheaval following the Russian Revolution. Built on a single-party communist system, it rapidly industrialized under central planning and became one of the world's leading powers. The USSR played a decisive role in the defeat of Nazi Germany during World War II and later entered a prolonged ideological conflict with the United States known as the Cold War. Despite achievements in science, heavy industry, and education, the Soviet system struggled with economic inefficiencies, shortages, and political repression. By the late 1980s, reforms under Mikhail Gorbachev—glasnost and perestroika—exposed deep structural issues. In 1991, rising nationalist movements and economic crises culminated in the dissolution of the USSR into fifteen independent republics.",
# docs = [
#     "The Soviet Union (USSR) emerged in 1922 after years of civil war and political upheaval following the Russian Revolution. Built on a single-party communist system, it rapidly industrialized under central planning and became one of the world's leading powers. The USSR played a decisive role in the defeat of Nazi Germany during World War II and later entered a prolonged ideological conflict with the United States known as the Cold War. Despite achievements in science, heavy industry, and education, the Soviet system struggled with economic inefficiencies, shortages, and political repression. By the late 1980s, reforms under Mikhail Gorbachev—glasnost and perestroika—exposed deep structural issues. In 1991, rising nationalist movements and economic crises culminated in the dissolution of the USSR into fifteen independent republics.",
#     "Eastern Europe underwent profound transformations during the 20th century. After World War II, much of the region fell under the political and military influence of the Soviet Union, leading to the establishment of socialist governments aligned with Moscow. These states participated in the Warsaw Pact and implemented centrally planned economies, but many faced lagging productivity, censorship, and social unrest. The 1980s brought growing dissatisfaction, most visibly in Poland through the Solidarity movement. The collapse of communist rule between 1989 and 1991 triggered rapid political, economic, and social transitions. Countries shifted toward democratic governance and market economies, though with varying levels of success. Today, many Eastern European nations are integrated into the European Union and NATO, reflecting a significant realignment from their Cold War positions.",
#     "The Eastern Orthodox Church is one of the oldest Christian traditions, rooted in the Byzantine Empire and characterized by its emphasis on liturgy, monasticism, and apostolic continuity. During the centuries of Ottoman, Russian, and later Soviet political control, Orthodox communities experienced periods of both protection and persecution. In the Soviet Union, the Orthodox Church faced severe repression: religious property was confiscated, clergy were imprisoned, and public worship was tightly restricted. However, the church survived through underground practice and later saw limited revival during World War II when Stalin temporarily eased restrictions. After the collapse of communist regimes in Eastern Europe and the USSR, the Orthodox Church re-emerged as an influential cultural and moral institution. Today it plays a major role in countries like Russia, Serbia, Greece, and Romania, where religion remains intertwined with national identity."
# ]

article_doc: Dict[str, Any] = {
    "id": "history_ussr_1922_1991",
    "language": "en",
    "source": "history",
    "title": "Rise and Fall of the Soviet Union",
    "text": (
        "The Soviet Union (USSR) emerged in 1922 after years of civil war and political "
        "upheaval following the Russian Revolution. Built on a single-party communist "
        "system, it rapidly industrialized under central planning and became one of the "
        "world's leading powers. The USSR played a decisive role in the defeat of Nazi "
        "Germany during World War II and later entered a prolonged ideological conflict "
        "with the United States known as the Cold War. \n\nDespite achievements in science, "
        "heavy industry, and education, the Soviet system struggled with economic "
        "inefficiencies, shortages, and political repression. By the late 1980s, reforms "
        "under Mikhail Gorbachev—glasnost and perestroika—exposed deep structural issues. "
        "In 1991, rising nationalist movements and economic crises culminated in the "
        "dissolution of the USSR into fifteen independent republics."
    ),
    "url": None,
    "metadata": {
        "period": "1922–1991",
        "region": "Eastern Europe / Eurasia",
        "topic": "political history"
    }
}
docs_list: List[Dict[str, Any]] = [
    {
        "id": "history_ussr_overview",
        "language": "en",
        "source": "history",
        "title": "Overview of the Soviet Union",
        "text": (
            "The Soviet Union (USSR) emerged in 1922 after years of civil war and political "
            "upheaval following the Russian Revolution. Built on a single-party communist "
            "system, it rapidly industrialized under central planning and became one of the "
            "world's leading powers. The USSR played a decisive role in the defeat of Nazi "
            "Germany during World War II and later entered a prolonged ideological conflict "
            "with the United States known as the Cold War. \n\nDespite achievements in science, "
            "heavy industry, and education, the Soviet system struggled with economic "
            "inefficiencies, shortages, and political repression. By the late 1980s, reforms "
            "under Mikhail Gorbachev—glasnost and perestroika—exposed deep structural issues. "
            "In 1991, rising nationalist movements and economic crises culminated in the "
            "dissolution of the USSR into fifteen independent republics."
        ),
        "url": None,
        "metadata": {
            "period": "1922–1991",
            "topic": "political history"
        }
    },
    {
        "id": "history_eastern_europe_postwar",
        "language": "en",
        "source": "history",
        "title": "Eastern Europe After World War II",
        "text": (
            "Eastern Europe underwent profound transformations during the 20th century. "
            "After World War II, much of the region fell under the political and military "
            "influence of the Soviet Union, leading to the establishment of socialist "
            "governments aligned with Moscow. These states participated in the Warsaw Pact "
            "and implemented centrally planned economies, but many faced lagging productivity, "
            "censorship, and social unrest. The 1980s brought growing dissatisfaction, most "
            "visibly in Poland through the Solidarity movement. \n\nThe collapse of communist "
            "rule between 1989 and 1991 triggered rapid political, economic, and social "
            "transitions. Countries shifted toward democratic governance and market economies, "
            "though with varying levels of success. Today, many Eastern European nations are "
            "integrated into the European Union and NATO."
        ),
        "url": None,
        "metadata": {
            "period": "1945–1991",
            "region": "Eastern Europe",
            "topic": "Cold War"
        }
    },
    {
        "id": "religion_eastern_orthodox_church",
        "language": "en",
        "source": "history",
        "title": "The Eastern Orthodox Church in Eastern Europe",
        "text": (
            "The Eastern Orthodox Church is one of the oldest Christian traditions, rooted "
            "in the Byzantine Empire and characterized by its emphasis on liturgy, "
            "monasticism, and apostolic continuity. During the centuries of Ottoman, "
            "Russian, and later Soviet political control, Orthodox communities experienced "
            "periods of both protection and persecution. In the Soviet Union, the Orthodox "
            "Church faced severe repression: religious property was confiscated, clergy were "
            "imprisoned, and public worship was tightly restricted. However, the church "
            "survived through underground practice and later saw limited revival during "
            "World War II. \n\nAfter the collapse of communist regimes in Eastern Europe and the "
            "USSR, the Orthodox Church re-emerged as an influential cultural and moral "
            "institution. Today it plays a major role in countries like Russia, Serbia, "
            "Greece, and Romania."
        ),
        "url": None,
        "metadata": {
            "topic": "religion",
            "region": "Eastern Europe"
        }
    }
]


settings = Settings()

smzr = Summarizer(settings)               # uses env or defaults


# ✅ Returns in Correct format for both single and multiple
res = smzr.summarize_docs(article_doc)
# print(f"\nOriginal: {article_doc["text"]}")
# print(f"\nSummary: {res[0]["summary"]}")

new_docs_list = smzr.summarize_docs(docs_list)
# print(f"\n\nSummary of all {len(new_docs_list)} docs: \n{[d["summary"] for d in new_docs_list]}")

from pprint import pprint
pprint(res, width=150)
print()
pprint(new_docs_list, width=150)


# TODO: To implement multiple document passing on one API call for Cohere and HuggingFace, instead of Iteration.



# OUTPUT Format
"""
{
    'id': 'history_ussr_1922_1991',
    'language': 'en',
    'metadata': {'period': '1922–1991', 'region': 'Eastern Europe / Eurasia', 'topic': 'political history'},
    'source': 'history',
    'summary': 'The Soviet Union (USSR) emerged in 1922 after years of civil war and political upheaval following the Russian Revolution. Built on a '
                'single-party communist system, it rapidly industrialized under central planning. In 1991, rising nationalist movements and economic '
                'crises culminated in the dissolution of the USSR into fifteen independent republics.',
    'text': 'The Soviet Union (USSR) emerged in 1922 after years of civil war and political upheaval following the Russian Revolution. Built on a '
            "single-party communist system, it rapidly industrialized under central planning and became one of the world's leading powers. The USSR "
            'played a decisive role in the defeat of Nazi Germany during World War II and later entered a prolonged ideological conflict with the '
            'United States known as the Cold War. Despite achievements in science, heavy industry, and education, the Soviet system struggled with '
            'economic inefficiencies, shortages, and political repression. By the late 1980s, reforms under Mikhail Gorbachev—glasnost and '
            'perestroika—exposed deep structural issues. In 1991, rising nationalist movements and economic crises culminated in the dissolution of '
            'the USSR into fifteen independent republics.',
    'title': 'Rise and Fall of the Soviet Union',
    'url': None
}
"""