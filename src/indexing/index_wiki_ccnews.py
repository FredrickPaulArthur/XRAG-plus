#!/usr/bin/env python3
"""
index_wiki_ccnews.py

Index all Wiki and CCNews JSON batches into Chroma using src.indexing.ChromaIndexer.

Indexing Commands:
-----------------
python -m src.indexing.index_wiki_ccnews --base_dir "data/index/hf_ccnews_extracted" --lang "en/de/ru/es/hi" --doc_batch_size 256 --workers 1 --chunking_method "token_chunking"

python -m src.indexing.index_wiki_ccnews --base_dir "data/index/hf_datasets_extracted" --lang "en/de/ru/es/hi" --doc_batch_size 256 --workers 1 --chunking_method "token_chunking"


Notes:
 - CCNews files: one JSON object per line, keys include "title", "text", "url", etc.
 - Wiki files: one JSON object per line, keys include "id", "title", "text", "url".
"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict
logging.getLogger("src").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("index_wiki_ccnews")


from src.indexing.indexer import ChromaIndexer
from main.main_config import MainConfig

from src.indexing.utils import make_doc_from_ccnews, make_doc_from_wiki, is_ccnews_record, is_wiki_record, iter_json_lines

from pathlib import Path
from typing import List, Dict




def index_ccnews(language_dir: Path | str, doc_batch_size: int, chunking_method: str, language: str):
    """Index CCNews data.

    This function accepts either a path pointing directly to a language folder
    (e.g. data/index/hf_ccnews_extracted/en) or a folder that contains multiple
    language subfolders. It will look for files named `batch_*.json` either
    directly under `language_dir` or under its subdirectories.
    """
    language_dir = Path(language_dir)
    # indexer_settings = IndexingSettings()
    indexer_settings = MainConfig().indexer
    if language == "hi":
        indexer_settings.CHROMA_PERSIST_DIRECTORY = f"C:/.chroma_db_ccnews_{language}"
    else:
        indexer_settings.CHROMA_PERSIST_DIRECTORY = f"./.chroma_db_ccnews_{language}"

    idx = ChromaIndexer(settings=indexer_settings)

    docs_buffer: List[Dict] = []
    total_indexed = 0

    if not language_dir.exists():
        logger.warning(f"CCNews {language} language directory does not exist: %s", language_dir)
        return

    # Collect batch files: either directly inside language_dir or inside its subdirs
    batch_files = sorted(language_dir.glob("batch_*.json"))
    if not batch_files:
        # fallback: scan subdirectories for batch_*.json
        for sub in sorted(language_dir.iterdir()):
            if not sub.is_dir():
                continue
            batch_files.extend(sorted(sub.glob("batch_*.json")))

    for file in batch_files:
        logger.info("\n\n🗃️ Reading Wiki file: %s\n", file)
        for i, obj in enumerate(iter_json_lines(file)):
            logger.info(f"File: {file} Json-line: {i}")

            if not is_ccnews_record(obj):
                logger.debug("Skipping non-ccnews-like record in %s", file)
                continue
            docs_buffer.append(make_doc_from_ccnews(obj, language))

            # logger.info(docs_buffer)
            # logger.info(type(docs_buffer))
            if len(docs_buffer) >= doc_batch_size:
                try:
                    res = idx.index_documents(docs_buffer, chunking_method=chunking_method)
                    logger.info("Indexed batch of %d docs. Res: %s", len(docs_buffer), res)
                except Exception as e:
                    logger.exception("Indexing batch failed: %s", e)
                total_indexed += len(docs_buffer)
                docs_buffer.clear()

    # Final flush
    if docs_buffer:
        try:
            res = idx.index_documents(docs_buffer, chunking_method=chunking_method)
            logger.info("Indexed final batch of %d docs. Res: %s", len(docs_buffer), res)
        except Exception as e:
            logger.exception("Indexing final batch failed: %s", e)
        total_indexed += len(docs_buffer)
        docs_buffer.clear()

    logger.info("CCNews indexing complete. Total documents indexed (approx): %d", total_indexed)



def index_wiki(language_dir: Path | str, doc_batch_size: int, chunking_method: str, language: str):
    """Index Wikipedia data.

    Accepts either a path that directly contains language-named folders
    (e.g. data/index/hf_datasets_extracted/) or a path pointing to a single
    language folder (e.g. data/index/hf_datasets_extracted/wikipedia_20231101_en).
    It will process files matching `*.json` (and only those starting with
    `wikipedia` or `batch` when encountered).
    """
    language_dir = Path(language_dir)
    docs_buffer: List[Dict] = []

    # indexer_settings = IndexingSettings()
    indexer_settings = MainConfig().indexer
    if language == "en":
        indexer_settings.CHROMA_PERSIST_DIRECTORY = f"C:/.chroma_db_ccnews_{language}"
    else:
        indexer_settings.CHROMA_PERSIST_DIRECTORY = f"./.chroma_db_ccnews_{language}"

    idx = ChromaIndexer(settings=indexer_settings)

    total_indexed = 0

    if not language_dir.exists():
        logger.warning("Wikipedia language directory does not exist: %s", language_dir)
        return

    # If the provided path contains json files directly, treat it as a language folder.
    direct_jsons = sorted(language_dir.glob("*.json"))
    if direct_jsons:
        # language folder like wikipedia_20231101_en
        language = language_dir.name.split("_")[-1]
        file_iter = direct_jsons
        logger.info("Processing Wikipedia folder: %s (language: %s)", language_dir.name, language)
    else:
        # Otherwise assume language_dir contains subfolders, each for a language
        file_iter = []
        for lang_dir in sorted(language_dir.iterdir()):
            if not lang_dir.is_dir():
                continue
            # directory names like wikipedia_20231101_en or wikipedia_20231101_de
            language = lang_dir.name.split("_")[-1]
            logger.info("Processing Wikipedia folder: %s (language: %s)", lang_dir.name, language)
            for file in sorted(lang_dir.glob("*.json")):
                if not file.name.startswith("wikipedia") and not file.name.startswith("batch"):
                    continue
                file_iter.append((file, language))

    # If we had direct jsons, normalize file_iter to tuples (file, language)
    if direct_jsons:
        file_iter = [(f, language) for f in direct_jsons]

    for file, language in file_iter:
        logger.info("\n\n🗃️ Reading Wiki file: %s\n", file)
        for i, obj in enumerate(iter_json_lines(file)):
            logger.info(f"File: {file} Json-line: {i}")
            if not is_wiki_record(obj):
                logger.debug("Skipping non-wiki-like record in %s", file)
                continue
            docs_buffer.append(make_doc_from_wiki(obj, language))
            if len(docs_buffer) >= doc_batch_size:
                try:
                    res = idx.index_documents(docs_buffer, chunking_method=chunking_method)
                    logger.info("Indexed batch of %d docs. Res: %s", len(docs_buffer), res)
                except Exception as e:
                    logger.exception("Indexing batch failed: %s", e)
                total_indexed += len(docs_buffer)
                docs_buffer.clear()

    # Final flush
    if docs_buffer:
        try:
            res = idx.index_documents(docs_buffer, chunking_method=chunking_method)
            logger.info("Indexed final batch of %d docs. Res: %s", len(docs_buffer), res)
        except Exception as e:
            logger.exception("Indexing final batch failed: %s", e)
        total_indexed += len(docs_buffer)
        docs_buffer.clear()

    logger.info("Indexing complete. Total documents indexed (approx): %d", total_indexed)






def parse_args():
    p = argparse.ArgumentParser(description="Index Wiki + CCNews into Chroma (XRAG+).")
    p.add_argument("--base_dir", type=str, help="Where to look for CCNews or Wikipedia Directory")
    p.add_argument("--lang", type=str, help="Language of the Articles to be indexed (en, de, hi, ru, es)")
    p.add_argument("--doc_batch_size", type=int, default=256, help="Number of docs to buffer before indexing")
    p.add_argument("--workers", type=int, default=2, help="Embedding worker processes (if supported by your indexer)")
    p.add_argument("--chunking_method", type=str, default=None,
                   help="Chunking method to use (token_chunking, sliding_window_chunking, paragraph_chunking, sentence_chunking). "
                        "If omitted, indexer default will be used.")
    return p.parse_args()

def main():
    args = parse_args()
    base_dir = Path(args.base_dir)  # "data/index/hf_ccnews_extracted" or "data/index/hf_datasets_extracted"
    language = Path(args.lang)

    logger.info(
        f"Starting Indexing with base_dir={args.base_dir} language={args.lang} doc_batch_size={args.doc_batch_size} workers={args.workers} chunking_method={args.chunking_method}"
    )

    if base_dir==Path("data/index/hf_ccnews_extracted"):
        language_dir = f"{base_dir}/{language}"
        index_ccnews(language_dir, args.doc_batch_size, args.chunking_method, args.lang)
    elif base_dir==Path("data/index/hf_datasets_extracted"):
        language_dir = f"{base_dir}/wikipedia_20231101_{language}"
        index_wiki(language_dir, args.doc_batch_size, args.chunking_method, args.lang)

if __name__ == "__main__":
    main()




"""
    Indexes the Data to ChromaDB storage after performing Chunking and Embedding using the modules.
    Config will be implemented with all_config.MainConfig



    # CCNews - each element in a separate line
    # location - .\data\hf_ccnews_extracted\de\batch_00000.json
    ccnews = {
        "title": "25 Jahre Ötzi - und noch immer Geheimnisse", 
        "text": "Umhausen. Als Erika Simon in den Ötztaler Alpen beinahe über eine in Schmelzwasser liegende Leiche stolpert, glaubt sie, einen erfrorenen Skitourengeher entdeckt zu haben. Sie und ihr Mann Helmut benachrichtigen den Wirt einer nahe gelegenen Hütte. Wenige Tage darauf erfährt das Paar aus Nürnberg: Sie haben eine Mumie gefunden. Die Geschichte vom Ötzi geht um die Welt. 25 Jahre später kommen Erika Simon, Wissenschaftler und der Polizist, der den Ötzi damals ausgegraben hat, im Ötzi-Museum im österreichischen Umhausen zusammen.\nUnzählige Male hat Erika Simon ihre Geschichte erzählt. In den Wochen vor dem Jahrestag könnte sie fünf Interviews am Tag geben, erzählt sie. Ihr Mann ist vor mehr als zehn Jahren gestorben. Die 75-Jährigen erinnert sich genau an den 19. September 1991 - und an die Aufregung, die der Fund ausgelöst hat. Wissenschaftler sprechen vom \"Jahrhundertfund\". Eine fast unversehrte, knapp 5300 Jahre alte Mumie - älter als Pharao Tutanchamun.\nBeinahe wäre Ötzi als namenloser Bergsteiger begraben worden. Anton Koler war einer der Ersten an der Fundstelle. Der Polizist sollte den Toten gemeinsam mit dem Hüttenwirt bergen. Mit einem Pressluftmeißel versuchen sie, ihn aus dem Eis zu befreien. \"Er hat einen üblen Geruch verbreitet und war ledrig\", erinnert sich Koler. Bei der Leiche liegt ein Pickel, der ihm spanisch vorkommt, wie er sagt. Daneben \"Gerümpel\". Den Pickel schätzt Koler damals auf 150 Jahre und nimmt ihn mit zur Polizeiwache. Auch Ernst Schöpf, Bürgermeister von Sölden, erinnert sich gut. Bei der Bergung wird die Leiche an der Hüfte verletzt und ihr steif gefrorener Arm gebrochen - sonst hätte der Tote nicht in den Sarg gepasst, erzählt Schöpf.\nDass die Mumie nicht begraben wurde, sei auch Extrembergsteiger Reinhold Messner zu verdanken, der sich den Gletschermann anschaut. Seine spontane Einschätzung: Der stammt aus dem Mittelalter. Später werden Untersuchungen ergeben, dass Ötzi mehr als 5000 Jahre alt ist. Der Tote wird zunächst nach Innsbruck gebracht. Gut sechs Jahre befindet sich Ötzi dort in einer Klimazelle, die die Gegebenheiten im Eis simuliert. Höchstens 20 Minuten und nur alle vier Wochen nehmen die Forscher die Mumie für Arbeiten aus der Zelle.\nStreit um Fundort\nWährend die Forscher die Mumie zu entschlüsseln beginnen, entbrennt ein Streit: Wem gehört der Ötzi? Österreich oder Südtirol? Der Fundort am Tisenjoch wird neu vermessen. Dann steht fest: Der Ötzi lag 92,56 Meter von der Grenze entfernt auf italienischem Boden. Inzwischen ist Ötzi im Südtiroler Archäologiemuseum ausgestellt.\nWissenschaftler versuchen, alles über ihn herauszufinden: Was hat er gegessen? Woher stammt er? Wie ist er gestorben? Zehn Jahre nach dem Fund steht fest: Ötzi ist ermordet worden, hinterrücks mit einem Pfeil niedergestreckt. Die Mumie wird zum Kriminalfall. Raubmord schließen die Forscher aus, Ötzi hatte noch seinen wertvollen Kupferpickel bei sich. Auf der Flucht war er auch nicht. Denn seinem Mageninhalt nach hatte er kurz vor seinem Tod ausgiebig und fettreich gegessen. Die Wissenschaftler finden noch viel heraus: Laktose-Intoleranz, Zahnprobleme, Anlage zu Herz-Kreislauferkrankungen und zahlreiche Tätowierungen.",
        "url": "http://www.fnweb.de/nachrichten/25-jahre-otzi-und-noch-immer-geheimnisse-1.2965885", 
        "date_publish": "", "_source_parquet": "0000.parquet", "_source_rowgroup": 0, 
        "_global_idx": 0
    }

    # Wiki - dict_keys(['id', 'url', 'title', 'text']) - each element in a separate line
    # location - .\data\hf_datasets_extracted\wikipedia_20231101_de\wikipedia_de_batch_00000.json
    wiki = {
        "id":"76",
        "url":"https:\/\/de.wikipedia.org\/wiki\/Achsensprung%20%28Film%29",
        "title":"Achsensprung (Film)",
        "text":"Ein Achsensprung ist ein Filmschnitt, mit dem die Beziehungsachse der Figuren oder Gruppen übersprungen wird. Blickachsen\noder Beziehungsachsen zwischen den Akteuren untereinander oder dem Point of Interest des Protagonisten bilden eine gedachte Linie. Auf die Leinwand projiziert, stellt diese Linie eine „links-rechts-“ und „oben-unten-Beziehung“ zwischen den Akteuren dar. Mit Achsensprung bezeichnet man einen Schnitt, bei dem sich dieses Verhältnis umkehrt. Es wird zwischen Seitenachsensprung und dem Höhenachsensprung unterschieden. Letzterer wird als weniger desorientierend vom Zuschauer empfunden, da die Leinwand weniger hoch als breit ist. \nEin Achsensprung kann beim Zuschauer Desorientierung verursachen, da die Anordnung und Blickrichtung der Akteure im Frame sich relativ zum Zuschauer zu verändern scheint.\n\nAktionsachse (Handlungsachse)\nist die gedachte Linie, in deren Richtung sich die Handlung oder das Inertialsystem der Filmwelt bewegt. Bei einer Autofahrt zum Beispiel ist die Aktionsachse so stark, dass die Beziehungsachsen an Bedeutung verlieren. Die Orientierung bleibt trotz eventuellem Achsensprung bewahrt. Wenn man aus der Fahrerseite filmt, bewegt sich die Landschaft scheinbar von rechts nach links; filmt man aus der Beifahrerseite, bewegt sie sich scheinbar von links nach rechts. Diese Änderung der Bewegungsrichtung ist aber nicht irritierend. Analog werden zwei Autos, die bei einer Parallelmontage in die gleiche Richtung fahren (oft von links nach rechts, weil das unserer Leserichtung entspricht), als einander verfolgend wahrgenommen; wenn eines jedoch von links nach rechts und das andere von rechts nach links fährt, erwartet der Zuschauer einen Zusammenstoß.\n\nIm Continuity Editing des klassischen Hollywoodkinos wird der Achsensprung als Fehler betrachtet und dementsprechend vermieden. \n\nDer Grundsatz, Achsensprünge zu vermeiden, wird 180-Grad-Regel genannt.\n\nBewusster Achsensprung \nIn manchen Fällen kann ein bewusster Achsensprung auch Stilmittel sein, um beispielsweise Verwirrung oder einen Kippmoment zu symbolisieren; Stanley Kubrick wird in diesem Zusammenhang häufig genannt. In Werbespots werden Achsensprünge oft verwendet, um einen rasanten Effekt zu bewirken. Bekannt ist auch eine Szene aus Herr der Ringe, in welcher Sméagol mit sich selbst spricht. Da er mit den Schnitten wechselnd von der einen zur anderen Seite spricht (Achsensprung), entsteht der Eindruck zweier gleich aussehender Personen, womit der gespaltene Charakter der Figur unterstrichen wird.\n\nAchsenwechsel \nIm Gegensatz zum Achsensprung handelt es sich hierbei um eine Bewegung der Kamera (Steadicam oder einer Dollyfahrt) über die Achse oder um eine Änderung der Bewegungsachse bzw. der Blickrichtung der Figuren, wodurch eine neue Achse definiert wird. Der Achsenwechsel wird vom Zuschauer nicht als störend wahrgenommen, weil sich die Bewegung fließend vollzieht. Diese Bewegung wird mitunter auch als Crab bezeichnet. Außerdem kann ein Zwischenschnitt in eine Totale eine Achsenüberschreitung möglich machen, da so die räumliche Anordnung der Akteure für den Zuschauer deutlich wird, oder der Zwischenschnitt auf einen Closeup, da sich der Betrachter danach wieder neu räumlich orientiert.\n\nAchsen im Film \n Die Handlungsachse gibt die Hauptrichtung der Handlung an. Meist ist sie die Verbindung der Akteure, bei einer Fußballübertragung die Verbindung der Tore.\n Die Blickachse gibt die Blickrichtung und den Blickwinkel (Verhältnis zwischen der Höhen- und Seitenachse) der Figuren an. Bei Gesprächen ist darauf zu achten, dass sich die Figuren anschauen, was bedeutet, dass, wenn eine Figur in einem Bild nach links oben schaut, der Gesprächspartner im anderen Bild (Gegenschuss) nach rechts unten schaut. Diese Richtungen und die beiden Winkel sollten nicht verändert werden, solange sich die reale Blickrichtung bzw. der Standpunkt der Figuren nicht ändert.\n Die Kameraachse ist die „Blickrichtung“ der Kamera. Bei einer subjektiven Perspektive (Point of View) ist sie mit der Blickachse identisch.\n\nWeblinks \n Erklärvideo zu Achsensprung\nFilmgestaltung\nPostproduktion"
    }
"""