import json
import logging
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("index_all")





def is_ccnews_record(obj: dict) -> bool:
    return "text" in obj and "title" in obj and ("_global_idx" in obj or "url" in obj)



def is_wiki_record(obj: dict) -> bool:
    return "text" in obj and ("id" in obj or "url" in obj) and "title" in obj



def make_doc_from_ccnews(obj: dict, language: str) -> Dict:
    doc_id = None
    if "_global_idx" in obj:
        doc_id = f"ccnews_{language}_{obj.get('_global_idx')}"
    else:
        # fallback unique-ish id
        doc_id = f"ccnews_{language}_{hash(obj.get('url', obj.get('title','')))}"
    return {
        "doc_id": doc_id,
        "language": language,
        "source": "ccnews",
        "title": obj.get("title", "")[:512],
        "text": obj.get("text", ""),
        "url": obj.get("url", ""),
        "meta": {k: v for k, v in obj.items() if k not in ("title", "text", "url")},
    }



def make_doc_from_wiki(obj: dict, language: str) -> Dict:
    doc_id = f"wiki_{language}_{obj.get('id', hash(obj.get('url', obj.get('title',''))))}"
    return {
        "doc_id": doc_id,
        "language": language,
        "source": "wiki",
        "title": obj.get("title", "")[:512],
        "text": obj.get("text", ""),
        "url": obj.get("url", ""),
        "meta": {k: v for k, v in obj.items() if k not in ("title", "text", "url", "id")},
    }



def iter_json_lines(file_path: Path):
    """Yield parsed JSON objects for a file where each line is a JSON object."""
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # some files may contain trailing commas or other minor issues;
                # try a safe fallback by stripping trailing commas
                try:
                    obj = json.loads(line.rstrip(","))
                    yield obj
                except Exception as e:
                    logger.warning("Failed to parse line in %s: %s", file_path, e)
                    continue