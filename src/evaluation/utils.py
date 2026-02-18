# src/evaluation/utils.py
import time, json, warnings, os
import psutil
import functools
from statistics import median
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.config import DATA_DIR



# -------------------------------------------------
# Query Loader
# -------------------------------------------------

def load_queries(lang: str, scenario: str) -> List[Dict]:
    """
    Load query examples from JSON/JSONL files.

    Expected flexible structure:
        DATA_DIR/qa/<dataset>/<lang>/<scenario>/*.json
        OR
        DATA_DIR/qa/<dataset>/<lang>/<scenario>/*.jsonl

    Returns list of dict:
        {question, answer, relevant_doc_ids, ...}
    """

    base_dir = os.path.join(DATA_DIR, lang, scenario)

    if not os.path.isdir(base_dir):
        raise RuntimeError(f"Query directory not found: {base_dir}")

    queries = []

    for fname in os.listdir(base_dir):
        path = os.path.join(base_dir, fname)

        if fname.endswith(".json"):
            data = _read_json(path)
            if isinstance(data, list):
                queries.extend(data)
            elif isinstance(data, dict):
                # Handle SQuAD-like structure
                queries.extend(_flatten_squad(data))

        elif fname.endswith(".jsonl"):
            queries.extend(_read_jsonl(path))

    return queries


# -------------------------------------------------
# Document Loader
# -------------------------------------------------

def load_documents(lang: str) -> List[Dict]:
    """
    Load document corpus for indexing.

    Searches:
        DATA_DIR/docs/<lang>/*.json
        DATA_DIR/docs/<lang>/*.jsonl

    Expected doc format:
        {
            "id": str,
            "text": str,
            "language": str,
            "source": str
        }
    """

    doc_dir = os.path.join(DATA_DIR, "docs", lang)

    if not os.path.isdir(doc_dir):
        raise RuntimeError(f"Document directory not found: {doc_dir}")

    docs = []

    for fname in os.listdir(doc_dir):
        path = os.path.join(doc_dir, fname)

        if fname.endswith(".json"):
            data = _read_json(path)
            if isinstance(data, list):
                docs.extend(data)
            elif isinstance(data, dict):
                docs.extend(data.get("documents", []))

        elif fname.endswith(".jsonl"):
            docs.extend(_read_jsonl(path))

    return docs



# -------------------------------------------------
# JSON Readers
# -------------------------------------------------

def _read_json(path: str, max_examples: int = None) -> List[Dict]:
    """
    Read JSON file or all JSON files inside a directory.

    Behaviour:
      - If `path` is a directory, iterate all .json files inside and aggregate results.
      - If a file contains a list -> extend results with that list.
      - If a file contains a dict and has 'data' or 'documents' keys -> extend with that list.
      - If a file contains a dict with other keys -> append the dict as a single example.
      - If json.load fails (file appears to be JSONL inside a .json) -> fallback to line-wise parsing.
    Returns a list of dict-like examples.
    """
    path = os.path.normpath(path)
    results: List[Dict] = []

    # Directory case: load *all* .json files in the directory
    if os.path.isdir(path):
        files = sorted([f for f in os.listdir(path) if f.endswith(".json")])
        if not files:
            raise RuntimeError(f"No JSON files found in directory: {path}")
        for file_name in files:
            file_path = os.path.join(path, file_name)
            results.extend(_read_json(file_path, max_examples))

            if len(results) >= max_examples:
                return results[:max_examples]

        return results

    # File case
    try:
        with open(path, "r", encoding="utf-8") as fh:
            try:
                obj = json.load(fh)
            except json.JSONDecodeError:
                # Fallback: maybe file is JSONL with .json extension
                fh.seek(0)
                items = []
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        warnings.warn(f"Skipping bad JSON line in {path}: {line[:80]}")

                return items

    except Exception as e:
        raise RuntimeError(f"Failed to open/read JSON file {path}: {e}")

    # Normalize loaded object to list of dicts
    if isinstance(obj, list):
        # List of objects
        for it in obj:
            if isinstance(it, dict):
                results.append(it)
            else:
                # wrap non-dict items into dict
                results.append({"value": it})
        return results

    if isinstance(obj, dict):
        # Common patterns: {"data": [...]} or {"documents": [...]} or SQuAD-style
        if "data" in obj and isinstance(obj["data"], list):
            # e.g., SQuAD / xquad style
            return obj["data"]
        if "documents" in obj and isinstance(obj["documents"], list):
            return obj["documents"]
        # If it's a single dict representing one example, return as single item list
        return [obj]

    # otherwise, unknown type — return empty
    return results



def _read_jsonl(path: str) -> List[Dict]:
    """
    Read JSONL data from file OR directory.
    Supports .jsonl and JSONL-style .json files.
    """
    items = []

    # ---------- Directory case ----------
    if os.path.isdir(path):
        for fname in os.listdir(path):
            if fname.endswith((".jsonl", ".json")):
                fpath = os.path.join(path, fname)
                items.extend(_read_jsonl(fpath))
        return items

    # ---------- File case ----------
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                warnings.warn(
                    f"Skipping bad JSON line in {path}: {line[:80]}"
                )
        return items


# -------------------------------------------------
# SQuAD/XQuAD Flattening Helper
# -------------------------------------------------

def _flatten_squad(data: Dict) -> List[Dict]:
    """
    Converts SQuAD-style nested JSON into flat list:
    {
        question, answer, context
    }
    """
    flat = []

    for article in data.get("data", []):
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                answers = qa.get("answers", [])
                flat.append({
                    "question": qa.get("question"),
                    "answer": answers[0]["text"] if answers else "",
                    "context": context,
                    "id": qa.get("id")
                })

    return flat



def compute_qa_scores(pred: str, golds: List[str]) -> Tuple[Optional[float], Optional[float]]:
    try:
        from src.evaluation.metrics.rag_metrics import exact_match, f1_score
    except Exception:
        import re, string
        def normalize(s: str):
            s = s.lower()
            s = re.sub(r'\b(a|an|the)\b', ' ', s)
            s = ''.join(ch for ch in s if ch not in set(string.punctuation))
            return " ".join(s.split())
        def exact_match(pred, gt):
            return int(normalize(pred) == normalize(gt))
        def f1_score(pred, gt):
            ptoks = normalize(pred).split()
            gtoks = normalize(gt).split()
            if not ptoks or not gtoks:
                return 0.0
            common = set(ptoks) & set(gtoks)
            if not common:
                return 0.0
            prec = len(common) / len(ptoks)
            rec = len(common) / len(gtoks)
            return 2 * prec * rec / (prec + rec)
    if not golds:
        return None, None
    ems = [exact_match(pred, g) for g in golds]
    f1s = [f1_score(pred, g) for g in golds]
    return float(max(ems)), float(max(f1s))



def timing(func):
    """
    Decorator to measure execution time of a function.
    """
    @functools.wraps(func)
    def wrapper_timing(*args, **kwargs):
        start = time.monotonic()
        result = func(*args, **kwargs)
        elapsed = time.monotonic() - start
        print(f"[TIMER] {func.__name__}: {elapsed:.2f}s")
        return result
    return wrapper_timing



def measure_latency(func, *args, repeats=10, **kwargs):
    """
    Measure median and p95 latency of a function over a number of repeats.
    Returns (median, p95) in seconds.
    """
    times = []
    for _ in range(repeats):
        start = time.monotonic()
        func(*args, **kwargs)
        times.append(time.monotonic() - start)
    return median(times), np.percentile(times, 95)



def get_memory_usage():
    """
    Return current process memory usage in MB.
    """
    process = psutil.Process()
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 * 1024)



def log_memory_usage(stage: str):
    """
    Print or log memory usage at a given stage of the pipeline.
    """
    mem_mb = get_memory_usage()
    print(f"[MEMORY] {stage}: {mem_mb:.1f} MB")



def format_dict(d: dict):
    """
    Pretty-format a dict for logging.
    """
    return ", ".join(f"{k}={v}" for k, v in d.items())

# Additional utilities (e.g., for rich logging) can be added here.






# -------------------------
# Helpers to find project pieces
# -------------------------
def _import_mainconfig_class():
    """
    Try common candidate import paths for MainConfig. Adjust candidates if your path differs.
    """
    candidates = [
        "main.main_config",     # e.g., project root main/main_config.py
        "src.main_config",      # alt
        "src.config.main_config",
        "src.main",             # sometimes defined here
        "main_config",          # direct module
    ]
    for path in candidates:
        try:
            mod = __import__(path, fromlist=["MainConfig"])
            if hasattr(mod, "MainConfig"):
                return getattr(mod, "MainConfig")
        except Exception:
            continue
    # if not found, try __main__ (maybe user provided in runtime)
    if "MainConfig" in globals():
        return globals()["MainConfig"]
    raise ImportError("Could not import MainConfig from candidates: " + ", ".join(candidates))


def _import_retriever_class():
    """
    Try common retriever class locations.
    """
    candidates = [
        ("src.retrieval", "Retriever"),
    ]
    for mod_path, cls_name in candidates:
        try:
            mod = __import__(mod_path, fromlist=[cls_name])
            if hasattr(mod, cls_name):
                return getattr(mod, cls_name)
        except Exception:
            continue
    return None


def _import_chroma_helpers():
    """
    Try to import project-provided Chroma helper functions from src.indexing.
    Expected helpers (optional):
      - create_or_get_chroma_collection(collection_name, persist_dir, embedding_dim)
      - load_chroma_collection(collection_name, persist_dir)
      - persist_chroma(client) or client.persist()
    If these are missing, we'll fall back to using chromadb directly.
    """
    helpers: Dict[str, Any] = {}
    candidates = [
        "src.indexing.chroma",       # src/indexing/chroma.py
        "src.indexing",              # src/indexing/__init__.py
        "src.indexing.utils",        # possible utils file
        "src.indexing.chroma_utils",
    ]
    for path in candidates:
        try:
            mod = __import__(path, fromlist=["create_or_get_chroma_collection"])
            # load whichever functions exist
            if hasattr(mod, "create_or_get_chroma_collection"):
                helpers["create_or_get_chroma_collection"] = getattr(mod, "create_or_get_chroma_collection")
            if hasattr(mod, "load_chroma_collection"):
                helpers["load_chroma_collection"] = getattr(mod, "load_chroma_collection")
            if hasattr(mod, "persist_chroma"):
                helpers["persist_chroma"] = getattr(mod, "persist_chroma")
            if hasattr(mod, "ChromaClientFactory"):
                helpers["ChromaClientFactory"] = getattr(mod, "ChromaClientFactory")
            # don't break — collect all available
        except Exception:
            continue
    return helpers


def _import_embedding_helpers():
    """
    Try to locate embedding utilities from your indexing/retriever modules.
    Expect functions like:
      - load_embedding_model(model_name) -> (tokenizer, model)
      - embed_texts(docs, tokenizer, model, batch_size=..)
    """
    helpers: Dict[str, Any] = {}
    candidate_modules = [
        ("src.indexing.embeddings", ["OpenAIEmbeddingProvider", "embed_documents"]),
        ("src.indexing.embeddings", ["CohereAIAIEmbeddingProvider", "embed_documents"]),
        ("src.indexing.embeddings", ["SentenceTransformersProvider", "embed_documents"]),
    ]
    for mod_path, names in candidate_modules:
        try:
            mod = __import__(mod_path, fromlist=names)
            for name in names:
                if hasattr(mod, name):
                    helpers[name] = getattr(mod, name)
        except Exception:
            continue
    return helpers


def _import_chunkers():
    """
    Try to import chunkers from src.chunker.chunkers
    """
    chunkers = {}
    try:
        mod = __import__("src.chunker.chunkers", fromlist=["SlidingWindowChunker", "fixed_chunker"])
        if hasattr(mod, "SlidingWindowChunker"):
            chunkers["SlidingWindowChunker"] = getattr(mod, "SlidingWindowChunker")
        if hasattr(mod, "fixed_chunker"):
            chunkers["fixed_chunker"] = getattr(mod, "fixed_chunker")
    except Exception:
        # best-effort fallback: try different names
        try:
            mod = __import__("src.chunker", fromlist=["chunkers"])
            if hasattr(mod, "chunkers"):
                cmod = getattr(mod, "chunkers")
                if hasattr(cmod, "SlidingWindowChunker"):
                    chunkers["SlidingWindowChunker"] = getattr(cmod, "SlidingWindowChunker")
        except Exception:
            pass
    return chunkers



# -------------------------
# System stats + QA scoring
# -------------------------
def _system_stats() -> Dict[str, float]:
    p = psutil.Process()
    return {
        "memory_mb": p.memory_info().rss / (1024.0 * 1024.0),
        "cpu_percent": p.cpu_percent(interval=0.01),
    }









# -------------------------
# Minimal Chroma Retriever wrapper (fallback)
# -------------------------
class ChromaRetriever:
    """
    Minimal wrapper to query a chroma collection (collection.query)
    Expects either a chroma Collection object or a client + collection_name pair.
    """
    def __init__(self, collection: Any = None, client: Any = None, collection_name: Optional[str] = None):
        self.collection = collection
        self.client = client
        self.collection_name = collection_name

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """
        Return list of (doc_idx, score). We store original doc_idx in metadata['doc_index'].
        """
        if self.collection is None and self.client and self.collection_name:
            try:
                self.collection = self.client.get_collection(self.collection_name)
            except Exception:
                pass
        if self.collection is None:
            raise RuntimeError("No chroma collection available for retrieval.")
        # try collection.query interface
        try:
            resp = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=['metadatas', 'distances', 'documents', 'ids']
            )
            # response structure may vary; try to extract
            docs = resp.get("documents", [[]])[0] if isinstance(resp, dict) else None
            metadatas = resp.get("metadatas", [[]])[0] if isinstance(resp, dict) else None
            distances = resp.get("distances", [[]])[0] if isinstance(resp, dict) else None
            ids = resp.get("ids", [[]])[0] if isinstance(resp, dict) else None
            results = []
            # prefer metadata doc_index
            if metadatas:
                for i, md in enumerate(metadatas):
                    di = md.get("doc_index") if isinstance(md, dict) else None
                    score = distances[i] if distances and i < len(distances) else 0.0
                    if di is None:
                        # fallback to id (if id encodes index)
                        try:
                            di = int(ids[i])
                        except Exception:
                            di = i
                    results.append((int(di), float(score)))
            else:
                # fallback: use ids as indices if numeric
                for i, _ in enumerate(docs or []):
                    try:
                        idx = int(ids[i])
                    except Exception:
                        idx = i
                    sc = distances[i] if distances and i < len(distances) else 0.0
                    results.append((idx, float(sc)))
            return results
        except Exception as e:
            raise RuntimeError(f"Chroma collection query failed: {e}")