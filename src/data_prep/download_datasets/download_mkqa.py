#!/usr/bin/env python3
"""
download_mkqa_from_hf.py
Download MKQA from Hugging Face (apple/mkqa) and export per-language JSONL.gz
Languages: en, es, de, ru (change LANGS as needed)
"""
import os, gzip, json, time, random, socket
from pathlib import Path
from datasets import load_dataset, DownloadConfig
from tqdm import tqdm
import httpx

# ---- CONFIG ----
LANGS = ["en", "es", "de", "ru"]
OUT_DIR = Path("../../data/mkqa")
CACHE_DIR = str(Path("../../data/hf_cache").resolve())
os.environ.setdefault("HF_DATASETS_CACHE", CACHE_DIR)
download_cfg = DownloadConfig(cache_dir=CACHE_DIR, max_retries=8)
MAX_ATTEMPTS = 6
BASE_DELAY = 1.0
JITTER = 0.3
# ----------------

def is_network_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPError):
        return True
    msg = str(exc).lower()
    if "getaddrinfo" in msg or "temporarily unavailable" in msg or "nodename" in msg:
        return True
    if isinstance(exc, (socket.gaierror, ConnectionResetError, ConnectionRefusedError)):
        return True
    return False

def retry(fn, attempts=MAX_ATTEMPTS):
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:
            attempt += 1
            if attempt >= attempts or not is_network_error(e):
                raise
            delay = BASE_DELAY * (2 ** (attempt - 1)) * (1 + random.uniform(-JITTER, JITTER))
            print(f"[retry] network error (attempt {attempt}/{attempts}): {e}. Sleeping {delay:.1f}s")
            time.sleep(delay)

def save_lang_jsonl_gz(dataset, lang, out_dir, split_name="train"):
    out_dir = Path(out_dir) / f"mkqa_{lang}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mkqa_{lang}_{split_name}.jsonl.gz"
    if out_path.exists():
        print(f"[skip] {out_path} exists")
        return
    count = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
        for ex in tqdm(dataset, desc=f"Export {lang}"):
            # mkqa dataset rows are already language-specific if you loaded per-config,
            # but double-check: if some rows contain a language field, you can filter here.
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} rows -> {out_path}")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Try to load by official dataset id first (apple/mkqa)
    print("Attempting to load 'apple/mkqa' from HF Hub...")
    for lang in LANGS:
        try:
            # apple/mkqa likely has a config per language (e.g., 'en', 'es' etc)
            ds = retry(lambda: load_dataset("apple/mkqa", lang, cache_dir=CACHE_DIR, download_config=download_cfg))
            # many MKQA variants use a single 'train' split only
            split = "train"
            # ds could be a Dataset or DatasetDict
            if hasattr(ds, "get"):
                # it's DatasetDict-like
                if isinstance(ds, dict) or "train" in ds:
                    ds_split = ds[split] if "train" in ds else next(iter(ds.values()))
                else:
                    ds_split = ds
            else:
                ds_split = ds

            print(f"Loaded apple/mkqa {lang}  |  Rows: {len(ds_split)}")
            save_lang_jsonl_gz(ds_split, lang, OUT_DIR, split_name=split)

        except Exception as e:
            print(f"[warn] failed to load apple/mkqa/{lang}: {e}")
            print("Trying fallback dataset id 'mkqa' with lang as config...")
            try:
                ds = retry(lambda: load_dataset("mkqa", lang, cache_dir=CACHE_DIR, download_config=download_cfg))
                ds_split = ds if not isinstance(ds, dict) else (ds["train"] if "train" in ds else next(iter(ds.values())))
                print(f"Loaded mkqa {lang}  |  Rows: {len(ds_split)}")
                save_lang_jsonl_gz(ds_split, lang, OUT_DIR, split_name="train")
            except Exception as e2:
                print(f"[error] fallback also failed for {lang}: {e2}. Skipping.")

if __name__ == "__main__":
    main()
