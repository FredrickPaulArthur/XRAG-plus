#!/usr/bin/env python3

# parquet_to_json.py

# Stream multiple parquet shards and write NDJSON batches (optionally gzipped).

# Usage:
#   python download_ccnews.py /
#     --parquet_dir "..\..\data\hf_ccnews\en" /
#     --out_dir "..\..\data\hf_ccnews_extracted\en" /
#     --batch_size 2000 /
#     --compress False /


# Notes:
# - Works on large shards using pyarrow. Keeps memory usage bounded.
# - Tries to detect common text columns: 'maintext','text','content','article','body'.
# - Keeps a small set of useful columns by default: id/title/text/url/date_publish/lang
# - Skips already-existing output files so runs are resumable.


import argparse
import json
import gzip
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
from dateutil import parser as dateparser


COMMON_TEXT_COLS = ["maintext", "text", "content", "article", "body", "raw_text"]
COMMON_ID_COLS = ["id", "page_id", "doc_id"]
COMMON_TITLE_COLS = ["title", "headline"]
COMMON_URL_COLS = ["url", "source_url", "link"]
COMMON_DATE_COLS = ["date_publish", "date_published", "date", "created_at", "published"]



def pick_column(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None



def normalize_row(row, col_map):
    """
    row : pandas Series
    col_map : dict mapping standard names -> actual column names (or None)
    returns dict with keys: id, title, text, url, date_publish, language (if available)
    """
    out = {}

    if col_map.get("id"):
        out["id"] = row.get(col_map["id"])
    if col_map.get("title"):
        out["title"] = row.get(col_map["title"])

    text_col = col_map.get("text")
    out["text"] = None
    if text_col:
        val = row.get(text_col)
        if val is None:
            out["text"] = None
        else:
            out["text"] = str(val)

    if col_map.get("url"):
        out["url"] = row.get(col_map["url"])
    if col_map.get("date_publish"):
        val = row.get(col_map["date_publish"])
        try:
            out["date_publish"] = str(dateparser.parse(val)) if val is not None else None
        except Exception:
            out["date_publish"] = str(val) if val is not None else None

    if col_map.get("language"):
        out["language"] = row.get(col_map["language"])
    return out



# ---------------------------
# Main conversion
# ---------------------------
def process_parquet_files(parquet_dir, out_dir, batch_size=2000, compress=True, min_text_len=50, keep_columns=None):
    parquet_dir = Path(parquet_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect all parquet files (.parquet, .parq) in folder (non-recursive) or recursively
    parquet_paths = sorted(parquet_dir.rglob("*.parquet"))
    if not parquet_paths:
        parquet_paths = sorted(parquet_dir.rglob("*.parq"))
    if not parquet_paths:
        raise SystemExit(f"No parquet files found under {parquet_dir}")

    # Accumulate docs across shards and flush to batches of `batch_size`.
    batch = []
    batch_idx = 0
    global_doc_counter = 0

    def flush_batch(batch_docs, batch_idx_local):
        if not batch_docs:
            return
        fname = f"batch_{batch_idx_local:05d}.json"
        if compress:
            fname = fname + ".gz"
        out_path = out_dir / fname
        if compress:
            with gzip.open(out_path, "wt", encoding="utf-8") as fh:
                for d in batch_docs:
                    fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        else:
            with open(out_path, "w", encoding="utf-8") as fh:
                for d in batch_docs:
                    fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"Wrote {len(batch_docs)} docs -> {out_path}")

    # iterate shards
    for shard_path in parquet_paths:
        print(f"Reading shard: {shard_path}")
        pqfile = pq.ParquetFile(str(shard_path))        # open parquet file with pyarrow for streaming
        schema_cols = [c for c in pqfile.schema.names]  # determine schema column names
        # determine candidate columns
        text_col = pick_column(schema_cols, COMMON_TEXT_COLS)
        id_col = pick_column(schema_cols, COMMON_ID_COLS)
        title_col = pick_column(schema_cols, COMMON_TITLE_COLS)
        url_col = pick_column(schema_cols, COMMON_URL_COLS)
        date_col = pick_column(schema_cols, COMMON_DATE_COLS)
        lang_col = "language" if "language" in schema_cols else None

        col_map = {"text": text_col, "id": id_col, "title": title_col, "url": url_col, "date_publish": date_col, "language": lang_col}

        # If user provided keep_columns, respect them
        if keep_columns:
            # override col_map if those names are present
            for std_name, want in [("text", "text"), ("id", "id"), ("title", "title"), ("url", "url"), ("date_publish","date_publish"), ("language","language")]:
                for candidate in keep_columns:
                    if candidate in schema_cols:
                        col_map[std_name] = candidate

        # iterate row-groups / record batches
        for rg in range(pqfile.num_row_groups):
            try:
                table = pqfile.read_row_group(rg)
            except Exception as e:
                print(f"Error reading row group {rg} in {shard_path}: {e}")
                continue
            df = table.to_pandas()
            # iterate rows in this small DataFrame
            for _, row in df.iterrows():
                doc = normalize_row(row, col_map)
                # require text (non-empty) and length threshold
                text = doc.get("text") or ""
                if not text or len(text.strip()) < min_text_len:
                    continue
                # optionally keep only certain keys and drop None values
                cleaned = {k: v for k, v in doc.items() if v is not None}
                # add provenance
                cleaned["_source_parquet"] = str(shard_path.name)
                cleaned["_source_rowgroup"] = rg
                cleaned["_global_idx"] = global_doc_counter
                batch.append(cleaned)
                global_doc_counter += 1

                if len(batch) >= batch_size:
                    flush_batch(batch, batch_idx)
                    batch_idx += 1
                    batch = []

    # flush remainder
    if batch:
        flush_batch(batch, batch_idx)

    print(f"\n✅Done. Total docs exported: {global_doc_counter}. Output batches in {out_dir}")







def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--parquet_dir", required=True, help="Directory containing parquet files (recursive)")
    p.add_argument("--out_dir", required=True, help="Output directory for NDJSON batches")
    p.add_argument("--batch_size", type=int, default=2000)
    p.add_argument("--compress", type=lambda s: s.lower() in ("1","true","yes"), default=True)
    p.add_argument("--min_text_len", type=int, default=50)
    p.add_argument("--keep_columns", nargs="*", default=None, help="If provided, will try to use these columns (space-separated names)")
    args = p.parse_args(argv)

    process_parquet_files(args.parquet_dir, args.out_dir, batch_size=args.batch_size, compress=args.compress, min_text_len=args.min_text_len, keep_columns=args.keep_columns)

if __name__ == "__main__":
    main()