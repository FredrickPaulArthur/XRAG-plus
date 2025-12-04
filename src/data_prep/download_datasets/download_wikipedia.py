import os
from datasets import load_dataset, DownloadConfig, DatasetDict
from tqdm import tqdm

exit()


# absolute path preferred
cache_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "hf_datasets"))
os.makedirs(cache_base, exist_ok=True)

# dataset cache; where datasets stores its parquet/files etc
os.environ["HF_DATASETS_CACHE"] = cache_base

# Also pass cache_dir / download_config explicitly to load_dataset
download_cfg = DownloadConfig(cache_dir=cache_base)



LANGUAGES = ["hi", "de", "es", "ru", "en"]

# where batched json files will be written
out_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "hf_datasets_extracted"))
os.makedirs(out_base, exist_ok=True)

BATCH_SIZE = 2000

for lang in LANGUAGES:
    print(f"\n=== Processing language: {lang} ===")
    ds_loaded = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", cache_dir=cache_base, download_config=download_cfg)

    if isinstance(ds_loaded, DatasetDict):  # Select Train split in the Dict
        split_name = "train" if "train" in ds_loaded.keys() else list(ds_loaded.keys())[0]
        ds = ds_loaded[split_name]
    else:
        ds = ds_loaded

    print(f"Dataset for {lang}: {ds}\nTotal rows: {len(ds)}")

    lang_outdir = os.path.join(out_base, f"wikipedia_20231101_{lang}")
    os.makedirs(lang_outdir, exist_ok=True)

    total = len(ds)
    if total == 0:
        print(f"Skipping {lang}: empty dataset")
        continue

    # Iterate in batches and write each batch to a separate JSON file
    batch_idx = 0
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        print(f"Writing {lang} batch {batch_idx:05d} (rows {start}..{end-1}) ...", end=" ")
        batch_ds = ds.select(range(start, end))
        out_file = os.path.join(lang_outdir, f"wikipedia_{lang}_batch_{batch_idx:05d}.json")
        batch_ds.to_json(out_file, lines=True, force_ascii=False)
        print("done")
        batch_idx += 1

    print(f"Finished language {lang}: wrote {batch_idx} files to {lang_outdir}")