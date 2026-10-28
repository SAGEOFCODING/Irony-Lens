"""Download, load and split the sarcasm datasets.

Every model reads the same cached train/val/test CSVs from data/processed,
so all results are directly comparable.

Datasets
  isarcasm   Twitter, SemEval-2022 Task 6. Labelled by the authors of the
             tweets themselves (intended sarcasm). Downloads automatically.
  reddit     Reddit SARC corpus (balanced, ~1M comments) with the parent
             comment as context. Must be downloaded from Kaggle once.
  headlines  News headlines (The Onion vs HuffPost). Clean labels, used as an
             easy reference point. Downloads automatically.

Usage:
  python -m src.data --dataset isarcasm
  python -m src.data --dataset reddit --sample-size 100000 --force
"""
import argparse
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import PROCESSED_DIR, RAW_DIR, SEED

SOURCES = {
    "isarcasm_train": "https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/train/train.En.csv",
    "isarcasm_test": "https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/test/task_A_En_test.csv",
    "headlines": "https://raw.githubusercontent.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection/master/Sarcasm_Headlines_Dataset.json",
}
REDDIT_FILE = RAW_DIR / "train-balanced-sarcasm.csv"
DATASETS = ("isarcasm", "reddit", "headlines")
SPLITS = ("train", "val", "test")
COLUMNS = ["text", "context", "label"]


def _download(key):
    url = SOURCES[key]
    dest = RAW_DIR / url.rsplit("/", 1)[-1]
    if not dest.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def _finalise(df):
    """Bring any source into the common schema: text, context, label."""
    df = df.copy()
    if "context" not in df:
        df["context"] = ""
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    # Duplicates across train and test would inflate scores (data leakage).
    df = df.drop_duplicates(subset=["text"])
    df["context"] = df["context"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df[COLUMNS].reset_index(drop=True)


def _split(df, val_size=0.1, test_size=0.1):
    """Stratified 80/10/10 split: every split keeps the original class ratio."""
    train, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=SEED)
    train, val = train_test_split(
        train, test_size=val_size / (1 - test_size), stratify=train["label"], random_state=SEED
    )
    return train, val, test


def build_isarcasm(sample_size=None):
    train = pd.read_csv(_download("isarcasm_train"), encoding_errors="replace")
    test = pd.read_csv(_download("isarcasm_test"), encoding_errors="replace")
    train = _finalise(train.rename(columns={"tweet": "text", "sarcastic": "label"}))
    test = _finalise(test.rename(columns={"sarcastic": "label"}))
    # The official test set is kept untouched; validation is carved from train.
    train = train[~train["text"].isin(set(test["text"]))]
    train, val = train_test_split(train, test_size=0.1, stratify=train["label"], random_state=SEED)
    return train, val, test


def build_reddit(sample_size=100_000):
    if not REDDIT_FILE.exists():
        raise SystemExit(
            f"Reddit data not found at {REDDIT_FILE}.\n"
            "Download 'train-balanced-sarcasm.csv' from https://www.kaggle.com/datasets/danofer/sarcasm\n"
            "(or run: kaggle datasets download -d danofer/sarcasm) and place it in data/raw/."
        )
    df = pd.read_csv(REDDIT_FILE, usecols=["label", "comment", "parent_comment"])
    df = _finalise(df.rename(columns={"comment": "text", "parent_comment": "context"}))
    if sample_size and len(df) > sample_size:
        # Equal number per class keeps the subset balanced like the full corpus.
        df = df.groupby("label", group_keys=False).sample(n=sample_size // 2, random_state=SEED)
    return _split(df)


def build_headlines(sample_size=None):
    df = pd.read_json(_download("headlines"), lines=True)
    return _split(_finalise(df.rename(columns={"headline": "text", "is_sarcastic": "label"})))


BUILDERS = {"isarcasm": build_isarcasm, "reddit": build_reddit, "headlines": build_headlines}


def load_splits(name, sample_size=100_000, force=False):
    """Return {"train": df, "val": df, "test": df}, building the cache if needed."""
    if name not in BUILDERS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {DATASETS}.")
    paths = {s: PROCESSED_DIR / f"{name}_{s}.csv" for s in SPLITS}
    if force or not all(p.exists() for p in paths.values()):
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for split, df in zip(SPLITS, BUILDERS[name](sample_size=sample_size)):
            df.to_csv(paths[split], index=False)
    # keep_default_na=False: an empty context stays "" and a post that says
    # "null" or "NaN" stays text instead of becoming a missing value.
    return {s: pd.read_csv(p, keep_default_na=False) for s, p in paths.items()}


def describe(splits):
    for split, df in splits.items():
        pos = df["label"].mean()
        print(f"{split:>5}: {len(df):>7,} rows | sarcastic {pos:6.1%} | avg words {df['text'].str.split().str.len().mean():.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="isarcasm")
    parser.add_argument("--sample-size", type=int, default=100_000, help="Reddit only: rows to keep")
    parser.add_argument("--force", action="store_true", help="Rebuild the cached splits")
    args = parser.parse_args()
    describe(load_splits(args.dataset, args.sample_size, args.force))


if __name__ == "__main__":
    main()
