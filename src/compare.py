"""Collect every model's test metrics for a dataset into one table and chart.

Usage:
  python -m src.compare --dataset isarcasm
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import REPORTS_DIR
from .data import DATASETS

SHOWN = ["accuracy", "precision_sarcastic", "recall_sarcastic", "f1_sarcastic", "f1_macro", "roc_auc"]


def load_results(dataset):
    rows = [json.loads(p.read_text()) for p in sorted((REPORTS_DIR / dataset).glob("*_metrics.json"))]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("model")[[c for c in SHOWN if c in rows[0]]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="isarcasm")
    args = parser.parse_args()

    table = load_results(args.dataset)
    if table.empty:
        raise SystemExit("No results yet - train at least one model first.")
    table = table.sort_values("f1_macro")
    out = REPORTS_DIR / args.dataset
    (out / "comparison.md").write_text(table.to_markdown(), encoding="utf8")
    print(table.to_markdown())

    ax = table[["accuracy", "f1_sarcastic", "f1_macro"]].plot.barh(figsize=(7, 1.2 + 0.9 * len(table)))
    ax.set_xlim(0, 1)
    ax.set_title(f"Test-set results - {args.dataset}")
    plt.tight_layout()
    plt.savefig(out / "comparison.png", dpi=150)
    print(f"\nSaved {out / 'comparison.md'} and comparison.png")


if __name__ == "__main__":
    main()
