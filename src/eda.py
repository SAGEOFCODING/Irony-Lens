"""Exploratory data analysis: class balance, post length, stylistic cues and
most frequent words per class. Figures go to reports/<dataset>/eda/.

Usage:
  python -m src.eda --dataset isarcasm
"""
import argparse
import re
from collections import Counter

import emoji
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .config import LABEL_NAMES, REPORTS_DIR
from .data import DATASETS, load_splits

CUES = {
    "exclamation (!)": lambda t: t.count("!"),
    "question (?)": lambda t: t.count("?"),
    "ellipsis (...)": lambda t: t.count("..."),
    "ALL-CAPS words": lambda t: len(re.findall(r"\b[A-Z]{2,}\b", t)),
    "elongated words": lambda t: len(re.findall(r"([a-zA-Z])\1{2,}", t)),
    "emojis": lambda t: emoji.emoji_count(t),
    "quotes": lambda t: t.count('"') + t.count("“"),
}
WORD_RE = re.compile(r"[a-z']{3,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="isarcasm")
    args = parser.parse_args()

    df = load_splits(args.dataset)["train"]
    out = REPORTS_DIR / args.dataset / "eda"
    out.mkdir(parents=True, exist_ok=True)
    df["class"] = df["label"].map(dict(enumerate(LABEL_NAMES)))
    df["words"] = df["text"].str.split().str.len()

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    df["class"].value_counts().reindex(LABEL_NAMES).plot.bar(ax=ax, color=["#2b6cb0", "#c53030"], rot=0)
    ax.set_title("Class distribution (train)")
    fig.tight_layout(); fig.savefig(out / "class_distribution.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for label, color in zip(LABEL_NAMES, ["#2b6cb0", "#c53030"]):
        df.loc[df["class"] == label, "words"].clip(upper=80).plot.hist(bins=40, alpha=0.6, ax=ax, color=color, label=label, density=True)
    ax.set_xlabel("words per post"); ax.legend(); ax.set_title("Post length")
    fig.tight_layout(); fig.savefig(out / "length_distribution.png", dpi=150); plt.close(fig)

    # Average count of each stylistic cue per post, by class.
    cues = pd.DataFrame({name: df["text"].map(fn) for name, fn in CUES.items()})
    cue_table = cues.groupby(df["class"]).mean().T[LABEL_NAMES].round(3)
    cue_table["ratio (sarc / not)"] = (cue_table[LABEL_NAMES[1]] / cue_table[LABEL_NAMES[0]].replace(0, float("nan"))).round(2)
    ax = cue_table[LABEL_NAMES].plot.barh(figsize=(6, 4), color=["#2b6cb0", "#c53030"])
    ax.set_title("Average stylistic cues per post")
    plt.tight_layout(); plt.savefig(out / "stylistic_cues.png", dpi=150); plt.close()

    top_words = {}
    for label in LABEL_NAMES:
        counts = Counter(
            w for t in df.loc[df["class"] == label, "text"].str.lower()
            for w in WORD_RE.findall(t) if w not in ENGLISH_STOP_WORDS and w not in {"http", "https", "user"}
        )
        top_words[label] = [f"{w} ({c})" for w, c in counts.most_common(20)]

    summary = [
        f"# EDA - {args.dataset} (train split)\n",
        f"Posts: {len(df):,}  |  sarcastic: {df['label'].mean():.1%}  |  "
        f"median words: {df['words'].median():.0f}\n",
        "## Stylistic cues (average per post)\n",
        cue_table.to_markdown(),
        "\n## Most frequent words (stop words removed)\n",
        pd.DataFrame(top_words).to_markdown(index=False),
    ]
    (out / "eda_summary.md").write_text("\n".join(summary), encoding="utf8")
    print("\n".join(summary))
    print(f"\nFigures saved in {out}")


if __name__ == "__main__":
    main()
