"""Evaluation shared by every model: metrics, confusion matrix, error list."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .config import LABEL_NAMES, REPORTS_DIR


def compute_metrics(y_true, y_pred, y_score=None):
    """Accuracy alone is misleading on imbalanced data, so we report the
    sarcastic-class precision/recall/F1 and macro-F1 (the SemEval metric)."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_sarcastic": p,
        "recall_sarcastic": r,
        "f1_sarcastic": f,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if y_score is not None and len(set(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    return {k: round(float(v), 4) for k, v in metrics.items()}


def save_report(model_name, dataset, texts, y_true, y_pred, y_score=None, extra=None):
    """Write metrics JSON, confusion-matrix PNG and the most confident errors."""
    out_dir = REPORTS_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(y_true, y_pred, y_score)
    metrics.update({"model": model_name, "dataset": dataset, **(extra or {})})
    (out_dir / f"{model_name}_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\n=== {model_name} on {dataset} (test set) ===")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, digits=4, zero_division=0))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=LABEL_NAMES, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(model_name)
    fig.tight_layout()
    fig.savefig(out_dir / f"{model_name}_confusion.png", dpi=150)
    plt.close(fig)

    # Error analysis: the posts the model got wrong with the highest confidence
    # are the most informative ones to read and discuss.
    errors = pd.DataFrame({"text": list(texts), "true": list(y_true), "pred": list(y_pred)})
    if y_score is not None:
        errors["score"] = list(y_score)
    errors = errors[errors["true"] != errors["pred"]]
    if "score" in errors:
        errors = errors.reindex((errors["score"] - 0.5).abs().sort_values(ascending=False).index)
    errors.head(200).to_csv(out_dir / f"{model_name}_errors.csv", index=False)
    return metrics
