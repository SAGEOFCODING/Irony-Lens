"""Model 1 - classical machine learning baseline.

TF-IDF features (word 1-2 grams + character 2-5 grams) fed to three linear
classifiers: Naive Bayes, Logistic Regression and Linear SVM. Hyper-parameters
are chosen on the validation set; the winner is evaluated once on the test set
and saved as a single scikit-learn Pipeline (cleaning + features + classifier).

Usage:
  python -m src.train_baseline --dataset isarcasm
"""
import argparse

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from .config import MODELS_DIR, REPORTS_DIR, SEED
from .data import DATASETS, load_splits
from .metrics import compute_metrics, save_report
from .preprocess import clean_batch

# class_weight="balanced" makes mistakes on the rarer class cost more, which
# matters for iSarcasm where only ~25% of tweets are sarcastic.
CANDIDATES = {
    "naive_bayes": (lambda **p: MultinomialNB(**p), {"alpha": [0.1, 0.3, 1.0]}),
    "logistic_regression": (
        lambda **p: LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED, **p),
        {"C": [0.5, 2.0, 8.0]},
    ),
    "linear_svm": (
        lambda **p: LinearSVC(class_weight="balanced", random_state=SEED, **p),
        {"C": [0.05, 0.2, 1.0]},
    ),
}


def build_features():
    """Word n-grams capture phrases ("yeah right", "oh great"); character
    n-grams capture spelling, elongation ("sooo"), punctuation runs ("!!!")
    and emoji tokens, and are robust to typos common on social media."""
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=200_000, sublinear_tf=True)),
    ])


def scores(clf, X):
    """Probability of 'sarcastic' when available, otherwise a margin score."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return 1 / (1 + np.exp(-clf.decision_function(X)))


def top_features(features, clf, k=20):
    names = features.get_feature_names_out()
    if hasattr(clf, "coef_"):
        weights = clf.coef_[0]
    else:  # Naive Bayes: log-probability ratio between the classes
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    order = np.argsort(weights)
    rows = [(names[i], weights[i], "sarcastic") for i in order[::-1][:k]]
    rows += [(names[i], weights[i], "not sarcastic") for i in order[:k]]
    return pd.DataFrame(rows, columns=["feature", "weight", "indicates"])


def plot_top_features(df, path):
    df = df.sort_values("weight")
    fig, ax = plt.subplots(figsize=(7, 9))
    colors = ["#2b6cb0" if w < 0 else "#c53030" for w in df["weight"]]
    ax.barh(df["feature"], df["weight"], color=colors)
    ax.set_title("Most informative features (red = sarcastic)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="isarcasm")
    args = parser.parse_args()

    data = load_splits(args.dataset)
    train, val, test = data["train"], data["val"], data["test"]

    print("Cleaning text and fitting TF-IDF features ...")
    cleaner = FunctionTransformer(clean_batch)
    features = build_features()
    X_train = features.fit_transform(clean_batch(train["text"]))
    X_val = features.transform(clean_batch(val["text"]))
    X_test = features.transform(clean_batch(test["text"]))
    print(f"Feature matrix: {X_train.shape[0]:,} posts x {X_train.shape[1]:,} features")

    # Model selection on the validation set only - the test set stays unseen.
    results, best = [], None
    for name, (make, grid) in CANDIDATES.items():
        for params in ParameterGrid(grid):
            clf = make(**params).fit(X_train, train["label"])
            m = compute_metrics(val["label"], clf.predict(X_val))
            results.append({"model": name, **params, **m})
            print(f"  {name:<20} {params}  val macro-F1 = {m['f1_macro']:.4f}")
            if best is None or m["f1_macro"] > best[0]:
                best = (m["f1_macro"], name, params, clf)

    _, best_name, best_params, best_clf = best
    print(f"\nBest on validation: {best_name} {best_params}")

    out_reports = REPORTS_DIR / args.dataset
    out_reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_reports / "baseline_model_selection.csv", index=False)

    save_report(
        "baseline_tfidf", args.dataset, test["text"], test["label"].tolist(),
        best_clf.predict(X_test).tolist(), scores(best_clf, X_test).tolist(),
        extra={"classifier": best_name, "params": best_params},
    )

    feats = top_features(features, best_clf)
    feats.to_csv(out_reports / "baseline_top_features.csv", index=False)
    plot_top_features(feats, out_reports / "baseline_top_features.png")

    # Fitted steps assembled into one object: raw text in, prediction out.
    pipeline = Pipeline([("clean", cleaner.fit(None)), ("features", features), ("clf", best_clf)])
    out_models = MODELS_DIR / args.dataset
    out_models.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_models / "baseline.joblib")
    print(f"Saved model to {out_models / 'baseline.joblib'}")


if __name__ == "__main__":
    main()
