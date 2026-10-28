"""Load the trained models for a dataset and classify new text.

Usage:
  python -m src.predict --dataset isarcasm "Oh great, another Monday. Just what I needed."
  python -m src.predict --dataset reddit "Yeah, because that worked so well last time" --context "Let's try it again"
"""
import argparse
import json

import joblib
import numpy as np

from .config import MODELS_DIR
from .data import DATASETS


class SarcasmDetector:
    """Wraps whichever of the three models exist for a dataset.

    torch / transformers are imported only when their model is present, so
    the baseline works on a machine with just scikit-learn installed.
    """

    def __init__(self, dataset):
        base = MODELS_DIR / dataset
        self.baseline = self.lstm = self.transformer = None

        if (base / "baseline.joblib").exists():
            self.baseline = joblib.load(base / "baseline.joblib")

        if (base / "lstm.pt").exists():
            from .lstm import load_lstm
            self.lstm, self.vocab, self.lstm_cfg = load_lstm(base / "lstm.pt")

        if (base / "transformer").exists():
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            path = base / "transformer"
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.transformer = AutoModelForSequenceClassification.from_pretrained(path).eval()
            self.tf_meta = json.loads((path / "project_meta.json").read_text())

    @property
    def available(self):
        names = {"baseline": self.baseline, "bilstm": self.lstm, "transformer": self.transformer}
        return [n for n, m in names.items() if m is not None]

    @property
    def uses_context(self):
        return self.transformer is not None and self.tf_meta.get("use_context", False)

    def predict(self, text, context=""):
        """Return {model_name: probability_of_sarcasm}."""
        out = {}
        if self.baseline is not None:
            clf = self.baseline.named_steps["clf"]
            if hasattr(clf, "predict_proba"):
                out["baseline"] = float(self.baseline.predict_proba([text])[0, 1])
            else:
                out["baseline"] = float(1 / (1 + np.exp(-self.baseline.decision_function([text])[0])))
        if self.lstm is not None:
            probs, _ = self._lstm_forward(text)
            out["bilstm"] = probs
        if self.transformer is not None:
            out["transformer"] = self._transformer_prob(text, context)
        return out

    def explain_baseline(self, text, k=8):
        """Features of this post that pushed the linear model most towards
        each class: weight x TF-IDF value. Positive = towards sarcastic."""
        clf = self.baseline.named_steps["clf"]
        if not hasattr(clf, "coef_"):
            return []
        X = self.baseline[:-1].transform([text]).tocoo()
        names = self.baseline.named_steps["features"].get_feature_names_out()
        contrib = [(names[j], float(v * clf.coef_[0][j])) for j, v in zip(X.col, X.data)]
        contrib.sort(key=lambda x: abs(x[1]), reverse=True)
        return contrib[:k]

    def explain_lstm(self, text):
        """(token, attention weight) pairs - which words the BiLSTM focused on."""
        from .lstm import tokenize
        _, weights = self._lstm_forward(text)
        tokens = tokenize(text)[: self.lstm_cfg["max_len"]] or ["<unk>"]
        return list(zip(tokens, weights))

    def _lstm_forward(self, text):
        import torch
        ids = self.vocab.encode(text, self.lstm_cfg["max_len"])
        with torch.no_grad():
            logits, attn = self.lstm(torch.tensor([ids]), torch.tensor([len(ids)]))
        return float(torch.softmax(logits, dim=1)[0, 1]), attn[0].tolist()

    def _transformer_prob(self, text, context=""):
        import torch
        from .preprocess import clean_text
        text = clean_text(text, demojize=False)
        max_len = self.tf_meta["max_len"]
        if self.uses_context:
            enc = self.tokenizer(text, context or "", truncation="longest_first", max_length=max_len, return_tensors="pt")
        else:
            enc = self.tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
        with torch.no_grad():
            logits = self.transformer(**enc).logits
        return float(torch.softmax(logits, dim=1)[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text")
    parser.add_argument("--dataset", choices=DATASETS, default="isarcasm")
    parser.add_argument("--context", default="", help="Parent comment (used if the transformer was trained with context)")
    args = parser.parse_args()

    detector = SarcasmDetector(args.dataset)
    if not detector.available:
        raise SystemExit(f"No trained models in {MODELS_DIR / args.dataset}. Train one first.")
    for name, p in detector.predict(args.text, args.context).items():
        verdict = "SARCASTIC" if p >= 0.5 else "not sarcastic"
        print(f"{name:<12} {p:6.1%}  {verdict}")
    if detector.baseline is not None:
        print("\nBaseline - strongest features in this text:")
        for feat, w in detector.explain_baseline(args.text):
            print(f"  {feat:<30} {w:+.3f}")


if __name__ == "__main__":
    main()
