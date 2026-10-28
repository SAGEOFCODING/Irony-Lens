"""Model 3 - transfer learning: fine-tune a pre-trained transformer.

Default model: cardiffnlp/twitter-roberta-base, a RoBERTa model further
pre-trained on ~58M tweets, so it already knows slang, hashtags and emojis.
A GPU is strongly recommended (Google Colab's free T4 is enough).

Usage:
  python -m src.train_transformer --dataset isarcasm
  python -m src.train_transformer --dataset reddit --use-context --max-train 40000
  python -m src.train_transformer --dataset isarcasm --model-name distilroberta-base   # faster, smaller
"""
import argparse
import json
import math
import shutil

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .config import MODELS_DIR, SEED
from .data import DATASETS, load_splits
from .metrics import compute_metrics, save_report
from .preprocess import clean_text


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy for imbalanced datasets."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = self.class_weights.to(outputs.logits.device)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def trainer_metrics(eval_pred):
    logits, labels = eval_pred
    return compute_metrics(labels, logits.argmax(axis=-1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="isarcasm")
    parser.add_argument("--model-name", default="cardiffnlp/twitter-roberta-base")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--use-context", action="store_true", help="Feed the parent comment as a second segment (Reddit)")
    parser.add_argument("--max-train", type=int, help="Subsample the training set to this many rows (faster)")
    args = parser.parse_args()

    data = load_splits(args.dataset)
    train = data["train"]
    if args.max_train and len(train) > args.max_train:
        train = train.sample(n=args.max_train, random_state=SEED)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def to_dataset(df):
        df = df.assign(text=[clean_text(t, demojize=False) for t in df["text"]]).reset_index(drop=True)
        ds = Dataset.from_pandas(df[["text", "context", "label"]])

        def tokenize(batch):
            if args.use_context:
                # Sentence-pair input: [CLS] reply [SEP] parent comment [SEP].
                # Self-attention lets every word of the reply look at the parent.
                return tokenizer(batch["text"], batch["context"], truncation="longest_first", max_length=args.max_len)
            return tokenizer(batch["text"], truncation=True, max_length=args.max_len)

        return ds.map(tokenize, batched=True, remove_columns=["text", "context"]).rename_column("label", "labels")

    train_ds, val_ds, test_ds = to_dataset(train), to_dataset(data["val"]), to_dataset(data["test"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2,
        id2label={0: "not_sarcastic", 1: "sarcastic"}, label2id={"not_sarcastic": 0, "sarcastic": 1},
    )

    counts = np.bincount(train["label"], minlength=2)
    class_weights = torch.tensor(len(train) / (2 * counts), dtype=torch.float)

    out_dir = MODELS_DIR / args.dataset / "transformer"
    ckpt_dir = MODELS_DIR / args.dataset / "transformer_checkpoints"
    # Warm-up over the first 10% of all optimisation steps.
    warmup_steps = int(0.1 * math.ceil(len(train) / args.batch_size) * args.epochs)
    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,           # small LR: nudge pre-trained weights, don't overwrite them
        weight_decay=0.01,
        warmup_steps=warmup_steps,       # ramp LR up gently at the start for stability
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=torch.cuda.is_available(),  # mixed precision: ~2x faster on GPU
        seed=SEED,
        logging_steps=50,
        report_to="none",
    )
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=trainer_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights,
    )
    trainer.train()

    logits = trainer.predict(test_ds).predictions
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    short_name = args.model_name.split("/")[-1]
    save_report(
        f"transformer_{short_name}", args.dataset, data["test"]["text"], data["test"]["label"].tolist(),
        (probs >= 0.5).astype(int).tolist(), probs.tolist(),
        extra={"base_model": args.model_name, "use_context": args.use_context, "train_rows": len(train)},
    )

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    meta = {"base_model": args.model_name, "use_context": args.use_context, "max_len": args.max_len}
    (out_dir / "project_meta.json").write_text(json.dumps(meta, indent=2))
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    print(f"Saved model to {out_dir}")


if __name__ == "__main__":
    main()
