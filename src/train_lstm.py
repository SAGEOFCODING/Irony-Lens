"""Model 2 - deep learning: BiLSTM with attention (PyTorch).

Usage:
  python -m src.train_lstm --dataset isarcasm
  python -m src.train_lstm --dataset isarcasm --glove data/raw/glove.twitter.27B.200d.txt
"""
import argparse
import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import MODELS_DIR, SEED
from .data import DATASETS, load_splits
from .lstm import BiLSTMAttention, Vocab, collate, load_glove
from .metrics import compute_metrics, save_report


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for ids, lengths, _ in loader:
        logits, _ = model(ids.to(device), lengths)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    return torch.cat(probs).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="isarcasm")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-len", type=int, default=64, help="Tokens kept per post")
    parser.add_argument("--embed-dim", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=3, help="Early-stopping patience in epochs")
    parser.add_argument("--glove", help="Optional path to GloVe vectors with --embed-dim dimensions")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_splits(args.dataset)

    vocab = Vocab.build(data["train"]["text"])
    print(f"Vocabulary: {len(vocab):,} tokens | device: {device}")

    def loader(df, shuffle):
        items = [(vocab.encode(t, args.max_len), y) for t, y in zip(df["text"], df["label"])]
        return DataLoader(items, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate)

    train_dl, val_dl, test_dl = loader(data["train"], True), loader(data["val"], False), loader(data["test"], False)

    model = BiLSTMAttention(len(vocab), args.embed_dim, args.hidden, args.dropout).to(device)
    if args.glove:
        model.embedding.weight.data.copy_(load_glove(args.glove, vocab, args.embed_dim))

    # Weighted loss: each class contributes equally even if one is rarer.
    counts = np.bincount(data["train"]["label"], minlength=2)
    class_weights = torch.tensor(len(data["train"]) / (2 * counts), dtype=torch.float, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for ids, lengths, labels in train_dl:
            optimizer.zero_grad()
            logits, _ = model(ids.to(device), lengths)
            loss = loss_fn(logits, labels.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # avoids exploding gradients in RNNs
            optimizer.step()
            total += loss.item() * len(labels)

        val_pred = (predict(model, val_dl, device) >= 0.5).astype(int)
        val_f1 = compute_metrics(data["val"]["label"], val_pred)["f1_macro"]
        print(f"epoch {epoch:>2} | train loss {total / len(data['train']):.4f} | val macro-F1 {val_f1:.4f}")

        # Early stopping: keep the epoch that generalised best, stop when the
        # validation score has not improved for `patience` epochs (overfitting).
        if val_f1 > best_f1:
            best_f1, best_state, bad_epochs = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    test_prob = predict(model, test_dl, device)
    save_report(
        "bilstm_attention", args.dataset, data["test"]["text"], data["test"]["label"].tolist(),
        (test_prob >= 0.5).astype(int).tolist(), test_prob.tolist(),
        extra={"glove": bool(args.glove), "best_val_f1_macro": best_f1},
    )

    out = MODELS_DIR / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    config = {k: getattr(args, k) for k in ("embed_dim", "hidden", "dropout", "max_len")}
    torch.save({"state_dict": model.state_dict(), "vocab": vocab.itos, "config": config}, out / "lstm.pt")
    print(f"Saved model to {out / 'lstm.pt'}")


if __name__ == "__main__":
    main()
