"""BiLSTM + attention model, its vocabulary and helpers (shared by training
and prediction)."""
import re
from collections import Counter

import torch
from torch import nn

from .preprocess import clean_text

# Tokens: emoji names (":face_with_rolling_eyes:"), words, runs of !?. and any
# other single symbol. Punctuation is kept because "!!!" and "?!" carry tone.
TOKEN_RE = re.compile(r":[a-z_]+:|[a-z0-9']+|[!?.]+|[^\sa-z0-9]")
PAD, UNK = "<pad>", "<unk>"


def tokenize(text):
    return TOKEN_RE.findall(clean_text(text).lower())


class Vocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {tok: i for i, tok in enumerate(itos)}

    @classmethod
    def build(cls, texts, min_freq=2, max_size=30_000):
        counts = Counter(tok for text in texts for tok in tokenize(text))
        kept = [tok for tok, c in counts.most_common(max_size) if c >= min_freq]
        return cls([PAD, UNK] + kept)

    def encode(self, text, max_len):
        ids = [self.stoi.get(tok, 1) for tok in tokenize(text)][:max_len]
        return ids or [1]

    def __len__(self):
        return len(self.itos)


def collate(batch):
    """Pad a list of (ids, label) to the longest sequence in the batch."""
    lengths = torch.tensor([len(ids) for ids, _ in batch])
    padded = torch.zeros(len(batch), int(lengths.max()), dtype=torch.long)
    for i, (ids, _) in enumerate(batch):
        padded[i, : len(ids)] = torch.tensor(ids)
    labels = torch.tensor([label for _, label in batch])
    return padded, lengths, labels


class BiLSTMAttention(nn.Module):
    """Embedding -> bidirectional LSTM -> attention pooling -> linear layer.

    The BiLSTM reads the post left-to-right and right-to-left, so each word's
    vector knows its full context. Attention learns which words matter most
    (e.g. "love" in "I love waiting 2 hours") and returns their weights, which
    we use to explain predictions.
    """

    def __init__(self, vocab_size, embed_dim=200, hidden=128, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(2 * hidden, 1)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, 2)

    def forward(self, ids, lengths):
        x = self.dropout(self.embedding(ids))
        # Packing makes the LSTM skip padding instead of reading it as words.
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=ids.size(1))
        scores = self.attention(h).squeeze(-1).masked_fill(ids == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = (weights.unsqueeze(-1) * h).sum(dim=1)
        return self.out(self.dropout(pooled)), weights


def load_glove(path, vocab, dim):
    """Initialise embeddings from GloVe (e.g. glove.twitter.27B.200d.txt).
    Words GloVe does not know keep their random initialisation."""
    weights = torch.randn(len(vocab), dim) * 0.1
    weights[0] = 0
    found = 0
    with open(path, encoding="utf8") as f:
        for line in f:
            word, *values = line.rstrip().split(" ")
            idx = vocab.stoi.get(word)
            if idx is not None and len(values) == dim:
                weights[idx] = torch.tensor([float(v) for v in values])
                found += 1
    print(f"GloVe: found vectors for {found:,}/{len(vocab):,} words")
    return weights


def load_lstm(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    vocab = Vocab(ckpt["vocab"])
    cfg = ckpt["config"]
    model = BiLSTMAttention(len(vocab), cfg["embed_dim"], cfg["hidden"], cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), vocab, cfg
