# -*- coding: utf-8 -*-
"""Approach 2: character-level BiLSTM error judgment model.

The earlier token-level BiLSTM was unreliable in environments without a Japanese
morphological analyzer because fallback tokenization could treat an entire sentence
as one token.  This version avoids that issue by working at the character level and
uses a BiLSTM to decide whether a known confusing span should be replaced in its
sentence context.
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ime_common import BASE_DATA, MANUAL_CONFUSION_SETS, build_confusion_map, evaluate_correction

torch.set_num_threads(1)

SEED = 42
N_TEST = 28
MAX_LEN = 48
PAD = "<PAD>"
UNK = "<UNK>"


def split_fixed_n(data: List[dict], n_test: int = N_TEST, seed: int = SEED):
    rng = random.Random(seed)
    rows = list(data)
    rng.shuffle(rows)
    return rows[n_test:], rows[:n_test]


def build_char_vocab(train_data: List[dict], confusion_map) -> Dict[str, int]:
    vocab = {PAD: 0, UNK: 1}
    def add_text(text: str):
        for ch in text:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    for item in train_data:
        add_text(item["noise"])
        add_text(item["clean"])
    for domain_map in confusion_map.values():
        for noisy, cleans in domain_map.items():
            add_text(noisy)
            for clean in cleans:
                add_text(clean)
    return vocab


def find_span(text: str, word: str) -> Tuple[int, int]:
    start = text.find(word)
    if start < 0:
        return 0, 0
    return start, start + len(word)


def build_judgment_rows(train_data: List[dict], confusion_map) -> List[Tuple[str, str, int]]:
    """Return rows of (sentence, target_span, label).

    label=1 means the target span is judged as an error in that sentence.
    label=0 means the span should be kept as it is.
    """
    rows: List[Tuple[str, str, int]] = []
    for item in train_data:
        domain = item["domain"]
        domain_map = confusion_map.get(domain, {})
        # Positive examples: noisy spans appearing in noisy sentences.
        for noisy in domain_map.keys():
            if noisy in item["noise"]:
                rows.append((item["noise"], noisy, 1))
        # Negative examples: correct spans appearing in clean sentences.
        for cleans in domain_map.values():
            for clean in cleans:
                if clean in item["clean"]:
                    rows.append((item["clean"], clean, 0))
    return rows


class SpanJudgmentDataset(Dataset):
    def __init__(self, rows, char2idx, max_len: int = MAX_LEN):
        self.rows = rows
        self.char2idx = char2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        sentence, word, label = self.rows[idx]
        start, end = find_span(sentence, word)
        ids = [self.char2idx.get(ch, self.char2idx[UNK]) for ch in sentence[: self.max_len]]
        marker = [0] * len(ids)
        for i in range(start, min(end, self.max_len)):
            marker[i] = 1
        ids += [self.char2idx[PAD]] * (self.max_len - len(ids))
        marker += [0] * (self.max_len - len(marker))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "marker": torch.tensor(marker, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class BiLSTMSpanJudge(nn.Module):
    def __init__(self, vocab_size: int, char_dim: int = 64, marker_dim: int = 8, hidden_dim: int = 96):
        super().__init__()
        self.char_embedding = nn.Embedding(vocab_size, char_dim, padding_idx=0)
        self.marker_embedding = nn.Embedding(2, marker_dim)
        self.lstm = nn.LSTM(char_dim + marker_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, input_ids, marker):
        x = torch.cat([self.char_embedding(input_ids), self.marker_embedding(marker)], dim=-1)
        hidden, _ = self.lstm(x)
        m = marker.float().unsqueeze(-1)
        denom = m.sum(dim=1).clamp_min(1.0)
        pooled = (hidden * m).sum(dim=1) / denom
        return self.classifier(pooled)


def train_model(model, rows, char2idx, epochs: int = 50, batch_size: int = 16, lr: float = 2e-3, device: str = "cpu"):
    ds = SpanJudgmentDataset(rows, char2idx)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_ok = 0
        total_n = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            marker = batch["marker"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, marker)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_ok += int((logits.argmax(dim=-1) == labels).sum().item())
            total_n += len(labels)
        if epoch in {0, 10, 20, epochs - 1}:
            print(f"epoch={epoch:03d} loss={total_loss / max(1, len(loader)):.4f} train_cls_acc={total_ok / max(1, total_n):.4f}")


def make_corrector(model, char2idx, confusion_map, threshold: float = 0.5, device: str = "cpu"):
    model.eval()

    def prob_error(sentence: str, word: str) -> float:
        ds = SpanJudgmentDataset([(sentence, word, 0)], char2idx)
        batch = ds[0]
        input_ids = batch["input_ids"].unsqueeze(0).to(device)
        marker = batch["marker"].unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.softmax(model(input_ids, marker), dim=-1)[0, 1].item()
        return float(prob)

    def correct(text: str, domain=None):
        corrected = text
        for noisy, cleans in confusion_map.get(domain, {}).items():
            if noisy not in corrected:
                continue
            if prob_error(corrected, noisy) >= threshold:
                corrected = corrected.replace(noisy, sorted(cleans)[0])
        return corrected

    return correct


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    train_data, test_data = split_fixed_n(BASE_DATA, N_TEST, SEED)
    confusion_map = build_confusion_map(train_data, MANUAL_CONFUSION_SETS, include_reverse=False)
    char2idx = build_char_vocab(train_data, confusion_map)
    rows = build_judgment_rows(train_data, confusion_map)
    print(f"train={len(train_data)}, test={len(test_data)}, judgment_rows={len(rows)}, vocab={len(char2idx)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BiLSTMSpanJudge(len(char2idx))
    train_model(model, rows, char2idx, epochs=50, batch_size=16, lr=2e-3, device=device)
    correct = make_corrector(model.to(device), char2idx, confusion_map, threshold=0.5, device=device)
    print("\n=== Approach 2: character-level BiLSTM span judgment ===")
    evaluate_correction(train_data, correct, name="train", verbose=0)
    evaluate_correction(test_data, correct, name="test", verbose=10)
    evaluate_correction(BASE_DATA, correct, name="all", verbose=0)


if __name__ == "__main__":
    main()
