# -*- coding: utf-8 -*-
"""Approach 3: BERT-based context-sensitive correction.

This version fixes three evaluation/design issues found in earlier drafts:
1. The test split is fixed to the same 28 examples used in the report.
2. The main evaluation does not pass the gold domain to the corrector; the domain
   classifier must predict the domain from the input text.  A known-domain result
   is printed only as an oracle/reference analysis.
3. Dynamic confusion pairs extracted from training data are filtered so that
   single-character accidental replacements such as 用 -> 要 do not cause broad
   substring over-correction.  Manual confusion sets are still treated as a closed
   candidate table, so the task remains known-confusion correction, not unknown
   error discovery.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from ime_common import (
    BASE_DATA,
    MANUAL_CONFUSION_SETS,
    DOMAIN_LABELS,
    ID_TO_DOMAIN,
    extract_replacements,
)

# Required packages:
# pip install torch transformers fugashi ipadic unidic_lite
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cl-tohoku/bert-base-japanese"
SEED = 42
N_TEST = 28
MAX_LEN_CLS = 64
MAX_LEN_WORD = 96


def split_fixed_n(data: List[dict], n_test: int = N_TEST, seed: int = SEED):
    rng = random.Random(seed)
    rows = list(data)
    rng.shuffle(rows)
    return rows[n_test:], rows[:n_test]


def build_safe_confusion_map(train_data: List[dict], manual_confusion_sets=None):
    """Build domain -> noisy span -> set(clean spans).

    Manual confusion sets are regarded as an externally supplied closed candidate
    table.  Training-data-derived pairs are added only when both sides have length
    >= 2, because single-character spans are too broad for string.replace and tend
    to produce false corrections inside unrelated words.
    """
    cmap = defaultdict(lambda: defaultdict(set))
    manual_confusion_sets = manual_confusion_sets or {}

    # Fix a likely typo in the hand-written table without mutating the source file.
    # The corpus contains 「契約内容を確認する」 vs 「契約内容を角認する」, so 角認
    # should map to 確認, not 契約.
    manual_overrides = {
        "law": [["確認", "角認"], ["法律", "法率"], ["根拠", "根居"], ["提出", "提主"]],
        "medical": [["診断", "診談"], ["立てる", "建てる"], ["注意", "中意"], ["改善", "改前"]],
        "math": [["空間", "空館"]],
    }

    for domain, entries in manual_confusion_sets.items():
        for entry in entries:
            if not entry:
                continue
            clean = entry[0]
            for noisy in entry[1:]:
                if noisy and clean and noisy != clean:
                    cmap[domain][noisy].add(clean)
    for domain, entries in manual_overrides.items():
        for entry in entries:
            clean = entry[0]
            for noisy in entry[1:]:
                cmap[domain][noisy].add(clean)

    for item in train_data:
        domain = item["domain"]
        for clean_span, noisy_span in extract_replacements(item["clean"], item["noise"]):
            if not clean_span or not noisy_span or clean_span == noisy_span:
                continue
            # Avoid broad single-character replacement rules.
            if len(clean_span) < 2 or len(noisy_span) < 2:
                continue
            cmap[domain][noisy_span].add(clean_span)
    return cmap


def replace_longest_first(text: str, replacements: List[Tuple[str, str]]) -> str:
    """Apply replacements in descending source-length order."""
    out = text
    for src, dst in sorted(replacements, key=lambda p: len(p[0]), reverse=True):
        out = out.replace(src, dst)
    return out


class PairDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        a, b, label = self.rows[idx]
        if b is None:
            enc = self.tokenizer(a, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        else:
            enc = self.tokenizer(a, b, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()} | {"labels": torch.tensor(label, dtype=torch.long)}


def build_training_rows(train_data, confusion_map):
    cls_rows = []
    word_rows = []
    for item in train_data:
        label = DOMAIN_LABELS[item["domain"]]
        cls_rows.append((item["clean"], None, label))
        cls_rows.append((item["noise"], None, label))

        domain_map = confusion_map.get(item["domain"], {})
        # Positive: noisy span in noisy sentence.
        for noisy in domain_map.keys():
            if noisy in item["noise"]:
                word_rows.append((item["noise"], noisy, 1))
        # Negative: clean candidate in clean sentence.
        for cleans in domain_map.values():
            for clean in cleans:
                if clean in item["clean"]:
                    word_rows.append((item["clean"], clean, 0))
    return cls_rows, word_rows


def train_classifier(model, dataset, epochs=3, batch_size=8, lr=3e-5, device="cpu"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_n = 0
        total_ok = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            total_ok += int((out.logits.argmax(dim=-1) == batch["labels"]).sum().item())
            total_n += len(batch["labels"])
        print(f"epoch={epoch} loss={total_loss / max(1, len(loader)):.4f} acc={total_ok / max(1, total_n):.4f}")


def make_corrector(tokenizer, cls_model, word_model, confusion_map, threshold=0.55, device="cpu"):
    cls_model.eval()
    word_model.eval()

    def predict_domain(text: str) -> str:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN_CLS).to(device)
        with torch.no_grad():
            pred = int(cls_model(**enc).logits.argmax(dim=-1).item())
        return ID_TO_DOMAIN[pred]

    def prob_error(text: str, word: str) -> float:
        enc = tokenizer(text, word, return_tensors="pt", truncation=True, max_length=MAX_LEN_WORD).to(device)
        with torch.no_grad():
            return float(torch.softmax(word_model(**enc).logits, dim=-1)[0, 1].item())

    def correct(text: str, domain: str | None = None):
        d = predict_domain(text) if domain is None else domain
        replacements = []
        for noisy, cleans in confusion_map.get(d, {}).items():
            if noisy not in text:
                continue
            if prob_error(text, noisy) >= threshold:
                # If multiple candidates exist, choose the shortest edit / lexicographically stable fallback.
                clean = sorted(cleans, key=lambda c: (abs(len(c) - len(noisy)), c))[0]
                replacements.append((noisy, clean))
        return replace_longest_first(text, replacements)

    return correct


def make_dictionary_baseline(confusion_map, use_gold_domain=True):
    def correct(text: str, domain: str | None = None):
        if domain is None:
            # No domain classifier in dictionary baseline; try all domains conservatively.
            domain_maps = confusion_map.values()
        else:
            domain_maps = [confusion_map.get(domain, {})]
        replacements = []
        for domain_map in domain_maps:
            for noisy, cleans in domain_map.items():
                if noisy in text:
                    clean = sorted(cleans, key=lambda c: (abs(len(c) - len(noisy)), c))[0]
                    replacements.append((noisy, clean))
        return replace_longest_first(text, replacements)
    return correct


def evaluate(data, correct_fn, name: str, pass_gold_domain: bool):
    total = len(data)
    correct_count = 0
    for item in data:
        pred = correct_fn(item["noise"], item["domain"] if pass_gold_domain else None)
        correct_count += int(pred == item["clean"])
    acc = correct_count / total if total else 0.0
    print(f"{name}: total={total}, correct={correct_count}, accuracy={acc:.4f}")
    return {"name": name, "total": total, "correct": correct_count, "accuracy": acc}


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    train_data, test_data = split_fixed_n(BASE_DATA, N_TEST, SEED)
    confusion_map = build_safe_confusion_map(train_data, MANUAL_CONFUSION_SETS)
    print(f"train={len(train_data)}, test={len(test_data)}")

    print("\n=== Dictionary baseline ===")
    dict_correct = make_dictionary_baseline(confusion_map)
    evaluate(test_data, dict_correct, "test / known-domain dictionary", pass_gold_domain=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    cls_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    word_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    cls_rows, word_rows = build_training_rows(train_data, confusion_map)
    print(f"cls_rows={len(cls_rows)}, word_rows={len(word_rows)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n=== Train domain classifier ===")
    train_classifier(cls_model, PairDataset(cls_rows, tokenizer, MAX_LEN_CLS), epochs=3, batch_size=8, lr=3e-5, device=device)
    print("\n=== Train word error classifier ===")
    train_classifier(word_model, PairDataset(word_rows, tokenizer, MAX_LEN_WORD), epochs=3, batch_size=8, lr=3e-5, device=device)

    bert_correct = make_corrector(tokenizer, cls_model.to(device), word_model.to(device), confusion_map, threshold=0.55, device=device)
    print("\n=== Approach 3: BERT correction ===")
    evaluate(test_data, bert_correct, "test / predicted-domain BERT", pass_gold_domain=False)
    evaluate(test_data, bert_correct, "test / known-domain BERT oracle", pass_gold_domain=True)


if __name__ == "__main__":
    main()
