# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import combinations

from ime_common import (
    BASE_DATA, MANUAL_CONFUSION_SETS, build_confusion_map, flatten_confusion_map,
    train_test_split, tokenize, detokenize, evaluate_correction
)

class NGramLM:
    def __init__(self, n: int = 2, alpha: float = 0.5):
        assert n in (2, 3)
        self.n = n
        self.alpha = alpha
        self.context_counts = Counter()
        self.ngram_counts = Counter()
        self.vocab = set()

    def train(self, texts):
        for text in texts:
            toks = tokenize(text)
            padded = ['<s>'] * (self.n - 1) + toks + ['</s>']
            self.vocab.update(toks)
            self.vocab.add('</s>')
            for i in range(self.n - 1, len(padded)):
                context = tuple(padded[i - self.n + 1:i])
                word = padded[i]
                self.context_counts[context] += 1
                self.ngram_counts[context + (word,)] += 1

    def score(self, text: str) -> float:
        toks = tokenize(text)
        padded = ['<s>'] * (self.n - 1) + toks + ['</s>']
        V = max(1, len(self.vocab))
        s = 0.0
        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[i - self.n + 1:i])
            word = padded[i]
            num = self.ngram_counts[context + (word,)] + self.alpha
            den = self.context_counts[context] + self.alpha * V
            s += math.log(num / den)
        return s


def generate_candidates(text: str, flat_confusion, max_changes: int = 2):
    """Generate no-change and replacement candidates.

    This fixes the original one-replacement-only tendency by allowing combinations of
    up to max_changes non-overlapping known noisy spans.
    """
    candidates = {text}
    matches = []
    for noisy, cleans in flat_confusion.items():
        start = text.find(noisy)
        if start >= 0:
            matches.append((start, start + len(noisy), noisy, list(cleans)))

    # Single replacements
    for _, _, noisy, cleans in matches:
        for clean in cleans:
            candidates.add(text.replace(noisy, clean))

    # Multiple replacements. Keep this small for speed and to avoid over-generation.
    for r in range(2, max_changes + 1):
        for combo in combinations(matches, r):
            spans = [(a, b) for a, b, _, _ in combo]
            # Skip overlapping spans
            if any(not (spans[i][1] <= spans[j][0] or spans[j][1] <= spans[i][0])
                   for i in range(len(spans)) for j in range(i + 1, len(spans))):
                continue
            current = text
            for _, _, noisy, cleans in combo:
                # Use the first candidate for this lightweight baseline.
                current = current.replace(noisy, cleans[0])
            candidates.add(current)
    return list(candidates)


def make_corrector(lm, flat_confusion, change_penalty: float = 0.25):
    def correct(text: str, domain=None):
        cands = generate_candidates(text, flat_confusion, max_changes=2)
        def rank_score(c):
            # penalize excessive changes so unchanged correct sentences are not destroyed
            edit_penalty = 0.0 if c == text else change_penalty
            return lm.score(c) - edit_penalty
        return max(cands, key=rank_score)
    return correct


def main():
    train_data, test_data = train_test_split(BASE_DATA, test_size=0.4, seed=42)
    cmap = build_confusion_map(train_data, MANUAL_CONFUSION_SETS, include_reverse=False)
    flat = flatten_confusion_map(cmap)

    train_clean = [x['clean'] for x in train_data]
    for n in (2, 3):
        lm = NGramLM(n=n, alpha=0.5)
        lm.train(train_clean)
        correct = make_corrector(lm, flat)
        print(f"\n=== Approach 1: {n}-gram ===")
        evaluate_correction(train_data, correct, name='train')
        evaluate_correction(test_data, correct, name='test')
        evaluate_correction(BASE_DATA, correct, name='all', verbose=0)

if __name__ == '__main__':
    main()
