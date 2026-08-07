"""
Evaluation metrics: Cross Entropy, Perplexity, Token Accuracy, BLEU-4, and ROUGE-L implemented from scratch.
"""

import math
from collections import Counter
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_perplexity(loss_val: float) -> float:
    """
    Computes Perplexity (PPL) from cross-entropy loss value: PPL = exp(loss).
    """
    try:
        return math.exp(min(loss_val, 100.0))  # Cap to prevent math overflow
    except OverflowError:
        return float("inf")


def calculate_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> float:
    """
    Calculates token-level top-1 accuracy percentage.

    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size].
        targets: Target token IDs [batch_size, seq_len].
        ignore_index: Target index to ignore in metric calculation.

    Returns:
        Accuracy score between 0.0 and 1.0.
    """
    predictions = torch.argmax(logits, dim=-1)
    mask = (targets != ignore_index)

    correct = (predictions == targets) & mask
    total_valid = mask.sum().item()

    if total_valid == 0:
        return 0.0

    return correct.sum().item() / total_valid


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Returns Counter of n-grams for token sequence."""
    return Counter([tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)])


def calculate_bleu(
    reference: List[str],
    candidate: List[str],
    max_n: int = 4
) -> float:
    """
    Calculates sentence-level BLEU score with brevity penalty from scratch.

    Args:
        reference: List of reference tokens.
        candidate: List of candidate tokens.
        max_n: Maximum n-gram order (4 for BLEU-4).

    Returns:
        BLEU score between 0.0 and 1.0.
    """
    if not candidate or not reference:
        return 0.0

    c_len = len(candidate)
    r_len = len(reference)

    # 1. Calculate Brevity Penalty (BP)
    if c_len > r_len:
        bp = 1.0
    else:
        bp = math.exp(1 - r_len / max(1, c_len))

    # 2. Compute modified n-gram precisions
    p_n: List[float] = []

    for n in range(1, max_n + 1):
        ref_ngrams = _get_ngrams(reference, n)
        cand_ngrams = _get_ngrams(candidate, n)

        clipped_count = 0
        total_cand_ngrams = sum(cand_ngrams.values())

        if total_cand_ngrams == 0:
            p_n.append(0.0)
            continue

        for ngram, count in cand_ngrams.items():
            clipped_count += min(count, ref_ngrams.get(ngram, 0))

        precision = clipped_count / total_cand_ngrams
        p_n.append(precision)

    if any(p == 0.0 for p in p_n):
        return 0.0

    # Geometric mean of modified precisions
    s = sum(math.log(p) for p in p_n) / max_n
    bleu_score = bp * math.exp(s)

    return bleu_score


def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Computes length of Longest Common Subsequence between two lists of tokens."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def calculate_rouge_l(reference: List[str], candidate: List[str]) -> Dict[str, float]:
    """
    Calculates ROUGE-L (Longest Common Subsequence precision, recall, F1).
    """
    if not reference or not candidate:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs_len = _lcs_length(reference, candidate)
    r_len = len(reference)
    c_len = len(candidate)

    precision = lcs_len / c_len if c_len > 0 else 0.0
    recall = lcs_len / r_len if r_len > 0 else 0.0

    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return {"precision": precision, "recall": recall, "f1": f1}
