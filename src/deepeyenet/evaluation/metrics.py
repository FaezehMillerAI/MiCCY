from __future__ import annotations

from collections import Counter
import math
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from deepeyenet.data.vocab import simple_tokenize


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(len(tokens) - n + 1, 0)))


def bleu_score(reference: str, prediction: str, max_n: int = 4) -> dict[str, float]:
    ref_tokens = simple_tokenize(reference)
    pred_tokens = simple_tokenize(prediction)
    if not pred_tokens:
        return {f"bleu_{n}": 0.0 for n in range(1, max_n + 1)}
    scores = {}
    for n in range(1, max_n + 1):
        ref_ngrams = _ngrams(ref_tokens, n)
        pred_ngrams = _ngrams(pred_tokens, n)
        overlap = sum((ref_ngrams & pred_ngrams).values())
        total = max(sum(pred_ngrams.values()), 1)
        precision = overlap / total
        bp = math.exp(min(0.0, 1 - len(ref_tokens) / max(len(pred_tokens), 1)))
        scores[f"bleu_{n}"] = bp * precision
    return scores


def rouge_l(reference: str, prediction: str) -> float:
    ref = simple_tokenize(reference)
    pred = simple_tokenize(prediction)
    if not ref or not pred:
        return 0.0
    dp = [[0] * (len(pred) + 1) for _ in range(len(ref) + 1)]
    for i in range(1, len(ref) + 1):
        for j in range(1, len(pred) + 1):
            if ref[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def keyword_overlap(reference: str, prediction: str) -> dict[str, float]:
    ref = set(simple_tokenize(reference))
    pred = set(simple_tokenize(prediction))
    overlap = len(ref & pred)
    precision = overlap / max(len(pred), 1)
    recall = overlap / max(len(ref), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"report_token_precision": precision, "report_token_recall": recall, "report_token_f1": f1}


def compute_generation_metrics(references: Iterable[str], predictions: Iterable[str]) -> dict[str, float]:
    references = list(references)
    predictions = list(predictions)
    aggregate = Counter()
    for reference, prediction in zip(references, predictions):
        aggregate.update(bleu_score(reference, prediction))
        aggregate["rouge_l"] += rouge_l(reference, prediction)
        aggregate.update(keyword_overlap(reference, prediction))
    total = max(len(references), 1)
    return {key: value / total for key, value in aggregate.items()}


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, keyword_names: list[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if y_true.size == 0:
        return metrics
    y_pred = (y_prob >= 0.5).astype(int)
    micro = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    metrics.update({
        "keyword_micro_precision": float(micro[0]),
        "keyword_micro_recall": float(micro[1]),
        "keyword_micro_f1": float(micro[2]),
        "keyword_macro_precision": float(macro[0]),
        "keyword_macro_recall": float(macro[1]),
        "keyword_macro_f1": float(macro[2]),
    })
    try:
        metrics["keyword_map"] = float(average_precision_score(y_true, y_prob, average="macro"))
    except ValueError:
        metrics["keyword_map"] = 0.0
    try:
        metrics["keyword_macro_auc"] = float(roc_auc_score(y_true, y_prob, average="macro"))
    except ValueError:
        metrics["keyword_macro_auc"] = 0.0
    return metrics
