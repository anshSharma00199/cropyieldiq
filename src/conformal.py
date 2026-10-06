"""Split-conformal helpers for classification sets and regression intervals.

Nonconformity for classification: 1 - p_hat(true class).
For regression: |y - y_hat|. Quantile uses the finite-sample correction
ceil((n+1)*(1-alpha))/n so that coverage is at least 1-alpha in expectation
under exchangeability (which temporal/spatial splits may violate).
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        raise ValueError("calibration scores are empty")
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, q_level, method="higher"))


def classification_scores(proba: np.ndarray, y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    index = {c: i for i, c in enumerate(classes)}
    return np.array([1.0 - proba[i, index[y[i]]] for i in range(len(y))], dtype=float)


def predict_sets(proba: np.ndarray, classes: np.ndarray, qhat: float) -> list[list[str]]:
    classes = np.asarray(classes)
    sets = []
    for row in proba:
        chosen = [str(classes[j]) for j in range(len(classes)) if (1.0 - row[j]) <= qhat]
        if not chosen:
            chosen = [str(classes[int(np.argmax(row))])]
        sets.append(chosen)
    return sets


def set_metrics(sets: list[list[str]], y_true) -> dict:
    y_true = [str(v) for v in y_true]
    covered = [yt in s for yt, s in zip(y_true, sets)]
    sizes = [len(s) for s in sets]
    return {
        "empirical_coverage": float(np.mean(covered)),
        "average_set_size": float(np.mean(sizes)),
        "n": int(len(y_true)),
    }


def interval_quantile(residuals: np.ndarray, alpha: float) -> float:
    return conformal_quantile(np.abs(np.asarray(residuals, dtype=float)), alpha)


def add_intervals(y_hat: np.ndarray, qhat: float) -> tuple[np.ndarray, np.ndarray]:
    y_hat = np.asarray(y_hat, dtype=float)
    return y_hat - qhat, y_hat + qhat


def interval_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_interval_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(np.asarray(hi, float) - np.asarray(lo, float)))
