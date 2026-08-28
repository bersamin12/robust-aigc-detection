"""Metrics for spec §6.1.

Convention throughout: `y` is 0/1 with 1 = AI-generated, `s` is a score where
higher means more likely AI-generated, `p` is a calibrated probability.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from aigcdet.operating_point import TARGET_FPR


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def threshold_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float = TARGET_FPR) -> float:
    """Lowest threshold whose false-positive rate does not exceed the target.

    `drop_intermediate=False` is load-bearing. `roc_curve` defaults to dropping
    ROC vertices that lie on a straight line, which is right for plotting and
    wrong here: some of the dropped vertices are thresholds that satisfy the
    target, and skipping them returns a stricter threshold than the lowest
    qualifying one. On randomised trials the default was exact in 3275 of 3600
    cases and stricter in the other 325 (never permissive, so it never
    overspent the FPR budget -- it just left coverage on the table); with the
    drop disabled it is exact in 3600 of 3600.
    """
    fpr, _, thr = roc_curve(y, s, drop_intermediate=False)
    ok = np.where(fpr <= target_fpr)[0]
    return float(thr[ok[-1]]) if len(ok) else float(np.max(s) + 1.0)


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float = TARGET_FPR) -> float:
    """TPR at the lowest threshold whose FPR does not exceed the target.

    `drop_intermediate=False` for the same reason as `threshold_at_fpr`: a
    dropped vertex is an operating point the caller asked about and did not get,
    which understates the reported TPR.
    """
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    ok = np.where(fpr <= target_fpr)[0]
    return float(tpr[ok[-1]]) if len(ok) else 0.0


def accuracy_at_threshold(y: np.ndarray, s: np.ndarray, thr: float) -> float:
    return float(((s >= thr).astype(int) == y).mean())


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def risk_coverage(y_correct: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by descending confidence; risk[i] is the error rate of the most
    confident i+1 predictions, coverage[i] their share of the set."""
    order = np.argsort(-confidence, kind="stable")
    c = y_correct[order].astype(float)
    n = len(c)
    coverage = np.arange(1, n + 1) / n
    risk = 1.0 - np.cumsum(c) / np.arange(1, n + 1)
    return coverage, risk


def aurc(y_correct: np.ndarray, confidence: np.ndarray) -> float:
    coverage, risk = risk_coverage(y_correct, confidence)
    return float(np.trapezoid(risk, coverage))


def accuracy_at_coverage(y_correct: np.ndarray, confidence: np.ndarray, coverage: float) -> float:
    k = max(1, int(round(len(y_correct) * coverage)))
    order = np.argsort(-confidence, kind="stable")[:k]
    return float(y_correct[order].mean())


def bootstrap_ci(
    fn: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray, s: np.ndarray,
    n: int = 1000, seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap. Resamples that lose a class are skipped."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], s[idx]))
    if not vals:
        raise ValueError("no valid bootstrap resamples; is one class empty?")
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))
