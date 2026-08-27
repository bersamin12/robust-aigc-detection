import numpy as np
import pytest

from aigcdet.eval import metrics as M


@pytest.fixture
def separable():
    rng = np.random.default_rng(0)
    y = np.array([0] * 500 + [1] * 500)
    s = np.concatenate([rng.normal(0.2, 0.1, 500), rng.normal(0.8, 0.1, 500)])
    return y, np.clip(s, 0, 1)


def test_perfect_separation_gives_auc_one():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert M.roc_auc(y, s) == pytest.approx(1.0)


def test_random_scores_give_auc_near_half(separable):
    rng = np.random.default_rng(1)
    y, _ = separable
    assert M.roc_auc(y, rng.random(len(y))) == pytest.approx(0.5, abs=0.05)


def test_auc_is_invariant_to_monotone_transform_of_scores(separable):
    y, s = separable
    transformed = 1.0 / (1.0 + np.exp(-5.0 * (s - 0.3)))  # strictly increasing, nonlinear
    assert M.roc_auc(y, transformed) == pytest.approx(M.roc_auc(y, s))


def test_tpr_at_fpr_is_between_zero_and_one(separable):
    y, s = separable
    v = M.tpr_at_fpr(y, s, 0.01)
    assert 0.0 <= v <= 1.0


def test_threshold_at_fpr_actually_holds_that_fpr(separable):
    y, s = separable
    thr = M.threshold_at_fpr(y, s, 0.01)
    realised = float(((s >= thr) & (y == 0)).sum() / (y == 0).sum())
    assert realised <= 0.02


def test_ece_is_zero_for_perfectly_calibrated_predictions():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.random(20000) < p).astype(int)
    assert M.expected_calibration_error(y, p, n_bins=15) < 0.02


def test_ece_is_large_for_overconfident_predictions():
    y = np.array([0, 1] * 500)
    p = np.array([0.99, 0.01] * 500)   # confidently wrong
    assert M.expected_calibration_error(y, p) > 0.9


def test_brier_of_perfect_prediction_is_zero():
    y = np.array([0, 1, 1, 0])
    assert M.brier(y, y.astype(float)) == pytest.approx(0.0)


def test_brier_of_always_half_predictor_is_one_quarter():
    y = np.array([0, 1, 1, 0, 1])
    p = np.full_like(y, 0.5, dtype=float)
    assert M.brier(y, p) == pytest.approx(0.25)


def test_risk_coverage_is_monotone_when_confidence_is_informative():
    correct = np.array([1] * 80 + [0] * 20)
    conf = np.concatenate([np.linspace(0.9, 1.0, 80), np.linspace(0.0, 0.5, 20)])
    cov, risk = M.risk_coverage(correct, conf)
    assert cov[-1] == pytest.approx(1.0)
    assert risk[0] <= risk[-1]          # deferring the unconfident lowers risk
    assert 0.0 <= M.aurc(correct, conf) <= 1.0


def test_accuracy_at_coverage_beats_full_coverage_when_confidence_is_informative():
    correct = np.array([1] * 80 + [0] * 20)
    conf = np.concatenate([np.linspace(0.9, 1.0, 80), np.linspace(0.0, 0.5, 20)])
    assert M.accuracy_at_coverage(correct, conf, 0.8) > M.accuracy_at_coverage(correct, conf, 1.0)


def test_bootstrap_ci_brackets_the_point_estimate_and_is_reproducible(separable):
    y, s = separable
    lo, hi = M.bootstrap_ci(M.roc_auc, y, s, n=200, seed=0)
    point = M.roc_auc(y, s)
    assert lo <= point <= hi
    assert (lo, hi) == M.bootstrap_ci(M.roc_auc, y, s, n=200, seed=0)
