import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from aigcdet.calibrate import INTERNAL_VAL_SPLIT, CalibrationError
from aigcdet.calibrate.eqi import EQI


def _val(n):
    """The per-row split column a legitimate caller passes."""
    return np.full(n, INTERNAL_VAL_SPLIT)


def test_eqi_tracks_correctness_probability():
    """Two degradation regimes with very different accuracy: EQI must separate."""
    rng = np.random.default_rng(0)
    n = 4000
    severe = (rng.random(n) < 0.5).astype(float)
    correct = np.where(severe > 0, rng.random(n) < 0.55, rng.random(n) < 0.95).astype(int)
    cond = severe[:, None]
    e = EQI().fit(cond, correct, split=_val(n))
    pred = e.predict(cond)
    assert pred[severe > 0].mean() < pred[severe == 0].mean()
    assert abs(pred[severe == 0].mean() - 0.95) < 0.08


def test_eqi_output_is_bounded():
    rng = np.random.default_rng(1)
    cond = rng.normal(size=(500, 4))
    correct = (rng.random(500) < 0.7).astype(int)
    p = EQI().fit(cond, correct, split=_val(500)).predict(cond)
    assert ((p >= 0) & (p <= 1)).all()


def test_eqi_is_reproducible():
    rng = np.random.default_rng(2)
    cond = rng.normal(size=(300, 2))
    correct = (rng.random(300) < 0.6).astype(int)
    a = EQI(seed=7).fit(cond, correct, split=_val(300)).predict(cond)
    b = EQI(seed=7).fit(cond, correct, split=_val(300)).predict(cond)
    np.testing.assert_allclose(a, b)


def test_eqi_predicts_on_unseen_rows_using_the_fitted_standardisation():
    """Fit and predict on disjoint rows: EQI is used at inference time."""
    rng = np.random.default_rng(3)
    n = 4000
    severe = (rng.random(n) < 0.5).astype(float)
    correct = np.where(severe > 0, rng.random(n) < 0.5, rng.random(n) < 0.9).astype(int)
    cond = severe[:, None]
    e = EQI().fit(cond[:3000], correct[:3000], split=_val(3000))
    pred = e.predict(cond[3000:])
    held = severe[3000:]
    assert pred[held > 0].mean() < pred[held == 0].mean()


def test_eqi_is_calibrated_enough_to_read_as_a_probability():
    """EQI's whole selling point is the reading "~40% usable evidence"."""
    rng = np.random.default_rng(9)
    n = 6000
    sev = rng.random(n)
    correct = (rng.random(n) < 0.95 - 0.5 * sev).astype(int)
    e = EQI().fit(sev[:, None], correct, split=_val(n))
    p = e.predict(sev[:, None])
    for lo, hi in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
        m = (sev >= lo) & (sev < hi)
        assert abs(p[m].mean() - correct[m].mean()) < 0.05


# --- fitting on the wrong split (spec §6.7) --------------------------------

@pytest.mark.parametrize("wrong", ["test", "train", "heldout_generator", "benchmark"])
def test_eqi_refuses_rows_from_any_split_but_internal_validation(wrong):
    rng = np.random.default_rng(4)
    cond = rng.normal(size=(200, 2))
    correct = (rng.random(200) < 0.6).astype(int)
    with pytest.raises(ValueError, match="val_internal"):
        EQI().fit(cond, correct, split=np.full(200, wrong))


def test_eqi_refuses_a_single_contaminating_row():
    rng = np.random.default_rng(4)
    cond = rng.normal(size=(200, 2))
    correct = (rng.random(200) < 0.6).astype(int)
    contaminated = _val(200)
    contaminated[3] = "test"
    with pytest.raises(ValueError, match=r"1 of 200 rows"):
        EQI().fit(cond, correct, split=contaminated)


def test_eqi_refuses_a_bare_string_promise():
    rng = np.random.default_rng(4)
    cond = rng.normal(size=(200, 2))
    correct = (rng.random(200) < 0.6).astype(int)
    with pytest.raises(ValueError, match="one label per row"):
        EQI().fit(cond, correct, split=INTERNAL_VAL_SPLIT)


def test_eqi_requires_the_split_column():
    rng = np.random.default_rng(4)
    cond = rng.normal(size=(200, 2))
    correct = (rng.random(200) < 0.6).astype(int)
    with pytest.raises(TypeError, match="split"):
        EQI().fit(cond, correct)


# --- unfitted, mis-shaped and degenerate use -------------------------------

def test_eqi_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        EQI().predict(np.zeros((3, 2)))


def test_eqi_predict_rejects_cond_of_the_wrong_width():
    rng = np.random.default_rng(5)
    cond = rng.normal(size=(200, 3))
    correct = (rng.random(200) < 0.6).astype(int)
    e = EQI().fit(cond, correct, split=_val(200))
    with pytest.raises(ValueError, match="3 columns"):
        e.predict(rng.normal(size=(10, 2)))


def test_eqi_can_be_refitted_at_a_different_conditioning_width():
    """Dropping the proxies from `cond` is a legitimate ablation; the stale
    fit must not block the new one."""
    rng = np.random.default_rng(5)
    correct = (rng.random(200) < 0.6).astype(int)
    e = EQI().fit(rng.normal(size=(200, 3)), correct, split=_val(200))
    two = rng.normal(size=(200, 2))
    e.fit(two, correct, split=_val(200))
    assert e.predict(two).shape == (200,)


def test_eqi_needs_both_outcomes():
    rng = np.random.default_rng(6)
    cond = rng.normal(size=(50, 2))
    with pytest.raises(ValueError, match="both"):
        EQI().fit(cond, np.ones(50, dtype=int), split=_val(50))


def test_eqi_rejects_too_few_rows_for_the_conditioning_width():
    rng = np.random.default_rng(7)
    correct = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="rows"):
        EQI().fit(rng.normal(size=(4, 6)), correct, split=_val(4))


def test_eqi_constant_cond_column_is_neutralised_rather_than_divided_by_zero():
    rng = np.random.default_rng(8)
    n = 2000
    severe = (rng.random(n) < 0.5).astype(float)
    correct = np.where(severe > 0, rng.random(n) < 0.5, rng.random(n) < 0.9).astype(int)
    cond = np.column_stack([severe, np.full(n, 4.0)])
    e = EQI().fit(cond, correct, split=_val(n))
    assert e.constant_columns == (1,)
    p = e.predict(cond)
    assert np.isfinite(p).all()
    shifted = cond.copy()
    shifted[:, 1] = -9.0
    np.testing.assert_allclose(e.predict(shifted), p)


@pytest.mark.parametrize("sd", [1e-9, 1e-7])
def test_eqi_near_constant_cond_column_cannot_saturate_the_prediction(sd):
    """The real hazard is NEAR-constant, not exactly constant. An exactly
    constant column gets a zero coefficient from the L2 penalty either way; a
    column with sd 1e-9 about a mean of 4.0 gets a small NON-zero one, and
    dividing by that 1e-9 spread puts a shifted row 1e10 scale-units out, where
    EQI reports exactly 1.0 on evidence it has never seen."""
    rng = np.random.default_rng(8)
    n = 2000
    severe = (rng.random(n) < 0.5).astype(float)
    correct = np.where(severe > 0, rng.random(n) < 0.5, rng.random(n) < 0.9).astype(int)
    cond = np.column_stack([severe, 4.0 + sd * rng.normal(size=n)])

    e = EQI().fit(cond, correct, split=_val(n))
    assert e.constant_columns == (1,)
    p = e.predict(cond)
    shifted = cond.copy()
    shifted[:, 1] = -9.0
    assert np.abs(e.predict(shifted) - p).max() < 1e-6


# --- convergence -----------------------------------------------------------

def test_eqi_reports_a_logistic_regression_that_did_not_converge():
    rng = np.random.default_rng(10)
    n = 2000
    sev = rng.random(n)
    correct = (rng.random(n) < 0.95 - 0.5 * sev).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        with pytest.raises(CalibrationError, match="without converging"):
            EQI(max_iter=1).fit(sev[:, None], correct, split=_val(n))


def test_eqi_rejects_a_non_positive_iteration_budget():
    with pytest.raises(ValueError, match="max_iter"):
        EQI(max_iter=0)
