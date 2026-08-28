import numpy as np
import pytest

from aigcdet.calibrate.eqi import EQI


def test_eqi_tracks_correctness_probability():
    """Two degradation regimes with very different accuracy: EQI must separate."""
    rng = np.random.default_rng(0)
    n = 4000
    severe = (rng.random(n) < 0.5).astype(float)
    correct = np.where(severe > 0, rng.random(n) < 0.55, rng.random(n) < 0.95).astype(int)
    cond = severe[:, None]
    e = EQI().fit(cond, correct)
    pred = e.predict(cond)
    assert pred[severe > 0].mean() < pred[severe == 0].mean()
    assert abs(pred[severe == 0].mean() - 0.95) < 0.08


def test_eqi_output_is_bounded():
    rng = np.random.default_rng(1)
    cond = rng.normal(size=(500, 4))
    correct = (rng.random(500) < 0.7).astype(int)
    p = EQI().fit(cond, correct).predict(cond)
    assert ((p >= 0) & (p <= 1)).all()


def test_eqi_is_reproducible():
    rng = np.random.default_rng(2)
    cond = rng.normal(size=(300, 2))
    correct = (rng.random(300) < 0.6).astype(int)
    a = EQI(seed=7).fit(cond, correct).predict(cond)
    b = EQI(seed=7).fit(cond, correct).predict(cond)
    np.testing.assert_allclose(a, b)


def test_eqi_predicts_on_unseen_rows_using_the_fitted_standardisation():
    """Fit and predict on disjoint rows: EQI is used at inference time."""
    rng = np.random.default_rng(3)
    n = 4000
    severe = (rng.random(n) < 0.5).astype(float)
    correct = np.where(severe > 0, rng.random(n) < 0.5, rng.random(n) < 0.9).astype(int)
    cond = severe[:, None]
    e = EQI().fit(cond[:3000], correct[:3000])
    pred = e.predict(cond[3000:])
    held = severe[3000:]
    assert pred[held > 0].mean() < pred[held == 0].mean()


@pytest.mark.parametrize("split", ["test", "train", "heldout_generator", "benchmark"])
def test_eqi_refuses_any_split_but_internal_validation(split):
    rng = np.random.default_rng(4)
    cond = rng.normal(size=(200, 2))
    correct = (rng.random(200) < 0.6).astype(int)
    with pytest.raises(ValueError, match="val_internal"):
        EQI().fit(cond, correct, split=split)


def test_eqi_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        EQI().predict(np.zeros((3, 2)))


def test_eqi_predict_rejects_cond_of_the_wrong_width():
    rng = np.random.default_rng(5)
    cond = rng.normal(size=(200, 3))
    correct = (rng.random(200) < 0.6).astype(int)
    e = EQI().fit(cond, correct)
    with pytest.raises(ValueError, match="3 columns"):
        e.predict(rng.normal(size=(10, 2)))


def test_eqi_needs_both_outcomes():
    rng = np.random.default_rng(6)
    cond = rng.normal(size=(50, 2))
    with pytest.raises(ValueError, match="both"):
        EQI().fit(cond, np.ones(50, dtype=int))


def test_eqi_rejects_too_few_rows_for_the_conditioning_width():
    rng = np.random.default_rng(7)
    correct = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="rows"):
        EQI().fit(rng.normal(size=(4, 6)), correct)


def test_eqi_constant_cond_column_is_neutralised_rather_than_divided_by_zero():
    rng = np.random.default_rng(8)
    n = 2000
    severe = (rng.random(n) < 0.5).astype(float)
    correct = np.where(severe > 0, rng.random(n) < 0.5, rng.random(n) < 0.9).astype(int)
    cond = np.column_stack([severe, np.full(n, 4.0)])
    e = EQI().fit(cond, correct)
    assert e.constant_columns == (1,)
    p = e.predict(cond)
    assert np.isfinite(p).all()
    shifted = cond.copy()
    shifted[:, 1] = -9.0
    np.testing.assert_allclose(e.predict(shifted), p)


def test_eqi_is_calibrated_enough_to_read_as_a_probability():
    """EQI's whole selling point is the reading "~40% usable evidence"."""
    rng = np.random.default_rng(9)
    n = 6000
    sev = rng.random(n)
    correct = (rng.random(n) < 0.95 - 0.5 * sev).astype(int)
    e = EQI().fit(sev[:, None], correct)
    p = e.predict(sev[:, None])
    for lo, hi in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
        m = (sev >= lo) & (sev < hi)
        assert abs(p[m].mean() - correct[m].mean()) < 0.05
