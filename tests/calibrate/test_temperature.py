import numpy as np
import pytest

from aigcdet.calibrate import CalibrationError
from aigcdet.calibrate.temperature import ConditionalTemperature, GlobalTemperature
from aigcdet.eval.metrics import expected_calibration_error


def _overconfident(n=4000, seed=0):
    """True probability is sigmoid(z), but the model reports sigmoid(3z)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return z * 3.0, y


def test_global_temperature_reduces_ece():
    logits, y = _overconfident()
    before = expected_calibration_error(y, 1 / (1 + np.exp(-logits)))
    cal = GlobalTemperature().fit(logits, y)
    after = expected_calibration_error(y, cal.transform(logits))
    assert after < before
    assert cal.temperature > 1.0          # shrinks over-confident logits


def test_global_temperature_preserves_ranking():
    logits, y = _overconfident()
    p = GlobalTemperature().fit(logits, y).transform(logits)
    assert np.array_equal(np.argsort(p), np.argsort(logits))


def test_conditional_temperature_beats_global_when_miscalibration_varies():
    """Two regimes with different overconfidence; a single scalar cannot fix both."""
    rng = np.random.default_rng(1)
    n = 4000
    z = rng.normal(0, 1.5, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    regime = (rng.random(n) < 0.5).astype(float)
    logits = z * np.where(regime > 0, 4.0, 1.2)
    cond = regime[:, None]

    g = GlobalTemperature().fit(logits, y)
    c = ConditionalTemperature(cond_dim=1).fit(logits, y, cond, epochs=400)
    ece_g = expected_calibration_error(y, g.transform(logits))
    ece_c = expected_calibration_error(y, c.transform(logits, cond))
    assert ece_c < ece_g


def test_conditional_temperature_is_always_positive():
    rng = np.random.default_rng(2)
    logits, y = _overconfident(1000, 2)
    cond = rng.normal(size=(1000, 3))
    c = ConditionalTemperature(cond_dim=3).fit(logits, y, cond, epochs=50)
    assert (c.temperatures(cond) > 0).all()


def test_transform_outputs_are_valid_probabilities():
    logits, y = _overconfident(500, 3)
    p = GlobalTemperature().fit(logits, y).transform(logits)
    assert ((p >= 0) & (p <= 1)).all()


def test_conditional_temperature_recovers_the_two_true_temperatures():
    """The fit is not merely "better than global": it lands on the truth."""
    rng = np.random.default_rng(1)
    n = 4000
    z = rng.normal(0, 1.5, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    regime = (rng.random(n) < 0.5).astype(float)
    logits = z * np.where(regime > 0, 4.0, 1.2)
    cond = regime[:, None]

    t = ConditionalTemperature(cond_dim=1).fit(logits, y, cond, epochs=400).temperatures(cond)
    assert t[regime > 0].mean() == pytest.approx(4.0, abs=0.35)
    assert t[regime == 0].mean() == pytest.approx(1.2, abs=0.35)


# --- fitting on the wrong split (spec §6.7) --------------------------------

@pytest.mark.parametrize("split", ["test", "train", "heldout_generator", "benchmark"])
def test_fit_refuses_any_split_but_internal_validation(split):
    logits, y = _overconfident(200, 4)
    with pytest.raises(ValueError, match="val_internal"):
        GlobalTemperature().fit(logits, y, split=split)
    with pytest.raises(ValueError, match="val_internal"):
        ConditionalTemperature(cond_dim=1).fit(
            logits, y, np.zeros((200, 1)), epochs=10, split=split)


# --- fits that must fail loudly rather than ship a bad temperature ---------

def test_degenerate_fit_collapsing_the_temperature_to_zero_is_rejected():
    """Perfectly separable validation drives T -> 0, i.e. infinite confidence."""
    logits, _ = _overconfident(1000, 5)
    separable = (logits > 0).astype(int)
    with pytest.raises(CalibrationError, match="temperature"):
        GlobalTemperature().fit(logits, separable)


def test_degenerate_fit_blowing_the_temperature_up_is_rejected():
    """Anti-correlated validation drives T -> inf, i.e. constant p = 0.5."""
    logits, _ = _overconfident(1000, 6)
    anti = (logits < 0).astype(int)
    with pytest.raises(CalibrationError, match="temperature"):
        GlobalTemperature().fit(logits, anti)


def test_fit_needs_both_classes():
    logits, _ = _overconfident(100, 7)
    with pytest.raises(ValueError, match="both classes"):
        GlobalTemperature().fit(logits, np.ones(100, dtype=int))


def test_fit_rejects_non_binary_labels():
    logits, _ = _overconfident(100, 8)
    y = np.full(100, 2)
    y[:50] = 0
    with pytest.raises(ValueError, match="0/1"):
        GlobalTemperature().fit(logits, y)


def test_fit_rejects_non_finite_logits():
    logits, y = _overconfident(100, 9)
    logits[0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        GlobalTemperature().fit(logits, y)


# --- unfitted use ----------------------------------------------------------

def test_global_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        GlobalTemperature().transform(np.zeros(4))


def test_conditional_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        ConditionalTemperature(cond_dim=2).transform(np.zeros(4), np.zeros((4, 2)))


# --- degenerate conditioning vectors ---------------------------------------

def test_conditional_fit_rejects_cond_of_the_wrong_width():
    logits, y = _overconfident(200, 10)
    with pytest.raises(ValueError, match="cond_dim"):
        ConditionalTemperature(cond_dim=3).fit(logits, y, np.zeros((200, 2)), epochs=10)


def test_conditional_transform_rejects_cond_of_the_wrong_width():
    logits, y = _overconfident(200, 11)
    c = ConditionalTemperature(cond_dim=2).fit(
        logits, y, np.tile(np.arange(200)[:, None] % 3, (1, 2)).astype(float), epochs=50)
    with pytest.raises(ValueError, match="cond_dim"):
        c.transform(logits, np.zeros((200, 5)))


def test_conditional_fit_rejects_far_too_few_rows_for_cond_dim():
    rng = np.random.default_rng(12)
    logits, y = _overconfident(6, 12)
    with pytest.raises(ValueError, match="rows"):
        ConditionalTemperature(cond_dim=8).fit(logits, y, rng.normal(size=(6, 8)), epochs=10)


def test_constant_cond_column_is_neutralised_rather_than_divided_by_zero():
    """A degradation family never seen in validation is a constant column: it
    must contribute exactly nothing, not a NaN and not a spurious weight."""
    rng = np.random.default_rng(13)
    n = 400
    z = rng.normal(0, 1.5, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    regime = (rng.random(n) < 0.5).astype(float)
    logits = z * np.where(regime > 0, 4.0, 1.2)
    cond = np.column_stack([regime, np.full(n, 7.0)])

    c = ConditionalTemperature(cond_dim=2).fit(logits, y, cond, epochs=200)
    assert c.constant_columns == (1,)
    assert float(c.net.weight.detach()[0, 1]) == 0.0
    t = c.temperatures(cond)
    assert np.isfinite(t).all()
    # A different constant in that column cannot move the temperature.
    shifted = cond.copy()
    shifted[:, 1] = -3.0
    np.testing.assert_allclose(c.temperatures(shifted), t)


def test_conditional_temperatures_are_bounded_and_finite():
    logits, y = _overconfident(1000, 14)
    rng = np.random.default_rng(14)
    cond = rng.normal(size=(1000, 2))
    t = ConditionalTemperature(cond_dim=2).fit(logits, y, cond, epochs=100).temperatures(cond)
    assert np.isfinite(t).all()
    assert (t > 0).all()


def test_conditional_transform_outputs_are_valid_probabilities():
    logits, y = _overconfident(500, 15)
    rng = np.random.default_rng(15)
    cond = rng.normal(size=(500, 2))
    c = ConditionalTemperature(cond_dim=2).fit(logits, y, cond, epochs=100)
    p = c.transform(logits, cond)
    assert ((p >= 0) & (p <= 1)).all()


def test_conditional_fit_is_reproducible():
    logits, y = _overconfident(800, 16)
    rng = np.random.default_rng(16)
    cond = rng.normal(size=(800, 2))
    a = ConditionalTemperature(cond_dim=2).fit(logits, y, cond, epochs=100).temperatures(cond)
    b = ConditionalTemperature(cond_dim=2).fit(logits, y, cond, epochs=100).temperatures(cond)
    np.testing.assert_allclose(a, b)


def test_fitting_does_not_disturb_the_global_torch_rng():
    """No global seeding: two fits either side of a draw must not change it."""
    import torch

    logits, y = _overconfident(200, 17)
    gen_state = torch.random.get_rng_state()
    GlobalTemperature().fit(logits, y)
    ConditionalTemperature(cond_dim=1).fit(logits, y, np.arange(200.0)[:, None] % 2, epochs=200)
    assert torch.equal(torch.random.get_rng_state(), gen_state)


def test_zero_iteration_budgets_are_rejected():
    logits, y = _overconfident(200, 18)
    with pytest.raises(ValueError, match="max_iter"):
        GlobalTemperature().fit(logits, y, max_iter=0)
    with pytest.raises(ValueError, match="epochs"):
        ConditionalTemperature(cond_dim=1).fit(
            logits, y, np.arange(200.0)[:, None] % 2, epochs=0)
