import warnings

import numpy as np
import pytest

from aigcdet.calibrate import INTERNAL_VAL_SPLIT, CalibrationError
from aigcdet.calibrate.temperature import ConditionalTemperature, GlobalTemperature
from aigcdet.eval.metrics import expected_calibration_error


def _val(n):
    """The per-row split column a legitimate caller passes."""
    return np.full(n, INTERNAL_VAL_SPLIT)


def _overconfident(n=4000, seed=0):
    """True probability is sigmoid(z), but the model reports sigmoid(3z)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return z * 3.0, y


def _two_regimes(n=4000, seed=1):
    """Two regimes with different overconfidence: true T is 4.0 and 1.2."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    regime = (rng.random(n) < 0.5).astype(float)
    logits = z * np.where(regime > 0, 4.0, 1.2)
    return logits, y, regime


def test_global_temperature_reduces_ece():
    logits, y = _overconfident()
    before = expected_calibration_error(y, 1 / (1 + np.exp(-logits)))
    cal = GlobalTemperature().fit(logits, y, split=_val(len(y)))
    after = expected_calibration_error(y, cal.transform(logits))
    assert after < before
    assert cal.temperature > 1.0          # shrinks over-confident logits


def test_global_temperature_preserves_ranking():
    logits, y = _overconfident()
    p = GlobalTemperature().fit(logits, y, split=_val(len(y))).transform(logits)
    assert np.array_equal(np.argsort(p), np.argsort(logits))


def test_conditional_temperature_beats_global_when_miscalibration_varies():
    """Two regimes with different overconfidence; a single scalar cannot fix both."""
    logits, y, regime = _two_regimes()
    cond = regime[:, None]
    split = _val(len(y))

    g = GlobalTemperature().fit(logits, y, split=split)
    c = ConditionalTemperature(cond_dim=1).fit(logits, y, cond, epochs=400, split=split)
    ece_g = expected_calibration_error(y, g.transform(logits))
    ece_c = expected_calibration_error(y, c.transform(logits, cond))
    assert ece_c < ece_g


def test_conditional_temperature_is_always_positive():
    rng = np.random.default_rng(2)
    logits, y = _overconfident(1000, 2)
    cond = rng.normal(size=(1000, 3))
    c = ConditionalTemperature(cond_dim=3).fit(
        logits, y, cond, epochs=50, split=_val(1000))
    assert (c.temperatures(cond) > 0).all()


def test_transform_outputs_are_valid_probabilities():
    logits, y = _overconfident(500, 3)
    p = GlobalTemperature().fit(logits, y, split=_val(500)).transform(logits)
    assert ((p >= 0) & (p <= 1)).all()


def test_conditional_temperature_recovers_the_two_true_temperatures():
    """The fit is not merely "better than global": it lands on the truth."""
    logits, y, regime = _two_regimes()
    cond = regime[:, None]
    t = ConditionalTemperature(cond_dim=1).fit(
        logits, y, cond, epochs=400, split=_val(len(y))).temperatures(cond)
    assert t[regime > 0].mean() == pytest.approx(4.0, abs=0.35)
    assert t[regime == 0].mean() == pytest.approx(1.2, abs=0.35)


# --- fitting on the wrong split (spec §6.7) --------------------------------
#
# `split` is a per-row column, not a promise: these check that a fit on
# contaminated rows is refused, not merely that a caller typed the right word.

@pytest.mark.parametrize("wrong", ["test", "train", "heldout_generator", "benchmark"])
def test_fit_refuses_rows_from_any_split_but_internal_validation(wrong):
    logits, y = _overconfident(200, 4)
    bad = np.full(200, wrong)
    with pytest.raises(ValueError, match="val_internal"):
        GlobalTemperature().fit(logits, y, split=bad)
    with pytest.raises(ValueError, match="val_internal"):
        ConditionalTemperature(cond_dim=1).fit(
            logits, y, np.zeros((200, 1)), epochs=10, split=bad)


def test_fit_refuses_a_single_contaminating_row():
    """One test row among 200 validation rows still poisons the fit."""
    logits, y = _overconfident(200, 4)
    contaminated = _val(200)
    contaminated[137] = "test"
    with pytest.raises(ValueError, match=r"1 of 200 rows"):
        GlobalTemperature().fit(logits, y, split=contaminated)


def test_fit_refuses_a_bare_string_promise():
    """The old scalar guard accepted this and checked nothing about the rows."""
    logits, y = _overconfident(200, 4)
    with pytest.raises(ValueError, match="one label per row"):
        GlobalTemperature().fit(logits, y, split=INTERNAL_VAL_SPLIT)


def test_fit_refuses_a_split_column_of_the_wrong_length():
    logits, y = _overconfident(200, 4)
    with pytest.raises(ValueError, match="expected 200 labels"):
        GlobalTemperature().fit(logits, y, split=_val(199))


def test_fit_requires_the_split_column():
    """No default: a caller with no split column has to construct one."""
    logits, y = _overconfident(200, 4)
    with pytest.raises(TypeError, match="split"):
        GlobalTemperature().fit(logits, y)
    with pytest.raises(TypeError, match="split"):
        ConditionalTemperature(cond_dim=1).fit(logits, y, np.zeros((200, 1)))


# --- fits that must fail loudly rather than ship a bad temperature ---------

def test_a_fit_that_has_not_converged_is_refused():
    """An exhausted iteration budget is a failure, not a temperature."""
    logits, y, regime = _two_regimes()
    with pytest.raises(CalibrationError, match="did not converge"):
        ConditionalTemperature(cond_dim=1).fit(
            logits, y, regime[:, None], epochs=3, split=_val(len(y)))


def test_a_global_fit_that_has_not_converged_is_refused():
    logits, y = _overconfident(4000, 5)
    with pytest.raises(CalibrationError, match="did not converge"):
        GlobalTemperature().fit(logits, y, split=_val(4000), max_iter=1)


def test_degenerate_fit_collapsing_the_temperature_to_zero_is_rejected():
    """Perfectly separable validation drives T -> 0, i.e. infinite confidence."""
    logits, _ = _overconfident(1000, 5)
    separable = (logits > 0).astype(int)
    with pytest.raises(CalibrationError, match="temperature"):
        GlobalTemperature().fit(logits, separable, split=_val(1000))


def test_degenerate_fit_blowing_the_temperature_up_is_rejected():
    """Anti-correlated validation drives T -> inf, i.e. constant p = 0.5."""
    logits, _ = _overconfident(1000, 6)
    anti = (logits < 0).astype(int)
    with pytest.raises(CalibrationError, match="temperature"):
        GlobalTemperature().fit(logits, anti, split=_val(1000))


def test_fit_needs_both_classes():
    logits, _ = _overconfident(100, 7)
    with pytest.raises(ValueError, match="both classes"):
        GlobalTemperature().fit(logits, np.ones(100, dtype=int), split=_val(100))


def test_fit_rejects_non_binary_labels():
    logits, _ = _overconfident(100, 8)
    y = np.full(100, 2)
    y[:50] = 0
    with pytest.raises(ValueError, match="0/1"):
        GlobalTemperature().fit(logits, y, split=_val(100))


def test_fit_rejects_non_finite_logits():
    logits, y = _overconfident(100, 9)
    logits[0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        GlobalTemperature().fit(logits, y, split=_val(100))


# --- unfitted use ----------------------------------------------------------

def test_global_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        GlobalTemperature().transform(np.zeros(4))


def test_conditional_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        ConditionalTemperature(cond_dim=2).transform(np.zeros(4), np.zeros((4, 2)))


def test_conditional_temperatures_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        ConditionalTemperature(cond_dim=2).temperatures(np.zeros((4, 2)))


def test_a_failed_fit_leaves_the_object_unusable_rather_than_half_fitted():
    logits, y, regime = _two_regimes()
    c = ConditionalTemperature(cond_dim=1)
    with pytest.raises(CalibrationError):
        c.fit(logits, y, regime[:, None], epochs=3, split=_val(len(y)))
    with pytest.raises(RuntimeError, match="not fitted"):
        c.temperatures(regime[:, None])


# --- extrapolation beyond the validation range -----------------------------

def test_extrapolated_cond_cannot_push_the_temperature_below_the_fitted_minimum():
    """A linear T(cond) extrapolates without limit, and downwards is lethal:
    T -> eps = 0.01 MULTIPLIES the logit by 100 and returns p = 1.000000 on an
    image cleaner than anything validation contained."""
    logits, y, regime = _two_regimes()
    cond = regime[:, None]
    c = ConditionalTemperature(cond_dim=1).fit(
        logits, y, cond, epochs=400, split=_val(len(y)))
    fitted = c.temperatures(cond)
    lo, hi = fitted.min(), fitted.max()
    assert lo > 1.0          # sanity: the fit really is in the 1.2 - 4.0 band

    far = np.array([[-10.0], [-2.0], [-0.5], [1.5], [10.0]])
    t = c.temperatures(far)
    assert (t >= lo).all()
    assert (t <= hi).all()

    # The consequence the clamp exists to prevent: without it, T = 0.0145 at
    # cond = -2 turns a +6 logit into p = 1.000000.
    p = c.transform(np.full(5, 6.0), far)
    assert (p < 0.999).all()


def test_the_clamp_does_not_disturb_temperatures_inside_the_fitted_range():
    logits, y, regime = _two_regimes()
    cond = regime[:, None]
    c = ConditionalTemperature(cond_dim=1).fit(
        logits, y, cond, epochs=400, split=_val(len(y)))
    t = c.temperatures(cond)
    # Interpolating between the two regimes stays strictly inside the range.
    mid = c.temperatures(np.array([[0.5]]))
    assert t.min() < mid[0] < t.max()


# --- degenerate conditioning vectors ---------------------------------------

def test_conditional_fit_rejects_cond_of_the_wrong_width():
    logits, y = _overconfident(200, 10)
    with pytest.raises(ValueError, match="cond_dim"):
        ConditionalTemperature(cond_dim=3).fit(
            logits, y, np.zeros((200, 2)), epochs=10, split=_val(200))


def test_conditional_transform_rejects_cond_of_the_wrong_width():
    logits, y = _overconfident(200, 11)
    c = ConditionalTemperature(cond_dim=2).fit(
        logits, y, np.tile(np.arange(200)[:, None] % 3, (1, 2)).astype(float),
        epochs=50, split=_val(200))
    with pytest.raises(ValueError, match="cond_dim"):
        c.transform(logits, np.zeros((200, 5)))


def test_conditional_fit_rejects_far_too_few_rows_for_cond_dim():
    rng = np.random.default_rng(12)
    logits, y = _overconfident(6, 12)
    with pytest.raises(ValueError, match="rows"):
        ConditionalTemperature(cond_dim=8).fit(
            logits, y, rng.normal(size=(6, 8)), epochs=10, split=_val(6))


def test_constant_cond_column_is_neutralised_rather_than_divided_by_zero():
    """A degradation family never seen in validation is a constant column: it
    must contribute exactly nothing, not a NaN and not a spurious weight."""
    logits, y, regime = _two_regimes(400, 13)
    cond = np.column_stack([regime, np.full(400, 7.0)])

    c = ConditionalTemperature(cond_dim=2).fit(
        logits, y, cond, epochs=200, split=_val(400))
    assert c.constant_columns == (1,)
    assert float(c.net.weight.detach()[0, 1]) == 0.0
    t = c.temperatures(cond)
    assert np.isfinite(t).all()
    # A different constant in that column cannot move the temperature.
    shifted = cond.copy()
    shifted[:, 1] = -3.0
    np.testing.assert_allclose(c.temperatures(shifted), t)


def test_near_constant_cond_column_is_neutralised_too():
    """sd = 1e-9 about a mean of 4.0 is constant in every sense that matters,
    and an absolute tolerance does not catch it."""
    rng = np.random.default_rng(13)
    logits, y, regime = _two_regimes(400, 13)
    cond = np.column_stack([regime, 4.0 + 1e-9 * rng.normal(size=400)])

    c = ConditionalTemperature(cond_dim=2).fit(
        logits, y, cond, epochs=200, split=_val(400))
    assert c.constant_columns == (1,)
    # Behavioural, not cosmetic: neutralised, the column is scaled by 1, so a
    # shift of -13 moves T by nothing. Scaled by its own 1e-9 spread instead,
    # the same shift lands 1.3e10 scale-units out and saturates T.
    shifted = cond.copy()
    shifted[:, 1] = -9.0
    assert np.abs(c.temperatures(shifted) - c.temperatures(cond)).max() < 1e-6


def test_conditional_temperatures_are_bounded_and_finite():
    logits, y = _overconfident(1000, 14)
    rng = np.random.default_rng(14)
    cond = rng.normal(size=(1000, 2))
    t = ConditionalTemperature(cond_dim=2).fit(
        logits, y, cond, epochs=100, split=_val(1000)).temperatures(cond)
    assert np.isfinite(t).all()
    assert (t > 0).all()


def test_conditional_transform_outputs_are_valid_probabilities():
    logits, y = _overconfident(500, 15)
    rng = np.random.default_rng(15)
    cond = rng.normal(size=(500, 2))
    c = ConditionalTemperature(cond_dim=2).fit(
        logits, y, cond, epochs=100, split=_val(500))
    p = c.transform(logits, cond)
    assert ((p >= 0) & (p <= 1)).all()


def test_saturating_logits_do_not_overflow():
    """Dividing a large logit by a temperature below 1 is exactly what this
    module makes. The naive 1/(1+exp(-x)) overflows above |x| = 709 and emits a
    RuntimeWarning on the way, so `warnings -> error` is the assertion here."""
    logits, y = _overconfident(500, 15)
    rng = np.random.default_rng(15)
    cond = rng.normal(size=(500, 2))
    c = ConditionalTemperature(cond_dim=2).fit(
        logits, y, cond, epochs=100, split=_val(500))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = c.transform(np.full(500, 1e4), cond)
        q = c.transform(np.full(500, -1e4), cond)
        g = GlobalTemperature().fit(
            logits, y, split=_val(500)).transform(np.array([1e4, -1e4]))
    assert ((p >= 0) & (p <= 1)).all()
    assert ((q >= 0) & (q <= 1)).all()
    assert ((g >= 0) & (g <= 1)).all()


# --- constructor and signature contract ------------------------------------

def _underconfident_regimes(n=4000, seed=21):
    """Two regimes the fit wants to give temperatures BELOW 1 (0.5 and 0.8):
    the only place an `eps` floor is observable rather than slack."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    regime = (rng.random(n) < 0.5).astype(float)
    return z * np.where(regime > 0, 0.5, 0.8), y, regime


def test_eps_floors_the_temperature():
    logits, y, regime = _underconfident_regimes()
    cond = regime[:, None]
    split = _val(len(y))

    free = ConditionalTemperature(cond_dim=1).fit(
        logits, y, cond, epochs=400, split=split).temperatures(cond)
    assert free.min() < 0.9        # unfloored, the fit really does go below 1

    floored = ConditionalTemperature(cond_dim=1, eps=1.0).fit(
        logits, y, cond, epochs=400, split=split).temperatures(cond)
    assert (floored >= 1.0).all()


@pytest.mark.parametrize("eps", [0.0, -1e-3])
def test_non_positive_eps_is_rejected(eps):
    with pytest.raises(ValueError, match="eps"):
        ConditionalTemperature(cond_dim=1, eps=eps)


@pytest.mark.parametrize("cond_dim", [0, -1, 1.5])
def test_invalid_cond_dim_is_rejected(cond_dim):
    with pytest.raises(ValueError, match="cond_dim"):
        ConditionalTemperature(cond_dim=cond_dim)


def test_fit_does_not_take_a_learning_rate():
    """L-BFGS's strong-Wolfe line search chooses its own step: an `lr` knob
    here would be inert, and an inert knob invites a caller to tune it."""
    logits, y, regime = _two_regimes(200, 16)
    with pytest.raises(TypeError, match="lr"):
        ConditionalTemperature(cond_dim=1).fit(
            logits, y, regime[:, None], epochs=100, lr=0.05, split=_val(200))


@pytest.mark.parametrize("budget", [0, -1])
def test_zero_iteration_budgets_are_rejected(budget):
    logits, y, regime = _two_regimes(200, 17)
    with pytest.raises(ValueError, match="max_iter"):
        GlobalTemperature().fit(logits, y, split=_val(200), max_iter=budget)
    with pytest.raises(ValueError, match="epochs"):
        ConditionalTemperature(cond_dim=1).fit(
            logits, y, regime[:, None], epochs=budget, split=_val(200))


def test_conditional_fit_is_reproducible():
    logits, y = _overconfident(800, 16)
    rng = np.random.default_rng(16)
    cond = rng.normal(size=(800, 2))
    a = ConditionalTemperature(cond_dim=2).fit(
        logits, y, cond, epochs=100, split=_val(800)).temperatures(cond)
    b = ConditionalTemperature(cond_dim=2).fit(
        logits, y, cond, epochs=100, split=_val(800)).temperatures(cond)
    np.testing.assert_allclose(a, b)


def test_fitting_does_not_disturb_the_global_torch_rng():
    """No global seeding: two fits either side of a draw must not change it."""
    import torch

    logits, y = _overconfident(200, 17)
    gen_state = torch.random.get_rng_state()
    GlobalTemperature().fit(logits, y, split=_val(200))
    ConditionalTemperature(cond_dim=1).fit(
        logits, y, np.arange(200.0)[:, None] % 2, epochs=200, split=_val(200))
    assert torch.equal(torch.random.get_rng_state(), gen_state)
