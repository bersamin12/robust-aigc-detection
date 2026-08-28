import numpy as np
import pytest

from aigcdet.calibrate import INTERNAL_VAL_SPLIT
from aigcdet.calibrate.policy import (
    Policy, auto_decided_fraction, decide, fit_policy, policy_report,
)


def _val(p):
    """The per-row split column a legitimate caller passes."""
    return np.full(len(p), INTERNAL_VAL_SPLIT)


def _population(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(int)
    eqi = rng.uniform(0.2, 1.0, n)
    # High-EQI images are scored well; low-EQI ones are near chance.
    p = np.where(rng.random(n) < eqi,
                 np.where(y == 1, rng.uniform(0.7, 1.0, n), rng.uniform(0.0, 0.3, n)),
                 rng.uniform(0.3, 0.7, n))
    return p, y, eqi


# A hand-checkable population with heavily TIED probabilities. Ties are where a
# `>=` / `<=` convention that does not survive a score flip shows itself.
#          idx:    0     1     2     3     4     5     6     7     8     9
_TIED_P = np.array([0.10, 0.10, 0.10, 0.20, 0.20, 0.30, 0.80, 0.90, 0.90, 0.95])
_TIED_Y = np.array([   0,    0,    0,    1,    1,    0,    1,    1,    1,    1])


def _reference_flag_threshold(p, y, target_fpr):
    """Lowest observed p at which at most `target_fpr` of authentic images flag.

    Brute force over every candidate, straight from the definition of the
    quantity `flag_threshold` claims to be.
    """
    authentic = p[y == 0]
    best = np.inf
    for c in np.unique(p):
        if (authentic >= c).mean() <= target_fpr:
            best = min(best, c)
    return float(best)


def _reference_clear_threshold(p, y, target_fpr):
    """Highest observed p at which at most `target_fpr` of AI images clear.

    Computed directly on the real-side quantity, with no label/score flip, so
    that it is an independent check on the flip the implementation performs.
    """
    ai = p[y == 1]
    best = -np.inf
    for c in np.unique(p):
        if (ai <= c).mean() <= target_fpr:
            best = max(best, c)
    return float(best)


# --------------------------------------------------------------------------
# The two thresholds
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("target_fpr", "expected_flag", "expected_clear"),
    [(0.0, 0.80, 0.10), (0.2, 0.80, 0.10), (0.4, 0.20, 0.30), (0.5, 0.20, 0.80)],
)
def test_thresholds_match_hand_computed_values_on_tied_scores(
        target_fpr, expected_flag, expected_clear):
    """Hand-worked reference. Authentic p = {.10,.10,.10,.30}; AI p =
    {.20,.20,.80,.90,.90,.95}. At target 0.4, at most 2 of 6 AI images may
    clear, so the clear threshold is 0.30 and not 0.80."""
    pol = fit_policy(_TIED_P, _TIED_Y, np.ones(10), target_fpr=target_fpr,
                     target_coverage=1.0, split=_val(_TIED_P))
    assert pol.flag_threshold == expected_flag
    assert pol.clear_threshold == expected_clear


@pytest.mark.parametrize("target_fpr", [0.0, 0.2, 0.4, 0.5])
def test_clear_threshold_equals_a_directly_computed_real_side_reference(target_fpr):
    """The implementation reaches the real-side threshold by flipping labels and
    scores. That is a trick, not an identity, so check it against the quantity
    computed from first principles on the unflipped data."""
    pol = fit_policy(_TIED_P, _TIED_Y, np.ones(10), target_fpr=target_fpr,
                     target_coverage=1.0, split=_val(_TIED_P))
    assert pol.clear_threshold == _reference_clear_threshold(
        _TIED_P, _TIED_Y, target_fpr)
    assert pol.flag_threshold == _reference_flag_threshold(
        _TIED_P, _TIED_Y, target_fpr)


def test_clear_threshold_is_an_observed_probability_not_a_rounded_complement():
    """`1 - (1 - p)` is not `p` in binary floating point. A clear threshold one
    ulp below an observed probability silently stops clearing that whole tie
    group; one ulp above can clear a group the target did not pay for."""
    pol = fit_policy(_TIED_P, _TIED_Y, np.ones(10), target_fpr=0.0,
                     target_coverage=1.0, split=_val(_TIED_P))
    assert pol.clear_threshold in _TIED_P
    d = decide(_TIED_P, np.ones(10), pol)
    # The three authentic images at exactly 0.10 must clear, not fall to review.
    assert list(d) == ["clear"] * 3 + ["review"] * 3 + ["flag"] * 4


@pytest.mark.parametrize("target_fpr", [0.0, 0.005, 0.01, 0.05, 0.1])
def test_neither_threshold_is_ever_more_permissive_than_the_exact_optimum(target_fpr):
    """`threshold_at_fpr` may drop collinear ROC vertices, which can make it
    stricter than the exact optimum. Stricter is safe; looser would overspend
    the FPR budget. Pin the direction on both sides."""
    p, y, eqi = _population(seed=5)
    pol = fit_policy(p, y, eqi, target_fpr=target_fpr, split=_val(p))
    assert pol.flag_threshold >= _reference_flag_threshold(p, y, target_fpr)
    assert pol.clear_threshold <= _reference_clear_threshold(p, y, target_fpr)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("target_fpr", [0.0, 0.01, 0.05])
def test_false_clear_rate_on_ai_images_respects_the_target(seed, target_fpr):
    p, y, eqi = _population(seed=seed)
    pol = fit_policy(p, y, eqi, target_fpr=target_fpr, split=_val(p))
    assert (p[y == 1] <= pol.clear_threshold).mean() <= target_fpr


def test_fit_policy_rejects_labels_with_only_one_class():
    """One class makes an FPR threshold meaningless: sklearn's ROC returns NaN
    rates and a warning, and the policy would ship a nonsense threshold."""
    p, _, eqi = _population(n=200)
    with pytest.raises(ValueError, match="both classes"):
        fit_policy(p, np.ones(200, dtype=int), eqi, split=_val(p))


@pytest.mark.parametrize("wrong", ["test", "train", "heldout_generator", "benchmark"])
def test_fit_policy_refuses_rows_from_any_split_but_internal_validation(wrong):
    p, y, eqi = _population(n=200)
    with pytest.raises(ValueError, match="val_internal"):
        fit_policy(p, y, eqi, split=np.full(200, wrong))


def test_fit_policy_refuses_a_single_contaminating_row():
    p, y, eqi = _population(n=200)
    contaminated = _val(p)
    contaminated[11] = "test"
    with pytest.raises(ValueError, match=r"1 of 200 rows"):
        fit_policy(p, y, eqi, split=contaminated)


def test_fit_policy_refuses_a_bare_string_promise():
    """The guard this replaced took the caller's word for the whole split."""
    p, y, eqi = _population(n=200)
    with pytest.raises(ValueError, match="one label per row"):
        fit_policy(p, y, eqi, split=INTERNAL_VAL_SPLIT)


def test_fit_policy_requires_the_split_column():
    p, y, eqi = _population(n=200)
    with pytest.raises(TypeError, match="split"):
        fit_policy(p, y, eqi)


@pytest.mark.parametrize("target_fpr", [-0.01, 1.5])
def test_fit_policy_rejects_out_of_range_target_fpr(target_fpr):
    p, y, eqi = _population(n=200)
    with pytest.raises(ValueError, match="target_fpr"):
        fit_policy(p, y, eqi, target_fpr=target_fpr, split=_val(p))


@pytest.mark.parametrize("target_coverage", [-0.01, 1.5])
def test_fit_policy_rejects_out_of_range_target_coverage(target_coverage):
    p, y, eqi = _population(n=200)
    with pytest.raises(ValueError, match="target_coverage"):
        fit_policy(p, y, eqi, target_coverage=target_coverage, split=_val(p))


def test_fit_policy_rejects_probabilities_outside_the_unit_interval():
    p, y, eqi = _population(n=200)
    p = p.copy()
    p[0] = 1.5
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fit_policy(p, y, eqi, split=_val(p))


def test_fit_policy_rejects_mismatched_lengths():
    p, y, eqi = _population(n=200)
    with pytest.raises(ValueError, match="same length"):
        fit_policy(p, y, eqi[:100], split=_val(p))


# --------------------------------------------------------------------------
# The EQI gate
# --------------------------------------------------------------------------

def test_full_target_coverage_leaves_no_image_behind_the_eqi_gate():
    p, y, eqi = _population()
    pol = fit_policy(p, y, eqi, target_coverage=1.0, split=_val(p))
    assert (eqi >= pol.eqi_threshold).all()


def test_zero_target_coverage_sends_every_image_to_review():
    """A quantile at 1.0 is the maximum EQI, which still admits the single
    most-evidenced image; zero coverage has to mean zero."""
    p, y, eqi = _population()
    pol = fit_policy(p, y, eqi, target_coverage=0.0, split=_val(p))
    d = decide(p, eqi, pol)
    assert (d == "review").all()
    assert auto_decided_fraction(d) == 0.0


def test_report_on_an_empty_auto_population_says_undefined_not_zero():
    """An empty denominator is not an accuracy of 0.0 and not a silent NaN."""
    p, y, eqi = _population()
    rep = policy_report(
        p, y, eqi, fit_policy(p, y, eqi, target_coverage=0.0, split=_val(p)))
    assert rep["auto_fraction"] == 0.0
    assert rep["review_fraction"] == 1.0
    assert rep["n_auto"] == 0
    assert rep["accuracy_on_auto"] is None
    assert rep["realised_fpr"] is None


def test_eqi_gate_beats_a_confident_probability():
    """A low-EQI image is reviewed however extreme its probability is."""
    pol = Policy(flag_threshold=0.9, clear_threshold=0.1, eqi_threshold=0.5)
    d = decide(np.array([0.99, 0.01, 0.99, 0.01]),
               np.array([0.9, 0.9, 0.1, 0.1]), pol)
    assert list(d) == ["flag", "clear", "review", "review"]


def test_overlapping_thresholds_route_to_review_rather_than_pick_a_winner():
    """A policy whose clear threshold sits above its flag threshold makes both
    rules fire on the same image. Answering "clear" (or "flag") there states a
    confident decision the evidence does not support."""
    pol = Policy(flag_threshold=0.4, clear_threshold=0.6, eqi_threshold=0.0)
    d = decide(np.array([0.3, 0.5, 0.7]), np.ones(3), pol)
    assert list(d) == ["clear", "review", "flag"]


def test_decide_rejects_mismatched_lengths():
    pol = Policy(flag_threshold=0.9, clear_threshold=0.1, eqi_threshold=0.5)
    with pytest.raises(ValueError, match="same length"):
        decide(np.array([0.5, 0.5]), np.array([0.5]), pol)


def test_auto_decided_fraction_rejects_an_empty_decision_array():
    with pytest.raises(ValueError, match="empty"):
        auto_decided_fraction(np.array([], dtype="<U6"))


# --------------------------------------------------------------------------
# The reported numbers
# --------------------------------------------------------------------------

def test_decisions_are_only_the_three_allowed_labels():
    p, y, eqi = _population()
    pol = fit_policy(p, y, eqi, split=_val(p))
    d = decide(p, eqi, pol)
    assert set(np.unique(d)) <= {"clear", "review", "flag"}
    assert len(d) == len(p)


def test_low_eqi_images_are_routed_to_review():
    p, y, eqi = _population()
    pol = fit_policy(p, y, eqi, split=_val(p))
    d = decide(p, eqi, pol)
    assert (d[eqi < pol.eqi_threshold] == "review").all()


def test_realised_fpr_on_auto_decided_images_respects_the_target():
    p, y, eqi = _population()
    pol = fit_policy(p, y, eqi, target_fpr=0.01, split=_val(p))
    rep = policy_report(p, y, eqi, pol)
    assert rep["realised_fpr"] <= 0.02


def test_auto_decided_fraction_is_between_zero_and_one():
    p, y, eqi = _population()
    d = decide(p, eqi, fit_policy(p, y, eqi, split=_val(p)))
    f = auto_decided_fraction(d)
    assert 0.0 <= f <= 1.0
    assert f == pytest.approx(1.0 - (d == "review").mean())


def test_accuracy_on_auto_decided_beats_accuracy_on_everything():
    p, y, eqi = _population()
    pol = fit_policy(p, y, eqi, split=_val(p))
    rep = policy_report(p, y, eqi, pol)
    all_acc = (((p >= 0.5).astype(int)) == y).mean()
    assert rep["accuracy_on_auto"] > all_acc


def test_report_fields_are_present():
    p, y, eqi = _population()
    rep = policy_report(p, y, eqi, fit_policy(p, y, eqi, split=_val(p)))
    assert {"auto_fraction", "realised_fpr", "accuracy_on_auto", "review_fraction"} <= set(rep)


def test_realised_fpr_denominator_is_the_auto_decided_authentic_images():
    """The headline number is a rate, and a rate is its denominator. Computed
    over ALL authentic images instead of the auto-decided ones it would be a
    different, smaller number reported under the same name."""
    p, y, eqi = _population()
    pol = fit_policy(p, y, eqi, split=_val(p))
    rep = policy_report(p, y, eqi, pol)
    d = decide(p, eqi, pol)
    auto_authentic = (d != "review") & (y == 0)

    assert rep["n_authentic_auto"] == int(auto_authentic.sum())
    assert rep["realised_fpr"] == pytest.approx(
        (d[auto_authentic] == "flag").mean())
    # A denominator of every authentic image gives a materially different rate.
    over_all_authentic = (d[y == 0] == "flag").mean()
    assert abs(rep["realised_fpr"] - over_all_authentic) > 1e-3


def test_accuracy_on_auto_denominator_is_the_auto_decided_images():
    p, y, eqi = _population()
    pol = fit_policy(p, y, eqi, split=_val(p))
    rep = policy_report(p, y, eqi, pol)
    d = decide(p, eqi, pol)
    auto = d != "review"

    assert rep["n_auto"] == int(auto.sum())
    assert rep["accuracy_on_auto"] == pytest.approx(
        ((d[auto] == "flag").astype(int) == y[auto]).mean())


def test_auto_and_review_fractions_are_over_the_whole_queue_and_sum_to_one():
    p, y, eqi = _population()
    pol = fit_policy(p, y, eqi, split=_val(p))
    rep = policy_report(p, y, eqi, pol)
    d = decide(p, eqi, pol)

    assert rep["auto_fraction"] == pytest.approx((d != "review").mean())
    assert rep["review_fraction"] == pytest.approx((d == "review").mean())
    assert rep["auto_fraction"] + rep["review_fraction"] == pytest.approx(1.0)
    assert rep["n_auto"] == int((d != "review").sum())
