import dataclasses

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


def _sigmoid_population(n=4000, seed=0):
    """Probabilities the way the pipeline makes them: a sigmoid of a logit.

    `_population`'s `rng.uniform` draws are k/2**53, for which `1 - (1 - p)` is
    exactly `p`; sigmoid outputs are not, which is where the complement form of
    the clear threshold goes wrong.
    """
    rng = np.random.default_rng(int(seed))
    y = (rng.random(n) < 0.5).astype(int)
    z = rng.normal(np.where(y == 1, 2.0, -2.0), 1.5)
    return 1.0 / (1.0 + np.exp(-z)), y


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
    group.

    Which probabilities it bites is not obvious and is worth stating, because it
    is why this fixture uses decimal literals rather than `rng.random()`: for
    raw uniform doubles the round trip is exact (0 of 5,000,000 measured), since
    they are all k/2**53. For the sigmoid outputs a calibrated detector actually
    produces it fails for 38.9% of values, and for 2-dp rounded scores 32.0%.

    The harm is one-directional: a pure COVERAGE LOSS that understates the
    Impact figure, never an FPR overclaim. Measured over 3000 randomised
    targets, all 497 differing cleared sets went the conservative way and none
    exceeded the budget -- gaining a member would need two observed
    probabilities within one ulp of each other, which float probabilities do
    not produce.
    """
    pol = fit_policy(_TIED_P, _TIED_Y, np.ones(10), target_fpr=0.0,
                     target_coverage=1.0, split=_val(_TIED_P))
    assert pol.clear_threshold in _TIED_P
    d = decide(_TIED_P, np.ones(10), pol)
    # The three authentic images at exactly 0.10 must clear, not fall to review.
    assert list(d) == ["clear"] * 3 + ["review"] * 3 + ["flag"] * 4


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("target_fpr", [0.005, 0.01, 0.05])
def test_clear_threshold_is_exact_on_probabilities_shaped_like_the_pipeline_s(
        target_fpr, seed):
    """The regime the detector actually ships in.

    `_population` draws `p` from `rng.uniform`, where the complement round trip
    happens to be exact, so it cannot see this bug at all. Real probabilities
    come off a sigmoid, where `1 - (1 - p) != p` for 38.9% of values: on 20
    seeds x 4 targets of this population the complement form put the threshold
    on a value no image had in 57 of 80 cases.
    """
    p, y = _sigmoid_population(seed=seed)
    pol = fit_policy(p, y, np.ones(len(p)), target_fpr=target_fpr,
                     target_coverage=1.0, split=_val(p))
    assert pol.clear_threshold in p
    assert pol.clear_threshold == _reference_clear_threshold(p, y, target_fpr)


@pytest.mark.parametrize("target_fpr", [0.0, 0.005, 0.01, 0.05, 0.1])
def test_both_thresholds_are_the_exact_optimum_not_merely_a_safe_one(target_fpr):
    """Equality on both sides, against the brute-force references.

    This was an inequality while `threshold_at_fpr` used `roc_curve`'s default
    `drop_intermediate=True`, which deletes collinear ROC vertices -- some of
    them thresholds that satisfy the target. That was safe (stricter, never
    looser: it spent less than the FPR budget) but it silently cost coverage,
    which is the Impact figure. With the drop disabled both sides are exact,
    and asserting equality here fails if that ever regresses.
    """
    p, y, eqi = _population(seed=5)
    pol = fit_policy(p, y, eqi, target_fpr=target_fpr, split=_val(p))
    assert pol.flag_threshold == _reference_flag_threshold(p, y, target_fpr)
    assert pol.clear_threshold == _reference_clear_threshold(p, y, target_fpr)


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
# The shipped operating point
# --------------------------------------------------------------------------

@pytest.mark.parametrize("target_coverage", [0.25, 0.5, 0.85])
def test_eqi_gate_defers_exactly_the_share_the_coverage_target_names(target_coverage):
    """`target_coverage` is the Impact figure's other half, and a percent-vs-
    fraction slip in the quantile argument would defer 92.5% instead of 15%
    while every other test stayed green. Pin the knob where it can fail: the
    two existing coverage tests sit at 1.0 and 0.0, and 1 - 1.0 == 0 is
    identical under every wrong formula."""
    p, y, eqi = _population()
    pol = fit_policy(p, y, eqi, target_coverage=target_coverage, split=_val(p))
    assert (eqi >= pol.eqi_threshold).mean() == pytest.approx(target_coverage)


def test_the_shipped_defaults_are_the_one_percent_eighty_five_percent_operating_point():
    """`target_fpr=0.01, target_coverage=0.85` are the numbers spec §1.3/§6.1
    states publicly. Every other test passes its targets explicitly, so both
    literals could be edited without a failure. Assert the behaviour the
    defaults produce rather than `__defaults__`, which pins the spelling
    instead of the operating point."""
    p, y, eqi = _population()
    pol = fit_policy(p, y, eqi, split=_val(p))

    # The FPR guarantee, over the population it is guaranteed on: all authentic.
    assert (p[y == 0] >= pol.flag_threshold).mean() <= 0.01
    # Spent symmetrically on the clear side.
    assert (p[y == 1] <= pol.clear_threshold).mean() <= 0.01
    # And 85% of the queue clears the EQI gate.
    assert (eqi >= pol.eqi_threshold).mean() == pytest.approx(0.85)


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


def test_realised_fpr_is_not_bounded_by_the_all_authentic_rate_in_either_direction():
    """The one inference a reader must not draw from `realised_fpr`.

    `target_fpr` is guaranteed over ALL authentic rows. `realised_fpr`
    conditions on the auto-decided subset afterwards, and that conditioning is
    not signed: across eight seeds of the same population it lands above the
    all-authentic rate five times and below it three times. A docstring or a
    write-up that says "the EQI gate removes the weakest images, which is why
    this is lower" turns a 1.28% into "below the 1% we targeted".

    This test asserts only that BOTH directions occur, so the day someone
    "fixes" the code or the prose to make one of them impossible, it fails. The
    targets are passed explicitly rather than defaulted, so that a change to the
    shipped defaults fails the operating-point test above and not this one.
    """
    higher = lower = 0
    for seed in range(8):
        p, y, eqi = _population(seed=seed)
        pol = fit_policy(p, y, eqi, target_fpr=0.01, target_coverage=0.85,
                         split=_val(p))
        rep = policy_report(p, y, eqi, pol)
        all_authentic_fpr = float((p[y == 0] >= pol.flag_threshold).mean())
        # The guarantee itself holds on every seed, on its own population.
        assert all_authentic_fpr <= 0.01
        higher += rep["realised_fpr"] > all_authentic_fpr
        lower += rep["realised_fpr"] < all_authentic_fpr
    assert higher > 0, "realised_fpr never exceeded the all-authentic rate"
    assert lower > 0, "realised_fpr never fell below the all-authentic rate"


# --------------------------------------------------------------------------
# Guards whose absence changes no other test
# --------------------------------------------------------------------------

def test_policy_is_frozen_so_a_report_cannot_describe_other_thresholds():
    """`policy_report` calls `decide` with the policy it was handed. A policy
    mutated between the two would silently describe decisions nobody made."""
    pol = Policy(flag_threshold=0.9, clear_threshold=0.1, eqi_threshold=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        pol.flag_threshold = 0.5


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("column", ["p", "eqi"])
def test_fit_policy_rejects_non_finite_scores(bad, column):
    """A NaN silently loses every comparison, so a NaN probability is neither
    flagged nor cleared and quietly inflates the review queue; an inf sails
    past both thresholds."""
    p, y, eqi = _population(n=200)
    p, eqi = p.copy(), eqi.copy()
    if column == "p":
        p[7] = bad
    else:
        eqi[7] = bad
    with pytest.raises(ValueError, match="finite"):
        fit_policy(p, y, eqi, split=_val(p))


def test_fit_policy_rejects_empty_inputs():
    """`np.quantile` of an empty array is a NaN threshold plus a RuntimeWarning,
    and every downstream rate becomes 0/0."""
    empty = np.array([], dtype=float)
    with pytest.raises(ValueError, match="empty"):
        fit_policy(empty, np.array([], dtype=int), empty, split=_val(empty))
