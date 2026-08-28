import numpy as np
import pytest

from aigcdet.baselines.aeroblade import (
    DEFAULT_DISTANCE, DISTANCES, aeroblade_score, aeroblade_scores,
)
from aigcdet.eval.metrics import roc_auc
from aigcdet.features.recon import RECON_FEATURE_NAMES

_L1 = RECON_FEATURE_NAMES.index("l1")
_N = len(RECON_FEATURE_NAMES)


# The brief's two tests. `distance="l1"` is the only edit: they write the value
# into slot 0, which is `l1`, and the default distance is now the published
# LPIPS. The l1 path must stay tested so both rows can be reported.
def test_lower_reconstruction_error_yields_a_higher_aigc_score():
    low = np.zeros(len(RECON_FEATURE_NAMES), dtype=np.float32)
    low[0] = 0.01                      # tiny L1 -> looks like a VAE round-trip
    high = np.zeros(len(RECON_FEATURE_NAMES), dtype=np.float32)
    high[0] = 0.20
    assert aeroblade_score(low, distance="l1") > aeroblade_score(high, distance="l1")


def test_score_is_finite_for_a_zero_vector():
    assert np.isfinite(aeroblade_score(np.zeros(len(RECON_FEATURE_NAMES), np.float32)))


# --------------------------------------------------------------------------

def _recon_vec(rng, distance, value):
    """A plausible `r` vector: every slot populated, only `distance` controlled."""
    v = rng.uniform(0.05, 2.0, _N).astype(np.float32)
    v[RECON_FEATURE_NAMES.index(distance)] = value
    return v


def test_the_default_distance_is_the_published_one():
    """LPIPS, not L1. The paper's own finding is that perceptual distance
    substantially beats pixel L1, and both are cached in the same vector, so
    defaulting to L1 would ship a knowingly weaker baseline in the one table
    where that understates the competition rather than ourselves."""
    assert DEFAULT_DISTANCE == "lpips"
    rng = np.random.default_rng(2)
    v = rng.uniform(0.05, 2.0, _N).astype(np.float32)
    assert aeroblade_score(v) == aeroblade_score(v, distance="lpips")
    assert aeroblade_score(v) != aeroblade_score(v, distance="l1")


@pytest.mark.parametrize("distance", DISTANCES)
def test_the_sign_convention_gives_ai_images_the_higher_score_not_the_lower(distance):
    """The failure this pins: dropping the negation puts every AUC in the
    report below 0.5, which reads as 'the baseline fails' when it means 'we
    inverted it'.

    Latent-diffusion images round-trip through their own autoencoder with LOW
    error, so the AI class is the low-distance class. Every other slot is
    random and overlapping between the classes, so only the chosen slot's
    readout and its sign can produce the ordering.
    """
    rng = np.random.default_rng(11)
    real = [_recon_vec(rng, distance, d) for d in rng.uniform(0.10, 0.20, 25)]
    ai = [_recon_vec(rng, distance, d) for d in rng.uniform(0.01, 0.05, 25)]
    y = np.array([0] * 25 + [1] * 25)
    s = np.array([aeroblade_score(v, distance=distance) for v in real + ai])
    assert roc_auc(y, s) == 1.0


@pytest.mark.parametrize("distance", DISTANCES)
def test_the_slot_is_resolved_by_name_at_call_time_not_frozen_at_import(
        monkeypatch, distance):
    """`RECON_FEATURE_NAMES` is the positional contract for the `r` bank. If it
    is ever reordered the readout must follow it.

    A literal index passes every other test in this file, because `l1` IS index
    0 today. Only reordering the tuple can tell the difference -- so this test
    reorders it, and it has to be a call-time lookup rather than a module
    constant computed at import for the reorder to be visible at all.
    """
    reordered = tuple(reversed(RECON_FEATURE_NAMES))
    monkeypatch.setattr("aigcdet.baselines.aeroblade.RECON_FEATURE_NAMES", reordered)
    v = np.arange(_N, dtype=np.float32) + 1.0     # every slot distinguishable
    assert aeroblade_score(v, distance=distance) == pytest.approx(
        -float(v[reordered.index(distance)]))
    assert reordered.index(distance) != RECON_FEATURE_NAMES.index(distance)


@pytest.mark.parametrize("distance", DISTANCES)
def test_score_depends_on_its_own_slot_alone(distance):
    """Training-free means exactly one number out of the cached vector. If any
    other slot leaked in, the baseline would no longer be the published
    method."""
    rng = np.random.default_rng(9)
    base = _recon_vec(rng, distance, 0.07)
    other = _recon_vec(rng, distance, 0.07)
    assert not np.allclose(np.delete(base, RECON_FEATURE_NAMES.index(distance)),
                           np.delete(other, RECON_FEATURE_NAMES.index(distance)))
    assert aeroblade_score(base, distance=distance) == aeroblade_score(
        other, distance=distance)


def test_score_is_strictly_decreasing_in_reconstruction_error():
    rng = np.random.default_rng(4)
    errors = np.sort(rng.uniform(0.0, 0.5, 30))
    scores = [aeroblade_score(_recon_vec(rng, "lpips", e)) for e in errors]
    assert all(a > b for a, b in zip(scores, scores[1:]))


def test_score_is_a_plain_float():
    """The signature says `-> float`. A numpy scalar propagates float32 into
    downstream metric code and silently changes rounding there."""
    v = np.ones(_N, dtype=np.float32)
    assert type(aeroblade_score(v)) is float


def test_an_unknown_distance_is_refused_rather_than_indexed():
    v = np.ones(_N, dtype=np.float32)
    with pytest.raises(ValueError, match="distance must be one of"):
        aeroblade_score(v, distance="err_std")
    with pytest.raises(ValueError, match="distance must be one of"):
        aeroblade_score(v, distance="nonsense")


def test_a_non_finite_slot_is_refused_rather_than_scored():
    v = np.ones(_N, dtype=np.float32)
    v[RECON_FEATURE_NAMES.index("lpips")] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        aeroblade_score(v)


# --------------------------------------------------------------------------
# Bank-level scoring, where the "column is silently unusable" failure lives.
# --------------------------------------------------------------------------

def test_bank_scores_are_the_negated_column():
    rng = np.random.default_rng(21)
    bank = rng.uniform(0.01, 1.0, (12, _N))
    s = aeroblade_scores(bank)
    assert s.shape == (12,)
    assert s == pytest.approx(-bank[:, RECON_FEATURE_NAMES.index("lpips")])
    assert s == pytest.approx([aeroblade_score(r) for r in bank])


def test_an_all_zero_distance_column_is_refused_loudly():
    """`lpips` is an optional extra and is deliberately uninstalled here, so a
    recon extraction run without it leaves that column all-zero. Scoring it
    returns a constant 0 for every row -- an AUC of exactly 0.5, which in the
    results table is indistinguishable from 'AEROBLADE does not work on our
    data'. This is the guard that stops that being reported as a finding.
    """
    rng = np.random.default_rng(22)
    bank = rng.uniform(0.01, 1.0, (10, _N))
    bank[:, RECON_FEATURE_NAMES.index("lpips")] = 0.0
    with pytest.raises(ValueError, match="all-zero"):
        aeroblade_scores(bank)
    # The l1 column is intact, so that row can still be reported.
    assert np.isfinite(aeroblade_scores(bank, distance="l1")).all()


def test_a_column_with_non_finite_entries_is_refused_loudly():
    rng = np.random.default_rng(23)
    bank = rng.uniform(0.01, 1.0, (10, _N))
    bank[3, RECON_FEATURE_NAMES.index("lpips")] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        aeroblade_scores(bank)


def test_a_single_zero_row_is_not_treated_as_an_unusable_column():
    """One image really can round-trip at distance ~0. Only an ENTIRELY zero
    column indicates a missing extraction, so the guard must not fire on a
    bank that merely contains a zero."""
    rng = np.random.default_rng(24)
    bank = rng.uniform(0.01, 1.0, (10, _N))
    bank[0, RECON_FEATURE_NAMES.index("lpips")] = 0.0
    assert aeroblade_scores(bank)[0] == 0.0


@pytest.mark.parametrize("bad", [
    np.zeros((4, 3)), np.zeros(12), np.zeros((2, 3, 12)), np.zeros((0, 12)),
])
def test_a_bank_of_the_wrong_shape_is_refused(bad):
    with pytest.raises(ValueError):
        aeroblade_scores(bad)
