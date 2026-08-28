import numpy as np
import pytest

from aigcdet.baselines.aeroblade import aeroblade_score
from aigcdet.eval.metrics import roc_auc
from aigcdet.features.recon import RECON_FEATURE_NAMES

_L1 = RECON_FEATURE_NAMES.index("l1")


def test_lower_reconstruction_error_yields_a_higher_aigc_score():
    low = np.zeros(len(RECON_FEATURE_NAMES), dtype=np.float32)
    low[0] = 0.01                      # tiny L1 -> looks like a VAE round-trip
    high = np.zeros(len(RECON_FEATURE_NAMES), dtype=np.float32)
    high[0] = 0.20
    assert aeroblade_score(low) > aeroblade_score(high)


def test_score_is_finite_for_a_zero_vector():
    assert np.isfinite(aeroblade_score(np.zeros(len(RECON_FEATURE_NAMES), np.float32)))


def _recon_vec(rng, l1):
    """A plausible `r` vector: every slot populated, only `l1` controlled."""
    v = rng.uniform(0.05, 2.0, len(RECON_FEATURE_NAMES)).astype(np.float32)
    v[_L1] = l1
    return v


def test_the_sign_convention_gives_ai_images_the_higher_score_not_the_lower():
    """The failure this pins: `+l1` instead of `-l1` puts every AUC in the
    report below 0.5, which reads as 'the baseline fails' when it means 'we
    inverted it'.

    Latent-diffusion images round-trip through their own autoencoder with LOW
    error, so the AI class is the low-`l1` class. Every other slot is random
    and overlapping between the classes, so only the `l1` readout and its sign
    can produce the ordering.
    """
    rng = np.random.default_rng(11)
    real = [_recon_vec(rng, l1) for l1 in rng.uniform(0.10, 0.20, 25)]
    ai = [_recon_vec(rng, l1) for l1 in rng.uniform(0.01, 0.05, 25)]
    y = np.array([0] * 25 + [1] * 25)
    s = np.array([aeroblade_score(v) for v in real + ai])
    assert roc_auc(y, s) == 1.0


def test_score_reads_the_l1_slot_by_name_not_by_a_hard_coded_position():
    """`RECON_FEATURE_NAMES` is the positional contract for the `r` bank. If
    that tuple is reordered, the readout must follow it; a literal index would
    silently start reporting `lpips` or `err_std`."""
    rng = np.random.default_rng(5)
    v = rng.uniform(0.05, 2.0, len(RECON_FEATURE_NAMES)).astype(np.float32)
    assert len(np.unique(v)) == len(v)          # every slot distinguishable
    assert aeroblade_score(v) == pytest.approx(-float(v[_L1]))


def test_score_depends_on_l1_alone():
    """Training-free means exactly one number out of the cached vector. If any
    other slot leaked in, the baseline would no longer be the published
    method."""
    rng = np.random.default_rng(9)
    base = _recon_vec(rng, 0.07)
    other = _recon_vec(rng, 0.07)
    assert not np.allclose(np.delete(base, _L1), np.delete(other, _L1))
    assert aeroblade_score(base) == aeroblade_score(other)


def test_score_is_strictly_decreasing_in_reconstruction_error():
    rng = np.random.default_rng(4)
    errors = np.sort(rng.uniform(0.0, 0.5, 30))
    scores = [aeroblade_score(_recon_vec(rng, e)) for e in errors]
    assert all(a > b for a, b in zip(scores, scores[1:]))


def test_score_is_a_plain_float():
    """The signature says `-> float`. A numpy scalar propagates float32 into
    downstream metric code and silently changes rounding there."""
    v = np.ones(len(RECON_FEATURE_NAMES), dtype=np.float32)
    assert type(aeroblade_score(v)) is float
