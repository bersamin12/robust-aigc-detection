"""`aigcdet.augment.geometric` -- dihedral augmentation.

The reason this module exists rather than a `cv2.warpAffine` is that it must
not touch the pixel VALUES, only their positions: `laplacian_var` is the
corpus's largest surviving confound at AUC 0.672, and an arbitrary-angle
rotation resamples and therefore attenuates exactly that channel. So the
load-bearing test here is not "does it rotate" but "does the histogram
survive".
"""
from __future__ import annotations

import numpy as np
import pytest

from aigcdet.augment.canonical import CanonPolicy, MODE_CROP, canonicalise
from aigcdet.augment.geometric import (
    DIHEDRAL_N, dihedral, geometric_rng, sample_dihedral,
)
from aigcdet.features.proxies import proxy_vector


def _square(side=64, seed=0):
    """A textured square. Random noise, because a smooth gradient can be
    symmetric under some group elements and would make the distinctness test
    pass vacuously."""
    return np.random.default_rng(seed).integers(
        0, 256, (side, side, 3), dtype=np.uint8)


# --------------------------------------------------------------------------
# the group
# --------------------------------------------------------------------------

def test_every_element_gives_a_distinct_image():
    """Eight orientations, not eight names for fewer."""
    img = _square()
    seen = {dihedral(img, k).tobytes() for k in range(DIHEDRAL_N)}
    assert len(seen) == DIHEDRAL_N


def test_the_identity_element_returns_the_image_unchanged():
    img = _square()
    assert np.array_equal(dihedral(img, 0), img)


def test_every_element_preserves_shape_and_dtype():
    img = _square()
    for k in range(DIHEDRAL_N):
        out = dihedral(img, k)
        assert out.shape == img.shape
        assert out.dtype == np.uint8


def test_four_rotations_return_to_the_start():
    """`k` decodes as rot90(k % 4), so applying k=1 four times is the identity
    -- the property that makes the eight elements a group rather than eight
    arbitrary permutations."""
    img = _square()
    out = img
    for _ in range(4):
        out = dihedral(out, 1)
    assert np.array_equal(out, img)


def test_the_flip_half_is_its_own_inverse():
    img = _square()
    assert np.array_equal(dihedral(dihedral(img, 4), 4), img)


def test_the_result_is_contiguous_so_opencv_and_pil_can_take_it():
    """`np.rot90` and `np.fliplr` return views with permuted or negative
    strides. `augment.ops` hands its input to `cv2` and to
    `PIL.Image.fromarray`, which either copy defensively or reject the buffer
    outright, so the copy belongs here once rather than at each of them."""
    for k in range(DIHEDRAL_N):
        assert dihedral(_square(), k).flags["C_CONTIGUOUS"]


def test_the_output_never_aliases_the_input():
    img = _square()
    out = dihedral(img, 0)
    out[0, 0, 0] ^= 0xFF
    assert img[0, 0, 0] != out[0, 0, 0]


# --------------------------------------------------------------------------
# the property the whole design rests on
# --------------------------------------------------------------------------

def test_no_element_changes_a_single_pixel_value():
    """An index permutation, not a resample. If this ever fails, the module
    has started attenuating the channel it was built to leave alone."""
    img = _square(seed=3)
    want = np.bincount(img.ravel(), minlength=256)
    for k in range(DIHEDRAL_N):
        got = np.bincount(dihedral(img, k).ravel(), minlength=256)
        assert np.array_equal(got, want)


def test_no_element_moves_the_two_proxies_the_confound_work_reads():
    """The measurable version of the test above, in the units that matter.

    `laplacian_var` and `noise_floor` are built from symmetric kernels -- a
    4-neighbour Laplacian and an isotropic Gaussian -- so they are invariant
    under the whole group. These are the two channels
    `docs/low_level_confounds.md` reads, and an arbitrary-angle rotation would
    move both.

    `laplacian_var` is asserted bit-exact. `noise_floor` is allowed one
    float32 ULP: OpenCV's separable Gaussian accumulates in a different order
    on a transposed buffer, so the last bit can differ. The tolerance is
    relative and tiny on purpose -- a real resample would move this by
    percent, not by an ULP."""
    img = _square(seed=5)
    want = proxy_vector(dihedral(img, 0))
    for k in range(1, DIHEDRAL_N):
        got = proxy_vector(dihedral(img, k))
        assert got[1] == want[1], f"laplacian_var moved at k={k}"
        assert got[2] == pytest.approx(want[2], rel=1e-6), (
            f"noise_floor moved at k={k} by more than float32 rounding")


def test_only_the_transposing_half_moves_jpeg_quality_and_only_slightly():
    """The one exception, pinned so it cannot grow unnoticed.

    `proxies._blockiness` anchors on the 8x8 JPEG grid and sums both gradient
    directions, so the four transposing elements (k odd) read a different
    value from the four that do not. The flip alone does not move it. The gap
    is under one quality point out of 100 -- two orders of magnitude below the
    estimator's own documented fallback error of 14-31 points -- and geometry
    runs BEFORE the recipe, so any JPEG the recipe applies is laid down on the
    final orientation anyway."""
    img = _square(seed=5)
    q = [proxy_vector(dihedral(img, k))[0] for k in range(DIHEDRAL_N)]
    # 180 degrees and the pure flip leave the grid alignment alone.
    for k in (2, 4, 6):
        assert q[k] == q[0], f"k={k} should not move jpeg_quality"
    # 90 and 270 do move it, by the same amount, and by very little.
    assert q[1] == q[3] == q[5] == q[7]
    assert 0 < abs(q[1] - q[0]) < 1.0


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------

def test_a_non_square_image_raises_rather_than_transposing_silently():
    """`rot90` by 90 degrees transposes. Every op in `augment.ops` is
    shape-preserving and the bank assumes one shape per image, so a transposed
    view would surface far from here or not at all."""
    tall = np.zeros((64, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="SQUARE"):
        dihedral(tall, 1)


def test_the_non_square_guard_fires_even_for_the_identity():
    """k=0 would happen to work. Accepting it would mean the guard passes or
    fails depending on a random draw, which is the worst of both."""
    with pytest.raises(ValueError, match="SQUARE"):
        dihedral(np.zeros((64, 32, 3), dtype=np.uint8), 0)


@pytest.mark.parametrize("bad", [-1, DIHEDRAL_N, 99])
def test_an_out_of_range_element_raises(bad):
    with pytest.raises(ValueError, match="k must be"):
        dihedral(_square(), bad)


def test_a_non_rgb_array_raises():
    with pytest.raises(ValueError, match="HxWx3"):
        dihedral(np.zeros((64, 64), dtype=np.uint8), 0)


# --------------------------------------------------------------------------
# sampling and the key
# --------------------------------------------------------------------------

def test_sampling_covers_the_whole_group_and_stays_inside_it():
    rng = np.random.default_rng(0)
    draws = [sample_dihedral(rng) for _ in range(4000)]
    assert set(draws) == set(range(DIHEDRAL_N))
    assert min(draws) >= 0 and max(draws) < DIHEDRAL_N


def test_sampling_is_roughly_uniform():
    rng = np.random.default_rng(1)
    counts = np.bincount([sample_dihedral(rng) for _ in range(8000)],
                         minlength=DIHEDRAL_N)
    assert counts.min() > 8000 / DIHEDRAL_N * 0.8


def test_sampling_consumes_exactly_one_draw():
    """Stated so a caller that ever shares a generator can reason about it."""
    a, b = np.random.default_rng(2), np.random.default_rng(2)
    sample_dihedral(a)
    b.integers(0, DIHEDRAL_N)
    assert a.integers(0, 1 << 30) == b.integers(0, 1 << 30)


def test_the_key_is_reproducible_from_seed_row_and_view():
    assert (sample_dihedral(geometric_rng(7, 42, 3))
            == sample_dihedral(geometric_rng(7, 42, 3)))


def test_the_key_separates_seeds_rows_and_views():
    """Each of the three components must matter on its own, or two different
    views share an orientation and the augmentation quietly does less than it
    claims."""
    base = [sample_dihedral(geometric_rng(7, 42, v)) for v in range(64)]
    rows = [sample_dihedral(geometric_rng(7, r, 0)) for r in range(64)]
    seeds = [sample_dihedral(geometric_rng(sd, 42, 0)) for sd in range(64)]
    for draws in (base, rows, seeds):
        assert len(set(draws)) > 1


def test_the_geometric_key_is_a_different_stream_from_the_recipe_key():
    """`extract.py` derives the recipe's sampling and apply generators from
    `[seed, row_id, view_idx]`. If geometry shared that key its draw would be
    correlated with the recipe's first draw for every view in the corpus."""
    shared = np.random.default_rng([7, 42, 3])
    geo = geometric_rng(7, 42, 3)
    assert geo.integers(0, 1 << 30) != shared.integers(0, 1 << 30)


# --------------------------------------------------------------------------
# the contract with canonicalisation
# --------------------------------------------------------------------------

def test_crop_mode_output_is_a_legal_input_to_dihedral():
    """The reason `CanonPolicy.is_square` exists. Crop mode is the only
    standardisation this module is legal under."""
    policy = CanonPolicy(mode=MODE_CROP, crop_side=64, nominal_side=128)
    # A crop policy is not held to band mode's contract: `band_side` defaults
    # to 200, which exceeds this nominal_side and would be illegal in band
    # mode. Crop mode never reads it, so it is not checked and does not reach
    # `as_record`.
    assert "band_side" not in policy.as_record()
    assert policy.is_square
    out = canonicalise(_square(200), policy=policy)
    assert out.shape[0] == out.shape[1]
    dihedral(out, 5)            # must not raise


def test_band_mode_output_is_refused_for_a_non_square_source():
    """Band mode preserves aspect ratio, so pairing it with dihedral is a
    configuration error that must be caught rather than silently transposing
    a third of the corpus."""
    policy = CanonPolicy()
    assert not policy.is_square
    tall = np.random.default_rng(0).integers(
        0, 256, (400, 250, 3), dtype=np.uint8)
    out = canonicalise(tall, policy=policy)
    assert out.shape[0] != out.shape[1]
    with pytest.raises(ValueError, match="SQUARE"):
        dihedral(out, 1)
