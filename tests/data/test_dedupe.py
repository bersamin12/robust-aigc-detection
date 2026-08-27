"""Leakage-guard tests.

The near-duplicate properties below are asserted across a RANGE of seeds, not
on one draw. The previous version asserted them on a single seed against a
fixture (`ops.blur` of uniform noise) that had no structure for JPEG to
preserve: at q40 that fixture's distance distribution over 40 seeds was
min 0 / median 4 / max 8 with 6 seeds FAILING outright, and the seed the test
used landed on exactly 4 -- passing with zero margin, recording a coincidence
rather than verifying a property. Any change moving a pixel by one level
flipped it.

See `_photo` for what replaced it, and the "known limitation" note below for
the content class on which this guard genuinely does not hold.
"""
import numpy as np
from PIL import Image

from aigcdet.augment import ops
from aigcdet.data.dedupe import build_hash_index, find_leaks, hamming, phash

#: Every distance property is measured over this many independent images.
_SEEDS = tuple(range(24))


def _photo(seed, size=256):
    """A stand-in for a PHOTOGRAPH: smooth low-frequency content plus a few
    hard edges.

    Built from PIL and numpy alone -- deliberately NOT from `ops.blur`, so a
    change to the augmentation ops cannot silently move the fixture these
    tests measure against, which is how the previous version came to depend
    on one lucky seed.

    A photograph is not noise. Bicubic upscaling an 8x8 field gives the
    smooth gradients real optics produce, and the rectangles give the hard
    edges real scenes contain. Both survive quantisation, so the DCT
    coefficients this hash reads are real signal rather than rounding noise
    -- which is precisely what the old blurred-noise fixture lacked.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.integers(30, 226, (8, 8, 3), dtype=np.uint8)
    img = np.asarray(Image.fromarray(coarse).resize((size, size), Image.BICUBIC),
                     dtype=np.uint8).copy()
    for _ in range(4):
        h = int(rng.integers(size // 8, size // 3))
        w = int(rng.integers(size // 8, size // 3))
        top = int(rng.integers(0, size - h))
        left = int(rng.integers(0, size - w))
        img[top:top + h, left:left + w] = rng.integers(0, 256, 3, dtype=np.uint8)
    return img


def _smooth_backdrop(seed, size=256):
    """Low-contrast but genuinely NOT constant: sky, wall, studio backdrop.
    Measured std 5.4, range 116-141, 26 distinct levels.

    This replaces a fixture (`ops.blur(rng.integers(118, 139, ...), 10.0)`)
    that measured std 0.49 over range 127-128 -- effectively a CONSTANT
    image. Its DCT AC coefficients were exactly zero on both sides of a
    brightness shift, so it was stable for a degenerate reason and never
    exercised the low-AC regime it was written for. Its test asserted
    robustness the method does not have on this content class; see
    `test_smooth_low_structure_content_is_not_matched_after_perturbation`
    for what is actually true.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.integers(118, 139, (4, 4, 3), dtype=np.uint8)
    return np.asarray(Image.fromarray(coarse).resize((size, size), Image.BICUBIC),
                      dtype=np.uint8)


def _distances(degrade, fixture=_photo, seeds=_SEEDS):
    """Hamming distance between each fixture image and its degraded self."""
    out = []
    for seed in seeds:
        img = fixture(seed)
        out.append(hamming(phash(img), phash(degrade(img))))
    return np.array(out)


def test_identical_images_hash_identically():
    img = _photo(0)
    assert phash(img) == phash(img.copy())


def test_recompressed_image_stays_within_threshold():
    """Over the 24 seeds asserted here: 17 at distance 0, 7 at 2, none above
    -- against the spec-mandated threshold of 4, so there is real margin
    rather than a boundary hit. (Offline over 40 seeds: same max of 2.)
    The median assertion catches a whole distribution drifting upward; the
    max assertion is the spec property."""
    d = _distances(lambda i: ops.jpeg(i, 40))
    assert d.max() <= 4
    assert np.median(d) <= 2


def test_recompressed_image_stays_within_threshold_at_the_harshest_eval_quality():
    """q30 is the harshest quality on the evaluation grid, so it bounds the
    recompression the guard has to survive. Over the 24 seeds asserted here:
    15 at 0, 9 at 2, max 2."""
    d = _distances(lambda i: ops.jpeg(i, 30))
    assert d.max() <= 4
    assert np.median(d) <= 2


def test_resized_image_stays_within_threshold():
    """Over the 24 seeds asserted here, at both evaluation scales: max 2
    (0.5 -> 22 at 0 and 2 at 2; 0.25 -> 21 at 0 and 3 at 2)."""
    for scale in (0.5, 0.25):
        d = _distances(lambda i: ops.resize_roundtrip(i, scale))
        assert d.max() <= 4, scale
        assert np.median(d) <= 2, scale


def test_brightness_shift_stays_within_threshold():
    # Colour jitter (+/-20% brightness) is one of the six degradation
    # families the whole detector is evaluated against; a demo image that
    # reached a training pool via an auto-enhance filter is exactly the
    # leakage case this guard must still catch.
    #
    # This is the TIGHTEST of the degradations: over the 24 seeds asserted
    # here the distribution is 8 at 0, 11 at 2 and 5 at 4 -- no failures, but
    # it reaches the threshold (40 seeds offline: 16/17/7, same max). The
    # cause is highlight clipping, which is NOT a DC shift: at +20%, ~5.8% of
    # pixels saturate at 255, and that genuinely destroys AC content.
    # Restricting the fixture to shades that cannot clip drops the maximum to
    # 2, which identifies clipping as the sole mechanism. No median assertion
    # here: unlike the other degradations this one legitimately sits at the
    # threshold, so only the spec property is pinned.
    #
    # Asserted on textured content only. The low-contrast backdrop is NOT
    # covered here: the guard genuinely does not hold on it, which
    # test_smooth_low_structure_content_is_not_matched_after_perturbation
    # pins as the measured limitation it is.
    d = _distances(lambda i: ops.jitter(i, 0.2, 0.0, 0.0))
    assert d.max() <= 4


def test_different_images_are_far_apart():
    """Separation, measured over every pair rather than one pair: distinct
    images must sit far above the threshold or the guard would drop training
    images it should keep."""
    hashes = [phash(_photo(seed)) for seed in _SEEDS]
    pairwise = [hamming(a, b)
                for i, a in enumerate(hashes) for b in hashes[i + 1:]]
    # Over all 276 pairs: min 20, median 32 -- five times the threshold.
    assert min(pairwise) > 10


def test_smooth_low_structure_content_is_not_matched_after_perturbation():
    """The guard's known limitation, pinned rather than left to be
    rediscovered.

    pHash at Hamming <= 4 is dependable only on content carrying
    high-frequency structure. On a smooth backdrop the AC coefficients
    supplying most of the 64 bits sit at quantisation-noise level, so those
    bits are arbitrary and ANY perturbation scatters them. Measured over the
    24 seeds asserted here, distance between a backdrop and a perturbed copy
    of ITSELF:

        +/-1 grey level   min  4, median 12, max 16   (1 of 24 within 4)
        brightness +20%   min 10, median 16, max 24   (0 of 24)
        JPEG q40          min 12, median 20, max 26   (0 of 24)
        resize 0.5        min  6, median  8, max 16   (0 of 24)

    A single grey level is enough. This is not specific to brightness, to
    JPEG, or to any rounding convention.

    If this test ever fails because the distances came DOWN, that is good
    news and the limitation note in the fix report should be revised -- do
    not simply loosen it. The threshold of 4 is spec-mandated (§4.1) and is
    not the thing to change here.
    """
    d_jpeg = _distances(lambda i: ops.jpeg(i, 40), fixture=_smooth_backdrop)
    d_bright = _distances(lambda i: ops.jitter(i, 0.2, 0.0, 0.0),
                          fixture=_smooth_backdrop)
    # Not one seed is recognised as a near-duplicate of itself.
    assert d_jpeg.min() > 4
    assert d_bright.min() > 4

    # Even the smallest possible perturbation -- one grey level -- typically
    # moves the hash further than the threshold allows.
    def _one_grey_level(img, seed):
        rng = np.random.default_rng(10_000 + seed)
        return np.clip(img.astype(np.int16) + rng.integers(-1, 2, img.shape),
                       0, 255).astype(np.uint8)

    d_one = np.array([
        hamming(phash(_smooth_backdrop(seed)),
                phash(_one_grey_level(_smooth_backdrop(seed), seed)))
        for seed in _SEEDS
    ])
    assert np.median(d_one) >= 8

    # The failure direction is FALSE NEGATIVE, not false positive: distinct
    # backdrops stay far apart, so nothing is wrongly dropped from training.
    # Measured min 18, median 32.
    hashes = [phash(_smooth_backdrop(seed)) for seed in _SEEDS]
    pairwise = [hamming(a, b)
                for i, a in enumerate(hashes) for b in hashes[i + 1:]]
    assert min(pairwise) > 10


def test_every_hash_bit_position_is_informative():
    """R17's regression pin, in the form that actually reproduces.

    R17 excluded the DC term from the packed bits. Its brightness test could
    never demonstrate the defect -- DC dwarfs the AC median, so the DC bit is
    set for every realistic image and a brightness shift cannot flip it. That
    is exactly WHY it was a defect: with DC packed in, one of the 64 bits is
    constant, so the hash carries 63 informative bits while every docstring
    calls it 64.

    Constant-ness is directly observable, and it discriminates cleanly.
    Measured over these 64 images: the current DC-excluded hash uses 64/64
    bit positions; the pre-R17 form that packed `d[:8, :8]` leaves bit 63
    (the DC bit, first in packing order) constant, 63/64. Verified against a
    local reimplementation of the pre-R17 form -- dedupe.py is untouched.
    """
    hashes = [phash(_photo(seed)) for seed in range(64)]
    bits = np.array([[(h >> i) & 1 for i in range(64)] for h in hashes])
    constant = np.where(bits.min(axis=0) == bits.max(axis=0))[0].tolist()
    assert constant == [], (
        f"bit position(s) {constant} never vary, so the hash carries fewer "
        f"than 64 informative bits")
    assert max(h.bit_length() for h in hashes) == 64


def test_find_leaks_flags_a_recompressed_duplicate(tmp_path):
    img = _photo(4)
    demo_p = tmp_path / "demo.png"
    cand_p = tmp_path / "cand.png"
    Image.fromarray(img).save(demo_p)
    Image.fromarray(ops.jpeg(img, 50)).save(cand_p)
    other_p = tmp_path / "other.png"
    Image.fromarray(_photo(77)).save(other_p)

    demo = build_hash_index([str(demo_p)])
    cand = build_hash_index([str(cand_p), str(other_p)])
    leaks = find_leaks(cand, demo, max_distance=4)
    assert str(cand_p) in leaks
    assert str(other_p) not in leaks


# --- Known limitation of the method, pinned by
# --- test_smooth_low_structure_content_is_not_matched_after_perturbation
# --- and written up for Plan 4's error-analysis note ---
#
# WHAT IT AFFECTS. pHash at Hamming <= 4 is a dependable near-duplicate guard
# only for content carrying high-frequency structure. On content with
# essentially no detail above the 32x32 downsample -- a plain sky, a studio
# backdrop, a smooth gradient -- the AC coefficients supplying most of the 64
# bits sit at quantisation-noise level, so those bits are arbitrary.
#
# MEASURED. Distance between a smooth backdrop (std 5.4) and a perturbed copy
# of itself, over 24 seeds: +/-1 grey level median 12, brightness +20% median
# 16, JPEG q40 median 20, resize 0.5 median 8 -- against a threshold of 4. A
# single grey level is enough; it is not specific to any one degradation.
#
# DIRECTION. False NEGATIVE. Distinct images stay far apart (min pairwise
# distance 18-20 in every measurement), so no training image is wrongly
# dropped. The risk is that a structureless demo image, re-encoded on its way
# into a training pool, is not recognised as a leak.
#
# WHAT BOUNDS THE RISK. Two things, both structural rather than statistical.
# First, since the C1 fix, COCO val2017 and DALL-E Advanced are excluded from
# training wholesale by the source registry (aigcdet.data.sources), not by
# hash -- both halves of the demo set carry exclude_from_training, asserted by
# test_sources.py::test_both_halves_of_the_demo_benchmark_are_excluded_from_training
# -- so this guard is the SECONDARY net, catching a demo image that arrives
# through some other source's pool, not the primary barrier. Second,
# both authentic photographs and generator outputs carry structure; a frame
# filled edge to edge with smooth backdrop is rare. So this is a real gap in
# a secondary net, not a hole in the main barrier.
