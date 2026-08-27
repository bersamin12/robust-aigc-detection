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


def _flat_photo(seed):
    """Near-uniform, low-texture image: a narrow-band, blurred backdrop
    (sky, wall, studio backdrop) rather than raw per-pixel noise. Real
    low-texture regions vary smoothly (lighting, vignetting), which is what
    the blur reproduces; unblurred per-pixel noise degrades to pure
    quantisation noise once resized down for hashing and is not
    representative of anything phash is expected to handle. This is the
    low-AC-energy regime where a DC leak into the hash bits would matter
    most, since the DC/AC magnitude gap is at its narrowest here.

    Measured caveat (see the known-limitation note below): a sigma-10 blur of
    118..139 noise lands at std 0.49, range 127-128 -- effectively a CONSTANT
    image, whose AC coefficients are exactly zero before and after a
    brightness shift. So this fixture is stable for a degenerate reason and
    does not actually exercise the low-AC regime it was written for.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(118, 139, (256, 256, 3), dtype=np.uint8)
    return ops.blur(base, 10.0)


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
    d = _distances(lambda i: ops.jitter(i, 0.2, 0.0, 0.0))
    assert d.max() <= 4

    d_flat = _distances(lambda i: ops.jitter(i, 0.2, 0.0, 0.0), fixture=_flat_photo)
    assert d_flat.max() <= 4


def test_different_images_are_far_apart():
    """Separation, measured over every pair rather than one pair: distinct
    images must sit far above the threshold or the guard would drop training
    images it should keep."""
    hashes = [phash(_photo(seed)) for seed in _SEEDS]
    pairwise = [hamming(a, b)
                for i, a in enumerate(hashes) for b in hashes[i + 1:]]
    # Over all 276 pairs: min 20, median 32 -- five times the threshold.
    assert min(pairwise) > 10


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


# --- Known limitation of the method, recorded here and in the fix report ---
#
# pHash at Hamming <= 4 is a dependable near-duplicate guard only for content
# carrying high-frequency structure. On content with essentially no detail
# above the 32x32 downsample -- a plain sky, a studio backdrop, a smooth
# gradient -- the AC coefficients supplying most of the 64 bits sit at
# quantisation-noise level, so those bits are arbitrary. Measured on smooth
# 4x4-upscaled backdrops over 20 seeds, the distance between an image and a
# perturbed copy of ITSELF is: jpeg q40 median 15, +/-20% brightness median
# 12, and a mere +/-1 GREY LEVEL of noise median 10 -- all far past 4. It is
# not specific to brightness, to JPEG, or to any rounding convention; any
# perturbation at all does it.
#
# Direction of the error: this is a FALSE NEGATIVE mode. Distinct images stay
# far apart (min pairwise distance 14-20 in every measurement), so nothing is
# wrongly dropped from training; the risk is that a structureless demo image
# re-encoded on its way into a training pool is NOT recognised. Most COCO
# val2017 photographs carry ample structure, so the exposure is narrow, but
# it is real and it is worth stating in the error-analysis note.
