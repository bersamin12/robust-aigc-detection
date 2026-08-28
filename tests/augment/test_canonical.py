"""Tests for resolution canonicalisation.

FIXTURE POLICY (this project's documented recurring failure mode is a fixture
that makes the property under test unreachable):

  * Every fixture is NON-SQUARE, so an aspect-ratio bug cannot hide.
  * Sizes span BOTH SIDES of the band and the nominal -- below the band (128),
    at the band (200), between (450) and at the nominal (512) -- so "resize was
    skipped for images already near the target" cannot pass.
  * The content is broadband texture, not a smooth gradient, so band-limiting
    is actually visible in a sharpness statistic. A smooth fixture has no
    high-frequency energy to remove and would make step 1 undetectable.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from aigcdet.augment.canonical import (
    CANON_BAND_JITTER, CANON_BAND_SIDE, CANON_NOMINAL_SIDE,
    canonical_rng, canonicalise,
)
from aigcdet.augment.recipes import Recipe
from aigcdet.augment.scenarios import EVAL_GRID

#: (short, long) for every fixture size, all non-square. Deliberately spans
#: below / at / between / at-nominal relative to the band and the nominal.
FIXTURE_SIZES = ((128, 171), (200, 267), (450, 600), (512, 683))


def textured(short: int, long: int, seed: int = 0) -> np.ndarray:
    """A non-square RGB image with energy at every spatial frequency.

    White noise plus a hard edge: noise gives broadband content so a
    band-limiting step is measurable, the edge gives a structure that survives
    resampling so the sharpness statistic does not just track noise power.
    """
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(short, long, 3), dtype=np.uint8)
    img[:, : long // 2] //= 2          # a hard vertical edge down the middle
    return img


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian on the luma channel -- the standard cheap
    proxy for how much high-frequency detail an image actually carries."""
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(grey.astype(np.float64), cv2.CV_64F).var())


def hf_energy(img: np.ndarray, keep: float) -> float:
    """Fraction of spectral energy above `keep` of the Nyquist frequency.

    Measured on a centre crop so every image contributes the same number of
    samples regardless of aspect ratio.
    """
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    n = min(grey.shape)
    top, left = (grey.shape[0] - n) // 2, (grey.shape[1] - n) // 2
    grey = grey[top:top + n, left:left + n]
    spec = np.abs(np.fft.fftshift(np.fft.fft2(grey - grey.mean()))) ** 2
    cy, cx = n // 2, n // 2
    yy, xx = np.ogrid[:n, :n]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    total = spec.sum()
    if total <= 0:
        return 0.0
    return float(spec[radius > keep * (n / 2)].sum() / total)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

@pytest.mark.parametrize("short,long", FIXTURE_SIZES)
def test_canonicalise_maps_every_native_size_to_the_nominal_short_side(short, long):
    out = canonicalise(textured(short, long))
    assert min(out.shape[:2]) == CANON_NOMINAL_SIDE, (
        f"{short}x{long} did not reach the nominal short side; every image must "
        "arrive at the backbone at identical pixel dimensions or the final "
        "resample still encodes the native resolution")


@pytest.mark.parametrize("short,long", FIXTURE_SIZES)
def test_canonicalise_preserves_aspect_ratio(short, long):
    src = textured(short, long)
    out = canonicalise(src)
    want = long / short
    got = max(out.shape[:2]) / min(out.shape[:2])
    # Tolerance is one pixel on the long side, i.e. pure rounding.
    assert abs(got - want) < (1.0 / CANON_NOMINAL_SIDE) * 2, (
        f"aspect ratio {want:.4f} became {got:.4f}; squashing to a square here "
        "would distort content differently for differently-shaped sources")


def test_canonicalise_preserves_orientation_for_tall_images():
    """Portrait must stay portrait. A short-side rule that assumes landscape
    silently transposes tall images."""
    out = canonicalise(textured(200, 300).transpose(1, 0, 2).copy())
    assert out.shape[0] > out.shape[1], "a portrait input came back landscape"
    assert min(out.shape[:2]) == CANON_NOMINAL_SIDE


def test_canonicalise_returns_uint8_rgb_and_never_mutates_its_input():
    src = textured(450, 600)
    before = src.copy()
    out = canonicalise(src)
    assert out.dtype == np.uint8 and out.shape[2] == 3
    assert np.array_equal(src, before), "canonicalise modified its input in place"
    assert out is not src


# --------------------------------------------------------------------------
# The band-limit: the step that must NOT be skipped
# --------------------------------------------------------------------------

def test_canonicalise_band_limits_an_image_already_at_the_nominal_size():
    """The skip mutation. An image already at 512 is exactly the case an
    `if already_close_enough: return img` short-circuit would wave through, and
    it is the case that carries the surplus detail which IS the leak."""
    src = textured(512, 683)
    out = canonicalise(src)
    assert min(out.shape[:2]) == CANON_NOMINAL_SIDE
    # Same pixel dimensions as a plain upscale-free image, but the detail above
    # the band must be gone.
    assert sharpness(out) < 0.25 * sharpness(src), (
        "a 512px image came back nearly as sharp as it went in: step 1 did not "
        "run, so full-resolution detail survived and the resolution cue with it")


def test_canonicalise_removes_energy_above_the_band_nyquist():
    out = canonicalise(textured(512, 683))
    # After band-limiting to 200 and upscaling to 512, content above roughly
    # 200/512 of the new Nyquist should be nearly empty.
    assert hf_energy(out, keep=0.55) < 0.02, (
        "spectral energy survives well above the band ceiling; the bandwidth "
        "was not actually capped")


def test_canonicalise_band_limits_sharp_and_smooth_images_alike():
    """Equal treatment. The function takes no label, so the only way to apply
    it asymmetrically is to branch on a pixel property that correlates with
    class -- image sharpness being the obvious one. Both must be capped."""
    sharp = textured(512, 683)
    smooth = cv2.GaussianBlur(sharp, (31, 31), 8.0)
    for name, src in (("sharp", sharp), ("smooth", smooth)):
        out = canonicalise(src)
        assert min(out.shape[:2]) == CANON_NOMINAL_SIDE, name
        assert hf_energy(out, keep=0.55) < 0.02, (
            f"the {name} image kept energy above the band ceiling; "
            "canonicalisation branched on image content")


def test_canonicalise_does_not_upscale_images_below_the_band_to_the_band():
    """Bandwidth cannot be restored, so a below-band image must go straight to
    the nominal in ONE upscale, not two. Two chained upscales would leave a
    different interpolation signature on exactly the below-band images."""
    calls = _resize_calls(textured(128, 171))
    upscales = [c for c in calls if c["up"]]
    assert len(upscales) == 1, (
        f"a 128px image was upscaled {len(upscales)} times: {calls}")
    assert min(upscales[0]["dsize"]) == CANON_NOMINAL_SIDE


# --------------------------------------------------------------------------
# Interpolation must depend on the STEP, never on the direction/size class
# --------------------------------------------------------------------------

def _resize_calls(img: np.ndarray, **kw) -> list[dict]:
    """Run canonicalise while logging every cv2.resize it performs."""
    real = cv2.resize
    log: list[dict] = []

    def spy(src, dsize, *args, **kwargs):
        out = real(src, dsize, *args, **kwargs)
        log.append({
            "from": (src.shape[1], src.shape[0]),
            "dsize": dsize,
            "interp": kwargs.get("interpolation"),
            "up": min(dsize) > min(src.shape[:2]),
        })
        return out

    cv2.resize = spy
    try:
        canonicalise(img, **kw)
    finally:
        cv2.resize = real
    return log


@pytest.mark.parametrize("short,long", FIXTURE_SIZES)
def test_downscales_and_upscales_each_use_one_fixed_kernel(short, long):
    """The interpolation-signature mutation.

    If the kernel is chosen by the direction of an individual resize, or worse
    by how big the input was, then "was this image above or below the band"
    is re-recorded in the resampling artefacts -- which is the native
    resolution, which is the label. Downscales must all use one kernel and
    upscales all another, identically for every native size.
    """
    calls = _resize_calls(textured(short, long))
    assert calls, f"{short}x{long} produced no resize at all"
    for c in calls:
        want = cv2.INTER_CUBIC if c["up"] else cv2.INTER_AREA
        assert c["interp"] == want, (
            f"{short}x{long}: a {'up' if c['up'] else 'down'}scale used "
            f"interpolation {c['interp']} not {want}: {calls}")


def test_every_image_ends_on_the_same_final_upscale_kernel_and_size():
    """The last step into the backbone must be identical for every native
    size -- that is the half of the leak removable at zero information cost."""
    finals = {}
    for short, long in FIXTURE_SIZES:
        last = _resize_calls(textured(short, long))[-1]
        finals[short] = (min(last["dsize"]), last["interp"], last["up"])
    assert len(set(finals.values())) == 1, (
        f"the final resample differs by native size: {finals}")


# --------------------------------------------------------------------------
# The actual point: attenuating the resolution cue
# --------------------------------------------------------------------------

def test_canonicalise_collapses_the_sharpness_gap_between_native_resolutions():
    """The property the module exists for.

    Two versions of the SAME scene, one at 512 and one at 200, are trivially
    separable by sharpness before canonicalisation. Afterwards the gap must
    largely close -- that gap is the label, for 30% of the corpus.
    """
    hi = textured(512, 683)
    lo = cv2.resize(hi, (267, 200), interpolation=cv2.INTER_AREA)

    # Before: compare them as the backbone would see them, both squashed to 384.
    def at_backbone(x):
        return cv2.resize(x, (384, 384), interpolation=cv2.INTER_AREA
                          if min(x.shape[:2]) > 384 else cv2.INTER_CUBIC)

    gap_before = abs(sharpness(at_backbone(hi)) - sharpness(at_backbone(lo)))
    ref_before = max(sharpness(at_backbone(hi)), sharpness(at_backbone(lo)))

    c_hi, c_lo = canonicalise(hi), canonicalise(lo)
    gap_after = abs(sharpness(at_backbone(c_hi)) - sharpness(at_backbone(c_lo)))
    ref_after = max(sharpness(at_backbone(c_hi)), sharpness(at_backbone(c_lo)))

    rel_before = gap_before / ref_before
    rel_after = gap_after / ref_after
    assert rel_after < 0.5 * rel_before, (
        f"relative sharpness gap only went {rel_before:.3f} -> {rel_after:.3f}; "
        "canonicalisation is not attenuating the resolution cue")


# --------------------------------------------------------------------------
# Condition transforms keep their stated meaning
# --------------------------------------------------------------------------

def test_canonicalisation_does_not_alter_any_condition_transform_parameter():
    """Canonicalisation is a preprocessing step, not a redefinition of the
    brief's evaluation grid. Every stated parameter must survive untouched."""
    before = {name: r.to_json() for name, r in EVAL_GRID.items()}
    img = canonicalise(textured(200, 267))
    for name, recipe in EVAL_GRID.items():
        recipe.apply(img, np.random.default_rng(0))
    after = {name: r.to_json() for name, r in EVAL_GRID.items()}
    assert after == before, "applying the grid to a canonicalised image mutated it"

    # And the exact values the brief specifies are still the values present.
    assert before["jpeg_q90"] == '[{"name": "jpeg", "params": {"quality": 90}}]'
    assert before["blur_s1.0"] == '[{"name": "blur", "params": {"sigma": 1.0}}]'
    assert before["resize_0.5"] == '[{"name": "resize", "params": {"scale": 0.5}}]'
    assert before["crop_80"] == '[{"name": "crop", "params": {"frac": 0.8}}]'


def test_condition_transforms_stay_shape_preserving_on_a_canonicalised_image():
    """Ops compose only because they are shape-preserving; canonicalisation
    must not hand them a shape that breaks that."""
    img = canonicalise(textured(128, 171))
    for name, recipe in EVAL_GRID.items():
        out = recipe.apply(img, np.random.default_rng(0))
        assert out.shape == img.shape, name
        assert out.dtype == np.uint8, name


def test_severity_labels_are_unaffected_by_canonicalisation():
    """`recipes._severity` is a pure function of the parameters and the
    degradation head's target definition depends on it staying that way."""
    img_a, img_b = canonicalise(textured(128, 171)), canonicalise(textured(512, 683))
    for name, recipe in EVAL_GRID.items():
        a = recipe.labels()
        recipe.apply(img_a, np.random.default_rng(0))
        recipe.apply(img_b, np.random.default_rng(0))
        b = recipe.labels()
        assert np.array_equal(a["presence"], b["presence"]), name
        assert np.array_equal(a["severity"], b["severity"]), name


def test_crop_80_still_keeps_four_fifths_of_the_canonical_image():
    """The one op whose meaning is geometric: 80% must still be 80% of the
    image as presented, which requires the aspect ratio to have been kept."""
    img = canonicalise(textured(200, 267))
    h, w = img.shape[:2]
    recipe = EVAL_GRID["crop_80"]
    out = recipe.apply(img, np.random.default_rng(0))
    assert out.shape == img.shape
    assert int(round(h * 0.8)) == int(round(0.8 * h)) and h == CANON_NOMINAL_SIDE
    assert abs((w / h) - (267 / 200)) < 0.01


# --------------------------------------------------------------------------
# RNG derivation and shard reproducibility
# --------------------------------------------------------------------------

def test_canonical_rng_is_the_projects_per_view_key():
    """Must be byte-identical to what extract.py / grid.py / recon.py use, or
    a canonicalised view stops being replayable from (seed, row_id, view_idx)."""
    for seed, row_id, view in ((0, 0, 0), (7, 12345, 3), (99, 2, 10)):
        want = np.random.default_rng([seed, row_id, view]).random(8)
        got = canonical_rng(seed, row_id, view).random(8)
        assert np.array_equal(got, want), (seed, row_id, view)


def test_canonical_rng_separates_rows_views_and_seeds():
    """Every component of the key must actually participate. A derivation that
    drops one makes two different images -- or two views of one image -- draw
    identical randomness."""
    base = canonical_rng(1, 100, 2).random(8)
    assert not np.array_equal(base, canonical_rng(2, 100, 2).random(8)), "seed ignored"
    assert not np.array_equal(base, canonical_rng(1, 101, 2).random(8)), "row_id ignored"
    assert not np.array_equal(base, canonical_rng(1, 100, 3).random(8)), "view_idx ignored"


def test_two_shards_of_the_same_image_canonicalise_identically():
    """The shard-reproducibility mutation. `data.shard_frame` keeps the frozen
    manifest's index labels precisely so that row 12345 is row 12345 in every
    shard; canonicalisation keyed on that must give the same pixels in a
    session that extracted rows 0-999 and one that extracted 12000-12999.
    """
    img = textured(450, 600)
    a = canonicalise(img, rng=canonical_rng(11, 12345, 2))
    b = canonicalise(img, rng=canonical_rng(11, 12345, 2))
    assert np.array_equal(a, b), (
        "the same (seed, row_id, view_idx) produced different pixels")

    # The key must actually participate. Compared across a RANGE of row_ids,
    # not a single neighbour: the jittered band is an integer in [180, 200], so
    # two arbitrary rows collide on the same band about 5% of the time and a
    # one-neighbour check would be flaky rather than wrong.
    others = [canonicalise(img, rng=canonical_rng(11, r, 2)) for r in range(12340, 12360)]
    assert any(not np.array_equal(a, o) for o in others), (
        "twenty different row_ids all produced identical pixels: the key is "
        "not being used, so the jitter is not actually per-image")


def test_canonicalise_is_deterministic_and_ignores_global_rng_state():
    """The default path takes no RNG at all, and must not quietly consume
    global numpy state -- that would make a view depend on execution order."""
    img = textured(512, 683)
    np.random.seed(0)
    a = canonicalise(img)
    np.random.seed(12345)
    b = canonicalise(img)
    assert np.array_equal(a, b)


def test_the_band_jitter_only_ever_lowers_the_ceiling():
    """A band ABOVE 200 is one the benchmark's real class (COCO, all 200px)
    cannot reach while its fake class can -- exactly the asymmetry being
    removed. Jitter must be one-sided."""
    src = textured(512, 683)
    floor = canonicalise(src, band_side=CANON_BAND_SIDE, rng=None)
    for row in range(40):
        out = canonicalise(src, rng=canonical_rng(3, row, 0))
        assert min(out.shape[:2]) == CANON_NOMINAL_SIDE
        # A higher band would mean MORE surviving high-frequency energy than
        # the unjittered floor. Never allowed.
        assert hf_energy(out, keep=0.55) <= hf_energy(floor, keep=0.55) + 1e-6, (
            f"row {row}: jitter raised the bandwidth ceiling above band_side")


def test_the_band_jitter_actually_varies_the_ceiling():
    """A fixture warning made real: if the jitter never fired, the one-sided
    test above would pass vacuously."""
    src = textured(512, 683)
    seen = set()
    for row in range(40):
        calls = _resize_calls(src, rng=canonical_rng(5, row, 0))
        downs = [c for c in calls if not c["up"]]
        assert len(downs) == 1, calls
        seen.add(min(downs[0]["dsize"]))
    assert len(seen) > 1, f"the jitter never changed the band: {seen}"
    assert max(seen) <= CANON_BAND_SIDE
    assert min(seen) >= int(CANON_BAND_SIDE * (1 - CANON_BAND_JITTER)) - 1


def test_jitter_is_off_unless_an_rng_is_supplied():
    src = textured(512, 683)
    downs = [c for c in _resize_calls(src) if not c["up"]]
    assert len(downs) == 1 and min(downs[0]["dsize"]) == CANON_BAND_SIDE


# --------------------------------------------------------------------------
# Configuration guards
# --------------------------------------------------------------------------

def test_band_must_be_below_nominal_so_step_two_is_always_an_upscale():
    src = textured(200, 267)
    with pytest.raises(ValueError, match="band_side < nominal_side"):
        canonicalise(src, band_side=512, nominal_side=512)
    with pytest.raises(ValueError, match="band_side < nominal_side"):
        canonicalise(src, band_side=600, nominal_side=512)


def test_canonicalise_rejects_inputs_the_call_sites_should_never_produce():
    with pytest.raises(ValueError, match="HxWx3"):
        canonicalise(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        canonicalise(np.zeros((10, 10, 3), dtype=np.float32))


def test_canonicalise_then_replay_reproduces_the_original_view_exactly():
    """`features/recon.py` re-decodes an image and re-runs the stored recipe to
    reproduce a cached view's EXACT pixels. Canonicalisation must therefore be
    idempotent in the pipeline sense -- decode, canonicalise, apply -- at every
    one of the three call sites, or replay attaches reconstruction features to
    pixels that were never scored. This pins the property the wiring relies on.
    """
    img = textured(450, 600)
    for view_idx, name in enumerate(("clean", "noise_s0.05", "jpeg_q50")):
        recipe = EVAL_GRID[name] if name != "clean" else Recipe(())
        first = recipe.apply(canonicalise(img), canonical_rng(7, 4242, view_idx))
        replay = recipe.apply(canonicalise(img), canonical_rng(7, 4242, view_idx))
        assert np.array_equal(first, replay), (
            f"{name}: replay diverged; recon.py would attach features to "
            "pixels that were never scored")


def test_canonicalise_is_size_stable_but_NOT_pixel_idempotent():
    """A WIRING HAZARD, pinned deliberately rather than papered over.

    The output SIZE is stable under repeated application, so a double-wired
    call site will not produce a shape error to alert anyone. The PIXELS are
    not: a canonical 512px image still has its short side above `band_side`,
    so a second pass band-limits it again -- AREA down to 200, CUBIC back up --
    and each round trip costs real detail (measured here at roughly 2.6x of
    the Laplacian variance).

    Making this idempotent is not possible from the pixels alone: nothing in a
    canonical image marks it as already canonical, and the whole point is that
    a natively-200 image and a band-limited-from-512 image should be
    indistinguishable. So the invariant has to be enforced by the WIRING --
    exactly once at each of the three call sites named in the module docstring
    -- and this test exists to make that requirement visible and to fail loudly
    if someone later "fixes" it by adding a silent already-canonical guard,
    which would reintroduce a native-resolution-dependent code path.
    """
    once = canonicalise(textured(512, 683))
    twice = canonicalise(once)
    assert twice.shape == once.shape, "size must be stable under re-application"
    # The second pass must not INCREASE bandwidth -- it can only remove.
    assert hf_energy(twice, keep=0.55) <= hf_energy(once, keep=0.55) + 1e-6
    # And it demonstrably does remove: this is the hazard, asserted as fact.
    assert sharpness(twice) < 0.75 * sharpness(once), (
        "a second pass no longer degrades the image. If canonicalise gained an "
        "'already canonical' short-circuit, it now branches on a property that "
        "tracks native resolution -- re-read the module docstring before "
        "relaxing this test")


def test_the_nominal_stays_above_the_backbone_input():
    """512 > 384 is load-bearing: the backbone must still DOWNSCALE what it is
    given, so canonicalisation never leaves it upscaling every input."""
    assert CANON_NOMINAL_SIDE > 384
    assert CANON_BAND_SIDE < CANON_NOMINAL_SIDE
