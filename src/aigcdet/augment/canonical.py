"""Resolution canonicalisation: strip the source image's native resolution
out of the pixels BEFORE any condition transform runs.

WHY THIS EXISTS
---------------
Native short side leaks the label, badly.

In the frozen manifest, 40,982 of 138,116 images (29.7%) sit at a short side
whose bucket is 100% one class: 128 -> 0 real / 1262 fake, 224 -> 0 / 7002,
256 -> 0 / 25728, 450 -> 0 / 6158. Short side alone scores ~72.6% against a
52.9% majority baseline. The organisers' scored benchmark is worse: COCO
(real) is short side 200 for EVERY image, while DALL-E 3 (fake) runs 618-1024
with median 1024. On that benchmark the two classes do not overlap in
resolution at all.

`data.normalize` was supposed to remove this. It does not: it caps the short
side at `SHORT_SIDE` (512) and explicitly never upscales, so every image below
512 keeps its native resolution. In this corpus nothing exceeds 512 and 92,767
images sit below it, so the cap is close to a no-op and the leak passes
straight through.

Absolute pixel dimensions never reach the backbone -- it squashes every input
to a fixed square (384). The leak survives as a RESAMPLING SIGNATURE. A 200px
image reaching a 384px backbone is upscaled 1.92x and arrives soft; a 512px
image is downscaled 0.75x and arrives sharp. That softness is near-perfectly
correlated with the label across a third of the training pool and perfectly
correlated on the benchmark. A model trained on it learns "soft = real" and
has learned nothing whatsoever about generation artefacts.

THE POLICY, AND WHAT IT COSTS
-----------------------------
Two steps, applied identically to every image, authentic and generated alike:

  1. If the short side exceeds `band_side`, downscale to `band_side` with
     INTER_AREA. This is the step that actually removes information.
  2. Resample to `nominal_side` with INTER_CUBIC. This is always an upscale
     (`band_side < nominal_side` is enforced).

The leak has two separable components, and they are not equally removable:

  * The LAST-STEP component -- the direction and magnitude of the final
    resample into the backbone's square. Removable at zero information cost,
    simply by handing the backbone the same pixel dimensions every time.
    Step 2 removes it completely.
  * The INTRINSIC BAND-LIMIT -- the native Nyquist ceiling. A 200px image has
    no detail finer than 200px and never will; upscaling cannot invent it.
    The ONLY way to equalise this across the corpus is to destroy the surplus
    detail in the high-resolution images. Step 1 does exactly that, and it is
    a real, irreversible cost.

`band_side` defaults to 200 because that is the bandwidth floor of the scored
benchmark's real class: every COCO image in it is short side 200. This is the
argument for paying the cost. Detail above 200px does not EXIST for any real
image in the scored set, so any forensic evidence a model learns up there is
evidence it can only ever find on fakes -- that is not detection, it is
resolution reading. You cannot legitimately use a frequency band in which one
class has no support. Band-limiting both classes to a common ceiling forces
whatever signal remains to live below 200px Nyquist, where both classes
genuinely do have support.

The cost is real and should be stated plainly: high-frequency generation
artefacts (GAN upsampling checkerboard, diffusion spectral peaks) live
precisely in the band being discarded, and for the 512px-native images this
throws that band away. The alternatives were weighed and are worse:

  * Downscale everything to a common SMALL side and stop there -- the backbone
    then upscales every input, so every image arrives soft and the surplus
    detail is destroyed without even buying uniform presentation.
  * Upscale everything to a common LARGE side without band-limiting -- costs
    no information and fixes the last-step component, but leaves the intrinsic
    band-limit fully intact, so a sharpness statistic still recovers the native
    resolution, and hence the label, almost perfectly. It attenuates nothing
    where it matters.

There is no option with no cost. This one pays information for a number that
can be believed on the organisers' benchmark.

MEASURED EFFECT (real images from `data/normalized/`, statistics computed on
the 384x384 the backbone actually receives; full method and numbers in the
handover notes):

  Content-controlled, 400 native-512 images each paired against a LANCZOS
  downscale of ITSELF to 200px, so only resolution differs:

                        before            after
    hf-energy   AUC 0.846, acc 78.8%   AUC 0.487, acc 51.9%   (chance 50%)
    var-Laplace AUC 0.821, acc 76.0%   AUC 0.433, acc 56.5%
    hi/lo ratio        3.22x                  0.95x

  The resolution cue is very largely gone: a single-threshold reader of
  high-frequency energy goes from 78.8% to 51.9%, i.e. to chance.

THIS ATTENUATES, IT DOES NOT ELIMINATE. Three residues survive:

  * Images natively BELOW `band_side` (the 128px bucket: 1262 images, 100%
    fake) cannot be raised to the common ceiling, because bandwidth cannot be
    restored. They stay softer than everything else and stay identifiable.
    Equalising them would mean band-limiting the whole corpus to 23px, its
    minimum, which is absurd. This is irreducible.
  * INTER_AREA is a box filter, not an ideal low-pass, so an image that
    REACHED `band_side` by downscaling keeps a slightly thinner spectral tail
    than one that was already there. It shows up as a small consistent bias:
    after canonicalisation the natively-200 arm is sharper than the
    band-limited-from-512 arm in 99.5% of PAIRS, yet the means differ by only
    ~17% against a much larger between-image spread, so a per-image threshold
    recovers just 56.5%. A paired comparison is not available to a classifier;
    a learned feature might still find more than a 1-D statistic does.
  * Only images ABOVE the band receive step 1 at all, so "was there a step-1
    kernel" is itself weakly tied to native resolution. Removing this would
    require band-limiting every image, i.e. a band below the corpus minimum.
    Also irreducible.

A SEPARATE SHORTCUT THIS DOES NOT TOUCH. At a SINGLE native resolution, where
no resolution cue can exist by construction, sharpness already predicts the
label: at 200px (WildFake, both classes) var-Laplacian scores AUC 0.238 /
68.5% BEFORE canonicalisation and 0.238 / 68.7% after. Real images in this
corpus are simply sharper than generated ones at matched resolution. That is a
content/source bias, not a resampling artefact, and canonicalisation neither
causes nor cures it -- but by removing the between-resolution variance that
partly masked it, canonicalisation can make it MORE visible to a 1-D statistic
(at 512px: 69.8% -> 74.0%). Addressing it needs source-balanced sampling or
content-matched pairs, not preprocessing.

WHAT THIS IS NOT
----------------
It is not a condition transform and it does not modify one. The brief's
evaluation transforms (JPEG q90/70/50/30, blur sigma 0.5/1.0/2.0, resize
0.5x/0.25x, noise, jitter, crop 80%) keep their stated parameters exactly;
canonicalisation runs BEFORE them, on the image they are then applied to.
`recipes._severity` is a pure function of those parameters and is untouched.

One consequence must be stated rather than buried: `blur` and `noise` are
defined in PIXEL units, so canonicalisation changes what "sigma = 1.0" means
relative to scene content for any image that was not already at
`nominal_side`. The parameter value is unaltered -- what changes is that
sigma = 1.0 now means the SAME physical thing for every image in the corpus,
which it emphatically did not before, when it meant one blur on a 200px image
and a quite different one on a 512px image. That is a gain in consistency, not
a redefinition, but it does move the numbers.

WIRING: THREE CALL SITES, NOT TWO
---------------------------------
This must run on the training path AND the evaluation path, at the same point
in each -- immediately after decode, immediately before the recipe is applied.
Canonicalising only at training time would build in a train/test mismatch that
costs more than the shortcut it removes. None of the call sites live in this
module, so they are named here explicitly:

  1. `features/extract.py`, `_prepare_image`  -- training bank.
  2. `eval/grid.py`, `extract_eval_bank`      -- evaluation bank.
  3. `features/recon.py`, `attach_recon_to_bank` -- REPLAY. This one is easy
     to miss and it is the dangerous one. It re-decodes the image and re-runs
     the stored recipe to "reproduce each view's exact cached pixels". If 1
     and 2 canonicalise and 3 does not, the replay silently produces
     DIFFERENT pixels from the ones the cached features were computed on, and
     reconstruction features get attached to the wrong images with no error
     raised anywhere. All three, or none.

In each the insertion is one line after the decode:

    base = np.asarray(im.convert("RGB"), dtype=np.uint8)
    base = canonicalise(base)                     # <- here, before r.apply

EXACTLY ONCE PER SITE. Canonicalisation is size-stable but NOT pixel-
idempotent: a canonical 512px image still has its short side above the band, so
a second pass band-limits it again and costs ~2.6x of the Laplacian variance.
Nothing about the output SIZE changes, so a double-wired call site raises no
error and simply produces quietly softer images. It cannot be made idempotent
from the pixels alone -- a canonical image is deliberately indistinguishable
from a natively-200 one, which is the entire objective -- so the invariant is
the wiring's responsibility. Pinned by
`test_canonicalise_is_size_stable_but_NOT_pixel_idempotent`.

DELIBERATELY NOT WIRED: `eval/controls.py`. Its `metadata_features` (width,
height, aspect ratio) and `_thumbnail_features` exist precisely TO measure how
much of the label is recoverable from the shortcut. Canonicalising their input
would blind the instrument that reports whether this module worked.
"""
from __future__ import annotations

import cv2
import numpy as np

#: The common bandwidth ceiling every image is reduced to. 200 is the short
#: side of every COCO (real) image in the organisers' scored benchmark, i.e.
#: the bandwidth floor of the real class -- see the module docstring.
CANON_BAND_SIDE: int = 200

#: The pixel size every canonicalised image is presented at. Must exceed
#: `CANON_BAND_SIDE`. 512 matches `data.normalize.SHORT_SIDE`, so downstream
#: sees the scale it already sees today, and it stays above the 384 backbone
#: input so the backbone still performs a downscale rather than an upscale.
CANON_NOMINAL_SIDE: int = 512

#: Optional one-sided jitter on `band_side` when an RNG is supplied, as a
#: fraction: the band is drawn from [(1 - j) * band_side, band_side].
#:
#: DOWNWARD ONLY, and that is not an aesthetic choice. A band above 200 is a
#: band the benchmark's real class cannot reach (COCO is 200 and no more)
#: while its fake class easily can, which reintroduces precisely the
#: asymmetry this module exists to remove. The jitter may only ever lower the
#: ceiling, never raise it.
CANON_BAND_JITTER: float = 0.10

#: Kernels are a property of the STEP, never of the direction of the resize.
#: Step 1 only ever downscales and step 2 only ever upscales, so no image is
#: ever routed through a different kernel because of how big it started --
#: which would re-record the native resolution as an interpolation signature
#: and undo the point of the exercise.
_DOWNSCALE_INTERP = cv2.INTER_AREA
_UPSCALE_INTERP = cv2.INTER_CUBIC


def canonical_rng(seed: int, row_id: int, view_idx: int) -> np.random.Generator:
    """The project's per-view key, and nothing new.

    `features/extract.py`, `eval/grid.py` and `features/recon.py` all derive a
    view's randomness as `np.random.default_rng([seed, row_id, view_idx])`,
    where `row_id` is the row's INDEX LABEL in the frozen manifest (never a
    positional index -- `data.shard_frame` deliberately does not reset the
    index, because a reset would restart every shard's key space at 0 and make
    two shards of the same image disagree on its pixels). Canonicalisation
    reuses that key rather than inventing a second one, so a canonicalised
    view stays reproducible from `(seed, row_id, view_idx)` alone.
    """
    return np.random.default_rng([int(seed), int(row_id), int(view_idx)])


def _resize_short_side(img: np.ndarray, target: int, interp: int) -> np.ndarray:
    """Resize so the SHORT side is exactly `target`, preserving aspect ratio.

    The short side is set exactly rather than scaled-and-rounded: a scale
    factor applied to both axes can round the short side to target +/- 1 on
    extreme aspect ratios, and "every image arrives at the same size" is the
    entire contract.
    """
    h, w = img.shape[:2]
    if h <= w:
        nh, nw = target, max(1, int(round(w * target / h)))
    else:
        nw, nh = target, max(1, int(round(h * target / w)))
    if (nh, nw) == (h, w):
        return img
    return cv2.resize(img, (nw, nh), interpolation=interp)


def canonicalise(img: np.ndarray, *,
                 band_side: int = CANON_BAND_SIDE,
                 nominal_side: int = CANON_NOMINAL_SIDE,
                 rng: np.random.Generator | None = None,
                 jitter: float = CANON_BAND_JITTER) -> np.ndarray:
    """Map any image to `nominal_side` through a common bandwidth ceiling.

    Takes no label, no source and no path: the transform is a pure function of
    the pixels and its configuration, so it cannot be applied asymmetrically to
    the two classes even by accident.

    `rng` defaults to None, which is the deterministic policy and the one BOTH
    the training and evaluation paths must use unless they are changed
    together. Supplying an RNG jitters the bandwidth ceiling downward (see
    `CANON_BAND_JITTER`), which discourages the head from memorising one exact
    resampling signature; derive it with `canonical_rng` so the result stays
    reproducible from `(seed, row_id, view_idx)`.

    Returns a fresh array; the input is never modified in place.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"canonicalise expects an HxWx3 RGB array, got shape {img.shape!r}; "
            "all three call sites decode with Image.convert('RGB')")
    if img.dtype != np.uint8:
        raise ValueError(
            f"canonicalise expects uint8, got {img.dtype!r}; every op in "
            "`augment.ops` is uint8 in / uint8 out and this runs before them")
    if not 0 < band_side < nominal_side:
        raise ValueError(
            f"need 0 < band_side < nominal_side, got band_side={band_side!r} "
            f"nominal_side={nominal_side!r}; step 2 must always be an upscale, "
            "otherwise the kernel used depends on the input's size and the "
            "native resolution is re-recorded as an interpolation signature")
    if not 0.0 <= jitter < 1.0:
        raise ValueError(f"jitter must be in [0, 1), got {jitter!r}")

    band = band_side
    if rng is not None and jitter > 0.0:
        # One-sided, downward only -- never above `band_side`. See the constant.
        band = int(round(band_side * float(rng.uniform(1.0 - jitter, 1.0))))
        band = max(1, min(band, band_side))

    out = img
    # Step 1: impose the common bandwidth ceiling. Images already at or below
    # it are left alone -- bandwidth cannot be restored, so raising them here
    # would only add a second interpolation to the ones that can least afford
    # it. There is deliberately NO early return for images "close enough" to
    # the nominal size: an image already at 512 must still be band-limited, or
    # it keeps the full-resolution detail that is the leak.
    if min(out.shape[:2]) > band:
        out = _resize_short_side(out, band, _DOWNSCALE_INTERP)
    # Step 2: present at the common size. Always an upscale, always the same
    # kernel, for every image in the corpus.
    out = _resize_short_side(out, nominal_side, _UPSCALE_INTERP)
    # Always a fresh array, never a view of the caller's: after step 1 the
    # short side is min(native, band) <= band < nominal_side, so step 2 always
    # performs a real resize and `cv2.resize` always allocates. (An earlier
    # revision carried an `img.copy()` fallback for the case where both steps
    # no-op; the `band_side < nominal_side` guard above makes that case
    # unreachable, and mutation testing found the branch was dead.)
    return out
