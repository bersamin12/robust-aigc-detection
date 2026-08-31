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

from dataclasses import dataclass

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


#: Side of the random window `mode="crop"` takes. 200 for the same reason
#: `CANON_BAND_SIDE` is 200 -- it is the short side of every COCO image in the
#: organisers' scored benchmark, so it is the most information either class is
#: guaranteed to have -- and because `min_short_side: 200` in a corpus preset
#: is what makes the window exist for every row.
CANON_CROP_SIDE: int = 200

#: Band-limit: downscale to a common ceiling, then upscale. Equalises the
#: bandwidth of the whole corpus at the cost of destroying detail above the
#: ceiling. This is the frozen stream's policy and the default.
MODE_BAND = "band"

#: Crop: take a square window at NATIVE resolution, then upscale. Equalises
#: the number of genuine pixels each image contributes without box-filtering
#: any of them away, so a generator's high-frequency signature survives inside
#: the window. Costs field of view instead: the window is a whole frame for a
#: 200px image and a detail for a 640px photograph, which trades a spectral
#: confound for a content one. See docs/dataset_presets.md.
MODE_CROP = "crop"

MODES = (MODE_BAND, MODE_CROP)


@dataclass(frozen=True)
class CanonPolicy:
    """How one stream standardises resolution, as one value that travels.

    A policy rather than a set of keyword arguments because it has to reach
    five production decode sites (`features/extract`, `eval/grid`,
    `features/recon`, `infer`, `explain/patch_heatmap`) and be written into
    the feature bank's config. That second part is what makes it safe:
    `BankWriter` treats every unrecognised config key as must-match, so
    recording the policy buys three refusals for free -- resuming a bank under
    a changed policy, merging shards built under different ones, and fusing
    two banks at A5 whose pixels were never comparable. Without it a crop bank
    and a band bank are indistinguishable on disk, and the failure is silent.
    """

    mode: str = MODE_BAND
    band_side: int = CANON_BAND_SIDE
    nominal_side: int = CANON_NOMINAL_SIDE
    crop_side: int = CANON_CROP_SIDE
    jitter: float = CANON_BAND_JITTER
    #: When the image is smaller than `crop_side`, take the largest square it
    #: DOES contain and let the upscale to `nominal_side` cover the rest,
    #: instead of raising.
    #:
    #: Off by default, and the default is the conservative one: with it on, how
    #: much an image is resampled becomes a function of its native resolution,
    #: and native resolution is not independent of the label. Measured on the
    #: plan manifest at crop_side=224, the upscale factor ALONE separates the
    #: classes at AUC 0.5430 -- against this corpus's own references of 0.5081
    #: for crop and 0.6105 for band. Small, real, and exactly the kind of
    #: low-level shortcut `npr_feature` was written to detect.
    #:
    #: Turning it on is therefore a measurement, not a convenience: gate it on
    #: `scripts/content_blind_probe.py` and report what that probe said.
    crop_clamp: bool = False

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(
                f"unknown canonicalisation mode {self.mode!r}; expected one of "
                f"{list(MODES)}")
        # Each mode is held to ITS OWN contract and not to the other's. A
        # crop policy carries a `band_side` it never reads, and forcing that
        # unused number to be legal against a small `nominal_side` would
        # reject perfectly good crop configurations. `as_record` drops the
        # unused fields for the same reason, and that one matters more: the
        # record is a must-match key on resume, merge and fusion, so an
        # ignored field left in it would fail comparisons over a number
        # neither bank's pixels depend on.
        if self.mode == MODE_BAND:
            if not 0 < self.band_side < self.nominal_side:
                raise ValueError(
                    f"need 0 < band_side < nominal_side, got band_side="
                    f"{self.band_side!r} nominal_side={self.nominal_side!r}; step 2 "
                    "must always be an upscale, otherwise the kernel used depends "
                    "on the input's size and the native resolution is re-recorded "
                    "as an interpolation signature")
            if not 0.0 <= self.jitter < 1.0:
                raise ValueError(f"jitter must be in [0, 1), got {self.jitter!r}")
        else:
            if self.crop_clamp and not 0 < self.crop_side <= self.nominal_side:
                raise ValueError(
                    f"need 0 < crop_side <= nominal_side when crop_clamp is "
                    f"set, got crop_side={self.crop_side!r} "
                    f"nominal_side={self.nominal_side!r}")
            if not self.crop_clamp and not 0 < self.crop_side < self.nominal_side:
                raise ValueError(
                    f"need 0 < crop_side < nominal_side, got crop_side="
                    f"{self.crop_side!r} nominal_side={self.nominal_side!r}; the "
                    "same argument as band_side -- the presentation resize must be "
                    "an upscale for every image alike")

    @property
    def is_square(self) -> bool:
        """Whether this policy's output is square for every input.

        `augment.geometric.dihedral` is legal only when it is: a 90-degree
        rotation transposes a non-square image and every op downstream is
        shape-preserving.
        """
        return self.mode == MODE_CROP

    def as_record(self) -> dict:
        """The blob written into the feature bank's config.

        Only the fields this mode actually reads. The record is compared
        key-by-key on resume, on merge and at A5 fusion, so carrying an
        ignored field would refuse two banks whose pixels are identical.
        """
        rec = {"mode": self.mode, "nominal_side": self.nominal_side}
        if self.mode == MODE_BAND:
            rec.update(band_side=self.band_side, jitter=self.jitter)
        else:
            rec.update(crop_side=self.crop_side)
            # Only recorded when set, so every bank written before this field
            # existed still compares equal to one written with it off.
            if self.crop_clamp:
                rec.update(crop_clamp=True)
        return rec

    @classmethod
    def from_record(cls, rec: dict) -> "CanonPolicy":
        """Rebuild a policy from a bank config or an exported bundle.

        A bank written before this field existed has no record at all; the
        caller passes `DEFAULT_POLICY` in that case rather than this being
        lenient about a missing mode, because "the key was absent" and "the
        key said band" must stay distinguishable.
        """
        return cls(**rec)


#: The frozen stream's policy. Every call site defaults to it, so nothing
#: changes for anything already built.
DEFAULT_POLICY = CanonPolicy()


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


def _random_square_crop(img: np.ndarray, side: int,
                        rng: np.random.Generator | None,
                        clamp: bool = False) -> np.ndarray:
    """Take a `side x side` window at NATIVE resolution.

    `rng is None` gives the CENTRE window. That is not a fallback, it is the
    policy for every deterministic path: inference must return the same score
    for the same file twice, and any replay site that re-derives pixels needs
    the draw to be reproducible or it is not a replay. Randomness is opt-in,
    supplied by `canonical_rng(seed, row_id, view_idx)` at the two extraction
    sites and nowhere else.

    Consumes exactly two draws when random, top then left, so the sequence is
    stable if a caller ever shares the generator.

    Returns a VIEW; the caller's resize allocates. `crop_side < nominal_side`
    is guaranteed by `CanonPolicy`, so that resize always happens and the
    result is never a view of the caller's array.
    """
    h, w = img.shape[:2]
    if clamp:
        # The largest square this image contains. The caller's resize to
        # nominal_side then upscales further than it would for a big image,
        # which is the whole cost of this option and is documented on
        # `CanonPolicy.crop_clamp`.
        side = min(side, h, w)
    if min(h, w) < side:
        raise ValueError(
            f"cannot take a {side}x{side} window from a {h}x{w} image. Crop "
            "standardisation has no way to invent the missing pixels, and "
            "upscaling to reach the window would reintroduce exactly the "
            "resampling signature this module removes. Set the corpus "
            f"preset's `min_short_side` to {side} so build_dataset drops these "
            "rows before they are ever normalised.")
    if rng is None:
        top, left = (h - side) // 2, (w - side) // 2
    else:
        top = int(rng.integers(0, h - side + 1))
        left = int(rng.integers(0, w - side + 1))
    return img[top:top + side, left:left + side]


def canonicalise(img: np.ndarray, *,
                 policy: CanonPolicy | None = None,
                 band_side: int = CANON_BAND_SIDE,
                 nominal_side: int = CANON_NOMINAL_SIDE,
                 rng: np.random.Generator | None = None,
                 jitter: float = CANON_BAND_JITTER) -> np.ndarray:
    """Map any image to `nominal_side` by one of two standardisation policies.

    Takes no label, no source and no path: the transform is a pure function of
    the pixels and its configuration, so it cannot be applied asymmetrically to
    the two classes even by accident.

    `rng` defaults to None, which is the deterministic policy and the one BOTH
    the training and evaluation paths must use unless they are changed
    together. Supplying an RNG jitters the bandwidth ceiling downward (see
    `CANON_BAND_JITTER`), which discourages the head from memorising one exact
    resampling signature; derive it with `canonical_rng` so the result stays
    reproducible from `(seed, row_id, view_idx)`.

    `policy` selects between the two (see `CanonPolicy`). The loose keyword
    arguments are the original signature, kept working so that no existing
    call site or test had to change; they build a band-mode policy. Passing a
    `policy` AND a loose argument raises rather than picking a winner -- two
    sources of truth for the same number is how a bank ends up describing
    pixels it does not contain.

    Returns a fresh array; the input is never modified in place.
    """
    loose = [n for n, v, d in (("band_side", band_side, CANON_BAND_SIDE),
                               ("nominal_side", nominal_side, CANON_NOMINAL_SIDE),
                               ("jitter", jitter, CANON_BAND_JITTER)) if v != d]
    if policy is not None and loose:
        raise ValueError(
            f"canonicalise got both policy={policy!r} and {loose}; put every "
            "setting in the policy, which is the object that reaches the bank "
            "config and can therefore be checked on resume, merge and fusion.")
    if policy is None:
        # Validation lives in CanonPolicy.__post_init__, so the loose path and
        # the policy path cannot drift into disagreeing about what is legal.
        policy = CanonPolicy(mode=MODE_BAND, band_side=band_side,
                             nominal_side=nominal_side, jitter=jitter)
    band_side, nominal_side = policy.band_side, policy.nominal_side
    jitter = policy.jitter

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"canonicalise expects an HxWx3 RGB array, got shape {img.shape!r}; "
            "all three call sites decode with Image.convert('RGB')")
    if img.dtype != np.uint8:
        raise ValueError(
            f"canonicalise expects uint8, got {img.dtype!r}; every op in "
            "`augment.ops` is uint8 in / uint8 out and this runs before them")
    # The band_side / nominal_side / jitter guards that used to live here now
    # live in `CanonPolicy.__post_init__`, which both entry paths go through:
    # the loose keyword arguments build a policy above, so the checks fire for
    # them too, with the same messages. Keeping a copy here would also have
    # applied band mode's contract to crop mode, where `band_side` is never
    # read -- which is what a policy is FOR.

    if policy.mode == MODE_CROP:
        # Step 1', the crop, replaces step 1, the band-limit. It removes no
        # detail from the pixels it keeps: a generator's high-frequency
        # signature survives inside the window instead of being box-filtered
        # away, which is the whole reason this mode exists. What it equalises
        # is the NUMBER of genuine pixels every image contributes, not their
        # bandwidth -- so a 200px image gives its whole frame and a 640px
        # photograph gives a detail, and semantic scale becomes correlated
        # with native resolution where spectral content no longer is. That
        # trade is recorded in docs/dataset_presets.md; it is not free, it is
        # different.
        #
        # Step 2 is unchanged and shared with band mode: one kernel, one size,
        # every image alike. `crop_side < nominal_side` makes it an upscale
        # for every input, so no image is routed through a different kernel
        # because of how big it started.
        window = _random_square_crop(img, policy.crop_side, rng,
                                     clamp=policy.crop_clamp)
        return _resize_short_side(window, nominal_side, _UPSCALE_INTERP)

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
