"""Dihedral (flip + 90-degree rotation) augmentation, applied per view.

WHY THIS AND NOT ARBITRARY ANGLES
---------------------------------
The obvious reading of "flip + random rotation" is a continuous angle through
`cv2.warpAffine`. That was rejected, and the reason is the same measurement
that shapes everything else in this corpus: **sharpness is the largest
surviving confound** (`laplacian_var` separates the classes at AUC 0.672 after
canonicalisation; `docs/low_level_confounds.md`). An arbitrary-angle rotation
resamples every pixel, which attenuates exactly that channel -- so it would
move the one statistic we are trying to read, in a stream whose whole purpose
is measuring what a different corpus does to it.

The dihedral group of the square costs nothing and moves almost nothing.
`np.rot90` and `np.fliplr` are index permutations: the output holds precisely
the input's pixel values, so the histogram is identical and every isotropic
statistic is too. Measured over all eight elements: `laplacian_var` is
**bit-exactly** unchanged and `noise_floor` agrees to one float32 ULP (both
are built from symmetric kernels -- a 4-neighbour Laplacian and an isotropic
Gaussian; the last-bit difference is OpenCV's separable blur accumulating in a
different order on a transposed buffer, not a real change). Those are the two
channels the confound work reads, and they do not move.

`jpeg_quality` is the one exception and it is worth stating exactly rather
than rounding to "invariant". `proxies._blockiness` anchors on the 8x8 JPEG
grid and sums the two gradient directions, so the four TRANSPOSING elements
(k odd, i.e. 90 and 270 degrees) read a different value from the four that do
not; the flip alone does not move it at all. Measured on random texture the
gap is **0.66 quality points out of 100** -- two orders of magnitude below the
estimator's own documented fallback error of 14-31 points. It is also mostly
moot in the pipeline, because geometry runs BEFORE the recipe, so any JPEG the
recipe applies is laid down on the final orientation and any JPEG history from
the source file has already been destroyed by the crop-and-upscale to 512.

WHY PER VIEW
------------
Stage A caches features once, so an augmentation cannot be resampled per epoch
the way it would be in an ordinary training loop -- whatever a view holds is
what the head sees for every epoch of Stage B. Giving each of the 11 views its
own orientation therefore buys 11 orientations per image at no extra
extraction cost, where a per-image transform would buy exactly one.

The consequence is worth stating rather than discovering later: the A3
consistency loss compares a clean view against a degraded one, and with a
per-view orientation those two views now also differ geometrically. A3 in this
stream is therefore asking for invariance to degradation AND to orientation.
That is a stronger objective, and arguably the right one for a detector -- a
flipped fake is still fake -- but it is NOT the same quantity A3 measures in
the band-mode stream, and the two numbers must not be read against each other.

WHY IT REFUSES A NON-SQUARE INPUT
---------------------------------
`np.rot90` by 90 or 270 degrees transposes the shape. Every op in
`augment.ops` is shape-preserving and the bank's view stack assumes one shape
per image, so a non-square input would either raise somewhere far downstream
or, worse, silently produce a transposed view. Square inputs come from
`canonical.CanonPolicy(mode="crop")`, which is the only standardisation this
module is legal under -- band mode preserves aspect ratio and does not
generally produce squares.
"""
from __future__ import annotations

import numpy as np

#: |D_4|: four rotations, each with and without a horizontal flip.
DIHEDRAL_N: int = 8


def dihedral(img: np.ndarray, k: int) -> np.ndarray:
    """Apply the `k`-th element of the dihedral group to a square image.

    `k` is decoded as `rot90(k % 4)` followed by a horizontal flip when
    `k >= 4`. The order matters only for which permutation each index names,
    not for the set of eight, but it is fixed because `k` is derived from
    `(seed, row_id, view_idx)` and a reordering would silently change every
    cached view's pixels without changing any shape.

    Returns a fresh C-contiguous array. `np.rot90`/`np.fliplr` return VIEWS
    with negative or permuted strides, and the ops downstream hand their
    output to OpenCV and to `PIL.Image.fromarray`, both of which either copy
    defensively or reject a negative-stride buffer. Copying once here is
    cheaper than discovering that at the third call site.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"dihedral expects an HxWx3 RGB array, got shape {img.shape!r}")
    if img.shape[0] != img.shape[1]:
        raise ValueError(
            f"dihedral expects a SQUARE image, got {img.shape[0]}x{img.shape[1]}. "
            "A 90 or 270 degree rotation transposes a non-square image, and "
            "every op in augment.ops is shape-preserving -- the mismatch would "
            "surface far from here or not at all. Square inputs come from "
            "canonical.CanonPolicy(mode='crop'); band mode preserves aspect "
            "ratio and must not be combined with this.")
    if not 0 <= int(k) < DIHEDRAL_N:
        raise ValueError(f"k must be in [0, {DIHEDRAL_N}), got {k!r}")

    k = int(k)
    out = np.rot90(img, k % 4)
    if k >= 4:
        out = np.fliplr(out)
    # `np.array` and not `np.ascontiguousarray`: for k == 0 the latter returns
    # the CALLER'S array unchanged, because `rot90(img, 0)` is `img` and it is
    # already contiguous. One element of eight silently aliasing its input is
    # exactly the kind of thing that shows up as a corrupted view a week later
    # -- `ops.jitter` works in place on its own float copy today, but nothing
    # here should depend on that staying true.
    return np.array(out, dtype=np.uint8, order="C")


def sample_dihedral(rng: np.random.Generator) -> int:
    """Draw one group element uniformly.

    Exactly one draw from `rng`, so a caller that shares a generator with
    something else can reason about how far this advances it. (The wiring in
    `features/extract.py` does not share -- it derives a fresh generator per
    view -- but the property is cheap to keep and expensive to recover.)
    """
    return int(rng.integers(0, DIHEDRAL_N))


def geometric_rng(seed: int, row_id: int, view_idx: int) -> np.random.Generator:
    """The per-view key for geometric augmentation.

    A SEPARATE stream from `canonical.canonical_rng` and from the recipe's
    sampling and apply generators, all of which key on the same
    `(seed, row_id, view_idx)`. The trailing literal is what separates them.

    The discipline is the one `features/extract.py` already documents: each
    step re-derives its own generator from the shared key rather than threading
    one stream through every step, so how many draws any step happens to
    consume can never shift where another step's draws land. That is what lets
    `recon.attach_recon_to_bank` replay a cached view's exact pixels from
    `(seed, row_id, view_idx)` and the stored recipe alone, with the crop
    offset and the orientation index stored nowhere at all.
    """
    return np.random.default_rng([int(seed), int(row_id), int(view_idx), 1])
