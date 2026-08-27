"""Model-free degradation proxies (spec §3.4).

Three numbers computed from pixels alone. They are cheap enough to run inside
predict.py, they validate the learned degradation head (report the Spearman
correlation on validation), and they are its fallback if it underperforms.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

PROXY_NAMES: tuple[str, ...] = ("jpeg_quality", "laplacian_var", "noise_floor")

# Standard JPEG luminance quantisation table at quality 50 (ITU-T T.81 Annex K).
_Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float64)


def _grey(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


_FLAT_EPS = 0.2  # below this total gradient, treat the image as featureless


def _blockiness(img: np.ndarray) -> float:
    """Relative excess of gradient energy on the 8x8 grid vs off it, in [-1, 1].

    JPEG quantisation concentrates discontinuities on block boundaries, but the
    raw on-minus-off *difference* is not monotone in true quality: under heavy
    quantisation of already-low-contrast content both the on- and off-grid
    gradients collapse toward zero *together*, so "no detectable block edge"
    becomes indistinguishable from "never compressed" (see the calibration
    notes by the lookup table below). Normalising by the total gradient —
    (on-off)/(on+off) — removes that scale confound and is empirically
    monotone against true JPEG quality across the offline sweep (this is left
    *unclipped* at 0: off > on is real signal, most often seen at very high
    quality where natural off-grid texture outweighs any residual block edge,
    not noise to floor away — clipping it to 0 was tried and produced a
    point-mass at the origin that made the calibration fit's near-zero region
    unstable across random seeds).

    A totally flat image (on+off below `_FLAT_EPS`) is the one case the ratio
    can't resolve on its own — 0/0 reads as "no signal", which the lookup
    table below would otherwise interpret as maximal quality for the most
    destroyed image the fallback will ever see. `_FLAT_EPS` is set well above
    the on+off noise floor actually observed for near-flat *compressed*
    content (order 0.01-0.1 in the calibration sweep) but far below the
    on+off floor of any content type with real texture even at its best
    quality (order 1.5+ throughout the sweep), so raising it only catches the
    genuinely featureless case, never a legitimately high-quality image with
    modest contrast. Such an image is treated as maximal blockiness (1.0)
    rather than the "no signal" default of 0 — a real photo essentially never
    collapses to bit-exact (or near-bit-exact) flatness short of destructive
    compression, so betting on "destroyed" is the safer of the two wrong
    answers for a proxy that exists to flag when evidence quality is bad.
    """
    # int16 differences are exact (no float64 array is needed to subtract two
    # bytes), and the on-grid columns are a strided slice rather than a
    # boolean mask, which avoids materialising two copies of the array.
    # Verified bit-identical to the float64/boolean-mask form it replaces.
    g = _grey(img).astype(np.int16)
    d = np.abs(np.diff(g, axis=1))
    if d.shape[1] < 16:
        return 0.0
    on_cols = d[:, 7::8]
    on_sum = int(on_cols.sum())
    n_on = on_cols.size
    n_off = d.size - n_on
    on = on_sum / n_on
    off = (int(d.sum()) - on_sum) / n_off
    total = on + off
    if total < _FLAT_EPS:
        return 1.0
    return float((on - off) / total)


# Fallback blockiness -> quality lookup (piecewise-linear via np.interp).
#
# Fit offline with isotonic regression (sklearn.isotonic.IsotonicRegression,
# increasing=False), bin-aggregated (80 equal-width bins over blockiness in
# [-1, 1], each bin's true quality averaged before fitting) to stop a single
# content/quality combination that happens to land on the same blockiness
# value from dominating the fit — an earlier unbinned version spiked at
# blockiness==0 exactly (a large pooled cluster of very-high-quality samples
# that happen to floor there) and dropped sharply for any nearby nonzero
# value, which inverted on nearly half of random high-quality texture seeds
# right at true q=70->90. Binning smooths that into a gradual, seed-stable
# decline instead of a cliff.
#
# Fit on `_blockiness` measured across true JPEG quality 1-98 plus an
# uncompressed (q=100) anchor, on four synthetic content families, 10 random
# seeds each: a near-flat low-contrast image (the augment test suite's own
# `_photo()` construction: blurred Gaussian noise), and three 1/f^alpha
# fractal textures (alpha=2.2 smooth, 1.6 medium, 1.0 high-detail). alpha=1.0
# was chosen over a flatter/whiter spectrum deliberately: alpha<~0.8 behaves
# like near-incompressible noise, which — per this module's own test fixture
# naming (`_photo`'s docstring: "pure noise has no blockiness structure") —
# is already known not to carry a usable blockiness signal; alpha~1.0
# ("pink" noise) is the standard stand-in for natural image texture and does
# carry one.
#
# Isotonic regression guarantees this table is non-increasing in blockiness
# by construction; combined with `_blockiness` being empirically
# non-increasing in true quality for every content family tested — verified
# at quality 5/10/20/30/50/70/90 across 80 random seeds per family with zero
# violations, after the fix above (see
# tests/features/test_proxies.py::test_estimated_jpeg_quality_fallback_is_non_inverting)
# — the composition is non-decreasing in true quality: the fallback does not
# invert on any content family or seed tested.
#
# The residual is real, not tuned away: leave-one-content-family-out MAE
# ranges from ~14 quality points (near-flat, smooth, medium) to ~31
# (high-detail texture, the hardest case to generalise to because it carries
# the most content-dependent variation). Blockiness alone cannot fully
# disambiguate content types from a single no-reference image; only the
# ordering (never a sign reversal) is guaranteed, not the absolute value.
_BLOCKINESS_GRID = np.linspace(-1.0, 1.0, 101)
_QUALITY_LOOKUP = np.array([
    91.71, 91.71, 91.71, 91.71, 91.71, 91.71, 91.71, 91.71, 91.71, 91.71,
    91.71, 91.71, 91.71, 91.71, 91.71, 91.71, 91.71, 91.71, 91.71, 91.71,
    91.71, 91.71, 91.71, 91.71, 91.71, 91.71, 91.71, 91.71, 91.71, 91.71,
    91.71, 91.71, 91.71, 91.71, 91.71, 91.71, 91.71, 91.71, 91.71, 91.71,
    91.71, 91.71, 91.71, 91.71, 91.71, 91.71, 91.71, 91.71, 91.71, 91.71,
    90.10, 85.34, 81.07, 77.95, 75.21, 73.37, 72.21, 70.92, 66.37, 65.93,
    64.70, 60.76, 54.46, 54.46, 54.41, 53.64, 52.93, 52.93, 52.93, 52.86,
    50.88, 48.06, 45.85, 42.18, 37.16, 34.76, 33.86, 32.46, 27.88, 25.34,
    24.45, 23.34, 20.57, 18.83, 18.43, 17.29, 15.68, 14.11, 14.11, 13.31,
    12.36, 11.82, 11.69, 10.11, 9.27, 8.07, 7.11, 6.94, 6.94, 5.26,
    3.71,
])


def estimate_jpeg_quality(img: np.ndarray, path: str | None = None) -> float:
    """Quality in [0, 100]. Exact when `path` is a JPEG, estimated otherwise.

    The pixel-only fallback is monotone in true quality but not calibrated to
    it: leave-one-family-out MAE is roughly 14 quality points on smooth,
    low-detail content and 31 on high-texture content, and it carries no
    usable signal on near-incompressible, noise-like content. Treat its
    output as an ordinal signal, not a calibrated quality figure.
    """
    if path is not None:
        try:
            with Image.open(path) as im:
                tables = getattr(im, "quantization", None)
                if tables:
                    tbl = np.asarray(tables[0], dtype=np.float64).reshape(8, 8)
                    # Invert the standard scaling: S = 5000/Q for Q<50 else 200-2Q
                    scale = float(np.median(tbl / _Q50)) * 100.0
                    q = (5000.0 / scale) if scale > 100.0 else ((200.0 - scale) / 2.0)
                    return float(np.clip(q, 1.0, 100.0))
        except (OSError, ValueError):
            # OSError covers a missing/unreadable file and PIL's
            # UnidentifiedImageError; ValueError covers a quantisation table
            # that is not 64 entries. Anything else is a real bug and must
            # not be turned into a silent fallback.
            pass  # fall through to the pixel-based estimate
    b = _blockiness(img)
    q = np.interp(b, _BLOCKINESS_GRID, _QUALITY_LOOKUP)
    return float(np.clip(q, 1.0, 100.0))


def laplacian_variance(img: np.ndarray) -> float:
    # Left on CV_64F deliberately: a float32 filter was measured at the same
    # cost (1.2 ms either way at 512x768), so there is nothing to buy here
    # and the more precise accumulator is free.
    return float(cv2.Laplacian(_grey(img), cv2.CV_64F).var())


def noise_floor(img: np.ndarray) -> float:
    """Median absolute deviation of a high-pass residual, robust to content."""
    # float32 rather than float64: the two medians dominate this function's
    # cost and the extra precision is discarded by proxy_vector's float32
    # anyway. `resid` is a local temporary, so the medians may partition it
    # in place and the deviation may be taken in place, leaving no full-size
    # temporaries at all. Verified to return the same value either way.
    g = _grey(img).astype(np.float32)
    resid = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    median = float(np.median(resid, overwrite_input=True))
    np.subtract(resid, median, out=resid)
    np.abs(resid, out=resid)
    return float(np.median(resid, overwrite_input=True) * 1.4826)


def proxy_vector(img: np.ndarray, path: str | None = None) -> np.ndarray:
    return np.array([
        estimate_jpeg_quality(img, path),
        laplacian_variance(img),
        noise_floor(img),
    ], dtype=np.float32)
