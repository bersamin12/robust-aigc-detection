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


def _blockiness(img: np.ndarray) -> float:
    """Mean gradient energy on the 8x8 grid minus that off it.

    JPEG quantisation concentrates discontinuities on block boundaries, so this
    difference grows as quality falls.
    """
    g = _grey(img).astype(np.float64)
    d = np.abs(np.diff(g, axis=1))
    if d.shape[1] < 16:
        return 0.0
    cols = np.arange(d.shape[1])
    on = d[:, cols % 8 == 7].mean()
    off = d[:, cols % 8 != 7].mean()
    return float(on - off)


def estimate_jpeg_quality(img: np.ndarray, path: str | None = None) -> float:
    """Quality in [0, 100]. Exact when `path` is a JPEG, estimated otherwise."""
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
        except Exception:
            pass  # fall through to the pixel-based estimate
    b = _blockiness(img)
    # Monotone decreasing map from blockiness to quality; the absolute scale is
    # unimportant because this feeds a learned calibrator, only the ordering is.
    return float(np.clip(100.0 - 20.0 * max(b, 0.0), 1.0, 100.0))


def laplacian_variance(img: np.ndarray) -> float:
    return float(cv2.Laplacian(_grey(img), cv2.CV_64F).var())


def noise_floor(img: np.ndarray) -> float:
    """Median absolute deviation of a high-pass residual, robust to content."""
    g = _grey(img).astype(np.float64)
    resid = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    return float(np.median(np.abs(resid - np.median(resid))) * 1.4826)


def proxy_vector(img: np.ndarray, path: str | None = None) -> np.ndarray:
    return np.array([
        estimate_jpeg_quality(img, path),
        laplacian_variance(img),
        noise_floor(img),
    ], dtype=np.float32)
