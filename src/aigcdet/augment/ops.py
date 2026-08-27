"""The six transform families from the brief.

Every op is shape-preserving and uint8 in / uint8 out, so ops compose freely
in any order. Parameters are the brief's own units: JPEG quality 0-100,
blur sigma in pixels, resize scale as a fraction, noise sigma on a [0,1]
intensity scale, jitter as a signed fraction, crop as a kept fraction.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


#: scipy.ndimage.gaussian_filter's default kernel truncation, reproduced so
#: the two implementations use the same support (radius = int(4*sigma + 0.5)).
_BLUR_TRUNCATE = 4.0


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """cv2 takes sigma directly, unlike PIL's GaussianBlur radius, so the
    brief's sigma values are reproduced exactly rather than approximated.

    This was `scipy.ndimage.gaussian_filter`, which cost ~15-25 ms per
    512x768 call against ~1-4 ms here -- and Plan 2 applies 11 views per
    image, serial with the GPU forward, so that difference is CPU-hours per
    feature bank. Kernel support and border mode are matched to scipy's
    (truncate=4.0, i.e. radius = int(4*sigma + 0.5), reflecting border), and
    the filter still runs on float32 and truncates on the way back to uint8,
    exactly as before: at the three evaluation sigmas the outputs agree with
    scipy to within ONE grey level, on at most ~0.1% of pixels.

    Filtering the uint8 array directly is a further ~8x faster, but cv2
    rounds instead of truncating there, which shifts roughly half of all
    pixels by one level. That is harmless in itself, yet it perturbs pHash
    enough to matter on very low-contrast content, and the whole-image cost
    difference is ~7% once the rest of the pipeline is counted -- not worth
    changing the numbers every downstream component was validated against.
    """
    if sigma <= 0:
        return img.copy()
    ksize = 2 * int(_BLUR_TRUNCATE * sigma + 0.5) + 1
    out = cv2.GaussianBlur(img.astype(np.float32), (ksize, ksize),
                           sigmaX=sigma, sigmaY=sigma,
                           borderType=cv2.BORDER_REFLECT)
    return np.clip(out, 0, 255).astype(np.uint8)


def resize_roundtrip(img: np.ndarray, scale: float) -> np.ndarray:
    """Downscale then upscale back — the thumbnail-generation analogue."""
    h, w = img.shape[:2]
    sh, sw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    # float32 throughout: the float64 draw and the float64 intermediate cost
    # roughly twice as much for precision that the uint8 result discards.
    # Note this draws a different (equally valid, equally reproducible)
    # stream from the same Generator than `rng.normal` did.
    n = rng.standard_normal(img.shape, dtype=np.float32)
    n *= sigma * 255.0
    n += img
    np.clip(n, 0, 255, out=n)
    return n.astype(np.uint8)


def jitter(img: np.ndarray, brightness: float, contrast: float, saturation: float) -> np.ndarray:
    """Signed fractional deltas: +0.2 means +20%."""
    # Arithmetic is identical to the obvious out-of-place form (verified
    # bit-for-bit); the in-place variants just stop allocating a fresh
    # HxWx3 float32 array at every step.
    x = img.astype(np.float32)
    x *= 1.0 + brightness
    mean = float(x.mean())
    x -= mean
    x *= 1.0 + contrast
    x += mean
    grey = (x @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
    x -= grey
    x *= 1.0 + saturation
    x += grey
    np.clip(x, 0, 255, out=x)
    return x.astype(np.uint8)


def center_crop(img: np.ndarray, frac: float) -> np.ndarray:
    """Crop the central `frac` of each side, then resize back to the original
    size so the op stays shape-preserving and composable."""
    h, w = img.shape[:2]
    ch, cw = max(1, int(round(h * frac))), max(1, int(round(w * frac)))
    top, left = (h - ch) // 2, (w - cw) // 2
    cropped = img[top:top + ch, left:left + cw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)


OP_FUNCS = {
    "jpeg": jpeg,
    "blur": blur,
    "resize": resize_roundtrip,
    "noise": noise,
    "jitter": jitter,
    "crop": center_crop,
}
