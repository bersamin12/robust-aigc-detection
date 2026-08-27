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
from scipy.ndimage import gaussian_filter


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    # scipy takes sigma directly, unlike PIL's GaussianBlur radius, so the
    # brief's sigma values are reproduced exactly rather than approximated.
    if sigma <= 0:
        return img.copy()
    out = gaussian_filter(img.astype(np.float32), sigma=(sigma, sigma, 0), mode="reflect")
    return np.clip(out, 0, 255).astype(np.uint8)


def resize_roundtrip(img: np.ndarray, scale: float) -> np.ndarray:
    """Downscale then upscale back — the thumbnail-generation analogue."""
    h, w = img.shape[:2]
    sh, sw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    n = rng.normal(0.0, sigma * 255.0, size=img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def jitter(img: np.ndarray, brightness: float, contrast: float, saturation: float) -> np.ndarray:
    """Signed fractional deltas: +0.2 means +20%."""
    x = img.astype(np.float32)
    x = x * (1.0 + brightness)
    mean = x.mean()
    x = (x - mean) * (1.0 + contrast) + mean
    grey = x @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    x = grey[..., None] + (x - grey[..., None]) * (1.0 + saturation)
    return np.clip(x, 0, 255).astype(np.uint8)


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
