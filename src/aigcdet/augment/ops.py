"""The six transform families from the brief.

Every op is shape-preserving and uint8 in / uint8 out, so ops compose freely
in any order. Parameters are the brief's own units: JPEG quality 0-100,
blur sigma in pixels, resize scale as a fraction, noise sigma on a [0,1]
intensity scale, jitter as a signed fraction, crop as a kept fraction.
"""
from __future__ import annotations

import importlib.util
import io
import threading

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


# ---------------------------------------------------------------------------
# The NTIRE 2026 transform families (arXiv 2604.11487, Table 3), grouped under
# the six family bits above. Every op keeps the contract at the top of this
# module: shape-preserving, uint8 in / uint8 out, composable in any order. Ops
# that draw random pixels take `rng`; everything else takes its randomness as
# PARAMETERS so a stored recipe replays byte-for-byte.
# ---------------------------------------------------------------------------

_INTERP = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR,
           "area": cv2.INTER_AREA, "cubic": cv2.INTER_CUBIC}


def _to_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


# --- jpeg family -----------------------------------------------------------

def jpeg_multi(img: np.ndarray, qualities: list) -> np.ndarray:
    """Successive JPEG encodes at each quality in order -- the "multiple
    compressions" condition: a re-shared image carries several grids."""
    out = img
    for q in qualities:
        out = jpeg(out, int(q))
    return out


def jpeg2000_available() -> bool:
    """Whether this OpenCV build can encode `.jp2`. Decided ONCE per process
    and pinned into the recipe pool literal by the caller, never probed per
    call: a pool that silently drops an op on one box produces different
    pixels from the same recipe on another."""
    return bool(cv2.haveImageWriter("probe.jp2"))


def jpeg2000(img: np.ndarray, rate: int) -> np.ndarray:
    """JPEG 2000 round trip. `rate` is OpenCV's compression-x1000 knob:
    1000 is lossless, lower is harsher."""
    ok, buf = cv2.imencode(".jp2", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, int(rate)])
    if not ok:
        raise RuntimeError("cv2 could not encode JPEG 2000; check "
                           "jpeg2000_available() before putting it in a pool")
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return np.ascontiguousarray(cv2.cvtColor(dec, cv2.COLOR_BGR2RGB))


# --- blur family -----------------------------------------------------------

def motion_blur(img: np.ndarray, length: int, angle: float) -> np.ndarray:
    """A line kernel of `length` pixels at `angle` degrees."""
    length = max(3, int(length))
    k = np.zeros((length, length), dtype=np.float32)
    c = (length - 1) / 2.0
    dx, dy = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    p0 = (int(round(c - dx * c)), int(round(c - dy * c)))
    p1 = (int(round(c + dx * c)), int(round(c + dy * c)))
    cv2.line(k, p0, p1, 1.0, thickness=1)
    k /= max(k.sum(), 1.0)
    out = cv2.filter2D(img.astype(np.float32), -1, k, borderType=cv2.BORDER_REFLECT)
    return _to_u8(out)


def lens_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Disc (bokeh) kernel of the given radius in pixels."""
    radius = max(1, int(radius))
    size = 2 * radius + 1
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    k = ((xx ** 2 + yy ** 2) <= radius ** 2).astype(np.float32)
    k /= k.sum()
    out = cv2.filter2D(img.astype(np.float32), -1, k, borderType=cv2.BORDER_REFLECT)
    return _to_u8(out)


def glass_blur(img: np.ndarray, sigma: float, max_delta: int, iters: int,
               rng: np.random.Generator) -> np.ndarray:
    """Frosted-glass: Gaussian blur, then locally shuffle pixels by up to
    `max_delta`, repeated `iters` times, then blur again (ImageNet-C's
    construction, vectorised: one random offset field per iteration)."""
    h, w = img.shape[:2]
    out = blur(img, sigma)
    ys, xs = np.mgrid[0:h, 0:w]
    for _ in range(max(1, int(iters))):
        dy = rng.integers(-max_delta, max_delta + 1, size=(h, w))
        dx = rng.integers(-max_delta, max_delta + 1, size=(h, w))
        sy = np.clip(ys + dy, 0, h - 1)
        sx = np.clip(xs + dx, 0, w - 1)
        out = out[sy, sx]
    return blur(np.ascontiguousarray(out), sigma)


# --- resize family ---------------------------------------------------------

def downscale(img: np.ndarray, scale: float, interp_down: str = "area",
              interp_up: str = "linear") -> np.ndarray:
    """`resize_roundtrip` with the resampler chosen per direction -- the
    interpolation is the artefact, so it is a parameter, not a constant."""
    h, w = img.shape[:2]
    sh, sw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = cv2.resize(img, (sw, sh), interpolation=_INTERP[interp_down])
    return cv2.resize(small, (w, h), interpolation=_INTERP[interp_up])


def downscale_keep(img: np.ndarray, scale: float, interp: str = "area") -> np.ndarray:
    """Resize by `scale` and KEEP it small -- NTIRE's `downscale` (factor
    0.3-0.8 of the full image, never resampled back). Not shape-preserving,
    so it is a PRE-CANONICAL op (`recipes.PRE_CANONICAL_OPS`): it acts on the
    decoded image before the crop window is taken, exactly where the
    organisers' transform acts, and `Recipe.apply` refuses it. The crop then
    sees a wider field of view at lower detail, and an image that falls below
    the window is taken whole under `crop_clamp`, as at scoring time."""
    h, w = img.shape[:2]
    sh, sw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    return np.ascontiguousarray(cv2.resize(img, (sw, sh), interpolation=_INTERP[interp]))


#: A learned image codec standing in for JPEG AI (ISO/IEC 6048), which the
#: challenge applies at four quality levels and whose reconstructions carry no
#: block grid -- textures come back smoothed and re-synthesised, which is the
#: artefact a JPEG-trained detector has never seen. compressai's
#: `bmshj2018_factorized` (Balle et al. 2018) is the cheapest hyperprior-free
#: learned codec with public weights: ~0.09 s per 224px image on one CPU
#: thread. Quality 1 is ~0.18 bpp, 4 is ~1 bpp; lower is harsher.
_JPEG_AI_ARCH = "bmshj2018_factorized"
_JPEG_AI_QUALITIES = (1, 2, 3, 4)
_jpeg_ai_models: dict = {}
_jpeg_ai_lock = threading.Lock()


def jpeg_ai_available() -> bool:
    """Whether compressai is installed. Decided ONCE per process by the caller
    and pinned by the pool name, as `jpeg2000_available`."""
    return importlib.util.find_spec("compressai") is not None


def _jpeg_ai_model(quality: int):
    if quality not in _JPEG_AI_QUALITIES:
        raise ValueError(f"jpeg_ai quality must be one of {_JPEG_AI_QUALITIES}, got {quality}")
    with _jpeg_ai_lock:
        net = _jpeg_ai_models.get(quality)
        if net is None:
            try:
                from compressai import zoo
            except ImportError as exc:
                raise ImportError(
                    "the jpeg_ai op needs compressai (`pip install compressai`); "
                    "check jpeg_ai_available() before putting it in a pool") from exc
            net = getattr(zoo, _JPEG_AI_ARCH)(quality=quality, pretrained=True).eval()
            for p in net.parameters():
                p.requires_grad_(False)
            _jpeg_ai_models[quality] = net
    return net


def jpeg_ai(img: np.ndarray, quality: int) -> np.ndarray:
    """Learned-codec round trip (analysis -> quantise -> synthesis; no
    entropy coding, which does not change the pixels). Padded to a multiple
    of 64 by edge replication and cropped back, so any size works."""
    import torch
    import torch.nn.functional as F

    net = _jpeg_ai_model(int(quality))
    h, w = img.shape[:2]
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)[None]
    ph, pw = (64 - h % 64) % 64, (64 - w % 64) % 64
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    with torch.no_grad():
        y = net(x)["x_hat"]
    y = y[0, :, :h, :w].clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
    return np.ascontiguousarray(y.permute(1, 2, 0).numpy())


def pixelate(img: np.ndarray, block: int) -> np.ndarray:
    """Average over `block`x`block` cells and hold each cell flat."""
    h, w = img.shape[:2]
    block = max(2, int(block))
    sh, sw = max(1, h // block), max(1, w // block)
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


# --- noise family ----------------------------------------------------------

def impulse(img: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Salt-and-pepper: `amount` of pixels set to 0 or 255, half each."""
    h, w = img.shape[:2]
    u = rng.random((h, w), dtype=np.float32)
    out = img.copy()
    out[u < amount / 2.0] = 0
    out[(u >= amount / 2.0) & (u < amount)] = 255
    return out


def iso_noise(img: np.ndarray, intensity: float, color_shift: float,
              rng: np.random.Generator) -> np.ndarray:
    """Sensor-like noise: signal-dependent luminance grain plus independent
    per-channel chroma noise."""
    x = img.astype(np.float32)
    luma = x @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    grain = rng.standard_normal(luma.shape, dtype=np.float32)
    grain *= intensity * np.sqrt(np.maximum(luma, 1.0)) * 2.0
    x += grain[..., None]
    chroma = rng.standard_normal(x.shape, dtype=np.float32)
    chroma *= color_shift * 255.0
    x += chroma
    return _to_u8(x)


# --- jitter family (colour / tone) -----------------------------------------

def tone_curve(img: np.ndarray, low_y: float, high_y: float) -> np.ndarray:
    """A cubic Bezier tone curve with control points (0.25, low_y) and
    (0.75, high_y); identity at low_y=0.25, high_y=0.75."""
    t = np.linspace(0.0, 1.0, 512, dtype=np.float64)
    bx = 3 * (1 - t) ** 2 * t * 0.25 + 3 * (1 - t) * t ** 2 * 0.75 + t ** 3
    by = 3 * (1 - t) ** 2 * t * low_y + 3 * (1 - t) * t ** 2 * high_y + t ** 3
    grid = np.arange(256, dtype=np.float64) / 255.0
    lut = _to_u8(np.interp(grid, bx, by) * 255.0)
    return lut[img]


def clahe(img: np.ndarray, clip: float, grid: int) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on the L channel."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    lab[..., 0] = c.apply(lab[..., 0])
    return np.ascontiguousarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def contrast(img: np.ndarray, alpha: float) -> np.ndarray:
    """Linear contrast about the image mean; identity at alpha=1."""
    x = img.astype(np.float32)
    mean = float(x.mean())
    return _to_u8((x - mean) * alpha + mean)


def brightness(img: np.ndarray, beta: float) -> np.ndarray:
    """Additive brightness in grey levels; identity at beta=0."""
    return _to_u8(img.astype(np.float32) + beta)


def quantize(img: np.ndarray, bits: int) -> np.ndarray:
    """Keep the top `bits` bits of every channel (posterise)."""
    shift = 8 - int(bits)
    if shift <= 0:
        return img.copy()
    return ((img >> shift) << shift).astype(np.uint8)


def channel_shift(img: np.ndarray, dr: float, dg: float, db: float) -> np.ndarray:
    """Per-channel additive offsets in grey levels."""
    x = img.astype(np.float32)
    x += np.array([dr, dg, db], dtype=np.float32)
    return _to_u8(x)


def saturation(img: np.ndarray, scale: float) -> np.ndarray:
    """Scale chroma about luminance; identity at scale=1."""
    x = img.astype(np.float32)
    grey = (x @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
    return _to_u8(grey + (x - grey) * scale)


# --- crop family (geometry) ------------------------------------------------

def aspect_crop(img: np.ndarray, top: float, left: float, h_frac: float,
                w_frac: float) -> np.ndarray:
    """An off-centre window of a different aspect ratio, resized back to the
    original size so the op stays shape-preserving."""
    h, w = img.shape[:2]
    ch, cw = max(1, int(round(h * h_frac))), max(1, int(round(w * w_frac)))
    t = min(max(0, int(round(h * top))), h - ch)
    l = min(max(0, int(round(w * left))), w - cw)
    cropped = img[t:t + ch, l:l + cw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)


def perspective(img: np.ndarray, offsets: list) -> np.ndarray:
    """Displace the four corners by `offsets` (8 fractions of width/height,
    in order TL.x, TL.y, TR.x, TR.y, BR.x, BR.y, BL.x, BL.y) and warp back
    onto the same canvas, reflecting at the border."""
    h, w = img.shape[:2]
    o = np.asarray(offsets, dtype=np.float32).reshape(4, 2)
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    dst = src + o * np.array([w, h], dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)


OP_FUNCS = {
    # The brief's six, one per family. Their names double as family names.
    "jpeg": jpeg,
    "blur": blur,
    "resize": resize_roundtrip,
    "noise": noise,
    "jitter": jitter,
    "crop": center_crop,
    # NTIRE additions, each grouped under one of the six by
    # `recipes.FAMILY_OF`.
    "jpeg_multi": jpeg_multi,
    "jpeg2000": jpeg2000,
    "motion_blur": motion_blur,
    "lens_blur": lens_blur,
    "glass_blur": glass_blur,
    "downscale": downscale,
    "pixelate": pixelate,
    "downscale_keep": downscale_keep,
    "jpeg_ai": jpeg_ai,
    "impulse": impulse,
    "iso_noise": iso_noise,
    "tone_curve": tone_curve,
    "clahe": clahe,
    "contrast": contrast,
    "brightness": brightness,
    "quantize": quantize,
    "channel_shift": channel_shift,
    "saturation": saturation,
    "aspect_crop": aspect_crop,
    "perspective": perspective,
}
