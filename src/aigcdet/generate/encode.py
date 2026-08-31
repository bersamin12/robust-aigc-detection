"""JPEG encoder parity between a real and its fake.

`docs/low_level_confounds.md` measured `jpeg_quality` alone separating the
frozen corpus at AUC 0.5532, and the Open Images reals are *thumbnails* --
already twice-compressed JPEGs. A fake saved as PNG, or as a JPEG at a fixed
quality, hands a detector the compression history as a free label. `docs/02`
§1 calls fixing that mandatory, and §5.2 makes it the first acceptance gate.

So the fake is not saved at "the same quality" as its real. It is saved through
its real's own **64 quantisation integers** and its real's chroma subsampling.
Quality is a lossy summary of a quantisation table -- two files at "quality 90"
from different encoders have different tables -- and it is the table a
DCT-domain feature sees. The branch that first did this measured `jpeg_quality`
AUC 0.5031 against the 0.5532 baseline.

The real, meanwhile, must reach the fake's cropped dimensions **without gaining
a compression generation**, or the parity is undone from the other side. That
is `emit_real_cropped`, and it has no fallback on purpose: see its docstring.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image

# Natural (row-major) order, the IJG standard luminance table at quality 50.
_Q50_NATURAL = (
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
)

# A JPEG FILE stores its quantisation tables in zigzag order, but PIL
# de-zigzags them: `Image.quantization` comes back row-major, matching
# `_Q50_NATURAL` above, and `Image.save(qtables=...)` re-zigzags on the way
# out. Comparing a zigzag table against a row-major one scrambles the estimate
# silently rather than raising -- it read q=50 as 42.9 -- so the ordering is
# asserted by round-tripping a known quality through PIL in `test_encode.py`
# rather than assumed from the format spec.
_Q50 = _Q50_NATURAL


def quality_of(qtable) -> float:
    """Estimate the IJG quality that produced `qtable` (one luminance table,
    zigzag order, 64 entries).

    Recorded per row so the confound gate can compare the two classes'
    *distributions*. It is an estimate -- the mapping is many-to-one, and an
    encoder with a custom table has no IJG quality at all -- so it is a
    diagnostic, never something the save path is driven by.
    """
    q = list(qtable)
    if len(q) != 64:
        raise ValueError(f"expected 64 coefficients, got {len(q)}")
    scales = [(v * 100.0 - 50.0) / b for v, b in zip(q, _Q50) if b]
    scale = sum(scales) / len(scales)
    quality = 5000.0 / scale if scale > 100.0 else (200.0 - scale) / 2.0
    return float(min(100.0, max(1.0, quality)))


def source_encoder(path: str | Path) -> dict:
    """Lift the encoder settings off a real: its quantisation tables, its chroma
    subsampling and whether it is progressive.

    Everything returned is fed straight back to `save_matched`. Nothing is
    normalised or rounded on the way through -- the point is byte-level
    settings parity, and a "sensible default" substituted for a missing value
    is exactly how one class ends up with a distribution the other lacks.
    """
    with Image.open(path) as im:
        if im.format != "JPEG":
            raise ValueError(f"{path} is {im.format}, not JPEG; encoder parity "
                             f"assumes both sides of the pair are JPEG")
        from PIL.JpegImagePlugin import get_sampling
        qt = {k: list(v) for k, v in im.quantization.items()}
        return {
            "qtables": qt,
            "subsampling": get_sampling(im),
            "progressive": bool(im.info.get("progressive")),
            "jpeg_quality": quality_of(qt[0]),
            "size": im.size,
            "mode": im.mode,
        }


def reproducible_encoder(enc: dict) -> bool:
    """Can `save_matched` reproduce these settings exactly?

    Two ways it cannot, both measured over 3,000 Open Images thumbnails:

    - **Grayscale or CMYK reals** (0.6%). One quantisation table, no chroma.
      A generator emits RGB, so the pair would differ in channel count -- a
      leak so total no detector would need anything else.
    - **A chroma sampling PIL cannot name** (7.8%). `get_sampling` returns -1
      for any layout outside 4:4:4 / 4:2:2 / 4:2:0, and PIL's writer reads -1
      as "use the default", quietly giving the fake 4:2:0 against a real that
      is something else.

    Both are dropped from the pool rather than special-cased. The pool is
    60,000 and the target is ~14,000 pairs, so 8.4% is affordable, and the
    alternative -- a fake whose encoder settings are *approximately* its
    real's -- is the confound this module exists to remove, just smaller.
    """
    return (enc["mode"] == "RGB" and len(enc["qtables"]) == 2
            and enc["subsampling"] in (0, 1, 2))


def assert_parity(real_path: str | Path, fake_path: str | Path) -> None:
    """Post-condition on an emitted pair: identical dimensions, identical
    quantisation tables, identical subsampling.

    Cheap, and it is the check that would have caught both failures `docs/03`
    §1 records -- the multiple-of-8 dimension leak and the DCT phase mismatch
    -- at the first image instead of at the gate. Run it on every pair, not on
    a sample.
    """
    r, f = source_encoder(real_path), source_encoder(fake_path)
    for key in ("size", "qtables", "subsampling"):
        if r[key] != f[key]:
            raise RuntimeError(
                f"encoder parity broken on {key}: real={r[key]!r} "
                f"fake={f[key]!r} ({real_path} vs {fake_path})")


def save_matched(img: Image.Image, path: str | Path, enc: dict,
                 *, passes: int = 1) -> None:
    """Save `img` through `enc`'s encoder settings.

    `passes` is 1. The reals are twice-compressed thumbnails, so a second pass
    is arguably closer -- but a re-encode of an already-quantised image is
    close to idempotent at the same table, while a second pass at a *different*
    table would introduce double-quantisation artefacts on one class only.
    Raise it only with a measurement.
    """
    if passes < 1:
        raise ValueError(f"passes must be >= 1, got {passes}")
    qtables = [enc["qtables"][k] for k in sorted(enc["qtables"])]
    kw = dict(format="JPEG", qtables=qtables, subsampling=enc["subsampling"],
              progressive=enc["progressive"], optimize=False)
    out = img.convert("RGB")
    for _ in range(passes - 1):
        import io
        buf = io.BytesIO()
        out.save(buf, **kw)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.save(path, **kw)


def emit_real_cropped(src: str | Path, dst: str | Path,
                      box: tuple[int, int, int, int]) -> None:
    """Crop the real to `box` losslessly with `jpegtran`, or raise.

    `box` must be MCU-aligned in BOTH offset and size (`geometry.crop_box`).
    On those boundaries `jpegtran -crop` rewrites no DCT coefficient: the
    output's quantisation tables, its subsampling and its coefficients are the
    input's, with blocks outside the box dropped.

    **There is deliberately no PIL fallback.** A fallback would re-encode the
    real -- adding a compression generation to the authentic class and only the
    authentic class -- and would do it silently, on whichever files happened to
    trip it. That is the confound this whole module exists to prevent, arriving
    through the error path. If `jpegtran` is missing, install it; if it refuses
    a file, drop the pair.
    """
    l, t, r, b = box
    w, h = r - l, b - t
    exe = shutil.which("jpegtran")
    if exe is None:
        raise RuntimeError(
            "jpegtran not on PATH. It is required, not optional: it is the "
            "only way to crop the real without re-encoding it. Debian/Ubuntu: "
            "libjpeg-turbo-progs.")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [exe, "-crop", f"{w}x{h}+{l}+{t}", "-copy", "none",
         "-outfile", str(dst), str(src)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"jpegtran failed on {src}: {proc.stderr.strip()}")
    with Image.open(dst) as im:
        if im.size != (w, h):
            raise RuntimeError(
                f"jpegtran produced {im.size} for a {w}x{h} crop of {src}. An "
                f"unaligned box is silently grown to the enclosing MCU; the "
                f"pair would then differ in dimensions, which is the leak "
                f"docs/03 §1 recorded.")
