"""Give every image identical resolution and encoding history before any
augmentation is applied, so the two classes cannot be told apart by their
container (spec §4.2).

Short side 512 because model input is 384: every expert must see a downscale,
never an upscale (spec §4.4).

This module also owns the project's policy for HOW an image reaches disk as a
PNG (`save_png`), because `scripts/acquire_data.py` has to write PNGs too when
a source streams decoded images rather than files, and a second, differently
opinionated copy of that policy is how classes start differing by container.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageOps

SHORT_SIDE: int = 512

#: Modes PNG stores without altering a single sample. Verified by a test that
#: actually round-trips each one through the encoder rather than trusting this
#: list -- Pillow's own table is private and has changed between releases.
#: Bare "I" is deliberately absent: it still encodes today but Pillow 12
#: deprecates it, so it is remapped below instead.
PNG_NATIVE_MODES: frozenset[str] = frozenset({"1", "L", "LA", "I;16", "P", "RGB", "RGBA"})

#: Non-native modes with a lossless PNG-native target. 32-bit integer
#: greyscale is remapped rather than converted to RGB, which would throw away
#: bit depth as well as the deprecation.
PNG_MODE_OVERRIDES: dict[str, str] = {"I": "I;16"}

#: One entry per input pair that could not be normalised: (src, reason).
Failure = tuple[str, str]

#: Ancillary PNG chunks this module refuses to carry from input to output.
#:
#: An embedded ICC profile is CONTAINER METADATA, and removing container
#: metadata that differs between the classes is the entire job of this module:
#: COCO's JPEGs carry an sRGB profile, generator PNGs usually carry none, so
#: "has an ICC profile" is a property of the source rather than of how the
#: image was made. Stripping it changes no sample value -- Pillow does not
#: apply a profile on decode, and `convert("RGB")` does not either.
#:
#: It is also a correctness fix, and that is how it was found. Pillow will
#: WRITE an arbitrarily large iCCP chunk and then refuse to READ it back
#: (`_safe_zlib_decompress` caps it at `MAX_TEXT_CHUNK`), so one SID_Set image
#: with an oversized profile produced a normalised PNG that normalisation
#: itself could not reopen. It survived the audit, which reads the RAW files,
#: and killed `build_hash_index` 95 minutes into a corpus build -- at the one
#: step whose comment says normalisation has "already proved the candidate
#: files decodable". It had proved the INPUTS decodable, not the outputs.
_STRIPPED_INFO_KEYS = ("icc_profile",)


def _without_stripped_metadata(im: Image.Image) -> dict:
    """Save keywords that suppress every chunk in `_STRIPPED_INFO_KEYS`.

    Passing the key explicitly as None is what suppresses it: Pillow reads
    `im.encoderinfo.get("icc_profile", im.info.get("icc_profile"))`, so the
    default is only consulted when the key is ABSENT from encoderinfo. Popping
    it from `im.info` instead would not work on an image whose profile arrived
    with the decoder.
    """
    return {k: None for k in _STRIPPED_INFO_KEYS}


def png_target_mode(mode: str, bands: tuple[str, ...]) -> str:
    """The mode `save_png` will write an image of `mode` in.

    The rule, and why each branch is what it is:

    - A PNG-native mode is written UNCHANGED. Acquisition is not the place to
      normalise pixels; `normalize_image` does that later, deliberately, to
      both classes at once. Converting here would apply an extra, invisible
      transform to whichever sources happen to arrive in an unusual mode --
      i.e. to one class -- inside a project whose premise is that pixel-level
      forensic cues survive or die at exactly these steps.
    - A non-native mode WITH an alpha band converts to RGBA, never to RGB.
      `convert("RGB")` composites alpha against black, which changes every
      transparent pixel to a colour that was never in the image. RGBA is
      PNG-native, so the alpha is simply kept and nothing is invented.
    - A non-native mode WITHOUT alpha converts to RGB. This is the CMYK case
      that crashed a real SID_Set run ("cannot write mode CMYK as PNG"), and
      it is the one branch that is genuinely lossy: PIL's CMYK->RGB is a naive
      per-channel inversion with no ICC transform, so the colours shift. There
      is no better option without the embedded profile, and CMYK cannot carry
      alpha, so nothing is flattened.

    Alpha is detected from the BANDS rather than from the mode string, so an
    unusual premultiplied mode ("RGBa", "La") is handled by the same rule
    instead of falling through a mode-name check to RGB.
    """
    if mode in PNG_NATIVE_MODES:
        return mode
    if mode in PNG_MODE_OVERRIDES:
        return PNG_MODE_OVERRIDES[mode]
    return "RGBA" if any(b in ("A", "a") for b in bands) else "RGB"


def save_png(im: Image.Image, dst: str) -> str:
    """Write `im` to `dst` as a PNG, converting only if PNG cannot hold its
    mode, and return the mode actually written.

    Writes through `dst + ".part"` and renames, so an interrupted run never
    leaves a truncated file at `dst` -- which matters because the acquisition
    scripts resume by testing `os.path.exists(dst)`, and a half-written image
    that resume treats as done is worse than no image at all.
    """
    target = png_target_mode(im.mode, im.getbands())
    if target != im.mode:
        im = im.convert(target)
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    tmp = dst + ".part"
    try:
        im.save(tmp, format="PNG", optimize=False,
                **_without_stripped_metadata(im))
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return target


def normalize_image(src: str, dst: str, short_side: int = SHORT_SIDE) -> tuple[int, int]:
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    with Image.open(src) as im:
        # Apply EXIF orientation FIRST, before anything is measured or
        # resized. Authentic photographs carry an orientation tag and
        # generator output does not, so ignoring it lands on one class only
        # -- inside the very step whose purpose is removing
        # class-distinguishing container properties. PNG drops the tag, so
        # an unrotated image bakes the wrong orientation in permanently and
        # records the wrong width/height in the manifest.
        im = ImageOps.exif_transpose(im) or im
        im = im.convert("RGB")
        w, h = im.size
        # Never upscale: inventing detail would fabricate the forensic evidence
        # this project is trying to measure.
        if min(w, h) > short_side:
            scale = short_side / min(w, h)
            w, h = max(1, round(w * scale)), max(1, round(h * scale))
            im = im.resize((w, h), Image.LANCZOS)
        im.save(dst, format="PNG", optimize=False,
                **_without_stripped_metadata(im))
    return (w, h)


def normalize_many(
    pairs: list[tuple[str, str]], workers: int = 8
) -> tuple[list[tuple[int, int] | None], list[Failure]]:
    """Normalise every pair, surviving individual unreadable files.

    Returns `(sizes, failures)`:

    - `sizes` has exactly one entry per input pair, IN INPUT ORDER, holding
      `(width, height)` for a file that normalised and `None` for one that
      did not. That order becomes the manifest's row order, and Plan 2's
      feature banks index against the manifest positionally, so a failure
      keeps its slot rather than shifting everything after it.
    - `failures` lists `(src, reason)` for each skipped file, in input order.

    A single truncated or zero-byte file used to abort the whole run --
    `list(ex.map(...))` propagates the first exception -- and at ~100k images
    streamed from HuggingFace and ModelScope at least one is near-certain.
    Failures are reported rather than swallowed: a run that quietly dropped
    3,000 unreadable images would be worse than one that died.
    """
    def _one(pair: tuple[str, str]) -> tuple[tuple[int, int] | None, str]:
        try:
            return normalize_image(*pair), ""
        except Exception as e:  # deliberately broad: one bad file must not end the run
            return None, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_one, pairs))  # ex.map preserves input order

    sizes = [size for size, _ in results]
    failures = [(src, reason) for (src, _), (size, reason) in zip(pairs, results)
                if size is None]
    return sizes, failures
