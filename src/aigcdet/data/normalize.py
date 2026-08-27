"""Give every image identical resolution and encoding history before any
augmentation is applied, so the two classes cannot be told apart by their
container (spec §4.2).

Short side 512 because model input is 384: every expert must see a downscale,
never an upscale (spec §4.4).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageOps

SHORT_SIDE: int = 512

#: One entry per input pair that could not be normalised: (src, reason).
Failure = tuple[str, str]


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
        im.save(dst, format="PNG", optimize=False)
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
