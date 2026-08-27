"""Give every image identical resolution and encoding history before any
augmentation is applied, so the two classes cannot be told apart by their
container (spec §4.2).

Short side 512 because model input is 384: every expert must see a downscale,
never an upscale (spec §4.4).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

SHORT_SIDE = 512


def normalize_image(src: str, dst: str, short_side: int = SHORT_SIDE) -> tuple[int, int]:
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    with Image.open(src) as im:
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


def normalize_many(pairs: list[tuple[str, str]], workers: int = 8) -> list[tuple[int, int]]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda ab: normalize_image(*ab), pairs))
