"""Crop geometry and content-addressed seeding for the AI-OV7 pairs.

WHY THIS IS A MODULE AND NOT FOUR LINES INLINE
----------------------------------------------
The first smoke run of this corpus (`docs/03` §1, 2026-08-30) emitted 6 pairs
and **6 of 6 leaked the label through geometry**: the fake came out a multiple
of 8 -- the VAE downsamples by that -- while the real was copied byte-for-byte
at its arbitrary native size. `width % 8 == 0` separated the classes at ~100%
without a classifier ever touching a pixel.

The confound gate then returned `jpeg_quality` AUC **0.0000** -- perfect
inverse separation -- on both families, despite quantisation tables being
copied correctly. The diagnosis is that `(w - cw) // 2` put each fake's fresh
8x8 DCT grid at a different *phase* from its real's preserved grid, and
copying quantisation tables cannot fix a phase difference.

Both bugs live in `crop_box`. Both are one arithmetic expression. Hence a
module with tests, rather than a notebook cell.
"""
from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

# Crop geometry aligns to a JPEG MCU, not merely to the VAE's 8. A crop on MCU
# boundaries can be applied to the REAL with `jpegtran -crop`, which rewrites
# no DCT coefficient -- so the real reaches the fake's dimensions without
# gaining a compression generation.
#
# 16 is the safe value, not the tight one. The MCU is 16x16 for 4:2:0, 16x8 for
# 4:2:2 and 8x8 for 4:4:4, and the Open Images thumbnails are a MIX (5 of the 6
# smoke reals were 4:4:4). Aligning to whatever subsampling each file happens to
# declare would trim less -- and would make the trim a function of the file's
# chroma format, which is a property of the camera and encoder, i.e. exactly
# the kind of thing this corpus exists to hold constant across a pair.
MCU_ALIGN = 16

# Skip reals below this short side. The Open Images thumbnails sit at p05 ~360,
# so this bites rarely; it exists so a stray tiny file cannot produce a fake
# generated far below any diffusion model's trained resolution.
MIN_SIDE = 320

# Cap the long side by CROPPING, never by resizing. Does not bind on the
# thumbnails (long side ~650). It binds the moment someone points this at Open
# Images originals -- which is precisely when nobody would be watching for it.
MAX_SIDE = 1024


def crop_box(w: int, h: int, *, mcu: int = MCU_ALIGN,
             max_side: int = MAX_SIDE) -> tuple[int, int, int, int]:
    """Centre box as (left, top, right, bottom), size AND offset multiples of
    `mcu`, aspect preserved.

    Crops only; never resizes. A resample leaves a spectral signature
    (`docs/resolution_shortcut.md`), and one class carrying that signature is
    the confound this corpus exists to remove.

    The cap scales both sides by one factor rather than clamping each
    independently. Clamping each would turn a 1200x1800 portrait into a
    1024x1024 crop -- silently shifting the aspect-ratio distribution of the
    class that is supposed to match its real on exactly that.

    The OFFSET is aligned as well as the size, which is the fix for the DCT
    phase mismatch: an unaligned offset is not a crop `jpegtran` can apply
    losslessly, and it leaves every fake's DCT grid at a different phase from
    its real's.
    """
    if w <= 0 or h <= 0:
        raise ValueError(f"non-positive dimensions: {w}x{h}")
    scale = min(1.0, max_side / max(w, h))
    cw, ch = int(w * scale), int(h * scale)
    cw, ch = cw - cw % mcu, ch - ch % mcu
    if cw < mcu or ch < mcu:
        raise ValueError(f"{w}x{h} has no {mcu}-aligned crop")
    left = ((w - cw) // 2 // mcu) * mcu
    top = ((h - ch) // 2 // mcu) * mcu
    return (left, top, left + cw, top + ch)


def seed_for(image_id: str, seed: int) -> int:
    """Content-addressed seed `blake2b(seed, image_id)`, never a counter.

    A counter makes the pixels a function of *where the image fell in the run*,
    so a rerun with a different total, a different shard boundary or one
    dropped file produces different images for the same real. Content
    addressing makes a rerun reproduce the corpus.
    """
    return int(hashlib.blake2b(f"{seed}:{image_id}".encode(),
                               digest_size=7).hexdigest(), 16)


def order_key(image_id: str, seed: int) -> str:
    """Shuffle key. Shards take contiguous blocks of this order, so two workers
    holding different shard numbers can never draw the same real -- and so the
    assignment does not depend on how many rows the pool happened to have."""
    return hashlib.blake2b(f"{seed}:{image_id}".encode(),
                           digest_size=8).hexdigest()


def shard_block(n: int, shard: int, n_shards: int) -> tuple[int, int]:
    """Half-open [start, stop) of a contiguous, disjoint block of `n` items.

    Contiguous rather than `i % n_shards` so that a partially-finished shard
    leaves a prefix of one block, not a comb across the whole pool.
    """
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard {shard} out of range for {n_shards}")
    per = n // n_shards
    extra = n % n_shards
    start = shard * per + min(shard, extra)
    return start, start + per + (1 if shard < extra else 0)


def box_mask(image_id: str, w: int, h: int, seed: int) -> Image.Image:
    """A deterministic rectangle covering 50-67% of each axis, mode "L".

    `docs/02` §3.1 wants a partially-synthetic class, and Open Images'
    segmentation and box CSVs are ~2 GB each -- more download than that arm is
    worth here. B-Free used plain rectangles for its different-category
    replacements anyway, so this is the honest version: it is a BOX, the name
    says box, and nothing claims the regenerated region follows an object.
    """
    rng = np.random.default_rng(seed_for(image_id, seed) % (2 ** 32))
    fw, fh = rng.uniform(0.5, 0.67, size=2)
    bw, bh = max(8, int(w * fw) // 8 * 8), max(8, int(h * fh) // 8 * 8)
    left = int(rng.integers(0, w - bw + 1))
    top = int(rng.integers(0, h - bh + 1))
    m = Image.new("L", (w, h), 0)
    m.paste(255, (left, top, left + bw, top + bh))
    return m
