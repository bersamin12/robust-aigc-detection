"""Leakage guard (spec §4.1).

Training data must not overlap COCO val2017 or DALL·E Advanced — the
organisers' demo benchmark. A byte-identity check is not enough: a demo image
that has been resized, re-encoded, or lightly recompressed on its way into a
training pool is still leakage, and its bytes will differ completely. A
perceptual hash catches that; a checksum would not.

Implemented directly rather than pulling in `imagehash`: it is a dozen lines,
it keeps the dependency list short, and it is worth having under test.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from scipy.fft import dct
from tqdm import tqdm


def phash(img: np.ndarray, hash_size: int = 8) -> int:
    """DCT-based perceptual hash. Robust to recompression and rescaling,
    which is exactly the overlap we need to catch."""
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(grey, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
    d = dct(dct(small.astype(np.float64), axis=0, norm="ortho"), axis=1, norm="ortho")
    # Take a (hash_size+1) x (hash_size+1) low-frequency block, then drop its
    # first row and column: that removes the DC term (mean brightness), so a
    # brightness shift (colour jitter is one of the six degradation families
    # this guard must survive) cannot flip a hash bit. Slicing from the
    # +1-sized block, rather than the hash_size x hash_size block, keeps the
    # remaining AC block at hash_size x hash_size, so the hash is still a
    # genuine 64-bit value at hash_size=8. The median and the packed bits
    # both come from this same DC-excluded AC block.
    ac = d[:hash_size + 1, :hash_size + 1][1:, 1:]
    med = np.median(ac)
    bits = (ac > med).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return int(bin(a ^ b).count("1"))


def build_hash_index(paths: list[str]) -> dict[str, int]:
    idx = {}
    # ~26 minutes at 100k images, entirely serial. `disable=None` shows the
    # bar on a TTY and stays silent in tests and logs.
    for p in tqdm(paths, desc="phash", unit="img", disable=None):
        with Image.open(p) as im:
            idx[p] = phash(np.asarray(im.convert("RGB"), dtype=np.uint8))
    return idx


def _pack(hashes: list[int]) -> np.ndarray:
    return np.array([[(h >> (8 * i)) & 0xFF for i in range(8)] for h in hashes], dtype=np.uint8)


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def find_leaks(
    candidate_hashes: dict[str, int],
    demo_hashes: dict[str, int],
    max_distance: int = 4,
) -> dict[str, str]:
    """Return {candidate_path: matching_demo_path} for every near-duplicate.

    Vectorised NumPy popcount over a packed uint8 array rather than a pure
    Python double loop: at scale (100k candidates x 14k demo images) the
    naive O(n*m) Python loop is ~1.4e9 comparisons, too slow to run.
    """
    if not candidate_hashes or not demo_hashes:
        return {}

    cps = list(candidate_hashes)
    chs = _pack(list(candidate_hashes.values()))
    dps = list(demo_hashes)
    dhs = _pack(list(demo_hashes.values()))

    leaks: dict[str, str] = {}
    chunk = 2048
    for start in range(0, len(cps), chunk):
        block = chs[start:start + chunk]
        dist = _POPCOUNT[block[:, None, :] ^ dhs[None, :, :]].sum(axis=2)
        hit_c, hit_d = np.where(dist <= max_distance)
        for ci, di in zip(hit_c, hit_d):
            leaks.setdefault(cps[start + ci], dps[di])
    return leaks
