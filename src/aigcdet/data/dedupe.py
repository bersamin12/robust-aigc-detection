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

from concurrent.futures import ProcessPoolExecutor

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


def _hash_one(p: str) -> tuple[str, int | None, str | None]:
    """Hash one path. Module-level and returning its error rather than
    raising, so it survives a process-pool boundary: an exception raised in
    a worker arrives at the parent stripped of the path that caused it,
    which is the one thing the caller needs.
    """
    try:
        with Image.open(p) as im:
            return p, phash(np.asarray(im.convert("RGB"), dtype=np.uint8)), None
    except Exception as e:
        return p, None, f"{type(e).__name__}: {e}"


def build_hash_index(paths: list[str], skip_unreadable: bool = False,
                     workers: int = 1) -> dict[str, int]:
    """Perceptual hash per path. A path absent from the result was skipped.

    `skip_unreadable=False` (the default) raises on the first file it cannot
    decode, which is right for the DEMO side: the leak guard is only as good
    as its coverage of the set being protected, so a demo image that cannot be
    hashed is a hole in the guard and must be fixed, not tolerated.

    `skip_unreadable=True` is for the CANDIDATE side, and the caller's
    obligation is the other half of the contract: a candidate that could not
    be hashed was never checked against the demo set, so it must be DROPPED
    from the corpus rather than kept unchecked. `build_dataset` does exactly
    that, and records the count.

    `workers` parallelises the decode, which is the whole cost. It was
    measured at ~26 minutes per 100k images serially -- 50 minutes of a
    105-minute corpus build, spent one core at a time on an embarrassingly
    parallel loop. `workers <= 1` keeps the serial path, which is what the
    tests use and what a caller without a fork-safe environment needs.

    The option exists because the alternative is worse in both directions.
    This runs ~90 minutes into a corpus build, after normalisation, audit and
    the caps -- and it used to raise there, discarding all of it, on a single
    odd file among 180,000. It did so once: a normalised PNG carrying an
    oversized ICC profile that Pillow would write but not read back. That
    particular bug is fixed at its root in `data.normalize`, but "one file in
    a third-party corpus will not decode" is a permanent condition, not an
    incident.
    """
    idx: dict[str, int] = {}
    # `disable=None` shows the bar on a TTY and stays silent in tests and logs.
    results = (map(_hash_one, paths) if workers <= 1 else
               ProcessPoolExecutor(max_workers=workers).map(
                   _hash_one, paths, chunksize=64))
    for p, h, err in tqdm(results, total=len(paths), desc="phash", unit="img",
                          disable=None):
        if err is None:
            idx[p] = h
        elif not skip_unreadable:
            raise ValueError(f"cannot hash {p}: {err}")
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
