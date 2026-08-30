"""Gate 0: prove the encoder-parity save path BEFORE any image is bought.

`docs/03-commercial-apis-on-open-images.md` §3.1 makes this the first gate, and
the reason is economic rather than scientific: a mis-encoded local generation is
regenerated for free, a mis-encoded purchased one is bought twice. So the path
that will store commercial API output is proved here, on task 02's free
open-weight images, against the same reals.

WHAT IT MEASURES
----------------
For every (real, generated) pair sharing an `ImageID` stem, three readings of
the same proxies (`features.proxies`):

  before   the generated file as the generator left it -- PNG, native
           resolution -- against the real. This is what the corpus would carry
           with no parity step, and it is the number the parity path has to beat.
  after    the generated image put through `data.encoder_parity`, which copies
           the paired real's quantisation tables, subsampling, progressive flag
           and exact pixel dimensions onto it.
  delta    how much of the gap the parity step actually closed.

Each is reported twice, and the pair matters:

  path-aware  `estimate_jpeg_quality(img, path)`, which reads the exact
              quantisation table when the file is a JPEG. This is how
              `data/audit.py` reads the corpus, so it is the honest "as stored"
              figure -- but BEFORE parity the reals take the exact branch and
              the generated PNGs take the pixel fallback, so the two classes are
              being measured by different instruments. `eval/controls.py` warns
              about exactly that mixture. A large `before` AUC here is partly an
              artefact of the split branch, which is itself the finding: the
              corpus cannot even be audited consistently until parity exists.
  pixel-only  the fallback on both sides, no path. One instrument, so `before`
              and `after` are comparable, and what it isolates is the
              compression history actually present in the samples.

`short_side` is reported alongside because `docs/resolution_shortcut.md`
measures it classifying at 72.6% in the training pool and ~100% on the
organisers' benchmark. Parity fixes it by construction -- the fake inherits the
real's exact dimensions -- so an `after` figure that is not ~0.5 means the
pairing is broken, not that the confound is stubborn.

THE THRESHOLD
-------------
`--max-auc` defaults to 0.60, which is task 02 §5's own gate ("if
`jpeg_quality` AUC alone is above ~0.60, fix the save path before looking at
anything else"). It is read against the pixel-only `after` figure: the
path-aware one is 0.5 by construction once both classes are JPEGs with
identical tables, so gating on it would be gating on a tautology.

WHAT A PASS DOES NOT PROVE
--------------------------
That the corpus is clean. Three pixel statistics are blind to content, and a
real is a re-encode of an already-compressed original while a parity-matched
fake carries single JPEG history -- the double-quantisation residual is real
and this cannot remove it. `scripts/gate_confounds.py` over the built manifest
and `eval/controls.py:content_blind_auc` over a bank are the next instruments.
Passing this is necessary, not sufficient.

USAGE
-----
    python scripts/prove_encoder_parity.py \\
        --reals     /mnt/berstorage/techjam/open_images/portrait \\
        --generated /mnt/berstorage/techjam/gen/flux_schnell \\
        --out       /mnt/berstorage/techjam/gen/flux_schnell_parity \\
        --n 400 --max-auc 0.60
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

from aigcdet.data.encoder_parity import (
    GEOMETRIES, GEOMETRY_RESAMPLE, ParityError, save_matched_to_real,
)
from aigcdet.features.proxies import PROXY_NAMES, estimate_jpeg_quality, proxy_vector

#: Read against the pixel-only `after` figure. Task 02 §5's own number.
DEFAULT_MAX_AUC = 0.60

#: Enough that an AUC's standard error is well under the effect being read
#: (~0.02 at 200 per class), small enough to run on a laptop in minutes.
DEFAULT_N = 400

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """Orientation-corrected AUC, identical to `gate_confounds.auc`.

    A confound that predicts the label BACKWARDS is exactly as usable to a head
    as one that predicts it forwards, so separability is the quantity, not
    direction. Kept byte-identical to the other gate so the two scripts'
    numbers can be compared without a footnote.
    """
    x, y = np.asarray(x, float), np.asarray(y, int)
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    if len(np.unique(y)) < 2:
        return float("nan")
    s = pd.Series(x).rank(method="average").to_numpy()
    n1 = int(y.sum())
    n0 = len(y) - n1
    a = (s[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(max(a, 1 - a))


def find_pairs(reals_dir: str, generated_dir: str) -> list[tuple[str, str, str]]:
    """(stem, real_path, generated_path) for every stem present in both.

    Pairing on the filename stem is the `ImageID` contract both handoffs state:
    "Record `ImageID` alongside every generated file so the pairing survives."
    A generated file with no matching real is skipped rather than paired against
    an arbitrary real -- an unpaired row is a smaller problem than a wrong one.
    """
    reals = {}
    for name in os.listdir(reals_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() in IMAGE_SUFFIXES:
            reals[stem] = os.path.join(reals_dir, name)
    pairs = []
    for name in sorted(os.listdir(generated_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() in IMAGE_SUFFIXES and stem in reals:
            pairs.append((stem, reals[stem], os.path.join(generated_dir, name)))
    return pairs


def _read(path: str) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(path) as im:
        img = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return img, (img.shape[1], img.shape[0])


def _measure(path: str) -> dict | None:
    """Both proxy readings plus geometry for one file.

    `proxy_vector(img)` with no path forces the pixel fallback on every file,
    which is what makes the two classes comparable; `estimate_jpeg_quality(img,
    path)` is added separately so the path-aware column can be reported beside
    it rather than instead of it.
    """
    try:
        img, (w, h) = _read(path)
    except Exception:
        return None
    row = dict(zip(PROXY_NAMES, proxy_vector(img)))
    row = {f"pix_{k}": float(v) for k, v in row.items()}
    row["path_jpeg_quality"] = float(estimate_jpeg_quality(img, path))
    row["short_side"] = float(min(w, h))
    return row


def _one_pair(task):
    """(stem, real, generated, out_dir, geometry) -> rows for one pair.

    Module-level and taking a plain tuple so it can go to a process pool; the
    work is decode- and encode-bound and embarrassingly parallel.
    """
    stem, real_path, gen_path, out_dir, geometry = task
    real_row = _measure(real_path)
    before_row = _measure(gen_path)
    if real_row is None or before_row is None:
        return stem, None, None, None, "unreadable"
    dst = os.path.join(out_dir, f"{stem}.jpg")
    try:
        with Image.open(gen_path) as im:
            im.load()
            save_matched_to_real(im, dst, real_path, geometry)
    except ParityError as e:
        return stem, real_row, before_row, None, str(e)
    except Exception as e:
        return stem, real_row, before_row, None, f"{type(e).__name__}: {e}"
    return stem, real_row, before_row, _measure(dst), ""


def _table(reals: pd.DataFrame, gens: pd.DataFrame, columns: list[str]) -> dict:
    y = np.r_[np.zeros(len(reals), int), np.ones(len(gens), int)]
    return {c: auc(np.r_[reals[c].to_numpy(), gens[c].to_numpy()], y) for c in columns}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reals", required=True, help="directory of real JPEGs, named <ImageID>.jpg")
    ap.add_argument("--generated", required=True, help="directory of generated images, named <ImageID>.*")
    ap.add_argument("--out", required=True, help="where parity-matched copies are written")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="pairs to sample")
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--geometry", choices=GEOMETRIES, default=GEOMETRY_RESAMPLE,
                    help="how a generated image is landed on its partner's exact "
                         "size. `resample` keeps the whole frame and pays a "
                         "resampling signature; `crop` resamples nothing and pays "
                         "field of view. docs/03 §3.1 says decide this on the "
                         "number: run both over the same purchased images and "
                         "compare. Irrelevant for local generation, which can "
                         "simply render at the target size.")
    ap.add_argument("--max-auc", type=float, default=DEFAULT_MAX_AUC,
                    help="refuse (exit 1) if pixel-only jpeg_quality AUC after "
                         "parity exceeds this. Task 02 §5's own gate is 0.60.")
    a = ap.parse_args(argv)

    pairs = find_pairs(a.reals, a.generated)
    if not pairs:
        raise SystemExit(f"no ImageID stems shared between {a.reals} and {a.generated}")
    if len(pairs) > a.n:
        rng = np.random.default_rng(a.seed)
        pairs = [pairs[i] for i in np.sort(rng.choice(len(pairs), a.n, replace=False))]

    os.makedirs(a.out, exist_ok=True)
    print(f"pairs: {len(pairs)}  geometry={a.geometry}  ->  {a.out}\n", flush=True)

    tasks = [(stem, r, g, a.out, a.geometry) for stem, r, g in pairs]
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            results = list(ex.map(_one_pair, tasks, chunksize=8))
    else:
        results = [_one_pair(t) for t in tasks]

    real_rows, before_rows, after_rows, failures = [], [], [], []
    for stem, real_row, before_row, after_row, err in results:
        if err:
            failures.append((stem, err))
        if real_row is None:
            continue
        if after_row is not None:
            # Only pairs that survived parity enter ANY table. Scoring `before`
            # over a wider set than `after` would let the improvement be a
            # change of sample rather than a change of encoding.
            real_rows.append(real_row)
            before_rows.append(before_row)
            after_rows.append(after_row)

    if not after_rows:
        for stem, err in failures[:10]:
            print(f"  {stem}: {err}")
        raise SystemExit("no pair survived the parity step")

    reals = pd.DataFrame(real_rows)
    before = pd.DataFrame(before_rows)
    after = pd.DataFrame(after_rows)
    n = len(after)

    pix_cols = [f"pix_{c}" for c in PROXY_NAMES]
    cols = pix_cols + ["path_jpeg_quality", "short_side"]
    b = _table(reals, before, cols)
    f = _table(reals, after, cols)

    label = {**{f"pix_{c}": f"{c} (pixel-only)" for c in PROXY_NAMES},
             "path_jpeg_quality": "jpeg_quality (path-aware)",
             "short_side": "short_side"}

    print(f"scored on {n} pairs that survived parity"
          + (f"; {len(failures)} skipped" if failures else "") + "\n")
    print(f"  {'proxy':<28} {'before':>8} {'after':>8} {'delta':>8}")
    print(f"  {'-' * 28} {'-' * 8} {'-' * 8} {'-' * 8}")
    for c in cols:
        print(f"  {label[c]:<28} {b[c]:>8.4f} {f[c]:>8.4f} {f[c] - b[c]:>+8.4f}")

    if failures:
        print(f"\n  first skipped pairs ({len(failures)} total):")
        for stem, err in failures[:5]:
            print(f"    {stem}: {err}")

    gate = f["pix_jpeg_quality"]
    print(f"\n  gate: pixel-only jpeg_quality after parity = {gate:.4f} "
          f"(max {a.max_auc:.2f})")

    if f["short_side"] > 0.55:
        print("  WARNING: short_side did not collapse to ~0.5. Parity assigns the "
              "real's exact dimensions, so this means the pairing is wrong, not "
              "that the confound is stubborn.", file=sys.stderr)

    if not np.isfinite(gate) or gate > a.max_auc:
        print(f"\nREFUSED: {gate:.4f} > {a.max_auc:.2f}. Fix the save path before "
              "spending anything.", file=sys.stderr)
        return 1
    print("\nPASS: the save path is safe to spend against.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
