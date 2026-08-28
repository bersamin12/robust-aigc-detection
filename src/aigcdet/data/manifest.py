"""The manifest is the contract every other component reads (spec §7.1).

Two kinds of column live here, and the difference is load-bearing:

* `MANIFEST_COLUMNS` are AUTHORED by whoever builds the dataset
  (`scripts/build_dataset.py`): path, label, provenance, dimensions, split.
* `MANIFEST_IDENTITY_COLUMNS` are DERIVED at the moment the manifest is
  frozen, by `write_manifest`, from the files that exist right then:
  `rel_path` and the two content digests. They are never hand-authored,
  because a hand-authored digest is a claim about bytes nobody checked.

`rel_path` is the manifest's portable identity. `path` is absolute and
therefore machine-specific: the same normalised dataset lives at
`/data/normalized/...` on the machine that built it and at
`/kaggle/input/<slug>/...` on each of the Kaggle sessions that extract
feature-bank shards from it. Every shard records a manifest fingerprint and
`FeatureBank.verify_against_manifest` refuses a bank whose fingerprint
disagrees -- so a fingerprint over absolute paths would refuse every shard
extracted anywhere but the machine that froze the manifest. The fingerprint
is taken over `rel_path` instead: identical on every machine, and still
changed by a reorder, an insertion, a deletion or a rename WITHIN the
dataset.

Relative paths are stored and the root is DERIVED from them
(`dataset_root`), not the other way round:

* Deriving `rel_path` on demand from the longest common prefix of `path`
  would be SUBSET-DEPENDENT. A shard covering only `wildfake/` rows has a
  longer common prefix than the whole manifest, and `df[df.split == "train"]`
  has another one again -- so shard fingerprints would not compose into the
  merged bank's fingerprint, which is exactly the workflow this exists to
  support. The common prefix is used ONCE, at freeze time, over the whole
  manifest, and its result is written down.
* A declared `root` COLUMN would be a second copy of a fact already implied
  by `path` and `rel_path`, and it is the one fact in a frozen file that goes
  stale the moment the dataset is copied elsewhere. The root is the part that
  is not invariant, so it is not frozen: `read_manifest(..., root=...)` (or
  the `AIGCDET_DATA_ROOT` environment variable, for entry points that take no
  root argument) rebases `path` onto wherever the data actually is, and the
  identity never moves.

Content digests answer a different question from the fingerprint, and the two
are deliberately not merged. The fingerprint answers "are these the same rows
in the same order?" -- it travels inside the frozen manifest, so it can never
tell you what is on this machine's disk. The digests answer "are the files
here the ones the manifest was frozen against?", and only
`aigcdet.data.verify.verify_images` can answer that, by recomputing them.
"""
from __future__ import annotations

import hashlib
import io
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

from aigcdet.data.splits import assign_splits

MANIFEST_COLUMNS = [
    "path",       # absolute path to the normalised PNG, on THIS machine
    "label",      # 0 = authentic, 1 = AI-generated
    "generator",  # e.g. "sdxl", "midjourney"; "" for authentic images
    "source",     # dataset of origin, e.g. "wildfake", "sid_set", "coco_val2017"
    "licence",    # licence string recorded at acquisition (spec §4.5)
    "width",
    "height",
    "split",      # "train" | "val_internal" | "heldout_generator" | "benchmark"
]

#: Derived at freeze time by `write_manifest`; see the module docstring.
#: `pixel_sha256` is "" on every row unless the manifest was frozen with
#: `digests="pixels"`.
MANIFEST_IDENTITY_COLUMNS = ["rel_path", "content_sha256", "pixel_sha256"]

SPLITS = ("train", "val_internal", "heldout_generator", "benchmark")

#: Environment variable read by `read_manifest` when no `root` is passed, so
#: entry points that take no --root flag (scripts/extract_features.py) can
#: still be pointed at a dataset mounted somewhere else.
ROOT_ENV_VAR = "AIGCDET_DATA_ROOT"

#: Distinct fake "generator families" in the synthetic fixture. Plural on
#: purpose: with a single generator name the fixture could not exercise the
#: held-out-generator split at all, and a bank built from it made Plan 2's
#: train_rung raise "bank has no val_internal rows".
DUMMY_GENERATORS = ("dummygen_a", "dummygen_b", "dummygen_c")

_HEX = set("0123456789abcdef")


# --- portable identity -----------------------------------------------------

def root_of(path: str, rel_path: str) -> str:
    """The directory `rel_path` is relative to, given the absolute `path`.

    Raises if `path` does not actually end with `rel_path` -- that pairing is
    the one thing that makes the identity checkable rather than asserted.
    """
    p = os.path.normpath(str(path))
    r = os.path.normpath(str(rel_path))
    suffix = os.sep + r
    if not p.endswith(suffix):
        raise ValueError(
            f"path {path!r} does not end with rel_path {rel_path!r}; the two "
            "columns describe different files, so this manifest's portable "
            "identity cannot be trusted")
    return p[: -len(suffix)] or os.sep


def dataset_root(df: pd.DataFrame) -> str | None:
    """The single directory every row's `path` sits under, or None if the
    frame carries no `rel_path` (an ad-hoc frame, not a frozen manifest).

    Raises if the rows imply more than one root: a manifest describes one
    materialised dataset tree, and two roots would make "the root" -- the
    thing a teammate is asked to point `read_manifest` at -- ambiguous.
    """
    if "rel_path" not in df.columns or df.empty:
        return None
    roots = {root_of(p, r) for p, r in zip(df["path"], df["rel_path"])}
    if len(roots) > 1:
        raise ValueError(
            f"manifest rows imply {len(roots)} different dataset roots, e.g. "
            f"{sorted(roots)[:3]}; a manifest must describe one tree so that "
            "it can be rebased onto one mount point")
    return roots.pop()


def derive_root(paths) -> str:
    """The deepest directory containing every path. Used ONCE, at freeze time,
    over the whole manifest -- never on a subset (see the module docstring)."""
    dirs = [os.path.dirname(os.path.abspath(str(p))) for p in paths]
    if not dirs:
        raise ValueError("cannot derive a dataset root from an empty manifest")
    return os.path.commonpath(dirs)


def relative_to_root(paths, root: str) -> list[str]:
    """Each path expressed relative to `root`, refusing any that escapes it."""
    root = os.path.abspath(str(root))
    out = []
    for p in paths:
        ap = os.path.abspath(str(p))
        rel = os.path.relpath(ap, root)
        if rel == os.pardir or rel.startswith(os.pardir + os.sep):
            raise ValueError(
                f"{p!r} is not under the dataset root {root!r}; every image in "
                "one manifest must live under one root, or the manifest cannot "
                "be rebased onto another machine's mount point")
        out.append(rel)
    return out


def rebase_manifest(df: pd.DataFrame, root: str) -> pd.DataFrame:
    """A copy of `df` whose absolute `path` points at `root` instead.

    This is what a teammate runs after attaching the published Kaggle Dataset:
    the frozen manifest's `path` column names directories that exist only on
    the machine that built it, while `rel_path` -- the identity every
    fingerprint is taken over -- is unchanged by the move.
    """
    if "rel_path" not in df.columns:
        raise ValueError(
            "cannot rebase a manifest with no rel_path column; it was frozen "
            "before portable identity existed, or it is not a manifest")
    out = df.copy()
    root = os.path.abspath(str(root))
    out["path"] = [os.path.join(root, str(r)) for r in out["rel_path"]]
    return out


# --- content digests -------------------------------------------------------

def file_digest(path: str) -> str:
    """sha256 of the file's bytes, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pixel_digest(path_or_bytes) -> str:
    """sha256 of the DECODED image: mode, size and raw pixel bytes.

    Deliberately not the file's bytes: this is the digest that survives a
    lossless re-encode and changes only when the pixels a model would
    actually see change.
    """
    src = (io.BytesIO(path_or_bytes)
           if isinstance(path_or_bytes, (bytes, bytearray))
           else path_or_bytes)
    with Image.open(src) as im:
        im.load()
        h = hashlib.sha256()
        h.update(f"{im.mode}|{im.width}x{im.height}|".encode())
        h.update(im.tobytes())
    return h.hexdigest()


def digest_row(path: str, want_pixels: bool = False) -> tuple[str, str]:
    """`(content_sha256, pixel_sha256)` for one file; the pixel digest is ""
    unless asked for.

    When both are wanted the file is read once and hashed twice -- a pixel
    digest costs a full read anyway, so the byte digest that comes with it is
    free, and a manifest frozen with `digests="pixels"` carries both.
    """
    if not want_pixels:
        return file_digest(path), ""
    with open(path, "rb") as f:
        raw = f.read()
    return hashlib.sha256(raw).hexdigest(), pixel_digest(raw)


def compute_digests(paths, want_pixels: bool = False,
                    workers: int = 8) -> list[tuple[str, str]]:
    """Digest every path, in order. Threaded: this is I/O bound, and hashlib
    and Pillow both release the GIL."""
    paths = [str(p) for p in paths]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} manifest path(s) do not exist and cannot be "
            f"digested, e.g. {missing[:3]}. The manifest is frozen against the "
            "files that exist when it is written; normalise them first.")
    if workers <= 1 or len(paths) < 2:
        return [digest_row(p, want_pixels) for p in paths]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda p: digest_row(p, want_pixels), paths))


def add_identity(df: pd.DataFrame, root: str | None = None,
                 digests: str | None = "bytes", workers: int = 8) -> pd.DataFrame:
    """A copy of `df` carrying `MANIFEST_IDENTITY_COLUMNS`.

    `digests`:
      * `"bytes"` (default) -- sha256 of each file's bytes. One streamed read
        per image, no decode. A byte digest never MISSES a pixel divergence
        (different pixels in the same file means different bytes); it only
        over-reports, flagging a lossless re-encode that left the pixels
        alone.
      * `"pixels"` -- additionally sha256 the decoded pixel array, which costs
        a full decode of every image. Opt-in, and worth paying when a byte
        mismatch has already been reported and the question is whether it
        actually changes what a model sees.
      * `None` -- no digests; `verify_images` can then only check presence.
    """
    if digests not in ("bytes", "pixels", None):
        raise ValueError(
            f"digests must be 'bytes', 'pixels' or None, got {digests!r}")
    out = df.copy()
    root = derive_root(out["path"]) if root is None else os.path.abspath(str(root))
    out["rel_path"] = relative_to_root(out["path"], root)
    if digests is None:
        out["content_sha256"] = ""
        out["pixel_sha256"] = ""
    else:
        pairs = compute_digests(out["path"], want_pixels=(digests == "pixels"),
                                workers=workers)
        out["content_sha256"] = [c for c, _ in pairs]
        out["pixel_sha256"] = [p for _, p in pairs]
    return out


# --- validation ------------------------------------------------------------

def validate_manifest(df: pd.DataFrame) -> None:
    """Fail loudly on a manifest that violates its own documented contract.

    Every one of these checks corresponds to a defect that reached the end of
    Plan 1 undetected: COCO val2017 carrying `label = 1`, a dataset name
    standing in for a generator family, and relative paths written under a
    column documented as absolute. They are cheap, and they are the last
    gate before the file is frozen — feature banks index against it
    positionally, so a manifest that is wrong is wrong for every later plan.

    The identity columns are checked only when present: they are absent on a
    frame that has not been through `write_manifest` yet, which is exactly the
    frame `build_dataset` validates just before writing it.
    """
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")

    problems: list[str] = []

    bad_labels = sorted(set(df["label"].unique()) - {0, 1})
    if bad_labels:
        problems.append(f"label must be 0 or 1, found {bad_labels}")

    bad_splits = sorted(set(df["split"].unique()) - set(SPLITS))
    if bad_splits:
        problems.append(f"split must be one of {list(SPLITS)}, found {bad_splits}")

    dupes = df["path"][df["path"].duplicated()].unique().tolist()
    if len(dupes):
        problems.append(
            f"{len(dupes)} duplicated path(s), e.g. {dupes[:3]}; rows must be "
            "one-per-image for positional indexing to be meaningful")

    relative = [p for p in df["path"] if not os.path.isabs(str(p))]
    if relative:
        problems.append(
            f"{len(relative)} relative path(s), e.g. {relative[:3]}; `path` is "
            "documented as absolute and is read from other working directories")

    problems += _identity_problems(df)

    if problems:
        raise ValueError("invalid manifest: " + "; ".join(problems))


def _identity_problems(df: pd.DataFrame) -> list[str]:
    present = [c for c in MANIFEST_IDENTITY_COLUMNS if c in df.columns]
    if not present:
        return []
    if len(present) != len(MANIFEST_IDENTITY_COLUMNS):
        return [f"identity columns are partial: has {present}, needs "
                f"{MANIFEST_IDENTITY_COLUMNS}; write_manifest writes them "
                "together and they mean nothing apart"]

    problems: list[str] = []
    rels = [str(r) for r in df["rel_path"]]
    bad = [r for r in rels
           if not r or os.path.isabs(r) or os.pardir in r.split(os.sep)]
    if bad:
        problems.append(
            f"{len(bad)} rel_path(s) are absolute, empty or contain '..', e.g. "
            f"{bad[:3]}; rel_path is the identity the fingerprint is taken "
            "over and must be a plain path inside the dataset root")
    rel_index = pd.Index(rels)
    rel_dupes = rel_index[rel_index.duplicated()].unique().tolist()
    if rel_dupes:
        problems.append(
            f"{len(rel_dupes)} duplicated rel_path(s), e.g. {rel_dupes[:3]}; "
            "two rows naming the same file inside the dataset would make the "
            "fingerprint blind to swapping them")
    if not bad:
        try:
            dataset_root(df)
        except ValueError as exc:
            problems.append(str(exc))

    for col in ("content_sha256", "pixel_sha256"):
        vals = [str(v) for v in df[col]]
        filled = [v for v in vals if v]
        if filled and len(filled) != len(vals):
            problems.append(
                f"{col} is set on {len(filled)} of {len(vals)} rows; a digest "
                "column is filled for every row or for none, or verify_images "
                "would silently skip the rows without one")
        malformed = [v for v in filled
                     if len(v) != 64 or not set(v.lower()) <= _HEX]
        if malformed:
            problems.append(
                f"{len(malformed)} malformed {col} value(s), e.g. "
                f"{malformed[:2]}; expected a 64-character sha256 hex digest")
    return problems


# --- read / write ----------------------------------------------------------

def write_manifest(df: pd.DataFrame, path: str, root: str | None = None,
                   digests: str | None = "bytes", workers: int = 8) -> None:
    """Freeze `df` to parquet, stamping the derived identity columns.

    `root` defaults to the deepest directory containing every image (see
    `derive_root`); pass it explicitly to pin a shallower root -- the
    directory a teammate will attach the published dataset at.

    The identity columns are written back onto `df` itself, in place. That is
    deliberate: the manifest is frozen HERE, so from this point on the
    caller's in-memory frame and the file on disk are the same manifest, and a
    caller that returns its frame (`build_dataset` does) would otherwise hand
    back something missing the identity its own file carries.
    """
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    stamped = add_identity(df, root=root, digests=digests, workers=workers)
    for c in MANIFEST_IDENTITY_COLUMNS:
        df[c] = stamped[c].to_numpy()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    stamped[MANIFEST_COLUMNS + MANIFEST_IDENTITY_COLUMNS].to_parquet(
        path, index=False)


def read_manifest(path: str, root: str | None = None) -> pd.DataFrame:
    """Read a frozen manifest, optionally rebasing `path` onto `root`.

    `root` falls back to `$AIGCDET_DATA_ROOT`, so an entry point with no
    --root flag still works on a machine where the dataset is mounted
    somewhere else (Kaggle attaches it under /kaggle/input/<slug>).
    """
    df = pd.read_parquet(path)
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest at {path} missing columns: {missing}")
    identity = [c for c in MANIFEST_IDENTITY_COLUMNS if c in df.columns]
    df = df[MANIFEST_COLUMNS + identity]
    if root is None:
        root = os.environ.get(ROOT_ENV_VAR) or None
    if root is not None:
        if not identity:
            raise ValueError(
                f"cannot rebase the manifest at {path} onto {root!r}: it has "
                "no rel_path column, so nothing records where each image sits "
                "inside the dataset. Rebuild it with a current write_manifest.")
        df = rebase_manifest(df, root)
    return df


def make_dummy_manifest(n: int, out_dir: str, rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic stand-in so downstream code can be built before real data lands.

    Fakes are given a mild low-pass bias so a trivial classifier can reach
    above-chance accuracy; that makes end-to-end training smoke tests meaningful.

    Fakes are spread over `DUMMY_GENERATORS`, and the splits are assigned by
    the real `assign_splits` with the last present generator held out, so a
    bank built from this fixture exercises the train / val_internal /
    heldout_generator paths rather than one uniform "train" block. n needs to
    be large enough for `val_fraction` to land at least one row in
    val_internal (a few dozen; the 500-row default is comfortable) and for
    every generator to appear at all.

    Paths recorded in the manifest are absolute, so they remain valid from any
    working directory when read by downstream tasks. Only the AUTHORED
    columns are produced: identity is stamped when the manifest is frozen, by
    `write_manifest` -- or by `add_identity` for a fixture that never goes to
    disk.

    Deterministic given `rng`: the split seed is drawn from it, not from
    global state.
    """
    out_dir_abs = os.path.abspath(out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)
    rows = []
    for i in range(n):
        label = int(i % 2)
        arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        if label == 1:
            arr = np.clip(arr.astype(np.float32) * 0.5 + 64, 0, 255).astype(np.uint8)
        p = os.path.abspath(os.path.join(out_dir_abs, f"dummy_{i:05d}.png"))
        Image.fromarray(arr).save(p)
        rows.append({
            "path": p,
            "label": label,
            "generator": DUMMY_GENERATORS[(i // 2) % len(DUMMY_GENERATORS)] if label else "",
            "source": "dummy",
            "licence": "CC0",
            "width": 64,
            "height": 64,
            "split": "",
        })
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)

    # Hold one generator out only when another remains to train on. At n=3
    # there is a single fake generator, and holding it out would leave train
    # with no fakes and (with n that small) no val_internal rows either --
    # the exact "bank has no val_internal rows" failure this fixture exists
    # to prevent, merely relocated to small n. Plans 2 and 3 call this with
    # n = 3, 4, 5 and 6.
    present = [g for g in DUMMY_GENERATORS if (df["generator"] == g).any()]
    heldout = present[-1:] if len(present) >= 2 else []
    df = assign_splits(df, heldout_generators=heldout,
                       seed=int(rng.integers(0, 2**31 - 1)))

    # val_fraction is a per-row probability, so at small n it can select
    # nothing at all. Guarantee one validation row whenever the pool has at
    # least two, deterministically (the last pool row), so every caller gets
    # a non-empty train AND a non-empty val_internal.
    pool = df.index[df["split"] != "heldout_generator"]
    if len(pool) >= 2 and not (df["split"] == "val_internal").any():
        df.loc[pool[-1], "split"] = "val_internal"
    return df
