"""On-disk feature bank: contract #2 from spec §7.1.

Layout:
    bank/config.json     backbone, dim, n_views, n_images, seed, manifest_sha256,
                         plus any writer-supplied extras (the eval bank records
                         its ordered `conditions` list here)
    bank/meta.parquet    N rows, image-level:
                         image_idx,row_id,path,label,generator,source,split
    bank/views.parquet   N*V rows: image_idx,view_idx,recipe_json
    bank/feats.npy       (N, V, D) float16   -- the ViT embedding
    bank/presence.npy    (N, V, 6) float32   -- degradation-head targets
    bank/severity.npy    (N, V, 6) float32
    bank/proxies.npy     (N, V, 3) float32   -- handcrafted h
    bank/recon.npy       (N, V, 12) float32  -- optional, attached later

Invariant: view 0 is always the undegraded view. The consistency loss and the
whole clean/degraded pairing depend on it, so it is checked, not assumed.

The bank is written once (on a GPU machine) and read many times elsewhere,
including on Kaggle, indexed positionally against the manifest it was built
from. `config.json` records backbone, seed, view count and row count, and
`meta.parquet` duplicates the manifest's own per-row columns at the same row
positions the arrays use, so `FeatureBank.verify_against_manifest` can catch
a bank built against a different (e.g. re-split) manifest with one call,
rather than requiring every caller to remember to check.

`row_id` is the row's index label in the FROZEN manifest, and it is stored
because it is load-bearing, not merely informative: `extract_bank` derives
every view's RNG from `(seed, row_id, view_idx)`, so it is the only key that
can replay a cached view's exact pixels. Before it was stored,
`recon.attach_recon_to_bank` had to recover it from the index of a manifest
the caller passed in -- making that caller's index an unverifiable,
silently-corrupting input (a `reset_index()`ed frame passed
`verify_against_manifest` and replayed every noise view wrongly).

`config.json` records `manifest_sha256`, a fingerprint of the manifest's
PORTABLE identity -- its `rel_path` column, the path of each image inside the
dataset tree -- so a bank carries an identity link back to the manifest it was
built from instead of relying on a human to hand `verify_against_manifest`
the right file.

That fingerprint used to be taken over the absolute `path` column, which made
it machine-specific and unusable for the workflow this project is built
around: one machine normalises the data and freezes the manifest, the data is
published as Kaggle Datasets, and several teammates extract shards of the bank
in Kaggle sessions where the same images live under `/kaggle/input/<slug>/`.
Every one of those shards fingerprinted differently from the manifest and from
each other, so `verify_against_manifest` refused all of them and `merge_banks`
produced a merged fingerprint that matched nothing. `meta.parquet` therefore
carries `rel_path` alongside `path`, and `config.json` carries
`manifest_root` -- the directory this bank's absolute paths were under when it
was extracted, which is per-shard information and deliberately NOT part of the
identity.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd

from aigcdet.augment.recipes import N_FAMILIES, Recipe  # noqa: F401  (re-exported)

N_VIEWS = 11          # 1 clean + 10 augmented (spec §3.1, K=10)
# N_FAMILIES is re-exported from aigcdet.augment.recipes, which owns FAMILIES.
RECON_DIM = 12


def identity_paths(df: pd.DataFrame) -> list[str]:
    """The row-identity strings of a manifest-shaped frame, in row order.

    `rel_path` -- each image's path INSIDE the dataset tree -- when the frame
    has one, which every frozen manifest and every bank's `meta.parquet`
    does. A frame that has none (an ad-hoc DataFrame built in a test, or a
    bank written without a `manifest_root`) falls back to `path`, which is
    the same string on one machine and is exactly as strong there; it is only
    not PORTABLE. The fallback is safe because it cannot make two different
    row sequences collide -- it can only make one row sequence fingerprint
    differently on a different machine, and the frames it applies to never
    leave one.
    """
    col = "rel_path" if "rel_path" in df.columns else "path"
    return [str(v) for v in df[col]]


def manifest_fingerprint(manifest_df: pd.DataFrame) -> str:
    """sha256 over the manifest's per-row identity, in row order.

    The bank is aligned to the manifest positionally, so the ordered identity
    list IS the bank's notion of "which manifest, which rows, in which order".
    A re-split, a re-filter, a re-ordering or a rename inside the dataset all
    change it; moving the whole dataset to another machine does not, because
    the identity is each image's path RELATIVE to the dataset root (see
    `aigcdet.data.manifest`).

    It deliberately does NOT cover the images' content. A manifest is frozen
    once and then copied to every machine that uses it, so any digest inside
    it is a copy of what the pixels were on the machine that froze it -- it
    can never tell you what is on this machine's disk, and folding it in here
    would only make a fingerprint mismatch ambiguous between "different rows"
    and "different pixels". Recomputing digests against the files that are
    actually present is `aigcdet.data.verify.verify_images`, and it is a
    separate check because it costs a full pass over the dataset.
    """
    h = hashlib.sha256()
    for ident in identity_paths(manifest_df):
        h.update(ident.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _rel_path(path: str, root: str | None) -> str:
    """`path` relative to the bank's `manifest_root`, or `path` itself when no
    root was recorded (see `identity_paths` for why that is safe)."""
    if not root:
        return str(path)
    return os.path.relpath(os.path.abspath(str(path)), os.path.abspath(str(root)))


#: How often `BankWriter` flushes its metadata to disk while extracting. The
#: `.npy` memmaps are preallocated at full size and survive a kill on their
#: own, but without meta/views/config they cannot be opened, so a Kaggle
#: session timeout used to lose the whole extraction.
CHECKPOINT_EVERY = 500


class BankWriter:
    """Streaming writer for one bank, checkpointed so a killed run can resume.

    `config.json` is written in `__init__`, before any image is processed, and
    `meta.parquet` / `views.parquet` are rewritten every `checkpoint_every`
    images (and again on `close`). Both parquet writes go through a temporary
    file and `os.replace`, so a kill mid-checkpoint leaves the previous
    checkpoint intact rather than a truncated file.

    With `resume=True` an existing bank directory is reopened in place: the
    memmaps are opened `r+` instead of being recreated, the already-written
    rows are read back from `meta.parquet`, and `completed` names their
    `image_idx` so the caller can skip them. The config recorded on disk must
    match the one asked for -- a resume against a different backbone, seed,
    view count, row count, manifest or `extra_config` is a different bank, not
    a continuation.
    """

    def __init__(self, out_dir: str, n_images: int, n_views: int, dim: int,
                 backbone: str, seed: int, manifest_sha256: str | None = None,
                 resume: bool = False, checkpoint_every: int = CHECKPOINT_EVERY,
                 extra_config: dict | None = None,
                 manifest_root: str | None = None):
        os.makedirs(out_dir, exist_ok=True)
        self.path = out_dir
        self.n_views = n_views
        self.checkpoint_every = max(1, int(checkpoint_every))
        # `manifest_root` is where this shard's images were mounted while it
        # was extracted. It is recorded so `write_image` can store each row's
        # path relative to it -- the identity that survives the move to
        # another machine -- and so a human debugging a shard can see which
        # copy of the data it came from. It is per-shard, never part of the
        # bank's identity: two shards of one bank extracted on two Kaggle
        # sessions legitimately have different roots.
        self.manifest_root = manifest_root
        self._config = {"backbone": backbone, "dim": dim,
                         "n_views": n_views, "n_images": n_images, "seed": seed,
                         "manifest_sha256": manifest_sha256,
                         "manifest_root": manifest_root}
        # `extra_config` is merged into `_config` -- NOT written separately
        # after close() -- precisely so it takes part in the resume equality
        # check below. `aigcdet.eval.grid` uses it to record the eval bank's
        # ordered condition list, and a resume against a DIFFERENT condition
        # list must be refused: the view axis would mean two different things
        # in one bank. Anything written outside `_config` would be silently
        # accepted as a continuation.
        if extra_config:
            clashing = sorted(set(extra_config) & set(self._config))
            if clashing:
                raise ValueError(
                    f"extra_config may not shadow the reserved bank config keys "
                    f"{clashing}; pass them through the named parameters instead")
            self._config.update(extra_config)

        cfg_path = os.path.join(out_dir, "config.json")
        resuming = resume and os.path.exists(cfg_path)
        if resuming:
            with open(cfg_path) as f:
                on_disk = json.load(f)
            # The union of both key sets, not just the REQUESTED keys: "a
            # different condition list" includes "no condition list". Iterating
            # `self._config` alone accepted a resume that simply omitted
            # `extra_config`, and the next checkpoint then rewrote config.json
            # WITHOUT it -- erasing an eval bank's `conditions` and leaving
            # `score_grid` to reject the bank as "not an eval bank".
            _MISSING = object()
            differing = {k: (on_disk.get(k, _MISSING), self._config.get(k, _MISSING))
                         for k in set(on_disk) | set(self._config)
                         if on_disk.get(k, _MISSING) != self._config.get(k, _MISSING)}
            if differing:
                raise ValueError(
                    f"cannot resume the bank at {out_dir}: its config.json "
                    f"disagrees with this call on {differing} (on_disk, requested). "
                    "A resume must continue the SAME extraction; extract to a new "
                    "directory instead.")
        elif resume:
            # Nothing to resume from -- a fresh start, not an error: this is
            # what the first session of a resumable run does.
            pass

        mode = "r+" if resuming else "w+"
        self.feats = self._memmap("feats.npy", mode, np.float16, (n_images, n_views, dim))
        self.presence = self._memmap("presence.npy", mode, np.float32,
                                      (n_images, n_views, N_FAMILIES))
        self.severity = self._memmap("severity.npy", mode, np.float32,
                                      (n_images, n_views, N_FAMILIES))
        self.proxies = self._memmap("proxies.npy", mode, np.float32,
                                     (n_images, n_views, 3))

        self._meta: list[dict] = []
        self._views: list[dict] = []
        self.completed: set[int] = set()
        if resuming:
            meta_path = os.path.join(out_dir, "meta.parquet")
            views_path = os.path.join(out_dir, "views.parquet")
            if os.path.exists(meta_path) and os.path.exists(views_path):
                self._meta = pd.read_parquet(meta_path).to_dict("records")
                self._views = pd.read_parquet(views_path).to_dict("records")
                self.completed = {int(r["image_idx"]) for r in self._meta}
        else:
            self._write_config()

    def _memmap(self, name: str, mode: str, dtype, shape):
        path = os.path.join(self.path, name)
        if mode == "r+":
            arr = np.lib.format.open_memmap(path, mode="r+")
            if arr.shape != shape or arr.dtype != dtype:
                raise ValueError(
                    f"cannot resume: {name} on disk is {arr.shape} {arr.dtype}, "
                    f"expected {shape} {np.dtype(dtype)}")
            return arr
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)

    def _write_config(self) -> None:
        with open(os.path.join(self.path, "config.json"), "w") as f:
            json.dump(self._config, f, indent=2)

    def _write_parquet(self, rows: list[dict], name: str, sort_by: str | None) -> None:
        """Write atomically: a kill mid-write must leave the last good
        checkpoint, not a truncated parquet file."""
        df = pd.DataFrame(rows)
        if sort_by is not None and not df.empty:
            df = df.sort_values(sort_by)
        final = os.path.join(self.path, name)
        tmp = final + ".tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, final)

    def checkpoint(self) -> None:
        """Flush the arrays and rewrite the metadata, so everything written so
        far is a complete, openable bank."""
        self.feats.flush()
        self.presence.flush()
        self.severity.flush()
        self.proxies.flush()
        self._write_config()
        self._write_parquet(self._meta, "meta.parquet", "image_idx")
        self._write_parquet(self._views, "views.parquet", None)

    def write_image(self, idx: int, meta_row: dict, feats: np.ndarray,
                     presence: np.ndarray, severity: np.ndarray,
                     proxies: np.ndarray, recipes: list[str],
                     row_id: int | None = None) -> None:
        """Write one image's views. `row_id` is its index label in the frozen
        manifest -- the RNG key every view's pixels are reproducible from. It
        defaults to `idx`, which is correct only for a bank built from a
        manifest with a contiguous 0..n-1 index; `extract_bank` always passes
        the real label.

        `rel_path` is derived from `manifest_root` and stored alongside
        `path`, unless `meta_row` already carries one -- which it does when
        `merge_banks` replays a shard's rows, and that shard's own root is the
        one that made its `rel_path` right. Re-deriving it here from the
        merged writer's root would overwrite a correct identity with a wrong
        one."""
        self.feats[idx] = feats.astype(np.float16)
        self.presence[idx] = presence
        self.severity[idx] = severity
        self.proxies[idx] = proxies
        row = dict(meta_row)
        if "path" in row:
            row.setdefault("rel_path", _rel_path(row["path"], self.manifest_root))
        self._meta.append({"image_idx": idx,
                            "row_id": idx if row_id is None else int(row_id),
                            **row})
        for v, rj in enumerate(recipes):
            self._views.append({"image_idx": idx, "view_idx": v, "recipe_json": rj})
        self.completed.add(idx)
        if len(self._meta) % self.checkpoint_every == 0:
            self.checkpoint()

    def close(self) -> None:
        self.checkpoint()


class FeatureBank:
    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "config.json")) as f:
            self.config = json.load(f)
        self.meta = pd.read_parquet(os.path.join(path, "meta.parquet"))
        self._views = pd.read_parquet(os.path.join(path, "views.parquet"))
        self.feats = np.load(os.path.join(path, "feats.npy"), mmap_mode="r")
        self.presence = np.load(os.path.join(path, "presence.npy"), mmap_mode="r")
        self.severity = np.load(os.path.join(path, "severity.npy"), mmap_mode="r")
        self.proxies = np.load(os.path.join(path, "proxies.npy"), mmap_mode="r")
        rp = os.path.join(path, "recon.npy")
        self.recon = np.load(rp, mmap_mode="r") if os.path.exists(rp) else None
        self._recipe_lookup: dict[tuple[int, int], str] | None = None

    @classmethod
    def open(cls, path: str) -> "FeatureBank":
        return cls(path)

    @property
    def row_ids(self) -> np.ndarray:
        """Each row's index label in the frozen manifest, in bank row order.

        This is the RNG key component every view's pixels were derived from
        (`extract_bank`), so replaying a cached view must read it from HERE and
        never from a manifest the caller happens to pass in.
        """
        if "row_id" not in self.meta.columns:
            raise ValueError(
                f"bank at {self.path} has no row_id column in meta.parquet; it "
                "predates row_id being stored and its views cannot be replayed "
                "reliably. Re-extract it.")
        return self.meta["row_id"].to_numpy()

    @property
    def rel_paths(self) -> list[str]:
        """Each row's path inside the dataset tree, in bank row order.

        This is the bank's half of the identity `manifest_sha256` is taken
        over, and the only path form that means the same thing in a shard
        extracted on Kaggle and in the manifest on the machine that froze it.
        """
        meta = self.meta.sort_values("image_idx")
        return identity_paths(meta)

    def recipe_json(self, image_idx: int, view_idx: int) -> str:
        """The recipe JSON for one view.

        The (image_idx, view_idx) -> json dict is built on FIRST use, not in
        `__init__`: at 100k images x 11 views it is a 1.1M-entry dict costing
        ~10 s and several hundred MB, and `train_rung` -- which opens the bank
        for every rung -- only ever needs `recipe_json(i, 0)` via
        `check_invariants`. Callers that never ask for a recipe now pay
        nothing.
        """
        if self._recipe_lookup is None:
            self._recipe_lookup = {
                (int(r.image_idx), int(r.view_idx)): r.recipe_json
                for r in self._views.itertuples()
            }
        return self._recipe_lookup[(image_idx, view_idx)]

    def attach_recon(self, arr: np.ndarray) -> None:
        expected = (len(self.meta), self.config["n_views"], RECON_DIM)
        if arr.shape != expected:
            raise ValueError(f"recon must be {expected}, got {arr.shape}")
        np.save(os.path.join(self.path, "recon.npy"), arr.astype(np.float32))
        self.recon = np.load(os.path.join(self.path, "recon.npy"), mmap_mode="r")

    def verify_against_manifest(self, manifest_df: pd.DataFrame) -> None:
        """Check the bank's rows are still positionally aligned with `manifest_df`.

        Rows are positional: array index i is manifest row i. A re-split after
        the bank was written silently misaligns labels against cached
        features (spec's "manifest is frozen once written" constraint) and
        produces a slightly worse number nobody can explain. This makes that
        failure loud instead of requiring a caller to think to check it.

        When the bank recorded a `manifest_sha256`, that fingerprint is checked
        first: it fails fast, and it names the mismatch as an identity problem
        ("this is not the manifest this bank was built from") rather than as a
        single misaligned row.

        Both checks compare PORTABLE identity (`rel_path`), not absolute
        paths, so a shard extracted on Kaggle verifies against the manifest on
        the machine that froze it. Whether the pixels at those paths are still
        the ones the manifest was frozen against is a different question, and
        an absolute path could not answer it either -- that is
        `aigcdet.data.verify.verify_images`.
        """
        recorded = self.config.get("manifest_sha256")
        if recorded is not None:
            actual = manifest_fingerprint(manifest_df)
            if actual != recorded:
                raise ValueError(
                    "this is not the manifest the bank was built from: bank "
                    f"config.json records manifest_sha256={recorded[:16]}..., "
                    f"the supplied manifest fingerprints to {actual[:16]}... "
                    "(the fingerprint covers each row's path inside the "
                    "dataset, in row order, so a re-split, re-filter, re-order "
                    "or rename all change it -- moving the dataset does not)")
        if len(manifest_df) != len(self.meta):
            raise ValueError(
                f"manifest has {len(manifest_df)} rows but bank has "
                f"{len(self.meta)} rows -- bank is not aligned with this manifest")
        # Compare like with like. A bank whose rows hold RELATIVE identity was
        # written with a `manifest_root` (or merged from shards that were), and
        # is compared against the manifest's identity. A bank whose rows hold
        # absolute paths -- a writer that passed no root, which every producer
        # in this repo now does pass, leaving ad-hoc frames and older banks --
        # is compared on absolute paths: that bank is simply not portable, and
        # comparing its paths against a manifest's rel_path would report every
        # row misaligned when nothing at all is wrong.
        bank_ids = self.rel_paths
        if bank_ids and not any(os.path.isabs(r) for r in bank_ids):
            manifest_ids = identity_paths(manifest_df)
        else:
            manifest_ids = [str(p) for p in manifest_df["path"]]
            bank_ids = [str(p) for p in
                        self.meta.sort_values("image_idx")["path"]]
        for i in range(len(manifest_ids)):
            if manifest_ids[i] != bank_ids[i]:
                raise ValueError(
                    f"manifest/bank row {i} misaligned: manifest path "
                    f"{manifest_ids[i]!r} != bank path {bank_ids[i]!r}")

    def check_invariants(self) -> None:
        # Every feature finite, checked FIRST because it is the cheapest check
        # here (one streamed pass over feats.npy, seconds on a 3 GB bank) and
        # the only one that looks at a feature value at all. 2026-08-29's
        # dinov3l bank was 131,116 rows of NaN -- a float16 overflow in the
        # backbone -- and it passed every other check in this method, the
        # chained job's row count and the trainer; sklearn's roc_auc was the
        # first thing to look at a value, thirty epochs later.
        bad_rows = non_finite_rows(self.feats)
        if len(bad_rows):
            raise ValueError(
                f"feats.npy holds non-finite values in {len(bad_rows)} of "
                f"{len(self.meta)} rows (first: {bad_rows[:5].tolist()}). A "
                "bank with NaN/inf features is unusable and must be "
                "re-extracted; the usual cause is the backbone overflowing "
                "its dtype (DINOv3-L needs bfloat16 -- see "
                "aigcdet.features.backbones.BackboneSpec.dtype)")
        # row_id keys every view's RNG, so a duplicate would mean two images
        # sharing one replay key -- the same class of defect extract_bank's
        # duplicated-index guard exists to prevent, checked again on the
        # written artefact.
        ids = self.row_ids
        if len(np.unique(ids)) != len(ids):
            raise ValueError(
                "meta.parquet has duplicate row_id values; each row's views are "
                "keyed on (seed, row_id, view_idx), so duplicates make two "
                "images replay to the same pixels")
        if float(np.asarray(self.presence)[:, 0, :].sum()) != 0.0:
            raise ValueError("view 0 must be the undegraded view, but it has "
                              "non-zero degradation presence")
        # presence and recipe_json are two independent encodings of the same
        # fact (what happened to a view); checking presence alone would miss
        # the two falling out of sync, so cross-check the recipe against it.
        n_images = len(self.meta)
        for i in range(n_images):
            recipe = Recipe.from_json(self.recipe_json(i, 0))
            if recipe.ops != ():
                raise ValueError(
                    f"view 0 must be the undegraded view, but image {i}'s "
                    f"recipe_json encodes a non-empty recipe: {recipe.ops!r}")
        if self.recon is not None and self.recon.shape[1] != self.config["n_views"]:
            # Unreachable through the public API -- attach_recon already
            # enforces this shape at write time -- so this guards against
            # external corruption of recon.npy, not a normal API path.
            raise ValueError("recon view coverage must match feats (spec §3.3)")


def non_finite_rows(feats, block: int = 4096) -> np.ndarray:
    """Indices of the rows of `feats` (N, V, D) holding any NaN or inf.

    Scanned in blocks of `block` rows so a memmapped multi-GB bank is never
    materialised at once (4096 x 11 x 1024 float16 is ~90 MB per block).
    """
    out: list[int] = []
    n = int(feats.shape[0])
    for start in range(0, n, block):
        blk = np.asarray(feats[start:start + block])
        finite = np.isfinite(blk).reshape(len(blk), -1).all(axis=1)
        out.extend((np.flatnonzero(~finite) + start).tolist())
    return np.asarray(out, dtype=np.int64)


#: Config keys every shard of one logical bank must agree on. `n_images` and
#: `manifest_sha256` are deliberately absent: shards cover different rows, so
#: those two are expected to differ and are recomputed for the merged bank.
#: Any key NOT in this tuple and not in `_MERGE_PER_SHARD` came from a
#: writer's `extra_config` and is treated the same way as the entries here --
#: it must agree across shards, and it is carried into the merged bank.
_MERGE_MUST_MATCH = ("backbone", "dim", "n_views", "seed")
#: Config keys that legitimately differ between shards of one bank.
#: `manifest_root` is here for the headline case: five teammates extract five
#: shards of one bank from five copies of one Kaggle Dataset, mounted at five
#: different paths. Requiring the roots to agree would refuse exactly the
#: merge this project exists to perform.
_MERGE_PER_SHARD = ("n_images", "manifest_sha256", "manifest_root")


def _extra_config(config: dict) -> dict:
    """The keys a `BankWriter` was given as `extra_config`, recovered from a
    written bank so `merge_banks` can carry them into the merged one. Without
    this, merging eval shards would drop `config["conditions"]` and the merged
    bank would no longer know what its view axis means.

    Note this treats EVERY unrecognised config key as a must-match extra. That
    is correct for both extras in the project today -- `conditions` (eval
    banks: the view axis would mean two different things) and
    `exclude_families` (training banks: a LOTO shard merged into a non-LOTO
    bank silently contaminates the A3-LOTO rung) -- but a future writer that
    recorded a per-run key -- a timestamp, a git sha, a hostname -- would make
    `merge_banks` refuse legitimate shards with a confusing "not part of the
    same bank" message. Add such a key to `_MERGE_PER_SHARD` when it appears.
    """
    known = set(_MERGE_MUST_MATCH) | set(_MERGE_PER_SHARD)
    return {k: v for k, v in config.items() if k not in known}


def merge_banks(bank_dirs: list[str], out_dir: str) -> str:
    """Concatenate shard banks into one bank at `out_dir`, in the given order.

    Sharding is safe by construction in this project -- every view's pixels
    depend only on `(seed, row_id, view_idx)`, never on which shard or session
    processed the image (see `aigcdet.features.extract`) -- but nothing could
    put the shards back together. This is that missing half.

    Refuses to merge shards that disagree on `backbone`, `dim`, `n_views` or
    `seed`, and refuses shards whose `row_id` sets overlap: an overlap means
    the same physical image appears twice, which breaks the bank's
    one-row-per-image contract and double-counts it in every split.

    `recon.npy` must be present on all shards or none; merging a bank where
    only some rows have reconstruction features would make A3-vs-A4 a
    comparison across different view coverage (spec §3.3).
    """
    if not bank_dirs:
        raise ValueError("merge_banks needs at least one bank directory")

    banks = [FeatureBank.open(d) for d in bank_dirs]
    ref = banks[0]
    ref_extra = _extra_config(ref.config)
    for d, b in zip(bank_dirs[1:], banks[1:]):
        differing = {k: (ref.config[k], b.config[k]) for k in _MERGE_MUST_MATCH
                     if ref.config[k] != b.config[k]}
        b_extra = _extra_config(b.config)
        differing.update({k: (ref_extra.get(k), b_extra.get(k))
                          for k in sorted(set(ref_extra) | set(b_extra))
                          if ref_extra.get(k) != b_extra.get(k)})
        if differing:
            raise ValueError(
                f"shard {d} is not part of the same bank as {bank_dirs[0]}: "
                f"{differing} (first, this one)")

    row_ids = np.concatenate([b.row_ids for b in banks])
    uniq, counts = np.unique(row_ids, return_counts=True)
    if len(uniq) != len(row_ids):
        clashing = uniq[counts > 1][:5].tolist()
        raise ValueError(
            f"shards overlap: {int((counts > 1).sum())} row_id(s) appear in more "
            f"than one shard, e.g. {clashing}. Each image must be extracted "
            "exactly once, or it is double-counted in every split.")

    has_recon = [b.recon is not None for b in banks]
    if any(has_recon) and not all(has_recon):
        missing = [d for d, h in zip(bank_dirs, has_recon) if not h]
        raise ValueError(
            f"some shards have recon.npy and some do not (missing: {missing}); "
            "attach it to every shard, or to none, before merging")

    n_total = sum(len(b.meta) for b in banks)
    # Over rel_path, not path: the shards may have been extracted from copies
    # of the dataset mounted at different absolute roots, and the merged bank
    # must fingerprint to what the ONE frozen manifest fingerprints to.
    merged_ids = pd.DataFrame(
        {"rel_path": [r for b in banks for r in b.rel_paths]})
    roots = {b.config.get("manifest_root") for b in banks}
    writer = BankWriter(out_dir, n_total, ref.config["n_views"], ref.config["dim"],
                        ref.config["backbone"], ref.config["seed"],
                        manifest_sha256=manifest_fingerprint(merged_ids),
                        checkpoint_every=max(1, n_total),
                        extra_config=ref_extra,
                        # Only meaningful if every shard came from the same
                        # mount; otherwise the merged bank's absolute paths
                        # genuinely have no single root, and its rows carry
                        # their own rel_path anyway.
                        manifest_root=roots.pop() if len(roots) == 1 else None)

    out_idx = 0
    for b in banks:
        n_views = b.config["n_views"]
        for i in range(len(b.meta)):
            row = b.meta.iloc[i].to_dict()
            row_id = int(row.pop("row_id"))
            row.pop("image_idx")
            writer.write_image(
                out_idx, row,
                feats=np.asarray(b.feats[i]),
                presence=np.asarray(b.presence[i]),
                severity=np.asarray(b.severity[i]),
                proxies=np.asarray(b.proxies[i]),
                recipes=[b.recipe_json(i, v) for v in range(n_views)],
                row_id=row_id)
            out_idx += 1
    writer.close()

    merged = FeatureBank.open(out_dir)
    if all(has_recon):
        merged.attach_recon(
            np.concatenate([np.asarray(b.recon) for b in banks], axis=0))
    merged.check_invariants()
    return out_dir
