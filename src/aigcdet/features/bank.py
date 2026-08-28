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
`path` column, so a bank carries an identity link back to the manifest it was
built from instead of relying on a human to hand `verify_against_manifest`
the right file.
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


def manifest_fingerprint(manifest_df: pd.DataFrame) -> str:
    """sha256 over the manifest's `path` column, in row order.

    The bank is aligned to the manifest positionally, so the ordered path list
    IS the bank's notion of "which manifest, which rows, in which order". A
    re-split, a re-filter or a re-ordering all change it.
    """
    h = hashlib.sha256()
    for path in manifest_df["path"]:
        h.update(str(path).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


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
                 extra_config: dict | None = None):
        os.makedirs(out_dir, exist_ok=True)
        self.path = out_dir
        self.n_views = n_views
        self.checkpoint_every = max(1, int(checkpoint_every))
        self._config = {"backbone": backbone, "dim": dim,
                         "n_views": n_views, "n_images": n_images, "seed": seed,
                         "manifest_sha256": manifest_sha256}
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
            differing = {k: (on_disk.get(k), v) for k, v in self._config.items()
                         if on_disk.get(k) != v}
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
        the real label."""
        self.feats[idx] = feats.astype(np.float16)
        self.presence[idx] = presence
        self.severity[idx] = severity
        self.proxies[idx] = proxies
        self._meta.append({"image_idx": idx,
                            "row_id": idx if row_id is None else int(row_id),
                            **meta_row})
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
        """
        recorded = self.config.get("manifest_sha256")
        if recorded is not None:
            actual = manifest_fingerprint(manifest_df)
            if actual != recorded:
                raise ValueError(
                    "this is not the manifest the bank was built from: bank "
                    f"config.json records manifest_sha256={recorded[:16]}..., "
                    f"the supplied manifest fingerprints to {actual[:16]}... "
                    "(the fingerprint covers the path column, in row order, so "
                    "a re-split, re-filter or re-order all change it)")
        if len(manifest_df) != len(self.meta):
            raise ValueError(
                f"manifest has {len(manifest_df)} rows but bank has "
                f"{len(self.meta)} rows -- bank is not aligned with this manifest")
        meta_sorted = self.meta.sort_values("image_idx").reset_index(drop=True)
        manifest_paths = manifest_df["path"].reset_index(drop=True)
        for i in range(len(manifest_df)):
            m_path = manifest_paths.iloc[i]
            b_path = meta_sorted.iloc[i]["path"]
            if m_path != b_path:
                raise ValueError(
                    f"manifest/bank row {i} misaligned: manifest path "
                    f"{m_path!r} != bank path {b_path!r}")

    def check_invariants(self) -> None:
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


#: Config keys every shard of one logical bank must agree on. `n_images` and
#: `manifest_sha256` are deliberately absent: shards cover different rows, so
#: those two are expected to differ and are recomputed for the merged bank.
#: Any key NOT in this tuple and not in `_MERGE_PER_SHARD` came from a
#: writer's `extra_config` and is treated the same way as the entries here --
#: it must agree across shards, and it is carried into the merged bank.
_MERGE_MUST_MATCH = ("backbone", "dim", "n_views", "seed")
#: Config keys that legitimately differ between shards of one bank.
_MERGE_PER_SHARD = ("n_images", "manifest_sha256")


def _extra_config(config: dict) -> dict:
    """The keys a `BankWriter` was given as `extra_config`, recovered from a
    written bank so `merge_banks` can carry them into the merged one. Without
    this, merging eval shards would drop `config["conditions"]` and the merged
    bank would no longer know what its view axis means."""
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
    merged_paths = pd.DataFrame(
        {"path": np.concatenate([b.meta["path"].to_numpy() for b in banks])})
    writer = BankWriter(out_dir, n_total, ref.config["n_views"], ref.config["dim"],
                        ref.config["backbone"], ref.config["seed"],
                        manifest_sha256=manifest_fingerprint(merged_paths),
                        checkpoint_every=max(1, n_total),
                        extra_config=ref_extra)

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
