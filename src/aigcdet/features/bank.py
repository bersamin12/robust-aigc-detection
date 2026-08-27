"""On-disk feature bank: contract #2 from spec §7.1.

Layout:
    bank/config.json     backbone, dim, n_views, n_images, seed, manifest_sha256
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

from aigcdet.augment.recipes import FAMILIES, Recipe

N_VIEWS = 11          # 1 clean + 10 augmented (spec §3.1, K=10)
N_FAMILIES = len(FAMILIES)   # presence/severity are per degradation family
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


class BankWriter:
    def __init__(self, out_dir: str, n_images: int, n_views: int, dim: int,
                 backbone: str, seed: int, manifest_sha256: str | None = None):
        os.makedirs(out_dir, exist_ok=True)
        self.path = out_dir
        self.n_views = n_views
        self.feats = np.lib.format.open_memmap(
            os.path.join(out_dir, "feats.npy"), mode="w+",
            dtype=np.float16, shape=(n_images, n_views, dim))
        self.presence = np.lib.format.open_memmap(
            os.path.join(out_dir, "presence.npy"), mode="w+",
            dtype=np.float32, shape=(n_images, n_views, N_FAMILIES))
        self.severity = np.lib.format.open_memmap(
            os.path.join(out_dir, "severity.npy"), mode="w+",
            dtype=np.float32, shape=(n_images, n_views, N_FAMILIES))
        self.proxies = np.lib.format.open_memmap(
            os.path.join(out_dir, "proxies.npy"), mode="w+",
            dtype=np.float32, shape=(n_images, n_views, 3))
        self._meta: list[dict] = []
        self._views: list[dict] = []
        self._config = {"backbone": backbone, "dim": dim,
                         "n_views": n_views, "n_images": n_images, "seed": seed,
                         "manifest_sha256": manifest_sha256}

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

    def close(self) -> None:
        self.feats.flush()
        self.presence.flush()
        self.severity.flush()
        self.proxies.flush()
        pd.DataFrame(self._meta).sort_values("image_idx").to_parquet(
            os.path.join(self.path, "meta.parquet"), index=False)
        pd.DataFrame(self._views).to_parquet(
            os.path.join(self.path, "views.parquet"), index=False)
        with open(os.path.join(self.path, "config.json"), "w") as f:
            json.dump(self._config, f, indent=2)


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
