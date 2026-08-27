"""On-disk feature bank: contract #2 from spec §7.1.

Layout:
    bank/config.json     backbone, dim, n_views, n_images, seed
    bank/meta.parquet    N rows, image-level: path,label,generator,source,split
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
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from aigcdet.augment.recipes import FAMILIES, Recipe

N_VIEWS = 11          # 1 clean + 10 augmented (spec §3.1, K=10)
N_FAMILIES = len(FAMILIES)   # presence/severity are per degradation family
RECON_DIM = 12


class BankWriter:
    def __init__(self, out_dir: str, n_images: int, n_views: int, dim: int,
                 backbone: str, seed: int):
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
                         "n_views": n_views, "n_images": n_images, "seed": seed}

    def write_image(self, idx: int, meta_row: dict, feats: np.ndarray,
                     presence: np.ndarray, severity: np.ndarray,
                     proxies: np.ndarray, recipes: list[str]) -> None:
        self.feats[idx] = feats.astype(np.float16)
        self.presence[idx] = presence
        self.severity[idx] = severity
        self.proxies[idx] = proxies
        self._meta.append({"image_idx": idx, **meta_row})
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
        self._recipe_lookup = {
            (int(r.image_idx), int(r.view_idx)): r.recipe_json
            for r in self._views.itertuples()
        }

    @classmethod
    def open(cls, path: str) -> "FeatureBank":
        return cls(path)

    def recipe_json(self, image_idx: int, view_idx: int) -> str:
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
        """
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
