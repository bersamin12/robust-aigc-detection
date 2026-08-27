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
from. `config.json` records backbone, seed, view count and row count so a
mismatched pairing -- a bank built against a different manifest -- is at
least detectable rather than silently assumed correct.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from aigcdet.augment.recipes import FAMILIES

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
        self.presence = np.load(os.path.join(path, "presence.npy"), mmap_mode="r+")
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

    def check_invariants(self) -> None:
        if float(np.asarray(self.presence)[:, 0, :].sum()) != 0.0:
            raise ValueError("view 0 must be the undegraded view, but it has "
                              "non-zero degradation presence")
        if self.recon is not None and self.recon.shape[1] != self.config["n_views"]:
            raise ValueError("recon view coverage must match feats (spec §3.3)")
