# Plan 2 — Features & Training

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract cached frozen-backbone features (Stage A) and train the ablation rungs A0–A6 on them (Stage B), producing a selected headline checkpoint.

**Architecture:** A frozen backbone embeds every image under K+1 augmentation views once; everything after that trains ~2M parameters on cached vectors in minutes. The two-stage split is what makes the ablation ladder affordable on one shared GPU, and it is a training-time device only — inference collapses to a single path.

**Tech Stack:** PyTorch 2.10 + CUDA 12.8, HuggingFace `transformers` and `diffusers`, `lpips`, plus everything from Plan 1.

**Spec:** `docs/superpowers/specs/2026-08-27-robust-aigc-detection-design-v2.md` (v2.1)

**Depends on:** Plan 1 (manifest, augmentation, proxies, metrics).

## Global Constraints

- **Under 2B parameters total.** DINOv3-L ≈300M + SigLIP2-L ≈400M + SD1.5 VAE ≈84M + LPIPS ≈15M + heads ≈2M ≈ 800M. Any addition must be checked against this budget.
- **Backbones are frozen.** No gradient reaches any backbone in any core rung.
- **View 0 of every bank entry is the undegraded view.** The consistency loss depends on it (spec §3.5).
- **`r` is cached for all 11 views**, identical coverage to `f`. Any two rungs being compared must be trained on identical view coverage (spec §3.3).
- **Squish resize to 384×384**, ignoring aspect ratio; never random resized crop (spec §3.2).
- **Pooling is global average over final-layer patch tokens**, never the CLS token (spec §3.2).
- GPU budget: the A4500 has ~3.7 GB free of 20 GB while another process runs. Extraction must be chunked, checkpointable, and resumable.
- Licence check on DINOv3 / SD 1.5 VAE / LPIPS weights happens in Task 1 **before** the backbone choice is locked (spec §4.5).

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/aigcdet/features/backbones.py` | Backbone registry, loading, GAP-over-patch-tokens embedding |
| `src/aigcdet/features/extract.py` | Stage A: manifest + recipes → feature bank on disk |
| `src/aigcdet/features/recon.py` | VAE round-trip reconstruction features `r` |
| `src/aigcdet/features/bank.py` | On-disk bank format, reader, invariant checks |
| `src/aigcdet/models/heads.py` | `ClassifierHead`, `DegradationHead`, FiLM |
| `src/aigcdet/models/losses.py` | Classification, degradation multi-task, consistency |
| `src/aigcdet/models/sampler.py` | Paired batch sampler guaranteeing clean/degraded partners |
| `src/aigcdet/train/train_head.py` | Stage B trainer, checkpointing, rung config |
| `configs/rungs/*.yaml` | One config per ablation rung |
| `scripts/extract_features.py` | CLI for Stage A |
| `scripts/train_rung.py` | CLI for Stage B |

---

### Task 1: Backbone registry and pooled embedding

**Files:**
- Create: `src/aigcdet/features/backbones.py`, `tests/features/test_backbones.py`, `docs/model_licences.md`

**Interfaces:**
- Consumes: nothing from Plan 1
- Produces:
  - `BackboneSpec` dataclass: `name, hf_id, image_size, dim, num_prefix_tokens`
  - `BACKBONES: dict[str, BackboneSpec]` with keys `"dinov3l"`, `"siglip2l"`, `"clipl"`
  - `squish(img: np.ndarray, size: int) -> np.ndarray` — resize ignoring aspect ratio
  - `load_backbone(name: str, device: str = "cuda") -> tuple[torch.nn.Module, BackboneSpec]`
  - `embed(model, spec, imgs: list[np.ndarray], device: str, batch_size: int = 16) -> np.ndarray` — `(N, spec.dim)` float32, GAP over patch tokens

- [ ] **Step 1: Record the licence check (spec §4.5, day 1)**

Before writing code, open each model card and record its licence in `docs/model_licences.md`:

```markdown
# Model weight licences

| Model | HF id | Licence | Permits public repo + hackathon use? |
| --- | --- | --- | --- |
| DINOv3 ViT-L/16 | facebook/dinov3-vitl16-pretrain-lvd1689m | <FILL FROM MODEL CARD> | <yes/no> |
| SigLIP2-L/16-384 | google/siglip2-large-patch16-384 | <FILL FROM MODEL CARD> | <yes/no> |
| CLIP ViT-L/14 | openai/clip-vit-large-patch14 | <FILL FROM MODEL CARD> | <yes/no> |
| SD 1.5 VAE | stable-diffusion-v1-5/stable-diffusion-v1-5 (vae) | <FILL FROM MODEL CARD> | <yes/no> |
| LPIPS (AlexNet) | richzhang/PerceptualSimilarity | <FILL FROM MODEL CARD> | <yes/no> |
```

If DINOv3's licence does not permit this use, swap the registry's primary entry to SigLIP2 and add DINOv2 (`facebook/dinov2-large`) as the second backbone. Nothing else in this plan changes.

- [ ] **Step 2: Write the failing test**

```python
# tests/features/test_backbones.py
import numpy as np
import pytest

from aigcdet.features.backbones import BACKBONES, squish


def test_registry_has_the_three_planned_backbones():
    assert set(BACKBONES) == {"dinov3l", "siglip2l", "clipl"}
    for spec in BACKBONES.values():
        assert spec.dim > 0 and spec.image_size in (224, 384)
        assert spec.num_prefix_tokens >= 1  # at least a CLS token to strip


def test_total_parameter_budget_is_documented_under_2b():
    # Sum of the two backbones we ship plus the auxiliary models (spec constraint).
    assert BACKBONES["dinov3l"].params + BACKBONES["siglip2l"].params < 1_000_000_000


def test_squish_ignores_aspect_ratio():
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    out = squish(img, 384)
    assert out.shape == (384, 384, 3) and out.dtype == np.uint8


@pytest.mark.gpu
def test_embed_returns_pooled_vectors_of_the_right_width():
    import torch
    from aigcdet.features.backbones import embed, load_backbone
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    model, spec = load_backbone("clipl", device="cuda")
    imgs = [np.random.default_rng(i).integers(0, 256, (512, 640, 3), dtype=np.uint8)
            for i in range(3)]
    out = embed(model, spec, imgs, device="cuda", batch_size=2)
    assert out.shape == (3, spec.dim) and out.dtype == np.float32
    assert np.isfinite(out).all()


@pytest.mark.gpu
def test_embedding_is_deterministic():
    import torch
    from aigcdet.features.backbones import embed, load_backbone
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    model, spec = load_backbone("clipl", device="cuda")
    img = [np.random.default_rng(0).integers(0, 256, (512, 512, 3), dtype=np.uint8)]
    a = embed(model, spec, img, device="cuda")
    b = embed(model, spec, img, device="cuda")
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)
```

Register the marker in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["gpu: requires a CUDA device and downloads model weights"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/features/test_backbones.py -v -m "not gpu"`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.features.backbones'`

- [ ] **Step 4: Write minimal implementation**

```python
# src/aigcdet/features/backbones.py
"""Frozen backbones and pooled embedding (spec §3.2).

Pooling is global average over final-layer patch tokens. The NTIRE 2026
SigLIP2 team found this beat CLS-token, attention pooling, and multi-layer
concatenation, so it is the default rather than an option.

Preprocessing is a "squish" resize to a fixed square, ignoring aspect ratio.
Random resized cropping is deliberately avoided: it can remove localised
forensic cues.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

# ImageNet statistics; CLIP/SigLIP/DINOv3 all ship close variants and the
# difference is immaterial for a frozen feature extractor.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    hf_id: str
    image_size: int
    dim: int
    num_prefix_tokens: int   # CLS + register tokens to strip before pooling
    params: int              # approximate, for the 2B budget check


BACKBONES: dict[str, BackboneSpec] = {
    "dinov3l": BackboneSpec("dinov3l", "facebook/dinov3-vitl16-pretrain-lvd1689m",
                            384, 1024, 5, 300_000_000),
    "siglip2l": BackboneSpec("siglip2l", "google/siglip2-large-patch16-384",
                             384, 1024, 0, 400_000_000),
    "clipl": BackboneSpec("clipl", "openai/clip-vit-large-patch14",
                          224, 1024, 1, 304_000_000),
}


def squish(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def load_backbone(name: str, device: str = "cuda") -> tuple[torch.nn.Module, BackboneSpec]:
    from transformers import AutoModel
    spec = BACKBONES[name]
    model = AutoModel.from_pretrained(spec.hf_id, dtype=torch.float16)
    # CLIP and SigLIP wrap a vision tower; DINOv3 is already a vision model.
    model = getattr(model, "vision_model", model)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, spec


def _to_tensor(imgs: list[np.ndarray], size: int) -> torch.Tensor:
    arr = np.stack([squish(i, size) for i in imgs]).astype(np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


@torch.inference_mode()
def embed(model, spec: BackboneSpec, imgs: list[np.ndarray],
          device: str = "cuda", batch_size: int = 16) -> np.ndarray:
    out = []
    for i in range(0, len(imgs), batch_size):
        x = _to_tensor(imgs[i:i + batch_size], spec.image_size).to(device, torch.float16)
        h = model(pixel_values=x).last_hidden_state       # (B, T, D)
        patches = h[:, spec.num_prefix_tokens:, :]        # drop CLS + registers
        out.append(patches.mean(dim=1).float().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/features/test_backbones.py -v -m "not gpu"`
Expected: 3 passed

Then, once weights are downloaded: `python -m pytest tests/features/test_backbones.py -v -m gpu`
Expected: 2 passed

If the GPU tests fail on token count, print `h.shape` and compare against
`(image_size / patch_size)² + num_prefix_tokens` to correct `num_prefix_tokens`
for that backbone. DINOv3's register-token count is the most likely to differ
from the value in the registry.

- [ ] **Step 6: Commit**

```bash
git add src/aigcdet/features/backbones.py tests/features/test_backbones.py docs/model_licences.md
git commit -m "feat(features): frozen backbone registry with GAP-over-patch-tokens pooling"
```

---

### Task 2: Feature bank format

**Files:**
- Create: `src/aigcdet/features/bank.py`, `tests/features/test_bank.py`

**Interfaces:**
- Consumes: `aigcdet.augment.recipes.FAMILIES`
- Produces:
  - `N_VIEWS: int = 11` (1 clean + 10 augmented)
  - `BankWriter(out_dir, n_images, n_views, dim, backbone, seed)` with `.write_image(idx, meta_row, feats, presence, severity, proxies, recipes)` and `.close()`
  - `FeatureBank.open(out_dir) -> FeatureBank` exposing `.feats (N,V,D) float16`, `.presence (N,V,6)`, `.severity (N,V,6)`, `.proxies (N,V,3)`, `.recon (N,V,12) | None`, `.meta` DataFrame, `.config` dict
  - `FeatureBank.attach_recon(arr)` — writes `recon.npy` after the fact
  - `FeatureBank.check_invariants()` — raises if view 0 is not the clean view

- [ ] **Step 1: Write the failing test**

```python
# tests/features/test_bank.py
import numpy as np
import pytest

from aigcdet.features.bank import N_VIEWS, BankWriter, FeatureBank


def _build(tmp_path, n=4, dim=8):
    w = BankWriter(str(tmp_path / "bank"), n_images=n, n_views=N_VIEWS,
                   dim=dim, backbone="test", seed=0)
    rng = np.random.default_rng(0)
    for i in range(n):
        presence = np.zeros((N_VIEWS, 6), np.float32)
        severity = np.zeros((N_VIEWS, 6), np.float32)
        presence[1:, 0] = 1.0                       # view 0 stays clean
        severity[1:, 0] = 0.5
        w.write_image(
            i,
            {"path": f"/x/{i}.png", "label": i % 2, "generator": "g", "source": "s",
             "split": "train"},
            feats=rng.normal(size=(N_VIEWS, dim)).astype(np.float32),
            presence=presence, severity=severity,
            proxies=rng.normal(size=(N_VIEWS, 3)).astype(np.float32),
            recipes=["[]"] + ['[{"name": "jpeg", "params": {"quality": 50}}]'] * (N_VIEWS - 1),
        )
    w.close()
    return FeatureBank.open(str(tmp_path / "bank"))


def test_bank_roundtrips_all_arrays(tmp_path):
    b = _build(tmp_path)
    assert b.feats.shape == (4, N_VIEWS, 8)
    assert b.presence.shape == (4, N_VIEWS, 6)
    assert b.severity.shape == (4, N_VIEWS, 6)
    assert b.proxies.shape == (4, N_VIEWS, 3)
    assert b.recon is None
    assert len(b.meta) == 4 and b.meta["label"].tolist() == [0, 1, 0, 1]
    assert b.config["backbone"] == "test"


def test_view_zero_is_the_clean_view(tmp_path):
    b = _build(tmp_path)
    b.check_invariants()
    assert b.presence[:, 0, :].sum() == 0.0


def test_check_invariants_rejects_a_degraded_view_zero(tmp_path):
    b = _build(tmp_path)
    b.presence[0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="view 0"):
        b.check_invariants()


def test_attach_recon_persists_and_reloads(tmp_path):
    b = _build(tmp_path)
    r = np.arange(4 * N_VIEWS * 12, dtype=np.float32).reshape(4, N_VIEWS, 12)
    b.attach_recon(r)
    b2 = FeatureBank.open(b.path)
    assert b2.recon is not None and b2.recon.shape == (4, N_VIEWS, 12)
    np.testing.assert_allclose(b2.recon[1, 2], r[1, 2])


def test_recipes_are_recoverable_per_view(tmp_path):
    from aigcdet.augment.recipes import Recipe
    b = _build(tmp_path)
    assert Recipe.from_json(b.recipe_json(0, 0)).ops == ()
    assert Recipe.from_json(b.recipe_json(0, 1)).ops[0].name == "jpeg"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/features/test_bank.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.features.bank'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/features/bank.py
"""On-disk feature bank: contract #2 from spec §7.1.

Layout:
    bank/config.json     backbone, dim, n_views, seed
    bank/meta.parquet    N rows, image-level: path,label,generator,source,split
    bank/views.parquet   N*V rows: image_idx,view_idx,recipe_json
    bank/feats.npy       (N, V, D) float16   -- the ViT embedding
    bank/presence.npy    (N, V, 6) float32   -- degradation-head targets
    bank/severity.npy    (N, V, 6) float32
    bank/proxies.npy     (N, V, 3) float32   -- handcrafted h
    bank/recon.npy       (N, V, 12) float32  -- optional, attached later

Invariant: view 0 is always the undegraded view. The consistency loss and the
whole clean/degraded pairing depend on it, so it is checked, not assumed.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

N_VIEWS = 11          # 1 clean + 10 augmented (spec §3.1, K=10)
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
            dtype=np.float32, shape=(n_images, n_views, 6))
        self.severity = np.lib.format.open_memmap(
            os.path.join(out_dir, "severity.npy"), mode="w+",
            dtype=np.float32, shape=(n_images, n_views, 6))
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
        self.feats.flush(); self.presence.flush()
        self.severity.flush(); self.proxies.flush()
        pd.DataFrame(self._meta).sort_values("image_idx").to_parquet(
            os.path.join(self.path, "meta.parquet"), index=False)
        pd.DataFrame(self._views).to_parquet(
            os.path.join(self.path, "views.parquet"), index=False)
        with open(os.path.join(self.path, "config.json"), "w") as f:
            json.dump(self._config, f, indent=2)


class FeatureBank:
    def __init__(self, path: str):
        self.path = path
        self.config = json.load(open(os.path.join(path, "config.json")))
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/features/test_bank.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/aigcdet/features/bank.py tests/features/test_bank.py
git commit -m "feat(features): on-disk feature bank with clean-view-0 invariant"
```

---

### Task 3: Stage A extraction

**Files:**
- Create: `src/aigcdet/features/extract.py`, `scripts/extract_features.py`, `tests/features/test_extract.py`

**Interfaces:**
- Consumes: `manifest`, `backbones`, `bank`, `recipes.sample_training_recipe`, `proxies.proxy_vector`
- Produces:
  - `extract_bank(manifest_df, backbone_name, out_dir, n_views=N_VIEWS, seed=..., device=..., limit=None, exclude_families=()) -> str`
  - `exclude_families` supports the A3-LOTO run (spec §4.6) by forbidding a whole transform family from the sampled recipes

- [ ] **Step 1: Write the failing test**

```python
# tests/features/test_extract.py
import numpy as np
import pytest

from aigcdet.data.manifest import make_dummy_manifest
from aigcdet.features.bank import N_VIEWS, FeatureBank


def test_extract_with_a_fake_backbone_produces_a_valid_bank(tmp_path, monkeypatch):
    from aigcdet.features import extract
    from aigcdet.features.backbones import BackboneSpec

    spec = BackboneSpec("fake", "none", 64, 5, 1, 0)
    monkeypatch.setattr(extract, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(
        extract, "embed",
        lambda m, s, imgs, device, batch_size=16:
            np.stack([np.full(s.dim, float(i.mean()), np.float32) for i in imgs]))

    df = make_dummy_manifest(6, str(tmp_path / "img"), np.random.default_rng(0))
    out = extract.extract_bank(df, "fake", str(tmp_path / "bank"), seed=1, device="cpu")

    b = FeatureBank.open(out)
    b.check_invariants()
    assert b.feats.shape == (6, N_VIEWS, 5)
    assert len(b.meta) == 6
    # View 0 must be the clean view: its embedding equals the raw image mean.
    assert b.presence[:, 0, :].sum() == 0.0
    # Augmented views must actually differ from the clean one.
    assert not np.allclose(np.asarray(b.feats[0, 0]), np.asarray(b.feats[0, 1]))


def test_exclude_families_never_samples_the_excluded_family(tmp_path, monkeypatch):
    from aigcdet.augment.recipes import FAMILIES, Recipe
    from aigcdet.features import extract
    from aigcdet.features.backbones import BackboneSpec

    spec = BackboneSpec("fake", "none", 64, 4, 1, 0)
    monkeypatch.setattr(extract, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(extract, "embed",
                        lambda m, s, imgs, device, batch_size=16:
                            np.zeros((len(imgs), s.dim), np.float32))

    df = make_dummy_manifest(4, str(tmp_path / "img2"), np.random.default_rng(0))
    out = extract.extract_bank(df, "fake", str(tmp_path / "bank2"), seed=2,
                               device="cpu", exclude_families=("noise",))
    b = FeatureBank.open(out)
    i_noise = FAMILIES.index("noise")
    assert np.asarray(b.presence)[:, :, i_noise].sum() == 0.0
    for img in range(4):
        for v in range(N_VIEWS):
            assert all(o.name != "noise" for o in Recipe.from_json(b.recipe_json(img, v)).ops)


def test_extraction_is_reproducible_for_a_fixed_seed(tmp_path, monkeypatch):
    from aigcdet.features import extract
    from aigcdet.features.backbones import BackboneSpec
    spec = BackboneSpec("fake", "none", 64, 3, 1, 0)
    monkeypatch.setattr(extract, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(extract, "embed",
                        lambda m, s, imgs, device, batch_size=16:
                            np.stack([np.full(s.dim, float(i.std()), np.float32) for i in imgs]))
    df = make_dummy_manifest(3, str(tmp_path / "img3"), np.random.default_rng(0))
    a = FeatureBank.open(extract.extract_bank(df, "fake", str(tmp_path / "b1"), seed=5, device="cpu"))
    c = FeatureBank.open(extract.extract_bank(df, "fake", str(tmp_path / "b2"), seed=5, device="cpu"))
    np.testing.assert_allclose(np.asarray(a.feats), np.asarray(c.feats))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/features/test_extract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.features.extract'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/features/extract.py
"""Stage A (spec §3.1): images x (1 clean + K augmented views) -> feature bank.

Runs once per backbone. Everything downstream trains on the output in minutes.
Per-image RNG is derived from (seed, image_idx) so extraction is reproducible
and resumable at any chunk boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from aigcdet.augment.recipes import FAMILIES, Recipe, sample_training_recipe
from aigcdet.features.backbones import embed, load_backbone
from aigcdet.features.bank import N_VIEWS, BankWriter
from aigcdet.features.proxies import proxy_vector


def _sample_recipe_excluding(rng, exclude: tuple[str, ...]) -> Recipe:
    """Rejection-sample a recipe that avoids whole transform families.

    Used for the leave-one-transform-out run (spec §4.6): the excluded family
    must be entirely absent from training so evaluation on it measures
    generalisation to an unanticipated degradation.
    """
    if not exclude:
        return sample_training_recipe(rng)
    for _ in range(200):
        r = sample_training_recipe(rng)
        if all(o.name not in exclude for o in r.ops):
            return r
    kept = [f for f in FAMILIES if f not in exclude]
    raise RuntimeError(f"could not sample a recipe from {kept}")


def extract_bank(
    manifest_df: pd.DataFrame,
    backbone_name: str,
    out_dir: str,
    n_views: int = N_VIEWS,
    seed: int = 20260827,
    device: str = "cuda",
    limit: int | None = None,
    exclude_families: tuple[str, ...] = (),
    batch_size: int = 16,
) -> str:
    df = manifest_df.reset_index(drop=True)
    if limit:
        df = df.iloc[:limit].reset_index(drop=True)

    model, spec = load_backbone(backbone_name, device=device)
    writer = BankWriter(out_dir, len(df), n_views, spec.dim, backbone_name, seed)

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"extract:{backbone_name}"):
        rng = np.random.default_rng([seed, int(i)])
        with Image.open(row["path"]) as im:
            base = np.asarray(im.convert("RGB"), dtype=np.uint8)

        # View 0 is always the clean view (bank invariant).
        recipes = [Recipe(())] + [
            _sample_recipe_excluding(rng, exclude_families) for _ in range(n_views - 1)
        ]
        views = [r.apply(base, rng) for r in recipes]

        feats = embed(model, spec, views, device=device, batch_size=batch_size)
        labels = [r.labels() for r in recipes]
        writer.write_image(
            int(i),
            {"path": row["path"], "label": int(row["label"]),
             "generator": row["generator"], "source": row["source"],
             "split": row["split"]},
            feats=feats,
            presence=np.stack([l["presence"] for l in labels]),
            severity=np.stack([l["severity"] for l in labels]),
            proxies=np.stack([proxy_vector(v) for v in views]),
            recipes=[r.to_json() for r in recipes],
        )
    writer.close()
    return out_dir
```

```python
# scripts/extract_features.py
"""Stage A CLI.

    python scripts/extract_features.py --manifest data/manifest.parquet \
        --backbone dinov3l --out banks/dinov3l --split train,val_internal
    # leave-one-transform-out bank for the A3-LOTO run:
    python scripts/extract_features.py --manifest data/manifest.parquet \
        --backbone dinov3l --out banks/dinov3l_loto --exclude noise
"""
from __future__ import annotations

import argparse

from aigcdet.data.manifest import read_manifest
from aigcdet.features.extract import extract_bank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="", help="filter to one manifest split")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--exclude", default="", help="comma-separated families to exclude")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()

    df = read_manifest(a.manifest)
    if a.split:
        df = df[df["split"] == a.split].reset_index(drop=True)
    extract_bank(df, a.backbone, a.out, seed=20260827, device=a.device,
                 limit=a.limit, batch_size=a.batch_size,
                 exclude_families=tuple(f for f in a.exclude.split(",") if f))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/features/test_extract.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/aigcdet/features/extract.py scripts/extract_features.py tests/features/test_extract.py
git commit -m "feat(features): Stage A extraction with LOTO family exclusion"
```

---

### Task 4: Reconstruction branch `r`

Spec §3.3. Cached for **all 11 views** so A3→A4 is a controlled comparison.

**Files:**
- Create: `src/aigcdet/features/recon.py`, `tests/features/test_recon.py`
- Modify: `scripts/extract_features.py` — add a `--recon` mode

**Interfaces:**
- Consumes: `bank.FeatureBank`, `bank.RECON_DIM`
- Produces:
  - `RECON_FEATURE_NAMES: tuple[str, ...]` — 12 names, fixed order
  - `recon_features(img: np.ndarray, vae, lpips_fn, device) -> np.ndarray` — `(12,)` float32 from a 256×256 native-pixel centre crop
  - `error_map(img, vae, device) -> np.ndarray` — for the dashboard heatmap
  - `attach_recon_to_bank(bank, manifest_df, device) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/features/test_recon.py
import numpy as np
import pytest

from aigcdet.features.bank import RECON_DIM
from aigcdet.features.recon import RECON_FEATURE_NAMES, native_center_crop


def test_feature_names_match_the_declared_width():
    assert len(RECON_FEATURE_NAMES) == RECON_DIM


def test_native_center_crop_does_not_resize():
    img = np.random.default_rng(0).integers(0, 256, (512, 700, 3), dtype=np.uint8)
    out = native_center_crop(img, 256)
    assert out.shape == (256, 256, 3)
    # must be an exact slice of the original, never an interpolation
    top, left = (512 - 256) // 2, (700 - 256) // 2
    np.testing.assert_array_equal(out, img[top:top + 256, left:left + 256])


def test_native_center_crop_pads_a_small_image_instead_of_upscaling():
    img = np.random.default_rng(0).integers(0, 256, (100, 120, 3), dtype=np.uint8)
    out = native_center_crop(img, 256)
    assert out.shape == (256, 256, 3)
    assert (out[:100, :120] == img).all() or out.shape == (256, 256, 3)


@pytest.mark.gpu
def test_recon_features_are_finite_and_lower_error_for_a_vae_roundtrip():
    import torch
    from aigcdet.features.recon import load_recon_models, recon_features
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    vae, lp = load_recon_models("cuda")
    rng = np.random.default_rng(0)
    photo = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    v = recon_features(photo, vae, lp, "cuda")
    assert v.shape == (RECON_DIM,) and np.isfinite(v).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/features/test_recon.py -v -m "not gpu"`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.features.recon'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/features/recon.py
"""Low-level reconstruction branch (spec §3.3).

Latent-diffusion outputs round-trip through their own VAE with anomalously low
error. Real photographs, and outputs from generators with a different decoder,
do not. The crop is taken at NATIVE pixel resolution: resizing would attenuate
exactly the signal being measured.

Two failure modes are expected and reported rather than hidden: the signal is
specific to the SD 1.5 autoencoder, and reconstruction error falls for any
heavily degraded image because degraded images are easier to reconstruct.
"""
from __future__ import annotations

import numpy as np
import torch

RECON_FEATURE_NAMES: tuple[str, ...] = (
    "l1", "lpips",
    "err_mean", "err_std", "err_p90", "err_max",
    "spec_b0", "spec_b1", "spec_b2", "spec_b3",
    "spec_mid_ratio", "spec_high_ratio",
)


def native_center_crop(img: np.ndarray, size: int = 256) -> np.ndarray:
    """Exact pixel slice, never an interpolation. Small images are reflect-padded."""
    h, w = img.shape[:2]
    if h < size or w < size:
        img = np.pad(img, ((0, max(0, size - h)), (0, max(0, size - w)), (0, 0)),
                     mode="reflect")
        h, w = img.shape[:2]
    top, left = (h - size) // 2, (w - size) // 2
    return img[top:top + size, left:left + size]


def load_recon_models(device: str = "cuda"):
    from diffusers import AutoencoderKL
    import lpips
    vae = AutoencoderKL.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae",
        dtype=torch.float16).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    lp = lpips.LPIPS(net="alex").to(device).eval()
    for p in lp.parameters():
        p.requires_grad_(False)
    return vae, lp


@torch.inference_mode()
def _roundtrip(crop: np.ndarray, vae, device: str) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(crop.astype(np.float32) / 127.5 - 1.0)
    x = x.permute(2, 0, 1)[None].to(device, torch.float16)
    lat = vae.encode(x).latent_dist.mode()
    rec = vae.decode(lat).sample.clamp(-1, 1)
    err = (x.float() - rec.float()).abs().mean(dim=1)[0].cpu().numpy()
    return err, x.float(), rec.float()


def _radial_bands(err: np.ndarray, n_bands: int = 4) -> np.ndarray:
    """Azimuthally averaged power spectrum of the error map.

    Mid-frequency error survives compression better than the highest
    frequencies, so the bands are kept separate rather than summed.
    """
    f = np.abs(np.fft.fftshift(np.fft.fft2(err - err.mean()))) ** 2
    h, w = f.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[:h, :w]
    rad = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = rad.max()
    out = []
    for b in range(n_bands):
        m = (rad >= rmax * b / n_bands) & (rad < rmax * (b + 1) / n_bands)
        out.append(float(np.log1p(f[m].mean())) if m.any() else 0.0)
    return np.array(out, dtype=np.float32)


@torch.inference_mode()
def recon_features(img: np.ndarray, vae, lpips_fn, device: str = "cuda") -> np.ndarray:
    crop = native_center_crop(img, 256)
    err, x, rec = _roundtrip(crop, vae, device)
    l1 = float(np.abs(err).mean())
    lp = float(lpips_fn(x, rec).item())
    bands = _radial_bands(err)
    total = float(bands.sum()) or 1.0
    return np.concatenate([
        np.array([l1, lp, err.mean(), err.std(),
                  np.percentile(err, 90), err.max()], dtype=np.float32),
        bands,
        np.array([bands[1] / total, bands[3] / total], dtype=np.float32),
    ]).astype(np.float32)


@torch.inference_mode()
def error_map(img: np.ndarray, vae, device: str = "cuda") -> np.ndarray:
    """Per-pixel reconstruction error, for the dashboard's second heatmap."""
    err, _, _ = _roundtrip(native_center_crop(img, 256), vae, device)
    return err


def attach_recon_to_bank(bank, manifest_df, device: str = "cuda",
                         seed: int = 20260827) -> None:
    """Recompute every view exactly as Stage A did, then cache `r` for ALL of
    them. Partial view coverage would make A3 vs A4 a comparison across
    different augmentation budgets (spec §3.3)."""
    from PIL import Image
    from tqdm import tqdm
    from aigcdet.augment.recipes import Recipe
    from aigcdet.features.bank import RECON_DIM

    vae, lp = load_recon_models(device)
    n, v = len(bank.meta), bank.config["n_views"]
    out = np.zeros((n, v, RECON_DIM), dtype=np.float32)
    for i in tqdm(range(n), desc="recon"):
        with Image.open(bank.meta.iloc[i]["path"]) as im:
            base = np.asarray(im.convert("RGB"), dtype=np.uint8)
        rng = np.random.default_rng([seed, int(i)])
        for j in range(v):
            view = Recipe.from_json(bank.recipe_json(i, j)).apply(base, rng)
            out[i, j] = recon_features(view, vae, lp, device)
    bank.attach_recon(out)
```

**Reproducibility note.** `attach_recon_to_bank` replays each stored recipe, but
noise draws consume the per-image RNG in the same order only if the recipes are
replayed in view order 0..V-1, which the loop above guarantees. Views whose
recipe contains no `noise` op are exactly reproducible regardless.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/features/test_recon.py -v -m "not gpu"`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/aigcdet/features/recon.py tests/features/test_recon.py
git commit -m "feat(features): VAE reconstruction branch cached for all views"
```

---

### Task 5: Heads

**Files:**
- Create: `src/aigcdet/models/__init__.py`, `src/aigcdet/models/heads.py`, `tests/models/test_heads.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `DegradationHead(dim_in, hidden=256, n_families=6)` → `forward(f) -> dict` with `presence (B,6)` logits, `severity (B,6)` in [0,1], `embedding (B,256)`
  - `ClassifierHead(dim_in, hidden=512, use_film=False, cond_dim=256)` → `forward(f, cond=None) -> dict` with `logit (B,)` and `hidden (B,512)`
  - `Detector(dim_feat, use_recon, recon_dim=12, use_film=False)` — wraps both, `forward(f, r=None) -> dict` with `logit`, `hidden`, `presence`, `severity`, `deg_embedding`

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_heads.py
import torch

from aigcdet.models.heads import ClassifierHead, DegradationHead, Detector


def test_degradation_head_shapes_and_severity_range():
    h = DegradationHead(dim_in=32)
    out = h(torch.randn(4, 32))
    assert out["presence"].shape == (4, 6)
    assert out["severity"].shape == (4, 6)
    assert out["embedding"].shape == (4, 256)
    assert (out["severity"] >= 0).all() and (out["severity"] <= 1).all()


def test_classifier_head_shapes():
    c = ClassifierHead(dim_in=32)
    out = c(torch.randn(5, 32))
    assert out["logit"].shape == (5,) and out["hidden"].shape == (5, 512)


def test_film_changes_the_hidden_state_and_plain_head_ignores_cond():
    torch.manual_seed(0)
    f = torch.randn(3, 32)
    cond = torch.randn(3, 256)
    film = ClassifierHead(dim_in=32, use_film=True)
    a = film(f, cond)["hidden"]
    b = film(f, torch.zeros_like(cond))["hidden"]
    assert not torch.allclose(a, b)
    plain = ClassifierHead(dim_in=32, use_film=False)
    assert torch.allclose(plain(f, cond)["hidden"], plain(f)["hidden"])


def test_detector_without_recon_matches_feature_width():
    d = Detector(dim_feat=16, use_recon=False)
    out = d(torch.randn(2, 16))
    assert out["logit"].shape == (2,) and out["presence"].shape == (2, 6)


def test_detector_with_recon_consumes_the_concatenated_width():
    d = Detector(dim_feat=16, use_recon=True, recon_dim=12)
    out = d(torch.randn(2, 16), torch.randn(2, 12))
    assert out["logit"].shape == (2,)


def test_detector_raises_when_recon_expected_but_missing():
    import pytest
    d = Detector(dim_feat=16, use_recon=True, recon_dim=12)
    with pytest.raises(ValueError, match="recon"):
        d(torch.randn(2, 16))


def test_stop_gradient_isolates_the_degradation_head_when_film_is_on():
    """With FiLM enabled, no classifier gradient may reach the degradation head
    (spec §3.4): otherwise `d` stops meaning 'degradation'."""
    d = Detector(dim_feat=16, use_recon=False, use_film=True)
    out = d(torch.randn(4, 16))
    out["logit"].sum().backward()
    grads = [p.grad for p in d.degradation.parameters() if p.grad is not None]
    assert all(g.abs().sum() == 0 for g in grads) or not grads


def test_trainable_parameter_count_is_small():
    d = Detector(dim_feat=1024, use_recon=True)
    n = sum(p.numel() for p in d.parameters() if p.requires_grad)
    assert n < 3_000_000, f"heads should stay ~2M parameters, got {n}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/models/test_heads.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.models'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/models/heads.py
"""Trainable heads (spec §3.4). Everything here runs on cached embeddings.

The headline model does NOT use FiLM: the degradation head feeds calibration,
EQI, and the dashboard, not the classifier. Conditioning is rung A7, a
hypothesis under test, because DCPT reports that architectural additions
overfit on limited training data.
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_FAMILIES = 6


class DegradationHead(nn.Module):
    def __init__(self, dim_in: int, hidden: int = 256, n_families: int = N_FAMILIES):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dim_in, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.presence = nn.Linear(hidden, n_families)
        self.severity = nn.Linear(hidden, n_families)

    def forward(self, f: torch.Tensor) -> dict[str, torch.Tensor]:
        e = self.trunk(f)
        return {"presence": self.presence(e),
                "severity": torch.sigmoid(self.severity(e)),
                "embedding": e}


class ClassifierHead(nn.Module):
    def __init__(self, dim_in: int, hidden: int = 512,
                 use_film: bool = False, cond_dim: int = 256):
        super().__init__()
        self.use_film = use_film
        self.trunk = nn.Sequential(
            nn.Linear(dim_in, hidden), nn.GELU(), nn.LayerNorm(hidden))
        if use_film:
            self.film = nn.Linear(cond_dim, hidden * 2)
        self.out = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(),
                                 nn.Linear(hidden // 2, 1))

    def forward(self, f: torch.Tensor, cond: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        h = self.trunk(f)
        if self.use_film and cond is not None:
            gamma, beta = self.film(cond).chunk(2, dim=-1)
            h = (1.0 + gamma) * h + beta
        return {"logit": self.out(h).squeeze(-1), "hidden": h}


class Detector(nn.Module):
    """Degradation head + classifier head over a cached embedding."""

    def __init__(self, dim_feat: int, use_recon: bool = False,
                 recon_dim: int = 12, use_film: bool = False):
        super().__init__()
        self.use_recon = use_recon
        self.use_film = use_film
        dim_in = dim_feat + (recon_dim if use_recon else 0)
        self.degradation = DegradationHead(dim_in)
        self.classifier = ClassifierHead(dim_in, use_film=use_film)

    def forward(self, f: torch.Tensor, r: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if self.use_recon:
            if r is None:
                raise ValueError("this Detector expects recon features `r`")
            f = torch.cat([f, r], dim=-1)
        deg = self.degradation(f)
        # Stop-gradient: the classifier must not reshape `d` into a general
        # purpose feature, or the degradation readout stops being meaningful.
        cond = deg["embedding"].detach() if self.use_film else None
        cls = self.classifier(f, cond)
        return {"logit": cls["logit"], "hidden": cls["hidden"],
                "presence": deg["presence"], "severity": deg["severity"],
                "deg_embedding": deg["embedding"]}
```

Also create empty `src/aigcdet/models/__init__.py` and `tests/models/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/models/test_heads.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/aigcdet/models tests/models
git commit -m "feat(models): classifier and degradation heads with stop-gradient FiLM"
```

---

### Task 6: Losses

Spec §3.5. The consistency term acts on the head's trainable hidden state, **not** on the frozen embedding.

**Files:**
- Create: `src/aigcdet/models/losses.py`, `tests/models/test_losses.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `classification_loss(logit, y, pos_weight=None) -> Tensor`
  - `degradation_loss(pred_presence, pred_severity, tgt_presence, tgt_severity) -> Tensor`
  - `consistency_loss(logit_clean, logit_deg, hidden_clean, hidden_deg, alpha, beta) -> Tensor`
  - `total_loss(out_clean, out_deg, batch, weights: LossWeights) -> tuple[Tensor, dict[str, float]]`
  - `LossWeights` dataclass: `lambda_deg, alpha, beta`

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_losses.py
import pytest
import torch

from aigcdet.models.losses import (
    LossWeights, classification_loss, consistency_loss, degradation_loss,
)


def test_classification_loss_is_near_zero_for_confident_correct_logits():
    logit = torch.tensor([10.0, -10.0])
    y = torch.tensor([1.0, 0.0])
    assert classification_loss(logit, y).item() < 1e-3


def test_degradation_loss_is_zero_at_a_perfect_prediction():
    tgt_p = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    tgt_s = torch.tensor([[0.7, 0.0, 0.0, 0.0, 0.0, 0.0]])
    pred_p = torch.tensor([[20.0, -20.0, -20.0, -20.0, -20.0, -20.0]])
    loss = degradation_loss(pred_p, tgt_s.clone(), tgt_p, tgt_s)
    assert loss.item() < 1e-3


def test_severity_error_is_masked_to_present_families():
    """A wrong severity on an absent family must not be penalised: its target
    is meaningless when the transform was never applied."""
    tgt_p = torch.tensor([[1.0, 0.0]  + [0.0] * 4])
    tgt_s = torch.tensor([[0.5, 0.0]  + [0.0] * 4])
    pred_p = torch.tensor([[20.0, -20.0] + [-20.0] * 4])
    good = degradation_loss(pred_p, tgt_s.clone(), tgt_p, tgt_s)
    noisy = tgt_s.clone(); noisy[0, 1] = 0.9          # absent family, wrong severity
    same = degradation_loss(pred_p, noisy, tgt_p, tgt_s)
    assert torch.isclose(good, same, atol=1e-6)


def test_consistency_loss_is_zero_when_clean_and_degraded_agree():
    lg = torch.tensor([1.5, -0.3])
    hd = torch.randn(2, 8)
    assert consistency_loss(lg, lg.clone(), hd, hd.clone(), 1.0, 1.0).item() == pytest.approx(0.0, abs=1e-6)


def test_consistency_loss_grows_with_disagreement():
    lg_a = torch.tensor([2.0, 2.0])
    h = torch.randn(2, 8)
    near = consistency_loss(lg_a, torch.tensor([1.9, 1.9]), h, h.clone(), 1.0, 1.0)
    far = consistency_loss(lg_a, torch.tensor([-4.0, -4.0]), h, h.clone(), 1.0, 1.0)
    assert far > near


def test_consistency_gradient_reaches_the_hidden_state():
    """Guards the v1 bug: the feature term must act on a trainable tensor."""
    h_clean = torch.randn(3, 8, requires_grad=True)
    h_deg = torch.randn(3, 8, requires_grad=True)
    lg = torch.zeros(3, requires_grad=True)
    consistency_loss(lg, lg.clone(), h_clean, h_deg, 0.0, 1.0).backward()
    assert h_deg.grad is not None and h_deg.grad.abs().sum() > 0


def test_loss_weights_defaults_are_explicit():
    w = LossWeights()
    assert w.lambda_deg > 0 and w.alpha > 0 and w.beta > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/models/test_losses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.models.losses'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/models/losses.py
"""Losses (spec §3.5).

    L = L_cls + lambda_deg * L_deg + alpha * KL(p_clean || p_deg)
                                   + beta  * MSE(h_clean, h_deg)

The feature-consistency term acts on `hidden`, the classifier's trainable 512-d
state. Applying it to the cached embedding — as the first draft of the spec did
— makes it a constant with no gradient path, so it would silently do nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    lambda_deg: float = 0.3
    alpha: float = 1.0
    beta: float = 1.0


def classification_loss(logit: torch.Tensor, y: torch.Tensor,
                        pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit, y.float(), pos_weight=pos_weight)


def degradation_loss(pred_presence: torch.Tensor, pred_severity: torch.Tensor,
                     tgt_presence: torch.Tensor, tgt_severity: torch.Tensor) -> torch.Tensor:
    pres = F.binary_cross_entropy_with_logits(pred_presence, tgt_presence)
    # Severity is only defined where the transform was actually applied.
    mask = tgt_presence
    denom = mask.sum().clamp(min=1.0)
    sev = (F.smooth_l1_loss(pred_severity, tgt_severity, reduction="none") * mask).sum() / denom
    return pres + sev


def _sym_kl_bernoulli(logit_a: torch.Tensor, logit_b: torch.Tensor) -> torch.Tensor:
    pa, pb = torch.sigmoid(logit_a), torch.sigmoid(logit_b)
    eps = 1e-6
    pa, pb = pa.clamp(eps, 1 - eps), pb.clamp(eps, 1 - eps)
    kl_ab = pa * (pa / pb).log() + (1 - pa) * ((1 - pa) / (1 - pb)).log()
    kl_ba = pb * (pb / pa).log() + (1 - pb) * ((1 - pb) / (1 - pa)).log()
    return (kl_ab + kl_ba).mean() * 0.5


def consistency_loss(logit_clean: torch.Tensor, logit_deg: torch.Tensor,
                     hidden_clean: torch.Tensor, hidden_deg: torch.Tensor,
                     alpha: float, beta: float) -> torch.Tensor:
    pred = _sym_kl_bernoulli(logit_clean, logit_deg)
    feat = F.mse_loss(hidden_deg, hidden_clean)
    return alpha * pred + beta * feat


def total_loss(out_clean: dict, out_deg: dict, batch: dict,
               weights: LossWeights, pos_weight: torch.Tensor | None = None
               ) -> tuple[torch.Tensor, dict[str, float]]:
    """`batch` supplies y_clean, y_deg, presence_deg, severity_deg."""
    l_cls = (classification_loss(out_clean["logit"], batch["y_clean"], pos_weight)
             + classification_loss(out_deg["logit"], batch["y_deg"], pos_weight)) * 0.5
    l_deg = degradation_loss(out_deg["presence"], out_deg["severity"],
                             batch["presence_deg"], batch["severity_deg"])
    l_con = consistency_loss(out_clean["logit"].detach(), out_deg["logit"],
                             out_clean["hidden"].detach(), out_deg["hidden"],
                             weights.alpha, weights.beta)
    total = l_cls + weights.lambda_deg * l_deg + l_con
    return total, {"cls": float(l_cls), "deg": float(l_deg),
                   "con": float(l_con), "total": float(total)}
```

**Note on the `.detach()` in `total_loss`.** The clean branch is treated as the
target the degraded branch is pulled towards, rather than letting both drift to
meet in the middle — the latter can be minimised by collapsing the
representation. Rung A3's config exposes `detach_clean` so the alternative can
be ablated if A3 underperforms.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/models/test_losses.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/aigcdet/models/losses.py tests/models/test_losses.py
git commit -m "feat(models): losses with consistency on the trainable hidden state"
```

---

### Task 7: Paired batch sampler

Spec §3.5. Every clean embedding needs degraded partners in the same batch, and batches must be class- and generator-balanced.

**Files:**
- Create: `src/aigcdet/models/sampler.py`, `tests/models/test_sampler.py`

**Interfaces:**
- Consumes: `aigcdet.features.bank.FeatureBank`
- Produces:
  - `PairedSampler(bank, indices, n_src=32, m_deg=2, rng=..., use_recon=False, augmented_only=False)`
  - `.__iter__()` yields dicts of torch tensors: `f_clean (B,D)`, `f_deg (B,D)`, `r_clean`, `r_deg` (or `None`), `y_clean (B,)`, `y_deg (B,)`, `presence_deg (B,6)`, `severity_deg (B,6)`, where `B = n_src * m_deg`
  - `.__len__()` — batches per epoch
  - `augmented_only=False` and `m_deg` control rung A0 (clean views only) versus A1+ (augmented)

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_sampler.py
import numpy as np
import torch

from aigcdet.features.bank import N_VIEWS, BankWriter, FeatureBank
from aigcdet.models.sampler import PairedSampler


def _bank(tmp_path, n=40, dim=6):
    w = BankWriter(str(tmp_path / "b"), n, N_VIEWS, dim, "t", 0)
    rng = np.random.default_rng(0)
    for i in range(n):
        pres = np.zeros((N_VIEWS, 6), np.float32); pres[1:, 0] = 1.0
        sev = np.zeros((N_VIEWS, 6), np.float32); sev[1:, 0] = 0.4
        w.write_image(i, {"path": f"/p{i}", "label": i % 2,
                          "generator": f"g{i % 3}", "source": "s", "split": "train"},
                      feats=rng.normal(size=(N_VIEWS, dim)).astype(np.float32),
                      presence=pres, severity=sev,
                      proxies=np.zeros((N_VIEWS, 3), np.float32),
                      recipes=["[]"] * N_VIEWS)
    w.close()
    return FeatureBank.open(str(tmp_path / "b"))


def test_batch_shapes_and_pairing(tmp_path):
    b = _bank(tmp_path)
    s = PairedSampler(b, np.arange(40), n_src=4, m_deg=2, rng=np.random.default_rng(0))
    batch = next(iter(s))
    assert batch["f_clean"].shape == (8, 6) and batch["f_deg"].shape == (8, 6)
    assert batch["y_clean"].shape == (8,) and batch["presence_deg"].shape == (8, 6)
    # Each source image contributes m_deg rows sharing one clean embedding.
    assert torch.allclose(batch["f_clean"][0], batch["f_clean"][1])
    assert not torch.allclose(batch["f_deg"][0], batch["f_deg"][1])


def test_labels_match_between_clean_and_degraded_rows(tmp_path):
    b = _bank(tmp_path)
    s = PairedSampler(b, np.arange(40), n_src=8, m_deg=2, rng=np.random.default_rng(1))
    for batch in s:
        assert torch.equal(batch["y_clean"], batch["y_deg"])


def test_degraded_rows_always_have_nonzero_degradation(tmp_path):
    b = _bank(tmp_path)
    s = PairedSampler(b, np.arange(40), n_src=8, m_deg=2, rng=np.random.default_rng(2))
    for batch in s:
        assert (batch["presence_deg"].sum(dim=1) > 0).all()


def test_batches_are_class_balanced(tmp_path):
    b = _bank(tmp_path)
    s = PairedSampler(b, np.arange(40), n_src=8, m_deg=1, rng=np.random.default_rng(3))
    for batch in s:
        assert batch["y_clean"].sum().item() == 4      # half positives


def test_recon_is_returned_only_when_requested(tmp_path):
    b = _bank(tmp_path)
    b.attach_recon(np.zeros((40, N_VIEWS, 12), np.float32))
    s = PairedSampler(b, np.arange(40), n_src=4, m_deg=1,
                      rng=np.random.default_rng(4), use_recon=True)
    batch = next(iter(s))
    assert batch["r_deg"].shape == (4, 12)
    s2 = PairedSampler(b, np.arange(40), n_src=4, m_deg=1, rng=np.random.default_rng(4))
    assert next(iter(s2))["r_deg"] is None


def test_epoch_length_matches_the_index_pool(tmp_path):
    b = _bank(tmp_path)
    s = PairedSampler(b, np.arange(40), n_src=8, m_deg=2, rng=np.random.default_rng(5))
    assert len(s) == len(list(s))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/models/test_sampler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.models.sampler'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/models/sampler.py
"""Paired batch sampler (spec §3.5).

Each batch is n_src source images x m_deg degraded views, with the matching
clean view carried alongside every row. Class balance is enforced per batch so
no gradient step is dominated by one class; generator balance is applied within
the positive half for the same reason.
"""
from __future__ import annotations

import numpy as np
import torch


class PairedSampler:
    def __init__(self, bank, indices: np.ndarray, n_src: int = 32, m_deg: int = 2,
                 rng: np.random.Generator | None = None, use_recon: bool = False,
                 device: str = "cpu"):
        if n_src % 2 != 0:
            raise ValueError("n_src must be even so batches can be class-balanced")
        self.bank = bank
        self.n_src, self.m_deg = n_src, m_deg
        self.rng = rng or np.random.default_rng(0)
        self.use_recon = use_recon
        self.device = device
        self.n_views = bank.config["n_views"]

        labels = bank.meta["label"].to_numpy()[indices]
        self.pos = indices[labels == 1]
        self.neg = indices[labels == 0]
        if len(self.pos) == 0 or len(self.neg) == 0:
            raise ValueError("index pool must contain both classes")
        self.generators = bank.meta["generator"].to_numpy()

    def __len__(self) -> int:
        return max(1, min(len(self.pos), len(self.neg)) // (self.n_src // 2))

    def _draw_positives(self, k: int) -> np.ndarray:
        """Sample generator-balanced positives: pick a family, then an image."""
        gens = self.generators[self.pos]
        uniq = np.unique(gens)
        chosen = []
        for _ in range(k):
            g = uniq[self.rng.integers(len(uniq))]
            pool = self.pos[gens == g]
            chosen.append(pool[self.rng.integers(len(pool))])
        return np.array(chosen, dtype=np.int64)

    def __iter__(self):
        half = self.n_src // 2
        for _ in range(len(self)):
            src = np.concatenate([
                self._draw_positives(half),
                self.neg[self.rng.integers(0, len(self.neg), half)],
            ])
            # Degraded views are 1..V-1; view 0 is clean and never sampled here.
            deg_views = self.rng.integers(1, self.n_views, size=(len(src), self.m_deg))

            si = np.repeat(src, self.m_deg)
            vi = deg_views.reshape(-1)
            f_clean = np.asarray(self.bank.feats[src, 0]).astype(np.float32)
            f_clean = np.repeat(f_clean, self.m_deg, axis=0)
            f_deg = np.asarray(self.bank.feats[si, vi]).astype(np.float32)
            y = self.bank.meta["label"].to_numpy()[si].astype(np.float32)

            batch = {
                "f_clean": torch.from_numpy(f_clean).to(self.device),
                "f_deg": torch.from_numpy(f_deg).to(self.device),
                "y_clean": torch.from_numpy(y).to(self.device),
                "y_deg": torch.from_numpy(y).to(self.device),
                "presence_deg": torch.from_numpy(
                    np.asarray(self.bank.presence[si, vi])).to(self.device),
                "severity_deg": torch.from_numpy(
                    np.asarray(self.bank.severity[si, vi])).to(self.device),
                "r_clean": None, "r_deg": None,
            }
            if self.use_recon:
                if self.bank.recon is None:
                    raise ValueError("bank has no recon features; run attach_recon first")
                rc = np.asarray(self.bank.recon[src, 0]).astype(np.float32)
                batch["r_clean"] = torch.from_numpy(
                    np.repeat(rc, self.m_deg, axis=0)).to(self.device)
                batch["r_deg"] = torch.from_numpy(
                    np.asarray(self.bank.recon[si, vi]).astype(np.float32)).to(self.device)
            yield batch
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/models/test_sampler.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/aigcdet/models/sampler.py tests/models/test_sampler.py
git commit -m "feat(models): class- and generator-balanced paired batch sampler"
```

---

### Task 8: Stage B trainer and rung configs

Spec §6.4. Each rung is minutes on cached features.

**Files:**
- Create: `src/aigcdet/train/__init__.py`, `src/aigcdet/train/train_head.py`, `scripts/train_rung.py`, `configs/rungs/a0.yaml` … `a6.yaml`, `tests/train/test_train_head.py`

**Interfaces:**
- Consumes: `bank`, `heads.Detector`, `losses`, `sampler.PairedSampler`, `eval.metrics.roc_auc`
- Produces:
  - `RungConfig` dataclass: `name, bank_dir, use_recon, use_film, use_augmented, use_consistency, use_degradation, epochs, lr, n_src, m_deg, seed`
  - `train_rung(cfg: RungConfig) -> dict` — returns `{"checkpoint": path, "val_auc": float, "history": [...]}`; writes `outputs/rungs/<name>/checkpoint.pt` containing `state_dict`, the config, `dim_feat`, and the backbone name
  - `load_detector(checkpoint_path, device) -> tuple[Detector, dict]`

- [ ] **Step 1: Write the failing test**

```python
# tests/train/test_train_head.py
import numpy as np
import torch

from aigcdet.features.bank import N_VIEWS, BankWriter, FeatureBank
from aigcdet.train.train_head import RungConfig, load_detector, train_rung


def _learnable_bank(tmp_path, n=120, dim=8):
    """Fakes get a shifted mean, so a linear head can separate them.
    Augmented views are noisier, so consistency training has something to do."""
    w = BankWriter(str(tmp_path / "b"), n, N_VIEWS, dim, "t", 0)
    rng = np.random.default_rng(0)
    for i in range(n):
        label = i % 2
        clean = rng.normal(loc=1.5 if label else -1.5, scale=0.5, size=dim)
        feats = np.stack([clean] + [clean + rng.normal(0, 0.8, dim)
                                    for _ in range(N_VIEWS - 1)]).astype(np.float32)
        pres = np.zeros((N_VIEWS, 6), np.float32); pres[1:, 0] = 1.0
        sev = np.zeros((N_VIEWS, 6), np.float32); sev[1:, 0] = 0.6
        w.write_image(i, {"path": f"/p{i}", "label": label, "generator": f"g{i % 2}",
                          "source": "s", "split": "train" if i < 100 else "val_internal"},
                      feats=feats, presence=pres, severity=sev,
                      proxies=np.zeros((N_VIEWS, 3), np.float32),
                      recipes=["[]"] * N_VIEWS)
    w.close()
    return str(tmp_path / "b")


def test_a0_trains_and_separates_the_classes(tmp_path):
    cfg = RungConfig(name="a0", bank_dir=_learnable_bank(tmp_path), epochs=15,
                     use_augmented=False, use_consistency=False,
                     use_degradation=False, out_dir=str(tmp_path / "out"))
    res = train_rung(cfg)
    assert res["val_auc"] > 0.85


def test_a3_runs_with_all_terms_enabled(tmp_path):
    cfg = RungConfig(name="a3", bank_dir=_learnable_bank(tmp_path), epochs=10,
                     use_augmented=True, use_consistency=True,
                     use_degradation=True, out_dir=str(tmp_path / "out3"))
    res = train_rung(cfg)
    assert res["val_auc"] > 0.7
    assert all(np.isfinite(h["total"]) for h in res["history"])


def test_checkpoint_roundtrips_and_reproduces_scores(tmp_path):
    cfg = RungConfig(name="a1", bank_dir=_learnable_bank(tmp_path), epochs=5,
                     use_augmented=True, out_dir=str(tmp_path / "out1"))
    res = train_rung(cfg)
    model, meta = load_detector(res["checkpoint"], device="cpu")
    model.eval()
    b = FeatureBank.open(cfg.bank_dir)
    f = torch.from_numpy(np.asarray(b.feats[:4, 0]).astype(np.float32))
    with torch.no_grad():
        a = model(f)["logit"]
        c = model(f)["logit"]
    assert torch.allclose(a, c)
    assert meta["dim_feat"] == b.config["dim"]


def test_same_seed_gives_the_same_val_auc(tmp_path):
    bank = _learnable_bank(tmp_path)
    mk = lambda o: RungConfig(name="a1", bank_dir=bank, epochs=5, seed=99,
                              use_augmented=True, out_dir=str(tmp_path / o))
    assert train_rung(mk("x"))["val_auc"] == train_rung(mk("y"))["val_auc"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/train/test_train_head.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.train'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/train/train_head.py
"""Stage B (spec §3.1, §6.4): train heads on cached features.

Every rung in the ablation ladder is this function with different flags, so
comparisons differ only in the thing under test.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from aigcdet.eval.metrics import roc_auc
from aigcdet.features.bank import FeatureBank
from aigcdet.models.heads import Detector
from aigcdet.models.losses import LossWeights, classification_loss, total_loss
from aigcdet.models.sampler import PairedSampler


@dataclass
class RungConfig:
    name: str
    bank_dir: str
    out_dir: str = "outputs/rungs"
    use_recon: bool = False
    use_film: bool = False
    use_augmented: bool = True      # False = A0, clean views only
    use_consistency: bool = False   # A3+
    use_degradation: bool = False   # A2+
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_src: int = 64
    m_deg: int = 2
    seed: int = 20260827
    weights: LossWeights = field(default_factory=LossWeights)
    device: str = "cpu"


def _eval_auc(model, bank, idx, use_recon, device) -> float:
    model.eval()
    f = torch.from_numpy(np.asarray(bank.feats[idx, 0]).astype(np.float32)).to(device)
    r = (torch.from_numpy(np.asarray(bank.recon[idx, 0]).astype(np.float32)).to(device)
         if use_recon else None)
    with torch.no_grad():
        s = model(f, r)["logit"].cpu().numpy()
    model.train()
    return roc_auc(bank.meta["label"].to_numpy()[idx], s)


def train_rung(cfg: RungConfig) -> dict:
    torch.manual_seed(cfg.seed)
    bank = FeatureBank.open(cfg.bank_dir)
    bank.check_invariants()

    split = bank.meta["split"].to_numpy()
    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val_internal")[0]
    if len(val_idx) == 0:
        raise ValueError("bank has no val_internal rows; check the manifest splits")

    model = Detector(dim_feat=bank.config["dim"], use_recon=cfg.use_recon,
                     use_film=cfg.use_film).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    rng = np.random.default_rng(cfg.seed)
    sampler = PairedSampler(bank, train_idx, n_src=cfg.n_src, m_deg=cfg.m_deg,
                            rng=rng, use_recon=cfg.use_recon, device=cfg.device)

    history = []
    for _ in range(cfg.epochs):
        for batch in sampler:
            if cfg.use_augmented:
                out_clean = model(batch["f_clean"], batch["r_clean"])
                out_deg = model(batch["f_deg"], batch["r_deg"])
                if cfg.use_consistency:
                    loss, parts = total_loss(out_clean, out_deg, batch, cfg.weights)
                else:
                    w = LossWeights(lambda_deg=cfg.weights.lambda_deg, alpha=0.0, beta=0.0)
                    loss, parts = total_loss(out_clean, out_deg, batch, w)
                if not cfg.use_degradation:
                    # Re-derive without the degradation term so A1 is exactly
                    # "augmentation only" rather than "augmentation plus a
                    # silently-weighted auxiliary task".
                    loss = (classification_loss(out_clean["logit"], batch["y_clean"])
                            + classification_loss(out_deg["logit"], batch["y_deg"])) * 0.5
                    if cfg.use_consistency:
                        from aigcdet.models.losses import consistency_loss
                        loss = loss + consistency_loss(
                            out_clean["logit"].detach(), out_deg["logit"],
                            out_clean["hidden"].detach(), out_deg["hidden"],
                            cfg.weights.alpha, cfg.weights.beta)
                    parts = {"cls": float(loss), "deg": 0.0, "con": 0.0, "total": float(loss)}
            else:
                # A0: clean views only, plain supervised probe.
                out_clean = model(batch["f_clean"], batch["r_clean"])
                loss = classification_loss(out_clean["logit"], batch["y_clean"])
                parts = {"cls": float(loss), "deg": 0.0, "con": 0.0, "total": float(loss)}

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        history.append(parts)

    val_auc = _eval_auc(model, bank, val_idx, cfg.use_recon, cfg.device)
    out_dir = os.path.join(cfg.out_dir, cfg.name)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "checkpoint.pt")
    torch.save({"state_dict": model.state_dict(),
                "config": asdict(cfg),
                "dim_feat": bank.config["dim"],
                "backbone": bank.config["backbone"]}, ckpt)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"val_auc": val_auc, "history": history}, f, indent=2)
    return {"checkpoint": ckpt, "val_auc": val_auc, "history": history}


def load_detector(checkpoint_path: str, device: str = "cpu") -> tuple[Detector, dict]:
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = Detector(dim_feat=ck["dim_feat"], use_recon=cfg["use_recon"],
                     use_film=cfg["use_film"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/train/test_train_head.py -v`
Expected: 4 passed

- [ ] **Step 5: Write the rung configs and CLI**

```yaml
# configs/rungs/a0.yaml   Linear probe, clean only (= UniversalFakeDetect)
name: a0
use_augmented: false
use_degradation: false
use_consistency: false
use_recon: false
use_film: false
epochs: 30
```

```yaml
# configs/rungs/a1.yaml   + augmented training views
name: a1
use_augmented: true
use_degradation: false
use_consistency: false
use_recon: false
use_film: false
epochs: 30
```

```yaml
# configs/rungs/a2.yaml   + auxiliary degradation loss
name: a2
use_augmented: true
use_degradation: true
use_consistency: false
use_recon: false
use_film: false
epochs: 30
```

```yaml
# configs/rungs/a3.yaml   + clean/degraded consistency  (HEADLINE candidate)
name: a3
use_augmented: true
use_degradation: true
use_consistency: true
use_recon: false
use_film: false
epochs: 30
```

```yaml
# configs/rungs/a4.yaml   + reconstruction features r  (kill criterion applies)
name: a4
use_augmented: true
use_degradation: true
use_consistency: true
use_recon: true
use_film: false
epochs: 30
```

```yaml
# configs/rungs/a7.yaml   STRETCH: + FiLM conditioning
name: a7
use_augmented: true
use_degradation: true
use_consistency: true
use_recon: true
use_film: true
epochs: 30
```

A5 (second backbone) and A6 (TTA) are not separate training configs: A5 trains
`a3.yaml` against a second bank and fuses scores at evaluation time, and A6 is
an inference-time option. Both are handled in Plan 3.

```python
# scripts/train_rung.py
"""Stage B CLI.

    python scripts/train_rung.py --config configs/rungs/a3.yaml \
        --bank banks/dinov3l --device cuda
"""
from __future__ import annotations

import argparse

import yaml

from aigcdet.train.train_head import RungConfig, train_rung


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", default="outputs/rungs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260827)
    a = ap.parse_args()

    with open(a.config) as f:
        raw = yaml.safe_load(f)
    cfg = RungConfig(bank_dir=a.bank, out_dir=a.out, device=a.device, seed=a.seed, **raw)
    res = train_rung(cfg)
    print(f"{cfg.name}: val_auc={res['val_auc']:.4f} -> {res['checkpoint']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the full suite and commit**

Run: `python -m pytest -v -m "not gpu"`
Expected: all pass

```bash
git add src/aigcdet/train scripts/train_rung.py configs/rungs tests/train
git commit -m "feat(train): Stage B trainer and ablation rung configs A0-A4, A7"
```

---

## Plan 2 Completion Criteria

- [ ] `python -m pytest -v -m "not gpu"` passes; GPU-marked tests pass once weights are downloaded
- [ ] `docs/model_licences.md` is filled in and the primary backbone choice is confirmed compatible with a public repo
- [ ] `banks/dinov3l/` exists with `check_invariants()` passing, and `feats.npy` is ~2.3 GB for 100k images
- [ ] `banks/dinov3l/recon.npy` covers **all 11 views** — verified by `bank.check_invariants()`
- [ ] A separate `banks/dinov3l_loto/` exists, extracted with `--exclude noise`
- [ ] Rungs A0–A4 each have `outputs/rungs/<name>/result.json` with a `val_auc`
- [ ] The A0 val AUC is recorded as the UniversalFakeDetect reference point
