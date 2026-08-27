# Plan 1 — Foundations & Data Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a normalised, deduplicated, split image dataset plus a tested augmentation, degradation-proxy, and metrics library that every later plan depends on.

**Architecture:** Pure-function libraries first (augmentation, proxies, metrics) so they can be built and tested against a synthetic dummy manifest on day 1 while the real datasets are still downloading. Data acquisition runs in the background from hour one. Everything downstream reads a single parquet manifest, which is the contract frozen on day 1.

**Tech Stack:** Python 3.13, NumPy, Pillow, SciPy, OpenCV, pandas + pyarrow, scikit-learn, pytest. No PyTorch in this plan — Plan 2 introduces it.

**Spec:** `docs/superpowers/specs/2026-08-27-robust-aigc-detection-design-v2.md` (v2.1)

## Global Constraints

- Models under 2B parameters total (no models in this plan; the constraint binds Plan 2).
- Images stored at **short side 512**, PNG, because model input is 384px and every expert must see a downscale, never an upscale (spec §4.4).
- Training recipes never draw **JPEG q ∈ [65, 75]** or **blur σ ∈ [0.85, 1.15]** — held-out severity bands (spec §4.6).
- Evaluation grid must reproduce the brief's parameter values **exactly**: JPEG q ∈ {90, 70, 50, 30}; blur σ ∈ {0.5, 1.0, 2.0}; resize scale ∈ {0.5, 0.25} then upscale to original; noise σ ∈ {0.02, 0.05, 0.10}; colour jitter ±20% brightness/contrast/saturation; centre crop 80%.
- Every dataset and model weight records a licence and source URL in the manifest/README (spec §4.5).
- No training data may overlap COCO val2017 or DALL·E Advanced (spec §4.1 leakage guard).
- Determinism: every stochastic function takes an explicit `numpy.random.Generator`; no global seeding.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `pyproject.toml` | Package metadata, dependencies, pytest config |
| `src/aigcdet/__init__.py` | Package marker, version |
| `src/aigcdet/augment/ops.py` | The six transform primitives, each exact-parameter and deterministic |
| `src/aigcdet/augment/recipes.py` | `Op`, `Recipe`, severity labels, training-recipe sampler with held-out bands |
| `src/aigcdet/augment/scenarios.py` | The brief's 14 eval conditions + 5 named composite scenarios |
| `src/aigcdet/features/proxies.py` | Handcrafted degradation proxies `h` (no model) |
| `src/aigcdet/eval/metrics.py` | AUC, TPR@FPR, ECE, Brier, risk-coverage, bootstrap CI |
| `src/aigcdet/data/manifest.py` | Manifest schema, read/write, dummy generator |
| `src/aigcdet/data/audit.py` | Per-class per-source format/resolution/quality audit table |
| `src/aigcdet/data/normalize.py` | Decode → short-side-512 → PNG |
| `src/aigcdet/data/dedupe.py` | pHash + leakage guard against the demo set |
| `src/aigcdet/data/splits.py` | Train / held-out-generator / internal-val assignment |
| `scripts/acquire_data.py` | Subset download from WildFake, SID_Set, COCO |
| `scripts/build_dataset.py` | Orchestrates audit → normalize → dedupe → splits |
| `tests/…` | One test module per source module |

Split by responsibility, not layer: `augment/` owns everything about transforms including their labels, because the severity label definition and the transform implementation must change together.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `src/aigcdet/__init__.py`, `tests/test_smoke.py`, `.gitignore`

**Interfaces:**
- Consumes: nothing
- Produces: importable package `aigcdet` with `aigcdet.__version__`; `pytest` runnable from repo root

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smoke.py
def test_package_imports():
    import aigcdet
    assert aigcdet.__version__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet'`

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "aigcdet"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26", "pillow>=10", "scipy>=1.11", "opencv-python-headless>=4.9",
    "pandas>=2.1", "pyarrow>=14", "scikit-learn>=1.4", "tqdm>=4.66", "pyyaml>=6",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```python
# src/aigcdet/__init__.py
__version__ = "0.1.0"
```

```
# .gitignore
__pycache__/
*.egg-info/
data/
banks/
outputs/
*.png
!docs/**/*.png
```

- [ ] **Step 4: Install and run test to verify it passes**

Run: `pip install -e ".[dev]" && python -m pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/aigcdet/__init__.py tests/test_smoke.py .gitignore
git commit -m "chore: scaffold aigcdet package with pytest"
```

---

### Task 2: Manifest schema and dummy generator

The manifest is contract #1 from spec §7.1 and must exist before anything else so Plans 2 and 3 can be built against synthetic data while real downloads run.

**Files:**
- Create: `src/aigcdet/data/__init__.py`, `src/aigcdet/data/manifest.py`, `tests/data/test_manifest.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `MANIFEST_COLUMNS: list[str]`
  - `write_manifest(df: pandas.DataFrame, path: str) -> None`
  - `read_manifest(path: str) -> pandas.DataFrame`
  - `make_dummy_manifest(n: int, out_dir: str, rng: numpy.random.Generator) -> pandas.DataFrame` — writes `n` real PNG files of random noise and returns the manifest describing them

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_manifest.py
import numpy as np
from aigcdet.data.manifest import (
    MANIFEST_COLUMNS, make_dummy_manifest, read_manifest, write_manifest,
)

def test_dummy_manifest_has_schema_and_real_files(tmp_path):
    rng = np.random.default_rng(0)
    df = make_dummy_manifest(20, str(tmp_path / "img"), rng)
    assert list(df.columns) == MANIFEST_COLUMNS
    assert len(df) == 20
    assert set(df["label"].unique()) <= {0, 1}
    # every path must exist and be readable
    from PIL import Image
    for p in df["path"]:
        assert Image.open(p).size[0] > 0

def test_manifest_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    df = make_dummy_manifest(8, str(tmp_path / "img"), rng)
    out = tmp_path / "m.parquet"
    write_manifest(df, str(out))
    back = read_manifest(str(out))
    assert list(back.columns) == MANIFEST_COLUMNS
    assert len(back) == 8
    assert back["path"].tolist() == df["path"].tolist()

def test_write_manifest_rejects_missing_column(tmp_path):
    import pandas as pd, pytest
    bad = pd.DataFrame({"path": ["a.png"]})
    with pytest.raises(ValueError, match="missing columns"):
        write_manifest(bad, str(tmp_path / "bad.parquet"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.data'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/data/manifest.py
"""The manifest is the contract every other component reads (spec §7.1)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PIL import Image

MANIFEST_COLUMNS = [
    "path",       # absolute path to the normalised PNG
    "label",      # 0 = authentic, 1 = AI-generated
    "generator",  # e.g. "sdxl", "midjourney"; "" for authentic images
    "source",     # dataset of origin, e.g. "wildfake", "sid_set", "coco_val2017"
    "licence",    # licence string recorded at acquisition (spec §4.5)
    "width",
    "height",
    "split",      # "train" | "val_internal" | "heldout_generator" | "benchmark"
]


def write_manifest(df: pd.DataFrame, path: str) -> None:
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    df[MANIFEST_COLUMNS].to_parquet(path, index=False)


def read_manifest(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)[MANIFEST_COLUMNS]


def make_dummy_manifest(n: int, out_dir: str, rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic stand-in so downstream code can be built before real data lands.

    Fakes are given a mild low-pass bias so a trivial classifier can reach
    above-chance accuracy; that makes end-to-end training smoke tests meaningful.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for i in range(n):
        label = int(i % 2)
        arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        if label == 1:
            arr = np.clip(arr.astype(np.float32) * 0.5 + 64, 0, 255).astype(np.uint8)
        p = os.path.join(out_dir, f"dummy_{i:05d}.png")
        Image.fromarray(arr).save(p)
        rows.append({
            "path": p,
            "label": label,
            "generator": "dummygen" if label else "",
            "source": "dummy",
            "licence": "CC0",
            "width": 64,
            "height": 64,
            "split": "train",
        })
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
```

Also create empty `src/aigcdet/data/__init__.py` and `tests/data/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/data/test_manifest.py -v`
Expected: 3 passed

- [ ] **Step 5: Generate and commit the shared dummy manifest**

```bash
python -c "
import numpy as np
from aigcdet.data.manifest import make_dummy_manifest, write_manifest
df = make_dummy_manifest(500, 'data/dummy/img', np.random.default_rng(20260827))
write_manifest(df, 'data/dummy/manifest.parquet')
print(len(df))
"
git add src/aigcdet/data tests/data
git commit -m "feat(data): manifest schema, IO, and dummy generator"
```

Note: `data/` is gitignored, so the dummy images are not committed. Each teammate regenerates them with the fixed seed `20260827`, which produces byte-identical files.

---

### Task 3: The six transform primitives

Spec §5. These must reproduce the brief's parameters exactly; §5 requires a unit test asserting that.

**Files:**
- Create: `src/aigcdet/augment/__init__.py`, `src/aigcdet/augment/ops.py`, `tests/augment/test_ops.py`

**Interfaces:**
- Consumes: nothing
- Produces, all `(img: np.ndarray[uint8, HWC]) -> np.ndarray[uint8, HWC]` and all shape-preserving:
  - `jpeg(img, quality: int)`
  - `blur(img, sigma: float)`
  - `resize_roundtrip(img, scale: float)`
  - `noise(img, sigma: float, rng: np.random.Generator)`
  - `jitter(img, brightness: float, contrast: float, saturation: float)`
  - `center_crop(img, frac: float)` — crops then resizes back to the original size, so the op is shape-preserving like the rest
  - `OP_FUNCS: dict[str, callable]`

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_ops.py
import numpy as np
import pytest

from aigcdet.augment import ops


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(128, 96, 3), dtype=np.uint8)


@pytest.mark.parametrize("q", [90, 70, 50, 30])
def test_jpeg_preserves_shape_and_degrades_monotonically(img, q):
    out = ops.jpeg(img, quality=q)
    assert out.shape == img.shape and out.dtype == np.uint8

def test_jpeg_lower_quality_is_further_from_original(img):
    d90 = np.abs(ops.jpeg(img, 90).astype(int) - img.astype(int)).mean()
    d30 = np.abs(ops.jpeg(img, 30).astype(int) - img.astype(int)).mean()
    assert d30 > d90

@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_blur_reduces_high_frequency_energy(img, sigma):
    out = ops.blur(img, sigma=sigma)
    assert out.shape == img.shape
    # Laplacian variance is a standard sharpness proxy; blur must reduce it
    import cv2
    sharp = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
    soft = cv2.Laplacian(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
    assert soft < sharp

def test_blur_sigma_zero_is_identity(img):
    assert np.array_equal(ops.blur(img, sigma=0.0), img)

@pytest.mark.parametrize("scale", [0.5, 0.25])
def test_resize_roundtrip_returns_original_shape(img, scale):
    out = ops.resize_roundtrip(img, scale=scale)
    assert out.shape == img.shape

@pytest.mark.parametrize("sigma", [0.02, 0.05, 0.10])
def test_noise_is_deterministic_given_rng_and_scales_with_sigma(img, sigma):
    a = ops.noise(img, sigma=sigma, rng=np.random.default_rng(7))
    b = ops.noise(img, sigma=sigma, rng=np.random.default_rng(7))
    assert np.array_equal(a, b)
    small = np.abs(ops.noise(img, 0.02, np.random.default_rng(1)).astype(int) - img.astype(int)).mean()
    big = np.abs(ops.noise(img, 0.10, np.random.default_rng(1)).astype(int) - img.astype(int)).mean()
    assert big > small

def test_jitter_identity_at_zero(img):
    assert np.array_equal(ops.jitter(img, 0.0, 0.0, 0.0), img)

def test_jitter_brightness_raises_mean(img):
    assert ops.jitter(img, 0.2, 0.0, 0.0).mean() > img.mean()

def test_center_crop_80_preserves_shape_and_drops_border(img):
    out = ops.center_crop(img, frac=0.8)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)

def test_op_funcs_covers_all_six_families():
    assert set(ops.OP_FUNCS) == {"jpeg", "blur", "resize", "noise", "jitter", "crop"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/augment/test_ops.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.augment'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/augment/ops.py
"""The six transform families from the brief.

Every op is shape-preserving and uint8 in / uint8 out, so ops compose freely
in any order. Parameters are the brief's own units: JPEG quality 0-100,
blur sigma in pixels, resize scale as a fraction, noise sigma on a [0,1]
intensity scale, jitter as a signed fraction, crop as a kept fraction.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    # scipy takes sigma directly, unlike PIL's GaussianBlur radius, so the
    # brief's sigma values are reproduced exactly rather than approximated.
    if sigma <= 0:
        return img.copy()
    out = gaussian_filter(img.astype(np.float32), sigma=(sigma, sigma, 0), mode="reflect")
    return np.clip(out, 0, 255).astype(np.uint8)


def resize_roundtrip(img: np.ndarray, scale: float) -> np.ndarray:
    """Downscale then upscale back — the thumbnail-generation analogue."""
    h, w = img.shape[:2]
    sh, sw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    n = rng.normal(0.0, sigma * 255.0, size=img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def jitter(img: np.ndarray, brightness: float, contrast: float, saturation: float) -> np.ndarray:
    """Signed fractional deltas: +0.2 means +20%."""
    x = img.astype(np.float32)
    x = x * (1.0 + brightness)
    mean = x.mean()
    x = (x - mean) * (1.0 + contrast) + mean
    grey = x @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    x = grey[..., None] + (x - grey[..., None]) * (1.0 + saturation)
    return np.clip(x, 0, 255).astype(np.uint8)


def center_crop(img: np.ndarray, frac: float) -> np.ndarray:
    """Crop the central `frac` of each side, then resize back to the original
    size so the op stays shape-preserving and composable."""
    h, w = img.shape[:2]
    ch, cw = max(1, int(round(h * frac))), max(1, int(round(w * frac)))
    top, left = (h - ch) // 2, (w - cw) // 2
    cropped = img[top:top + ch, left:left + cw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)


OP_FUNCS = {
    "jpeg": jpeg,
    "blur": blur,
    "resize": resize_roundtrip,
    "noise": noise,
    "jitter": jitter,
    "crop": center_crop,
}
```

Also create empty `src/aigcdet/augment/__init__.py` and `tests/augment/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/augment/test_ops.py -v`
Expected: 16 passed

- [ ] **Step 5: Commit**

```bash
git add src/aigcdet/augment tests/augment
git commit -m "feat(augment): six transform primitives with exact brief parameters"
```

---

### Task 4: Recipes, severity labels, and held-out bands

Spec §4.6 and §5. The severity vector produced here **is** the degradation head's training label in Plan 2, so its definition is locked now.

**Files:**
- Create: `src/aigcdet/augment/recipes.py`, `tests/augment/test_recipes.py`

**Interfaces:**
- Consumes: `aigcdet.augment.ops.OP_FUNCS`
- Produces:
  - `FAMILIES: tuple[str, ...]` — `("jpeg", "blur", "resize", "noise", "jitter", "crop")`, the fixed label ordering
  - `Op` dataclass with `name: str`, `params: dict`
  - `Recipe` dataclass with `ops: tuple[Op, ...]`, methods `apply(img, rng) -> np.ndarray`, `to_json() -> str`, `from_json(s) -> Recipe`, `labels() -> dict[str, np.ndarray]` returning `{"presence": (6,) float32, "severity": (6,) float32}`
  - `HELDOUT_JPEG_Q: tuple[int, int]` = `(65, 75)`; `HELDOUT_BLUR_SIGMA: tuple[float, float]` = `(0.85, 1.15)`
  - `sample_training_recipe(rng, max_ops: int = 3) -> Recipe`

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_recipes.py
import numpy as np
import pytest

from aigcdet.augment.recipes import (
    FAMILIES, HELDOUT_BLUR_SIGMA, HELDOUT_JPEG_Q, Op, Recipe,
    sample_training_recipe,
)


def test_families_order_is_fixed():
    assert FAMILIES == ("jpeg", "blur", "resize", "noise", "jitter", "crop")


def test_empty_recipe_is_identity_and_labels_all_zero():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    r = Recipe(ops=())
    assert np.array_equal(r.apply(img, rng), img)
    lab = r.labels()
    assert lab["presence"].shape == (6,) and lab["severity"].shape == (6,)
    assert lab["presence"].sum() == 0 and lab["severity"].sum() == 0


def test_labels_mark_presence_and_normalised_severity():
    r = Recipe(ops=(Op("jpeg", {"quality": 30}), Op("blur", {"sigma": 2.0})))
    lab = r.labels()
    i_jpeg, i_blur = FAMILIES.index("jpeg"), FAMILIES.index("blur")
    assert lab["presence"][i_jpeg] == 1.0 and lab["presence"][i_blur] == 1.0
    # q=30 is the harshest listed quality -> severity 1.0; sigma=2.0 likewise
    assert lab["severity"][i_jpeg] == pytest.approx(1.0)
    assert lab["severity"][i_blur] == pytest.approx(1.0)
    assert lab["presence"][FAMILIES.index("noise")] == 0.0


def test_severity_is_monotone_in_harshness():
    s90 = Recipe((Op("jpeg", {"quality": 90}),)).labels()["severity"][0]
    s30 = Recipe((Op("jpeg", {"quality": 30}),)).labels()["severity"][0]
    assert s30 > s90


def test_recipe_json_roundtrip():
    r = Recipe((Op("jpeg", {"quality": 50}), Op("noise", {"sigma": 0.05})))
    back = Recipe.from_json(r.to_json())
    assert back == r


def test_apply_is_deterministic_for_a_given_rng_seed():
    img = np.random.default_rng(3).integers(0, 256, (48, 48, 3), dtype=np.uint8)
    r = Recipe((Op("noise", {"sigma": 0.05}),))
    a = r.apply(img, np.random.default_rng(11))
    b = r.apply(img, np.random.default_rng(11))
    assert np.array_equal(a, b)


def test_sampler_never_draws_heldout_bands():
    rng = np.random.default_rng(1234)
    lo_q, hi_q = HELDOUT_JPEG_Q
    lo_s, hi_s = HELDOUT_BLUR_SIGMA
    for _ in range(3000):
        for op in sample_training_recipe(rng).ops:
            if op.name == "jpeg":
                assert not (lo_q <= op.params["quality"] <= hi_q)
            if op.name == "blur":
                assert not (lo_s <= op.params["sigma"] <= hi_s)


def test_sampler_chains_one_to_three_distinct_families():
    rng = np.random.default_rng(5)
    for _ in range(500):
        r = sample_training_recipe(rng)
        names = [o.name for o in r.ops]
        assert 1 <= len(names) <= 3
        assert len(set(names)) == len(names)  # distinct families


def test_sampler_output_applies_cleanly():
    rng = np.random.default_rng(9)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    for _ in range(50):
        out = sample_training_recipe(rng).apply(img, rng)
        assert out.shape == img.shape and out.dtype == np.uint8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/augment/test_recipes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.augment.recipes'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/augment/recipes.py
"""Recipes compose ops and carry their own supervision labels.

The severity normalisation below IS the degradation head's target definition
(spec §3.4), so it is fixed here once and imported everywhere else.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from aigcdet.augment.ops import OP_FUNCS

FAMILIES: tuple[str, ...] = ("jpeg", "blur", "resize", "noise", "jitter", "crop")

# Held-out severity bands (spec §4.6): the brief's q=70 and sigma=1.0
# conditions must be unseen severities at evaluation time.
HELDOUT_JPEG_Q = (65, 75)
HELDOUT_BLUR_SIGMA = (0.85, 1.15)

# Ranges the training sampler draws from, chosen to cover the brief's values.
_JPEG_RANGE = (30, 98)
_BLUR_RANGE = (0.2, 2.2)
_RESIZE_RANGE = (0.25, 0.9)
_NOISE_RANGE = (0.005, 0.11)
_JITTER_RANGE = (0.05, 0.20)
_CROP_RANGE = (0.75, 0.98)


def _severity(name: str, p: dict) -> float:
    """Map raw parameters to a comparable [0, 1] harshness scale."""
    if name == "jpeg":
        return float(np.clip((100.0 - p["quality"]) / 70.0, 0.0, 1.0))
    if name == "blur":
        return float(np.clip(p["sigma"] / 2.0, 0.0, 1.0))
    if name == "resize":
        return float(np.clip((1.0 - p["scale"]) / 0.75, 0.0, 1.0))
    if name == "noise":
        return float(np.clip(p["sigma"] / 0.10, 0.0, 1.0))
    if name == "jitter":
        worst = max(abs(p["brightness"]), abs(p["contrast"]), abs(p["saturation"]))
        return float(np.clip(worst / 0.20, 0.0, 1.0))
    if name == "crop":
        return float(np.clip((1.0 - p["frac"]) / 0.20, 0.0, 1.0))
    raise KeyError(name)


@dataclass(frozen=True)
class Op:
    name: str
    params: dict

    def __post_init__(self):
        if self.name not in OP_FUNCS:
            raise KeyError(f"unknown op {self.name!r}")


@dataclass(frozen=True)
class Recipe:
    ops: tuple[Op, ...] = ()

    def apply(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = img
        for op in self.ops:
            fn = OP_FUNCS[op.name]
            out = fn(out, rng=rng, **op.params) if op.name == "noise" else fn(out, **op.params)
        return out

    def labels(self) -> dict[str, np.ndarray]:
        presence = np.zeros(len(FAMILIES), dtype=np.float32)
        severity = np.zeros(len(FAMILIES), dtype=np.float32)
        for op in self.ops:
            i = FAMILIES.index(op.name)
            presence[i] = 1.0
            # If a family appears twice in a chain, keep the harsher instance.
            severity[i] = max(severity[i], _severity(op.name, op.params))
        return {"presence": presence, "severity": severity}

    def to_json(self) -> str:
        return json.dumps([{"name": o.name, "params": o.params} for o in self.ops])

    @classmethod
    def from_json(cls, s: str) -> "Recipe":
        return cls(tuple(Op(d["name"], d["params"]) for d in json.loads(s)))


def _sample_params(name: str, rng: np.random.Generator) -> dict:
    if name == "jpeg":
        lo, hi = HELDOUT_JPEG_Q
        while True:
            q = int(rng.integers(_JPEG_RANGE[0], _JPEG_RANGE[1] + 1))
            if not (lo <= q <= hi):
                return {"quality": q}
    if name == "blur":
        lo, hi = HELDOUT_BLUR_SIGMA
        while True:
            s = float(rng.uniform(*_BLUR_RANGE))
            if not (lo <= s <= hi):
                return {"sigma": s}
    if name == "resize":
        return {"scale": float(rng.uniform(*_RESIZE_RANGE))}
    if name == "noise":
        return {"sigma": float(rng.uniform(*_NOISE_RANGE))}
    if name == "jitter":
        m = _JITTER_RANGE
        return {k: float(rng.uniform(m[0], m[1]) * rng.choice([-1.0, 1.0]))
                for k in ("brightness", "contrast", "saturation")}
    if name == "crop":
        return {"frac": float(rng.uniform(*_CROP_RANGE))}
    raise KeyError(name)


def sample_training_recipe(rng: np.random.Generator, max_ops: int = 3) -> Recipe:
    """1 to `max_ops` chained ops from distinct families (spec §5, p = 1.0)."""
    k = int(rng.integers(1, max_ops + 1))
    chosen = rng.choice(np.array(FAMILIES), size=k, replace=False)
    return Recipe(tuple(Op(str(n), _sample_params(str(n), rng)) for n in chosen))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/augment/test_recipes.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/aigcdet/augment/recipes.py tests/augment/test_recipes.py
git commit -m "feat(augment): recipes, severity labels, held-out severity bands"
```

---

### Task 5: The evaluation grid

Spec §5 requires the eval grid to reproduce the brief's values exactly, and §5 explicitly asks for a unit test asserting it.

**Files:**
- Create: `src/aigcdet/augment/scenarios.py`, `tests/augment/test_scenarios.py`

**Interfaces:**
- Consumes: `aigcdet.augment.recipes.{Op, Recipe}`
- Produces:
  - `CORE_CONDITIONS: dict[str, Recipe]` — 15 entries: `"clean"` plus the brief's 14
  - `COMPOSITE_SCENARIOS: dict[str, Recipe]` — the 5 named chains
  - `EVAL_GRID: dict[str, Recipe]` — the union, 20 entries
  - `HELDOUT_SEVERITY_CONDITIONS: frozenset[str]` — condition names that fall inside a held-out band and must be flagged as unseen severities in the robustness table

- [ ] **Step 1: Write the failing test**

```python
# tests/augment/test_scenarios.py
import numpy as np

from aigcdet.augment.recipes import HELDOUT_BLUR_SIGMA, HELDOUT_JPEG_Q
from aigcdet.augment.scenarios import (
    CORE_CONDITIONS, COMPOSITE_SCENARIOS, EVAL_GRID, HELDOUT_SEVERITY_CONDITIONS,
)


def test_core_has_clean_plus_the_briefs_fourteen():
    assert len(CORE_CONDITIONS) == 15
    assert CORE_CONDITIONS["clean"].ops == ()


def test_brief_parameters_reproduced_exactly():
    for q in (90, 70, 50, 30):
        assert CORE_CONDITIONS[f"jpeg_q{q}"].ops[0].params == {"quality": q}
    for s in (0.5, 1.0, 2.0):
        assert CORE_CONDITIONS[f"blur_s{s}"].ops[0].params == {"sigma": s}
    for sc in (0.5, 0.25):
        assert CORE_CONDITIONS[f"resize_{sc}"].ops[0].params == {"scale": sc}
    for s in (0.02, 0.05, 0.10):
        assert CORE_CONDITIONS[f"noise_s{s}"].ops[0].params == {"sigma": s}
    assert CORE_CONDITIONS["crop_80"].ops[0].params == {"frac": 0.8}
    j = CORE_CONDITIONS["jitter_20"].ops[0].params
    assert j == {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2}


def test_five_named_composites_exist_and_chain_two_or_three_ops():
    assert set(COMPOSITE_SCENARIOS) == {
        "social_repost", "messaging_app", "screenshot", "filtered_upload", "low_light_share",
    }
    for r in COMPOSITE_SCENARIOS.values():
        assert 2 <= len(r.ops) <= 3


def test_eval_grid_is_the_union_of_twenty_conditions():
    assert len(EVAL_GRID) == 20
    assert set(EVAL_GRID) == set(CORE_CONDITIONS) | set(COMPOSITE_SCENARIOS)


def test_heldout_severity_conditions_are_flagged():
    # q=70 and sigma=1.0 sit inside the bands the training sampler excludes
    assert "jpeg_q70" in HELDOUT_SEVERITY_CONDITIONS
    assert "blur_s1.0" in HELDOUT_SEVERITY_CONDITIONS
    # and the two composites that use q=70 (spec §5: "which is deliberate")
    assert "social_repost" in HELDOUT_SEVERITY_CONDITIONS
    assert "filtered_upload" in HELDOUT_SEVERITY_CONDITIONS
    assert "jpeg_q30" not in HELDOUT_SEVERITY_CONDITIONS
    lo, hi = HELDOUT_JPEG_Q
    assert lo <= 70 <= hi
    lo, hi = HELDOUT_BLUR_SIGMA
    assert lo <= 1.0 <= hi


def test_every_condition_applies_cleanly():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    for name, r in EVAL_GRID.items():
        out = r.apply(img, np.random.default_rng(0))
        assert out.shape == img.shape, name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/augment/test_scenarios.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.augment.scenarios'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/augment/scenarios.py
"""The evaluation grid: the brief's exact conditions plus five named chains.

Kept separate from `recipes.sample_training_recipe` so evaluation conditions
can never be silently drawn from the training distribution (spec §5).
"""
from __future__ import annotations

from aigcdet.augment.recipes import (
    HELDOUT_BLUR_SIGMA, HELDOUT_JPEG_Q, Op, Recipe,
)

CORE_CONDITIONS: dict[str, Recipe] = {
    "clean": Recipe(()),
    **{f"jpeg_q{q}": Recipe((Op("jpeg", {"quality": q}),)) for q in (90, 70, 50, 30)},
    **{f"blur_s{s}": Recipe((Op("blur", {"sigma": s}),)) for s in (0.5, 1.0, 2.0)},
    **{f"resize_{sc}": Recipe((Op("resize", {"scale": sc}),)) for sc in (0.5, 0.25)},
    **{f"noise_s{s}": Recipe((Op("noise", {"sigma": s}),)) for s in (0.02, 0.05, 0.10)},
    "jitter_20": Recipe((Op("jitter", {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2}),)),
    "crop_80": Recipe((Op("crop", {"frac": 0.8}),)),
}

COMPOSITE_SCENARIOS: dict[str, Recipe] = {
    "social_repost": Recipe((Op("resize", {"scale": 0.5}), Op("jpeg", {"quality": 70}))),
    "messaging_app": Recipe((Op("resize", {"scale": 0.25}), Op("jpeg", {"quality": 30}))),
    "screenshot": Recipe((Op("crop", {"frac": 0.8}), Op("resize", {"scale": 0.5}),
                          Op("jpeg", {"quality": 50}))),
    "filtered_upload": Recipe((Op("jitter", {"brightness": 0.2, "contrast": 0.2,
                                             "saturation": 0.2}),
                               Op("jpeg", {"quality": 70}))),
    "low_light_share": Recipe((Op("noise", {"sigma": 0.05}), Op("jpeg", {"quality": 50}))),
}

EVAL_GRID: dict[str, Recipe] = {**CORE_CONDITIONS, **COMPOSITE_SCENARIOS}


def _touches_heldout_band(recipe: Recipe) -> bool:
    for op in recipe.ops:
        if op.name == "jpeg" and HELDOUT_JPEG_Q[0] <= op.params["quality"] <= HELDOUT_JPEG_Q[1]:
            return True
        if op.name == "blur" and HELDOUT_BLUR_SIGMA[0] <= op.params["sigma"] <= HELDOUT_BLUR_SIGMA[1]:
            return True
    return False


#: Conditions the training sampler can never have seen at this severity.
#: The robustness table marks these rows so the distinction is visible.
HELDOUT_SEVERITY_CONDITIONS = frozenset(
    name for name, r in EVAL_GRID.items() if _touches_heldout_band(r)
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/augment/test_scenarios.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/aigcdet/augment/scenarios.py tests/augment/test_scenarios.py
git commit -m "feat(augment): eval grid with exact brief params and named scenarios"
```

---

### Task 6: Handcrafted degradation proxies `h`

Spec §3.4. Three numbers, no model, computed in `predict.py`. They validate the learned degradation head on day 4 and act as its fallback.

**Files:**
- Create: `src/aigcdet/features/__init__.py`, `src/aigcdet/features/proxies.py`, `tests/features/test_proxies.py`

**Interfaces:**
- Consumes: nothing (pure numpy/PIL/cv2)
- Produces:
  - `PROXY_NAMES: tuple[str, ...]` = `("jpeg_quality", "laplacian_var", "noise_floor")`
  - `estimate_jpeg_quality(img: np.ndarray, path: str | None = None) -> float` — reads the quantisation table when `path` is a JPEG, otherwise falls back to a blockiness estimate; returns a value in [0, 100]
  - `laplacian_variance(img: np.ndarray) -> float`
  - `noise_floor(img: np.ndarray) -> float`
  - `proxy_vector(img: np.ndarray, path: str | None = None) -> np.ndarray` — shape `(3,)` float32

- [ ] **Step 1: Write the failing test**

```python
# tests/features/test_proxies.py
import numpy as np
from PIL import Image

from aigcdet.augment import ops
from aigcdet.features.proxies import (
    PROXY_NAMES, laplacian_variance, noise_floor, proxy_vector,
    estimate_jpeg_quality,
)


def _photo(seed=0):
    """Smooth-ish synthetic image; pure noise has no blockiness structure."""
    rng = np.random.default_rng(seed)
    base = rng.normal(128, 40, (256, 256, 3))
    return np.clip(ops.blur(np.clip(base, 0, 255).astype(np.uint8), 2.0), 0, 255)


def test_proxy_vector_shape_and_names():
    v = proxy_vector(_photo())
    assert v.shape == (3,) and v.dtype == np.float32
    assert PROXY_NAMES == ("jpeg_quality", "laplacian_var", "noise_floor")


def test_laplacian_variance_drops_with_blur():
    img = _photo()
    assert laplacian_variance(ops.blur(img, 2.0)) < laplacian_variance(img)


def test_noise_floor_rises_with_added_noise():
    img = _photo()
    clean = noise_floor(img)
    noisy = noise_floor(ops.noise(img, 0.10, np.random.default_rng(0)))
    assert noisy > clean


def test_estimated_jpeg_quality_tracks_true_quality_from_file(tmp_path):
    img = _photo()
    est = {}
    for q in (30, 90):
        p = tmp_path / f"q{q}.jpg"
        Image.fromarray(img).save(p, format="JPEG", quality=q)
        est[q] = estimate_jpeg_quality(np.asarray(Image.open(p).convert("RGB")), str(p))
    assert est[90] > est[30]


def test_estimated_jpeg_quality_without_path_still_ranks_pixels(tmp_path):
    img = _photo()
    low = estimate_jpeg_quality(ops.jpeg(img, 30))
    high = estimate_jpeg_quality(ops.jpeg(img, 95))
    assert high > low


def test_proxies_are_finite_on_flat_image():
    flat = np.full((64, 64, 3), 128, dtype=np.uint8)
    assert np.all(np.isfinite(proxy_vector(flat)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/features/test_proxies.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.features'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/features/proxies.py
"""Model-free degradation proxies (spec §3.4).

Three numbers computed from pixels alone. They are cheap enough to run inside
predict.py, they validate the learned degradation head (report the Spearman
correlation on validation), and they are its fallback if it underperforms.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

PROXY_NAMES: tuple[str, ...] = ("jpeg_quality", "laplacian_var", "noise_floor")

# Standard JPEG luminance quantisation table at quality 50 (ITU-T T.81 Annex K).
_Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float64)


def _grey(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _blockiness(img: np.ndarray) -> float:
    """Mean gradient energy on the 8x8 grid minus that off it.

    JPEG quantisation concentrates discontinuities on block boundaries, so this
    difference grows as quality falls.
    """
    g = _grey(img).astype(np.float64)
    d = np.abs(np.diff(g, axis=1))
    if d.shape[1] < 16:
        return 0.0
    cols = np.arange(d.shape[1])
    on = d[:, cols % 8 == 7].mean()
    off = d[:, cols % 8 != 7].mean()
    return float(on - off)


def estimate_jpeg_quality(img: np.ndarray, path: str | None = None) -> float:
    """Quality in [0, 100]. Exact when `path` is a JPEG, estimated otherwise."""
    if path is not None:
        try:
            with Image.open(path) as im:
                tables = getattr(im, "quantization", None)
                if tables:
                    tbl = np.asarray(tables[0], dtype=np.float64).reshape(8, 8)
                    # Invert the standard scaling: S = 5000/Q for Q<50 else 200-2Q
                    scale = float(np.median(tbl / _Q50)) * 100.0
                    q = (5000.0 / scale) if scale > 100.0 else ((200.0 - scale) / 2.0)
                    return float(np.clip(q, 1.0, 100.0))
        except Exception:
            pass  # fall through to the pixel-based estimate
    b = _blockiness(img)
    # Monotone decreasing map from blockiness to quality; the absolute scale is
    # unimportant because this feeds a learned calibrator, only the ordering is.
    return float(np.clip(100.0 - 20.0 * max(b, 0.0), 1.0, 100.0))


def laplacian_variance(img: np.ndarray) -> float:
    return float(cv2.Laplacian(_grey(img), cv2.CV_64F).var())


def noise_floor(img: np.ndarray) -> float:
    """Median absolute deviation of a high-pass residual, robust to content."""
    g = _grey(img).astype(np.float64)
    resid = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    return float(np.median(np.abs(resid - np.median(resid))) * 1.4826)


def proxy_vector(img: np.ndarray, path: str | None = None) -> np.ndarray:
    return np.array([
        estimate_jpeg_quality(img, path),
        laplacian_variance(img),
        noise_floor(img),
    ], dtype=np.float32)
```

Also create empty `src/aigcdet/features/__init__.py` and `tests/features/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/features/test_proxies.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/aigcdet/features tests/features
git commit -m "feat(features): model-free degradation proxies"
```

---

### Task 7: Metrics

Spec §6.1. Built on day 1 against dummy scores so the evaluation harness is off the critical path.

**Files:**
- Create: `src/aigcdet/eval/__init__.py`, `src/aigcdet/eval/metrics.py`, `tests/eval/test_metrics.py`

**Interfaces:**
- Consumes: nothing
- Produces, all taking `y: np.ndarray[int]` (0/1) and `s: np.ndarray[float]` (higher = more likely AIGC):
  - `roc_auc(y, s) -> float`
  - `tpr_at_fpr(y, s, target_fpr: float = 0.01) -> float`
  - `threshold_at_fpr(y, s, target_fpr: float = 0.01) -> float`
  - `accuracy_at_threshold(y, s, thr: float) -> float`
  - `expected_calibration_error(y, p, n_bins: int = 15) -> float`
  - `brier(y, p) -> float`
  - `risk_coverage(y_correct: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — returns `(coverage, risk)`
  - `aurc(y_correct, confidence) -> float`
  - `accuracy_at_coverage(y_correct, confidence, coverage: float) -> float`
  - `bootstrap_ci(fn, y, s, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_metrics.py
import numpy as np
import pytest

from aigcdet.eval import metrics as M


@pytest.fixture
def separable():
    rng = np.random.default_rng(0)
    y = np.array([0] * 500 + [1] * 500)
    s = np.concatenate([rng.normal(0.2, 0.1, 500), rng.normal(0.8, 0.1, 500)])
    return y, np.clip(s, 0, 1)


def test_perfect_separation_gives_auc_one():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert M.roc_auc(y, s) == pytest.approx(1.0)


def test_random_scores_give_auc_near_half(separable):
    rng = np.random.default_rng(1)
    y, _ = separable
    assert M.roc_auc(y, rng.random(len(y))) == pytest.approx(0.5, abs=0.05)


def test_tpr_at_fpr_is_between_zero_and_one(separable):
    y, s = separable
    v = M.tpr_at_fpr(y, s, 0.01)
    assert 0.0 <= v <= 1.0


def test_threshold_at_fpr_actually_holds_that_fpr(separable):
    y, s = separable
    thr = M.threshold_at_fpr(y, s, 0.01)
    realised = float(((s >= thr) & (y == 0)).sum() / (y == 0).sum())
    assert realised <= 0.02


def test_ece_is_zero_for_perfectly_calibrated_predictions():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.random(20000) < p).astype(int)
    assert M.expected_calibration_error(y, p, n_bins=15) < 0.02


def test_ece_is_large_for_overconfident_predictions():
    y = np.array([0, 1] * 500)
    p = np.array([0.99, 0.01] * 500)   # confidently wrong
    assert M.expected_calibration_error(y, p) > 0.9


def test_brier_of_perfect_prediction_is_zero():
    y = np.array([0, 1, 1, 0])
    assert M.brier(y, y.astype(float)) == pytest.approx(0.0)


def test_risk_coverage_is_monotone_when_confidence_is_informative():
    correct = np.array([1] * 80 + [0] * 20)
    conf = np.concatenate([np.linspace(0.9, 1.0, 80), np.linspace(0.0, 0.5, 20)])
    cov, risk = M.risk_coverage(correct, conf)
    assert cov[-1] == pytest.approx(1.0)
    assert risk[0] <= risk[-1]          # deferring the unconfident lowers risk
    assert 0.0 <= M.aurc(correct, conf) <= 1.0


def test_accuracy_at_coverage_beats_full_coverage_when_confidence_is_informative():
    correct = np.array([1] * 80 + [0] * 20)
    conf = np.concatenate([np.linspace(0.9, 1.0, 80), np.linspace(0.0, 0.5, 20)])
    assert M.accuracy_at_coverage(correct, conf, 0.8) > M.accuracy_at_coverage(correct, conf, 1.0)


def test_bootstrap_ci_brackets_the_point_estimate_and_is_reproducible(separable):
    y, s = separable
    lo, hi = M.bootstrap_ci(M.roc_auc, y, s, n=200, seed=0)
    point = M.roc_auc(y, s)
    assert lo <= point <= hi
    assert (lo, hi) == M.bootstrap_ci(M.roc_auc, y, s, n=200, seed=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.eval'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/eval/metrics.py
"""Metrics for spec §6.1.

Convention throughout: `y` is 0/1 with 1 = AI-generated, `s` is a score where
higher means more likely AI-generated, `p` is a calibrated probability.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def threshold_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float = 0.01) -> float:
    """Lowest threshold whose false-positive rate does not exceed the target."""
    fpr, _, thr = roc_curve(y, s)
    ok = np.where(fpr <= target_fpr)[0]
    return float(thr[ok[-1]]) if len(ok) else float(np.max(s) + 1.0)


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    ok = np.where(fpr <= target_fpr)[0]
    return float(tpr[ok[-1]]) if len(ok) else 0.0


def accuracy_at_threshold(y: np.ndarray, s: np.ndarray, thr: float) -> float:
    return float(((s >= thr).astype(int) == y).mean())


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def risk_coverage(y_correct: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by descending confidence; risk[i] is the error rate of the most
    confident i+1 predictions, coverage[i] their share of the set."""
    order = np.argsort(-confidence, kind="stable")
    c = y_correct[order].astype(float)
    n = len(c)
    coverage = np.arange(1, n + 1) / n
    risk = 1.0 - np.cumsum(c) / np.arange(1, n + 1)
    return coverage, risk


def aurc(y_correct: np.ndarray, confidence: np.ndarray) -> float:
    coverage, risk = risk_coverage(y_correct, confidence)
    return float(np.trapezoid(risk, coverage))


def accuracy_at_coverage(y_correct: np.ndarray, confidence: np.ndarray, coverage: float) -> float:
    k = max(1, int(round(len(y_correct) * coverage)))
    order = np.argsort(-confidence, kind="stable")[:k]
    return float(y_correct[order].mean())


def bootstrap_ci(
    fn: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray, s: np.ndarray,
    n: int = 1000, seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap. Resamples that lose a class are skipped."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], s[idx]))
    if not vals:
        raise ValueError("no valid bootstrap resamples; is one class empty?")
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))
```

Also create empty `src/aigcdet/eval/__init__.py` and `tests/eval/__init__.py`.

Note: `np.trapezoid` requires NumPy ≥ 2.0. If the environment pins NumPy 1.x, use `np.trapz`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/eval/test_metrics.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/aigcdet/eval tests/eval
git commit -m "feat(eval): AUC, TPR@FPR, ECE, Brier, risk-coverage, bootstrap CI"
```

---

### Task 8: Data acquisition and audit

Spec §4.1, §4.2, §4.5. **Start the download before implementing anything else on day 1** — it is the critical path.

**Files:**
- Create: `scripts/acquire_data.py`, `src/aigcdet/data/audit.py`, `tests/data/test_audit.py`

**Interfaces:**
- Consumes: `aigcdet.features.proxies.estimate_jpeg_quality`
- Produces:
  - `audit_table(paths: list[str], labels: list[int], sources: list[str]) -> pandas.DataFrame` with one row per (source, label) and columns `n, fmt_top, width_median, height_median, jpeg_q_median`
  - `audit_flags(df: pandas.DataFrame) -> list[str]` — human-readable warnings where the two classes differ materially

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_audit.py
import numpy as np
from PIL import Image

from aigcdet.data.audit import audit_flags, audit_table


def _write(p, size, fmt, quality=None):
    arr = np.random.default_rng(0).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(p, format=fmt, **({"quality": quality} if quality else {}))
    return str(p)


def test_audit_table_reports_per_class_shape(tmp_path):
    paths, labels, sources = [], [], []
    for i in range(4):
        paths.append(_write(tmp_path / f"r{i}.jpg", (640, 480), "JPEG", 75))
        labels.append(0); sources.append("coco")
    for i in range(4):
        paths.append(_write(tmp_path / f"f{i}.png", (1024, 1024), "PNG"))
        labels.append(1); sources.append("sdxl")
    df = audit_table(paths, labels, sources)
    assert len(df) == 2
    real = df[df["label"] == 0].iloc[0]
    fake = df[df["label"] == 1].iloc[0]
    assert real["fmt_top"] == "JPEG" and fake["fmt_top"] == "PNG"
    assert fake["width_median"] > real["width_median"]


def test_audit_flags_detects_the_confound(tmp_path):
    paths, labels, sources = [], [], []
    for i in range(4):
        paths.append(_write(tmp_path / f"r{i}.jpg", (640, 480), "JPEG", 75))
        labels.append(0); sources.append("coco")
    for i in range(4):
        paths.append(_write(tmp_path / f"f{i}.png", (1024, 1024), "PNG"))
        labels.append(1); sources.append("sdxl")
    flags = audit_flags(audit_table(paths, labels, sources))
    assert any("format" in f.lower() for f in flags)
    assert any("resolution" in f.lower() for f in flags)


def test_audit_flags_empty_when_classes_match(tmp_path):
    paths, labels, sources = [], [], []
    for i in range(4):
        paths.append(_write(tmp_path / f"r{i}.png", (512, 512), "PNG"))
        labels.append(0); sources.append("a")
    for i in range(4):
        paths.append(_write(tmp_path / f"f{i}.png", (512, 512), "PNG"))
        labels.append(1); sources.append("b")
    assert audit_flags(audit_table(paths, labels, sources)) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_audit.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.data.audit'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/data/audit.py
"""Profile the classes before normalising them (spec §4.2 defence 1).

The output table goes in docs/data_audit.md and in the README; it is the
figure that motivates the whole normalisation step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from aigcdet.features.proxies import estimate_jpeg_quality


def audit_table(paths: list[str], labels: list[int], sources: list[str]) -> pd.DataFrame:
    rows = []
    for p, lab, src in zip(paths, labels, sources):
        with Image.open(p) as im:
            fmt, (w, h) = im.format, im.size
            q = estimate_jpeg_quality(np.asarray(im.convert("RGB")), p)
        rows.append({"source": src, "label": lab, "fmt": fmt,
                     "width": w, "height": h, "jpeg_q": q})
    df = pd.DataFrame(rows)
    return (df.groupby(["source", "label"], as_index=False)
              .agg(n=("fmt", "size"),
                   fmt_top=("fmt", lambda s: s.mode().iloc[0]),
                   width_median=("width", "median"),
                   height_median=("height", "median"),
                   jpeg_q_median=("jpeg_q", "median")))


def audit_flags(df: pd.DataFrame) -> list[str]:
    """Warn where authentic and generated images differ in ways a detector
    could exploit without looking at content at all."""
    flags: list[str] = []
    real, fake = df[df["label"] == 0], df[df["label"] == 1]
    if real.empty or fake.empty:
        return flags
    if set(real["fmt_top"]) != set(fake["fmt_top"]):
        flags.append(
            f"Format confound: authentic {sorted(set(real['fmt_top']))} vs "
            f"generated {sorted(set(fake['fmt_top']))}")
    rw, fw = real["width_median"].median(), fake["width_median"].median()
    if max(rw, fw) / max(1.0, min(rw, fw)) > 1.5:
        flags.append(f"Resolution confound: median width {rw:.0f} vs {fw:.0f}")
    rq, fq = real["jpeg_q_median"].median(), fake["jpeg_q_median"].median()
    if abs(rq - fq) > 10:
        flags.append(f"JPEG-quality confound: median q {rq:.0f} vs {fq:.0f}")
    return flags
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/data/test_audit.py -v`
Expected: 3 passed

- [ ] **Step 5: Write the acquisition script**

```python
# scripts/acquire_data.py
"""Subset download for WildFake, SID_Set, and COCO val2017 (spec §4.1).

Pulls selected generator folders and parquet shards rather than whole
repositories — download volume is the binding risk, not compute (spec §4.4).
Records each dataset's licence, which the manifest and README require (§4.5).

Usage:
    python scripts/acquire_data.py --dataset sid_set --limit 30000 --out data/raw
    python scripts/acquire_data.py --dataset wildfake --generators sdxl,sd15,midjourney \
        --limit 30000 --out data/raw
    python scripts/acquire_data.py --dataset coco_val2017 --out data/raw
"""
from __future__ import annotations

import argparse
import json
import os

LICENCES = {
    "sid_set": "see https://huggingface.co/datasets/saberzl/SID_Set — confirm before use",
    "wildfake": "see https://modelscope.cn/datasets/hy2628982280/WildFake — confirm before use",
    "coco_val2017": "CC BY 4.0 (images: Flickr terms) — https://cocodataset.org/#termsofuse",
}


def acquire_sid_set(out: str, limit: int) -> None:
    from datasets import load_dataset  # pip install datasets
    ds = load_dataset("saberzl/SID_Set", split="train", streaming=True)
    os.makedirs(out, exist_ok=True)
    n = 0
    for rec in ds:
        if n >= limit:
            break
        # SID_Set labels: 0 real, 1 fully synthetic, 2 tampered.
        # Tampered is out of scope for the binary task (spec §4.1).
        if rec.get("label") == 2:
            continue
        sub = "real" if rec["label"] == 0 else "fake"
        d = os.path.join(out, "sid_set", sub)
        os.makedirs(d, exist_ok=True)
        rec["image"].save(os.path.join(d, f"{n:07d}.png"))
        n += 1
    print(f"sid_set: wrote {n}")


def acquire_wildfake(out: str, limit: int, generators: list[str]) -> None:
    from modelscope.msdatasets import MsDataset  # pip install modelscope
    raise SystemExit(
        "WildFake layout must be inspected before subsetting. Run:\n"
        "  python -c \"from modelscope.hub.api import HubApi; "
        "print(HubApi().get_dataset_files('hy2628982280/WildFake'))\"\n"
        f"then pull only the folders for: {generators}, writing to {out}/wildfake/<generator>/."
    )


def acquire_coco_val2017(out: str) -> None:
    import urllib.request
    import zipfile
    os.makedirs(out, exist_ok=True)
    zp = os.path.join(out, "val2017.zip")
    if not os.path.exists(zp):
        urllib.request.urlretrieve("http://images.cocodataset.org/zips/val2017.zip", zp)
    with zipfile.ZipFile(zp) as z:
        z.extractall(os.path.join(out, "coco_val2017"))
    print("coco_val2017: extracted")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["sid_set", "wildfake", "coco_val2017"])
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--limit", type=int, default=30000)
    ap.add_argument("--generators", default="")
    a = ap.parse_args()

    if a.dataset == "sid_set":
        acquire_sid_set(a.out, a.limit)
    elif a.dataset == "wildfake":
        acquire_wildfake(a.out, a.limit,
                         [g for g in a.generators.split(",") if g])
    else:
        acquire_coco_val2017(a.out)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "LICENCES.json"), "a") as f:
        f.write(json.dumps({a.dataset: LICENCES[a.dataset]}) + "\n")


if __name__ == "__main__":
    main()
```

`acquire_wildfake` deliberately raises with instructions rather than guessing the
repository layout: the folder structure must be inspected once, by a human, on
day 1. Replace the body with the concrete paths at that point.

- [ ] **Step 6: Commit**

```bash
git add src/aigcdet/data/audit.py tests/data/test_audit.py scripts/acquire_data.py
git commit -m "feat(data): acquisition scripts and pre-normalisation confound audit"
```

---

### Task 9: Normalisation to short-side-512 PNG

Spec §4.2 defence 1 and §4.4.

**Files:**
- Create: `src/aigcdet/data/normalize.py`, `tests/data/test_normalize.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `SHORT_SIDE: int = 512`
  - `normalize_image(src: str, dst: str, short_side: int = SHORT_SIDE) -> tuple[int, int]` — returns `(width, height)` written; always writes PNG; never upscales beyond the source (records the true size instead)
  - `normalize_many(pairs: list[tuple[str, str]], workers: int = 8) -> list[tuple[int, int]]`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_normalize.py
import numpy as np
from PIL import Image

from aigcdet.data.normalize import SHORT_SIDE, normalize_image, normalize_many


def _src(tmp_path, name, size, fmt="JPEG"):
    arr = np.random.default_rng(0).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    p = tmp_path / name
    Image.fromarray(arr).save(p, format=fmt)
    return str(p)


def test_short_side_is_512_and_exceeds_model_input():
    assert SHORT_SIDE == 512 and SHORT_SIDE > 384


def test_downscales_large_image_to_short_side_and_writes_png(tmp_path):
    src = _src(tmp_path, "big.jpg", (2048, 1024))
    dst = str(tmp_path / "out.png")
    w, h = normalize_image(src, dst)
    assert min(w, h) == 512
    assert (w, h) == (1024, 512)          # aspect ratio preserved
    with Image.open(dst) as im:
        assert im.format == "PNG"


def test_does_not_upscale_a_small_image(tmp_path):
    src = _src(tmp_path, "small.jpg", (300, 200))
    dst = str(tmp_path / "small.png")
    w, h = normalize_image(src, dst)
    assert (w, h) == (300, 200)


def test_converts_greyscale_and_rgba_to_rgb(tmp_path):
    p = tmp_path / "g.png"
    Image.fromarray(np.zeros((600, 600), dtype=np.uint8), mode="L").save(p)
    dst = str(tmp_path / "g_out.png")
    normalize_image(str(p), dst)
    with Image.open(dst) as im:
        assert im.mode == "RGB"


def test_normalize_many_processes_all_pairs(tmp_path):
    pairs = [(_src(tmp_path, f"i{i}.jpg", (800, 600)), str(tmp_path / f"o{i}.png"))
             for i in range(5)]
    sizes = normalize_many(pairs, workers=2)
    assert len(sizes) == 5
    assert all(min(w, h) == 512 for w, h in sizes)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.data.normalize'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/data/normalize.py
"""Give every image identical resolution and encoding history before any
augmentation is applied, so the two classes cannot be told apart by their
container (spec §4.2).

Short side 512 because model input is 384: every expert must see a downscale,
never an upscale (spec §4.4).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

SHORT_SIDE = 512


def normalize_image(src: str, dst: str, short_side: int = SHORT_SIDE) -> tuple[int, int]:
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        # Never upscale: inventing detail would fabricate the forensic evidence
        # this project is trying to measure.
        if min(w, h) > short_side:
            scale = short_side / min(w, h)
            w, h = max(1, round(w * scale)), max(1, round(h * scale))
            im = im.resize((w, h), Image.LANCZOS)
        im.save(dst, format="PNG", optimize=False)
    return (w, h)


def normalize_many(pairs: list[tuple[str, str]], workers: int = 8) -> list[tuple[int, int]]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda ab: normalize_image(*ab), pairs))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/data/test_normalize.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/aigcdet/data/normalize.py tests/data/test_normalize.py
git commit -m "feat(data): short-side-512 PNG normalisation"
```

---

### Task 10: Perceptual-hash leakage guard

Spec §4.1. Training data must not overlap the demo set, or the demo-set number measures memorisation.

**Files:**
- Create: `src/aigcdet/data/dedupe.py`, `tests/data/test_dedupe.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `phash(img: np.ndarray, hash_size: int = 8) -> int` — 64-bit perceptual hash as a Python int
  - `hamming(a: int, b: int) -> int`
  - `build_hash_index(paths: list[str]) -> dict[str, int]`
  - `find_leaks(candidate_hashes: dict[str, int], demo_hashes: dict[str, int], max_distance: int = 4) -> dict[str, str]` — maps leaked candidate path to the demo path it matches

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_dedupe.py
import numpy as np
from PIL import Image

from aigcdet.augment import ops
from aigcdet.data.dedupe import build_hash_index, find_leaks, hamming, phash


def _photo(seed):
    rng = np.random.default_rng(seed)
    return ops.blur(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8), 3.0)


def test_identical_images_hash_identically():
    img = _photo(0)
    assert phash(img) == phash(img.copy())


def test_recompressed_image_stays_within_threshold():
    img = _photo(1)
    assert hamming(phash(img), phash(ops.jpeg(img, 40))) <= 4


def test_resized_image_stays_within_threshold():
    img = _photo(2)
    assert hamming(phash(img), phash(ops.resize_roundtrip(img, 0.5))) <= 4


def test_different_images_are_far_apart():
    assert hamming(phash(_photo(3)), phash(_photo(99))) > 10


def test_find_leaks_flags_a_recompressed_duplicate(tmp_path):
    img = _photo(4)
    demo_p = tmp_path / "demo.png"
    cand_p = tmp_path / "cand.png"
    Image.fromarray(img).save(demo_p)
    Image.fromarray(ops.jpeg(img, 50)).save(cand_p)
    other_p = tmp_path / "other.png"
    Image.fromarray(_photo(77)).save(other_p)

    demo = build_hash_index([str(demo_p)])
    cand = build_hash_index([str(cand_p), str(other_p)])
    leaks = find_leaks(cand, demo, max_distance=4)
    assert str(cand_p) in leaks
    assert str(other_p) not in leaks
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_dedupe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.data.dedupe'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/data/dedupe.py
"""Leakage guard (spec §4.1).

Implemented directly rather than pulling in `imagehash`: it is a dozen lines,
it keeps the dependency list short, and it is worth having under test.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from scipy.fft import dct


def phash(img: np.ndarray, hash_size: int = 8) -> int:
    """DCT-based perceptual hash. Robust to recompression and rescaling,
    which is exactly the overlap we need to catch."""
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(grey, (hash_size * 4, hash_size * 4), interpolation=cv2.INTER_AREA)
    d = dct(dct(small.astype(np.float64), axis=0, norm="ortho"), axis=1, norm="ortho")
    low = d[:hash_size, :hash_size]
    med = np.median(low[1:, 1:])          # skip DC, which only encodes brightness
    bits = (low > med).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return int(bin(a ^ b).count("1"))


def build_hash_index(paths: list[str]) -> dict[str, int]:
    idx = {}
    for p in paths:
        with Image.open(p) as im:
            idx[p] = phash(np.asarray(im.convert("RGB"), dtype=np.uint8))
    return idx


def find_leaks(
    candidate_hashes: dict[str, int],
    demo_hashes: dict[str, int],
    max_distance: int = 4,
) -> dict[str, str]:
    """Return {candidate_path: matching_demo_path} for every near-duplicate."""
    demo_items = list(demo_hashes.items())
    leaks: dict[str, str] = {}
    for cp, ch in candidate_hashes.items():
        for dp, dh in demo_items:
            if hamming(ch, dh) <= max_distance:
                leaks[cp] = dp
                break
    return leaks
```

Note: `find_leaks` is O(n·m). For 100k candidates against 14k demo images that is
1.4 · 10⁹ comparisons — too slow in pure Python. Once the test passes, replace the
inner loop with a vectorised NumPy popcount over a packed `uint8` array:

```python
def _pack(hashes: list[int]) -> np.ndarray:
    return np.array([[(h >> (8 * i)) & 0xFF for i in range(8)] for h in hashes], dtype=np.uint8)

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def find_leaks(candidate_hashes, demo_hashes, max_distance: int = 4):
    cps, chs = list(candidate_hashes), _pack(list(candidate_hashes.values()))
    dps, dhs = list(demo_hashes), _pack(list(demo_hashes.values()))
    leaks = {}
    for start in range(0, len(cps), 2048):                    # chunk to bound memory
        block = chs[start:start + 2048]
        dist = _POPCOUNT[block[:, None, :] ^ dhs[None, :, :]].sum(axis=2)
        hit_c, hit_d = np.where(dist <= max_distance)
        for ci, di in zip(hit_c, hit_d):
            leaks.setdefault(cps[start + ci], dps[di])
    return leaks
```

Keep the same tests; they must still pass after the swap.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/data/test_dedupe.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/aigcdet/data/dedupe.py tests/data/test_dedupe.py
git commit -m "feat(data): pHash leakage guard against the demo set"
```

---

### Task 11: Splits and the dataset build orchestrator

Spec §4.6. Splits are frozen before any training and never revisited.

**Files:**
- Create: `src/aigcdet/data/splits.py`, `scripts/build_dataset.py`, `tests/data/test_splits.py`

**Interfaces:**
- Consumes: `aigcdet.data.manifest.{MANIFEST_COLUMNS, read_manifest, write_manifest}`
- Produces:
  - `assign_splits(df, heldout_generators: list[str], val_fraction: float = 0.1, seed: int = 20260827) -> pandas.DataFrame` — returns a copy with `split` filled
  - `choose_heldout_generators(df, n: int = 2, seed: int = 20260827) -> list[str]` — picks the 2 held-out families deterministically from generators with enough images
  - `split_report(df) -> pandas.DataFrame` — counts per split × label, for the README

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_splits.py
import numpy as np
import pandas as pd
import pytest

from aigcdet.data.manifest import MANIFEST_COLUMNS
from aigcdet.data.splits import assign_splits, choose_heldout_generators, split_report


def _df(n_per_gen=250, gens=("g1", "g2", "g3", "g4")):  # >=200/gen: see MIN_HELDOUT_IMAGES
    rows = []
    for g in gens:
        for i in range(n_per_gen):
            rows.append({"path": f"/f/{g}/{i}.png", "label": 1, "generator": g,
                         "source": "wildfake", "licence": "x", "width": 512,
                         "height": 512, "split": ""})
    for i in range(n_per_gen * len(gens)):
        rows.append({"path": f"/r/{i}.png", "label": 0, "generator": "",
                     "source": "sid_set", "licence": "x", "width": 512,
                     "height": 512, "split": ""})
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def test_choose_heldout_generators_is_deterministic_and_returns_two():
    df = _df()
    a = choose_heldout_generators(df, n=2, seed=1)
    b = choose_heldout_generators(df, n=2, seed=1)
    assert a == b and len(a) == 2 and set(a) <= {"g1", "g2", "g3", "g4"}


def test_heldout_generator_images_never_land_in_train():
    df = _df()
    out = assign_splits(df, heldout_generators=["g1", "g2"])
    assert set(out[out["generator"].isin(["g1", "g2"])]["split"]) == {"heldout_generator"}
    assert "g1" not in set(out[out["split"] == "train"]["generator"])


def test_splits_are_exhaustive_and_disjoint():
    out = assign_splits(_df(), heldout_generators=["g1"])
    assert (out["split"] != "").all()
    assert set(out["split"]) <= {"train", "val_internal", "heldout_generator"}
    assert len(out) == len(_df())


def test_validation_fraction_is_approximately_respected():
    out = assign_splits(_df(), heldout_generators=["g1"], val_fraction=0.1)
    pool = out[out["split"].isin(["train", "val_internal"])]
    frac = (pool["split"] == "val_internal").mean()
    assert frac == pytest.approx(0.1, abs=0.03)


def test_assignment_is_reproducible_with_the_same_seed():
    a = assign_splits(_df(), ["g1"], seed=7)["split"].tolist()
    b = assign_splits(_df(), ["g1"], seed=7)["split"].tolist()
    assert a == b


def test_split_report_counts_by_split_and_label():
    rep = split_report(assign_splits(_df(), ["g1"]))
    assert {"split", "label", "n"} <= set(rep.columns)
    assert rep["n"].sum() == len(_df())


def test_raises_when_a_heldout_generator_is_absent():
    with pytest.raises(ValueError, match="not present"):
        assign_splits(_df(), heldout_generators=["nope"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/data/test_splits.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.data.splits'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/data/splits.py
"""Splits, frozen before any training (spec §4.6).

The held-out-transform-family split (A3-LOTO) is NOT here: it is a property of
the training recipe sampler, not of the image set, and Plan 2 configures it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_SEED = 20260827
MIN_HELDOUT_IMAGES = 200


def choose_heldout_generators(df: pd.DataFrame, n: int = 2, seed: int = DEFAULT_SEED) -> list[str]:
    """Pick n generator families to exclude from training entirely.

    Restricted to families with at least MIN_HELDOUT_IMAGES images so the
    held-out evaluation has enough support for a usable confidence interval.
    """
    counts = df[df["label"] == 1]["generator"].value_counts()
    eligible = sorted(counts[counts >= MIN_HELDOUT_IMAGES].index.tolist())
    if len(eligible) < n:
        raise ValueError(
            f"need {n} generators with >={MIN_HELDOUT_IMAGES} images, have {len(eligible)}")
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(np.array(eligible), size=n, replace=False).tolist())


def assign_splits(
    df: pd.DataFrame,
    heldout_generators: list[str],
    val_fraction: float = 0.1,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    present = set(df["generator"].unique())
    missing = [g for g in heldout_generators if g not in present]
    if missing:
        raise ValueError(f"held-out generators not present in manifest: {missing}")

    out = df.copy()
    out["split"] = ""
    held = out["generator"].isin(heldout_generators)
    out.loc[held, "split"] = "heldout_generator"

    rest = ~held
    rng = np.random.default_rng(seed)
    draws = rng.random(int(rest.sum()))
    out.loc[rest, "split"] = np.where(draws < val_fraction, "val_internal", "train")
    return out


def split_report(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["split", "label"], as_index=False)
              .size().rename(columns={"size": "n"}))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/data/test_splits.py -v`
Expected: 7 passed

- [ ] **Step 5: Write the orchestrator**

```python
# scripts/build_dataset.py
"""audit -> normalise -> dedupe -> split -> manifest (spec §4).

Usage:
    python scripts/build_dataset.py --raw data/raw --out data/normalized \
        --demo-dir data/raw/demo --manifest data/manifest.parquet
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

from aigcdet.data.audit import audit_flags, audit_table
from aigcdet.data.dedupe import build_hash_index, find_leaks
from aigcdet.data.manifest import MANIFEST_COLUMNS, write_manifest
from aigcdet.data.normalize import normalize_many
from aigcdet.data.splits import assign_splits, choose_heldout_generators, split_report

IMG_EXT = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")


def _scan(root: str) -> list[str]:
    out = []
    for ext in IMG_EXT:
        out += glob.glob(os.path.join(root, "**", ext), recursive=True)
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--demo-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    # Directory convention from acquire_data.py: raw/<source>/<real|fake>/...
    # and for WildFake, raw/wildfake/<generator>/...
    rows = []
    for p in _scan(a.raw):
        rel = os.path.relpath(p, a.raw).split(os.sep)
        source = rel[0]
        bucket = rel[1] if len(rel) > 1 else ""
        label = 0 if bucket == "real" else 1
        generator = "" if label == 0 else (bucket if source == "wildfake" else source)
        rows.append({"src": p, "label": label, "generator": generator, "source": source})
    raw = pd.DataFrame(rows)
    print(f"scanned {len(raw)} raw images")

    # 1. Audit BEFORE normalising: the table is the figure for the README.
    at = audit_table(raw["src"].tolist(), raw["label"].tolist(), raw["source"].tolist())
    flags = audit_flags(at)
    os.makedirs("docs", exist_ok=True)
    with open("docs/data_audit.md", "w") as f:
        f.write("# Pre-normalisation data audit\n\n")
        f.write(at.to_markdown(index=False) + "\n\n## Flags\n\n")
        f.write("\n".join(f"- {x}" for x in flags) if flags else "- none\n")
    print(f"audit flags: {flags}")

    # 2. Normalise.
    pairs, dsts = [], []
    for i, r in raw.iterrows():
        dst = os.path.join(a.out, r["source"], r["generator"] or "real", f"{i:07d}.png")
        pairs.append((r["src"], dst))
        dsts.append(dst)
    sizes = normalize_many(pairs, workers=a.workers)
    raw["path"], raw["width"], raw["height"] = dsts, [s[0] for s in sizes], [s[1] for s in sizes]

    # 3. Leakage guard against the demo set.
    demo = build_hash_index(_scan(a.demo_dir))
    cand = build_hash_index(raw["path"].tolist())
    leaks = find_leaks(cand, demo, max_distance=4)
    print(f"dropping {len(leaks)} images that near-duplicate the demo set")
    raw = raw[~raw["path"].isin(leaks)].reset_index(drop=True)
    # Spec §4.1(2): no COCO-derived authentic source may be trained on.
    raw = raw[~((raw["label"] == 0) & (raw["source"].str.contains("coco")))].reset_index(drop=True)

    licences = {}
    lp = os.path.join(a.raw, "LICENCES.json")
    if os.path.exists(lp):
        for line in open(lp):
            licences.update(json.loads(line))
    raw["licence"] = raw["source"].map(lambda s: licences.get(s, "UNRECORDED"))
    raw["split"] = ""

    df = raw[MANIFEST_COLUMNS]
    held = choose_heldout_generators(df, n=2)
    print(f"held-out generators: {held}")
    df = assign_splits(df, heldout_generators=held)
    write_manifest(df, a.manifest)
    print(split_report(df).to_string(index=False))
    with open("docs/splits.json", "w") as f:
        json.dump({"heldout_generators": held, "seed": 20260827}, f, indent=2)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the full test suite and commit**

Run: `python -m pytest -v`
Expected: all tests pass

```bash
git add src/aigcdet/data/splits.py scripts/build_dataset.py tests/data/test_splits.py
git commit -m "feat(data): splits and end-to-end dataset build orchestrator"
```

---

## Plan 1 Completion Criteria

- [ ] `python -m pytest -v` passes with no failures
- [ ] `data/dummy/manifest.parquet` exists (500 images, seed 20260827) so Plans 2 and 3 can be built without real data
- [ ] `docs/data_audit.md` exists and its flags are quoted in the README draft
- [ ] `docs/splits.json` records the two held-out generators and the seed
- [ ] The real manifest reports ≥20k images (the spec's floor) with both classes present in every split
- [ ] The leakage-guard drop count is recorded — it is a number the README should state
