# Plan 3 — Calibration & Evaluation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the robustness table, the calibration and abstention results, the content-blind control, the baselines, and the error-analysis note — the four evidence deliverables the submission is judged on.

**Architecture:** Evaluation reuses the Stage A machinery with *fixed* recipes instead of sampled ones, producing an eval bank of shape `(N, 20 conditions, D)`. Every rung then scores the whole grid in seconds. Calibration and the decision policy are fitted on internal validation only and applied unchanged everywhere else.

**Tech Stack:** Everything from Plans 1 and 2, plus `scipy.stats` for Spearman and `matplotlib` for the heatmap.

**Spec:** `docs/superpowers/specs/2026-08-27-robust-aigc-detection-design-v2.md` (v2.1)

**Depends on:** Plan 1 (augmentation, metrics), Plan 2 (banks, detector, trainer).

## Global Constraints

- **Two-tier evaluation cap (spec §4.4a), stated in every table.** Ablation/selection tier: 5k internal validation + a 5k stratified benchmark subsample, full 20-condition grid. Final-report tier: the complete 13.8k benchmark on the 15 core conditions, run **once**, on day 6.
- **Model selection rule, fixed before results exist (spec §6.4):** the headline model is the rung in A3–A6 with the highest **robust TPR @ 1% FPR on internal validation, held-out generators**. Not clean AUC, not the external benchmark.
- All thresholds, temperatures, and EQI fits come from internal validation only (spec §6.7).
- Bootstrap 95% CIs on every AUC, 1000 resamples, seed recorded.
- The stratified benchmark subsample seed is fixed at **20260827** and committed to `docs/eval_subsample.json`.
- `HELDOUT_SEVERITY_CONDITIONS` rows are marked in the robustness table as unseen severities.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/aigcdet/eval/grid.py` | Eval bank extraction over fixed conditions; grid scoring |
| `src/aigcdet/calibrate/temperature.py` | Global and degradation-conditional temperature scaling |
| `src/aigcdet/calibrate/eqi.py` | Evidence Quality Index fitting |
| `src/aigcdet/calibrate/policy.py` | Clear/Review/Flag thresholds and auto-decided fraction |
| `src/aigcdet/eval/controls.py` | Content-blind control experiment |
| `src/aigcdet/eval/report.py` | Robustness table, heatmap, degradation-head validation |
| `src/aigcdet/baselines/*.py` | UnivFD, NPR, AEROBLADE |
| `scripts/run_ablation.py` | Trains and evaluates every rung, emits the table |
| `scripts/make_error_sheet.py` | Error-analysis contact sheets |

---

### Task 1: Eval bank over fixed conditions

**Files:**
- Create: `src/aigcdet/eval/grid.py`, `tests/eval/test_grid.py`

**Interfaces:**
- Consumes: `augment.scenarios.EVAL_GRID`, `features.bank.BankWriter`, `features.extract.{load_backbone, embed}`
- Produces:
  - `extract_eval_bank(manifest_df, backbone_name, out_dir, conditions=EVAL_GRID, device, seed, batch_size) -> str` — bank whose view axis is the condition axis; `config["conditions"]` lists names in order; view 0 is `"clean"`
  - `score_grid(model, bank, use_recon, device) -> pandas.DataFrame` with columns `condition, image_idx, label, generator, source, score`
  - `stratified_subsample(meta_df, n, seed) -> np.ndarray` — indices balanced across class × generator × source

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_grid.py
import numpy as np
import pandas as pd
import pytest

from aigcdet.augment.scenarios import EVAL_GRID
from aigcdet.data.manifest import make_dummy_manifest
from aigcdet.features.bank import FeatureBank


def test_eval_bank_view_axis_is_the_condition_axis(tmp_path, monkeypatch):
    from aigcdet.eval import grid
    from aigcdet.features.backbones import BackboneSpec
    spec = BackboneSpec("fake", "none", 64, 4, 1, 0)
    monkeypatch.setattr(grid, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(grid, "embed",
                        lambda m, s, imgs, device, batch_size=16:
                            np.stack([np.full(s.dim, float(i.mean()), np.float32) for i in imgs]))

    df = make_dummy_manifest(5, str(tmp_path / "img"), np.random.default_rng(0))
    out = grid.extract_eval_bank(df, "fake", str(tmp_path / "eb"), device="cpu")
    b = FeatureBank.open(out)
    assert b.config["conditions"][0] == "clean"
    assert list(b.config["conditions"]) == list(EVAL_GRID)
    assert b.feats.shape == (5, len(EVAL_GRID), 4)
    b.check_invariants()   # view 0 clean


def test_score_grid_returns_one_row_per_image_and_condition(tmp_path, monkeypatch):
    import torch
    from aigcdet.eval import grid
    from aigcdet.features.backbones import BackboneSpec
    from aigcdet.models.heads import Detector
    spec = BackboneSpec("fake", "none", 64, 4, 1, 0)
    monkeypatch.setattr(grid, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(grid, "embed",
                        lambda m, s, imgs, device, batch_size=16:
                            np.zeros((len(imgs), s.dim), np.float32))
    df = make_dummy_manifest(4, str(tmp_path / "img2"), np.random.default_rng(0))
    b = FeatureBank.open(grid.extract_eval_bank(df, "fake", str(tmp_path / "eb2"), device="cpu"))
    model = Detector(dim_feat=4, use_recon=False)
    out = grid.score_grid(model, b, use_recon=False, device="cpu")
    assert len(out) == 4 * len(EVAL_GRID)
    assert set(out.columns) >= {"condition", "image_idx", "label", "generator", "source", "score"}
    assert out["condition"].nunique() == len(EVAL_GRID)


def test_stratified_subsample_preserves_class_balance_and_is_reproducible():
    from aigcdet.eval.grid import stratified_subsample
    meta = pd.DataFrame({
        "label": [0] * 60 + [1] * 60,
        "generator": [""] * 60 + ["g1"] * 30 + ["g2"] * 30,
        "source": ["coco"] * 60 + ["wf"] * 60,
    })
    a = stratified_subsample(meta, 40, seed=1)
    b = stratified_subsample(meta, 40, seed=1)
    assert np.array_equal(a, b)
    assert len(a) == 40
    picked = meta.iloc[a]
    assert abs((picked["label"] == 1).mean() - 0.5) < 0.15
    assert picked["generator"].nunique() >= 2


def test_stratified_subsample_returns_everything_when_n_exceeds_pool():
    from aigcdet.eval.grid import stratified_subsample
    meta = pd.DataFrame({"label": [0, 1, 0, 1], "generator": ["", "g", "", "g"],
                         "source": ["a"] * 4})
    assert len(stratified_subsample(meta, 100, seed=0)) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_grid.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.eval.grid'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/eval/grid.py
"""Evaluation over the fixed condition grid (spec §6.1-6.2).

The eval bank reuses the Stage A layout, but its view axis is the CONDITION
axis: view j is always condition j, identically for every image. That makes
every rung's grid score a single pass over cached vectors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from aigcdet.augment.scenarios import EVAL_GRID
from aigcdet.features.backbones import embed, load_backbone
from aigcdet.features.bank import BankWriter
from aigcdet.features.proxies import proxy_vector


def extract_eval_bank(manifest_df: pd.DataFrame, backbone_name: str, out_dir: str,
                      conditions: dict | None = None, device: str = "cuda",
                      seed: int = 20260827, batch_size: int = 16) -> str:
    conditions = conditions or EVAL_GRID
    names = list(conditions)
    if names[0] != "clean":
        raise ValueError("condition 0 must be 'clean' to preserve the bank invariant")

    df = manifest_df.reset_index(drop=True)
    model, spec = load_backbone(backbone_name, device=device)
    w = BankWriter(out_dir, len(df), len(names), spec.dim, backbone_name, seed)
    w._config["conditions"] = names

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"eval:{backbone_name}"):
        with Image.open(row["path"]) as im:
            base = np.asarray(im.convert("RGB"), dtype=np.uint8)
        # Fixed per-image seed so noise conditions are reproducible run to run.
        views = [conditions[n].apply(base, np.random.default_rng([seed, int(i)]))
                 for n in names]
        labels = [conditions[n].labels() for n in names]
        w.write_image(
            int(i),
            {"path": row["path"], "label": int(row["label"]),
             "generator": row["generator"], "source": row["source"],
             "split": row["split"]},
            feats=embed(model, spec, views, device=device, batch_size=batch_size),
            presence=np.stack([l["presence"] for l in labels]),
            severity=np.stack([l["severity"] for l in labels]),
            proxies=np.stack([proxy_vector(v) for v in views]),
            recipes=[conditions[n].to_json() for n in names],
        )
    w.close()
    return out_dir


@torch.no_grad()
def score_grid(model, bank, use_recon: bool = False, device: str = "cpu",
               batch_size: int = 4096) -> pd.DataFrame:
    names = bank.config["conditions"]
    meta = bank.meta
    rows = []
    model.eval()
    for j, cond in enumerate(names):
        feats = np.asarray(bank.feats[:, j, :]).astype(np.float32)
        recon = (np.asarray(bank.recon[:, j, :]).astype(np.float32)
                 if use_recon and bank.recon is not None else None)
        scores = []
        for s in range(0, len(feats), batch_size):
            f = torch.from_numpy(feats[s:s + batch_size]).to(device)
            r = torch.from_numpy(recon[s:s + batch_size]).to(device) if recon is not None else None
            scores.append(model(f, r)["logit"].cpu().numpy())
        scores = np.concatenate(scores)
        rows.append(pd.DataFrame({
            "condition": cond,
            "image_idx": np.arange(len(feats)),
            "label": meta["label"].to_numpy(),
            "generator": meta["generator"].to_numpy(),
            "source": meta["source"].to_numpy(),
            "score": scores,
        }))
    return pd.concat(rows, ignore_index=True)


def stratified_subsample(meta_df: pd.DataFrame, n: int, seed: int = 20260827) -> np.ndarray:
    """Balanced across class x generator x source (spec §4.4a).

    A uniform random subsample would under-represent small generator families,
    which are exactly the ones the held-out evaluation cares about.
    """
    if n >= len(meta_df):
        return np.arange(len(meta_df))
    rng = np.random.default_rng(seed)
    keys = (meta_df["label"].astype(str) + "|" + meta_df["generator"].astype(str)
            + "|" + meta_df["source"].astype(str))
    groups = {k: np.where(keys.to_numpy() == k)[0] for k in keys.unique()}
    per = max(1, n // len(groups))
    picked: list[int] = []
    for idx in groups.values():
        take = min(per, len(idx))
        picked.extend(rng.choice(idx, size=take, replace=False).tolist())
    # Top up to exactly n from whatever is left, so the tier size is exact.
    remaining = np.setdiff1d(np.arange(len(meta_df)), np.array(picked))
    if len(picked) < n and len(remaining):
        extra = rng.choice(remaining, size=min(n - len(picked), len(remaining)), replace=False)
        picked.extend(extra.tolist())
    return np.sort(np.array(picked[:n], dtype=np.int64))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/eval/test_grid.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/aigcdet/eval/grid.py tests/eval/test_grid.py
git commit -m "feat(eval): fixed-condition eval bank, grid scoring, stratified subsample"
```

---

### Task 2: Calibration and EQI

**Files:**
- Create: `src/aigcdet/calibrate/__init__.py`, `src/aigcdet/calibrate/temperature.py`, `src/aigcdet/calibrate/eqi.py`, `tests/calibrate/test_temperature.py`, `tests/calibrate/test_eqi.py`

**Interfaces:**
- Consumes: `eval.metrics.expected_calibration_error`
- Produces:
  - `GlobalTemperature.fit(logits, y) -> self`; `.transform(logits) -> probs`; `.temperature: float`
  - `ConditionalTemperature(cond_dim).fit(logits, y, cond) -> self`; `.transform(logits, cond) -> probs`
  - `EQI.fit(cond, correct) -> self`; `.predict(cond) -> np.ndarray` in [0,1]

- [ ] **Step 1: Write the failing tests**

```python
# tests/calibrate/test_temperature.py
import numpy as np
import pytest

from aigcdet.calibrate.temperature import ConditionalTemperature, GlobalTemperature
from aigcdet.eval.metrics import expected_calibration_error


def _overconfident(n=4000, seed=0):
    """True probability is sigmoid(z), but the model reports sigmoid(3z)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return z * 3.0, y


def test_global_temperature_reduces_ece():
    logits, y = _overconfident()
    before = expected_calibration_error(y, 1 / (1 + np.exp(-logits)))
    cal = GlobalTemperature().fit(logits, y)
    after = expected_calibration_error(y, cal.transform(logits))
    assert after < before
    assert cal.temperature > 1.0          # shrinks over-confident logits


def test_global_temperature_preserves_ranking():
    logits, y = _overconfident()
    p = GlobalTemperature().fit(logits, y).transform(logits)
    assert np.array_equal(np.argsort(p), np.argsort(logits))


def test_conditional_temperature_beats_global_when_miscalibration_varies():
    """Two regimes with different overconfidence; a single scalar cannot fix both."""
    rng = np.random.default_rng(1)
    n = 4000
    z = rng.normal(0, 1.5, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    regime = (rng.random(n) < 0.5).astype(float)
    logits = z * np.where(regime > 0, 4.0, 1.2)
    cond = regime[:, None]

    g = GlobalTemperature().fit(logits, y)
    c = ConditionalTemperature(cond_dim=1).fit(logits, y, cond, epochs=400)
    ece_g = expected_calibration_error(y, g.transform(logits))
    ece_c = expected_calibration_error(y, c.transform(logits, cond))
    assert ece_c < ece_g


def test_conditional_temperature_is_always_positive():
    rng = np.random.default_rng(2)
    logits, y = _overconfident(1000, 2)
    cond = rng.normal(size=(1000, 3))
    c = ConditionalTemperature(cond_dim=3).fit(logits, y, cond, epochs=50)
    assert (c.temperatures(cond) > 0).all()


def test_transform_outputs_are_valid_probabilities():
    logits, y = _overconfident(500, 3)
    p = GlobalTemperature().fit(logits, y).transform(logits)
    assert ((p >= 0) & (p <= 1)).all()
```

```python
# tests/calibrate/test_eqi.py
import numpy as np

from aigcdet.calibrate.eqi import EQI


def test_eqi_tracks_correctness_probability():
    """Two degradation regimes with very different accuracy: EQI must separate."""
    rng = np.random.default_rng(0)
    n = 4000
    severe = (rng.random(n) < 0.5).astype(float)
    correct = np.where(severe > 0, rng.random(n) < 0.55, rng.random(n) < 0.95).astype(int)
    cond = severe[:, None]
    e = EQI().fit(cond, correct)
    pred = e.predict(cond)
    assert pred[severe > 0].mean() < pred[severe == 0].mean()
    assert abs(pred[severe == 0].mean() - 0.95) < 0.08


def test_eqi_output_is_bounded():
    rng = np.random.default_rng(1)
    cond = rng.normal(size=(500, 4))
    correct = (rng.random(500) < 0.7).astype(int)
    p = EQI().fit(cond, correct).predict(cond)
    assert ((p >= 0) & (p <= 1)).all()


def test_eqi_is_reproducible():
    rng = np.random.default_rng(2)
    cond = rng.normal(size=(300, 2))
    correct = (rng.random(300) < 0.6).astype(int)
    a = EQI(seed=7).fit(cond, correct).predict(cond)
    b = EQI(seed=7).fit(cond, correct).predict(cond)
    np.testing.assert_allclose(a, b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/calibrate -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.calibrate'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/calibrate/temperature.py
"""Calibration (spec §3.7).

Global temperature scaling is the standard baseline. The conditional variant
lets the temperature depend on the estimated degradation, which is the point:
a detector that stays accurate but becomes wildly overconfident at JPEG-30 is
dangerous in a moderation pipeline, and one scalar cannot fix both regimes.

Fitted on internal validation only, with the classifier frozen.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class GlobalTemperature:
    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "GlobalTemperature":
        lg = torch.tensor(logits, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                lg / log_t.exp(), yt)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = float(log_t.exp().item())
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        return _sigmoid(logits / self.temperature)


class ConditionalTemperature:
    """T(cond) = softplus(Linear(cond)) + eps, so temperature is always positive."""

    def __init__(self, cond_dim: int, eps: float = 1e-2, seed: int = 20260827):
        torch.manual_seed(seed)
        self.eps = eps
        self.net = nn.Linear(cond_dim, 1)
        nn.init.zeros_(self.net.weight)
        nn.init.constant_(self.net.bias, 0.5414)   # softplus(0.5414) ~= 1.0

    def temperatures(self, cond: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = nn.functional.softplus(
                self.net(torch.tensor(cond, dtype=torch.float32))).squeeze(-1)
        return t.numpy() + self.eps

    def fit(self, logits: np.ndarray, y: np.ndarray, cond: np.ndarray,
            epochs: int = 300, lr: float = 0.05) -> "ConditionalTemperature":
        lg = torch.tensor(logits, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        c = torch.tensor(cond, dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            t = nn.functional.softplus(self.net(c)).squeeze(-1) + self.eps
            loss = nn.functional.binary_cross_entropy_with_logits(lg / t, yt)
            loss.backward()
            opt.step()
        return self

    def transform(self, logits: np.ndarray, cond: np.ndarray) -> np.ndarray:
        return _sigmoid(logits / self.temperatures(cond))
```

```python
# src/aigcdet/calibrate/eqi.py
"""Evidence Quality Index (spec §3.6).

EQI is fitted, not hand-defined: it is the model's probability of being correct
given the degradation evidence, estimated on validation data. That makes it
interpretable ("this image retains ~40% usable evidence") and directly usable
for abstention, rather than a hand-tuned severity score.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class EQI:
    def __init__(self, seed: int = 20260827):
        self.seed = seed
        self.model = LogisticRegression(max_iter=2000, random_state=seed)
        self._mu = None
        self._sd = None

    def _z(self, cond: np.ndarray) -> np.ndarray:
        return (cond - self._mu) / self._sd

    def fit(self, cond: np.ndarray, correct: np.ndarray) -> "EQI":
        self._mu = cond.mean(axis=0, keepdims=True)
        self._sd = cond.std(axis=0, keepdims=True) + 1e-6
        self.model.fit(self._z(cond), correct.astype(int))
        return self

    def predict(self, cond: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._z(cond))[:, 1]
```

Also create empty `src/aigcdet/calibrate/__init__.py` and `tests/calibrate/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/calibrate -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/aigcdet/calibrate tests/calibrate
git commit -m "feat(calibrate): global and degradation-conditional temperature, EQI"
```

---

### Task 3: Decision policy and the reviewer-load number

Spec §1.3 and §6.1 — this produces the Impact figure.

**Files:**
- Create: `src/aigcdet/calibrate/policy.py`, `tests/calibrate/test_policy.py`

**Interfaces:**
- Consumes: `eval.metrics.threshold_at_fpr`
- Produces:
  - `Policy` dataclass: `flag_threshold, clear_threshold, eqi_threshold`
  - `fit_policy(p, y, eqi, target_fpr=0.01, target_coverage=0.85) -> Policy`
  - `decide(p, eqi, policy) -> np.ndarray[str]` of `"clear" | "review" | "flag"`
  - `auto_decided_fraction(decisions) -> float`
  - `policy_report(p, y, eqi, policy) -> dict` with `auto_fraction, realised_fpr, accuracy_on_auto, review_fraction`

- [ ] **Step 1: Write the failing test**

```python
# tests/calibrate/test_policy.py
import numpy as np
import pytest

from aigcdet.calibrate.policy import (
    auto_decided_fraction, decide, fit_policy, policy_report,
)


def _population(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(int)
    eqi = rng.uniform(0.2, 1.0, n)
    # High-EQI images are scored well; low-EQI ones are near chance.
    p = np.where(rng.random(n) < eqi,
                 np.where(y == 1, rng.uniform(0.7, 1.0, n), rng.uniform(0.0, 0.3, n)),
                 rng.uniform(0.3, 0.7, n))
    return p, y, eqi


def test_decisions_are_only_the_three_allowed_labels():
    p, y, eqi = _population()
    pol = fit_policy(p, y, eqi)
    d = decide(p, eqi, pol)
    assert set(np.unique(d)) <= {"clear", "review", "flag"}
    assert len(d) == len(p)


def test_low_eqi_images_are_routed_to_review():
    p, y, eqi = _population()
    pol = fit_policy(p, y, eqi)
    d = decide(p, eqi, pol)
    assert (d[eqi < pol.eqi_threshold] == "review").all()


def test_realised_fpr_on_auto_decided_images_respects_the_target():
    p, y, eqi = _population()
    pol = fit_policy(p, y, eqi, target_fpr=0.01)
    rep = policy_report(p, y, eqi, pol)
    assert rep["realised_fpr"] <= 0.02


def test_auto_decided_fraction_is_between_zero_and_one():
    p, y, eqi = _population()
    d = decide(p, eqi, fit_policy(p, y, eqi))
    f = auto_decided_fraction(d)
    assert 0.0 <= f <= 1.0
    assert f == pytest.approx(1.0 - (d == "review").mean())


def test_accuracy_on_auto_decided_beats_accuracy_on_everything():
    p, y, eqi = _population()
    pol = fit_policy(p, y, eqi)
    rep = policy_report(p, y, eqi, pol)
    all_acc = (((p >= 0.5).astype(int)) == y).mean()
    assert rep["accuracy_on_auto"] > all_acc


def test_report_fields_are_present():
    p, y, eqi = _population()
    rep = policy_report(p, y, eqi, fit_policy(p, y, eqi))
    assert {"auto_fraction", "realised_fpr", "accuracy_on_auto", "review_fraction"} <= set(rep)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/calibrate/test_policy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.calibrate.policy'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/calibrate/policy.py
"""Clear / Review / Flag (spec §1.3, §3.7).

The output number this exists to produce: the share of a moderation queue the
system decides without a human, while holding false positives on authentic
images at the target rate. That is the Impact figure the rubric asks for.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aigcdet.eval.metrics import threshold_at_fpr


@dataclass
class Policy:
    flag_threshold: float      # p >= this  -> flag as AI-generated
    clear_threshold: float     # p <= this  -> clear as authentic
    eqi_threshold: float       # below this -> always review, whatever p says


def fit_policy(p: np.ndarray, y: np.ndarray, eqi: np.ndarray,
               target_fpr: float = 0.01, target_coverage: float = 0.85) -> Policy:
    """Flag threshold holds the FPR target; the EQI threshold buys back coverage
    by deferring the least-evidenced images."""
    flag = threshold_at_fpr(y, p, target_fpr)
    # Clear the authentic side symmetrically: same FPR target with labels flipped.
    clear = 1.0 - threshold_at_fpr(1 - y, 1 - p, target_fpr)
    eqi_thr = float(np.quantile(eqi, max(0.0, 1.0 - target_coverage)))
    return Policy(flag_threshold=float(flag), clear_threshold=float(clear),
                  eqi_threshold=eqi_thr)


def decide(p: np.ndarray, eqi: np.ndarray, policy: Policy) -> np.ndarray:
    out = np.full(len(p), "review", dtype=object)
    confident = eqi >= policy.eqi_threshold
    out[confident & (p >= policy.flag_threshold)] = "flag"
    out[confident & (p <= policy.clear_threshold)] = "clear"
    return out.astype(str)


def auto_decided_fraction(decisions: np.ndarray) -> float:
    return float((decisions != "review").mean())


def policy_report(p: np.ndarray, y: np.ndarray, eqi: np.ndarray, policy: Policy) -> dict:
    d = decide(p, eqi, policy)
    auto = d != "review"
    pred = np.where(d == "flag", 1, 0)
    authentic_auto = auto & (y == 0)
    return {
        "auto_fraction": float(auto.mean()),
        "review_fraction": float((~auto).mean()),
        "realised_fpr": float((pred[authentic_auto] == 1).mean()) if authentic_auto.any() else 0.0,
        "accuracy_on_auto": float((pred[auto] == y[auto]).mean()) if auto.any() else 0.0,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/calibrate/test_policy.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/aigcdet/calibrate/policy.py tests/calibrate/test_policy.py
git commit -m "feat(calibrate): Clear/Review/Flag policy and reviewer-load reporting"
```

---

### Task 4: Content-blind control

Spec §4.2 defence 2 and §6.5. This is the experiment that tells you whether your headline numbers are real.

**Files:**
- Create: `src/aigcdet/eval/controls.py`, `tests/eval/test_controls.py`

**Interfaces:**
- Consumes: `eval.metrics.roc_auc`
- Produces:
  - `thumbnail_features(paths, size=16) -> np.ndarray` — `(N, size*size*3)` float32
  - `metadata_features(paths) -> np.ndarray` — `(N, 4)`: width, height, aspect, estimated JPEG quality
  - `content_blind_auc(features, labels, seed=..., n_splits=5) -> dict` with `auc, auc_ci, verdict`
  - `VERDICT_THRESHOLDS = {"broken": 0.85, "suspect": 0.70}`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_controls.py
import numpy as np
from PIL import Image

from aigcdet.eval.controls import (
    VERDICT_THRESHOLDS, content_blind_auc, metadata_features, thumbnail_features,
)


def _write(p, size, fmt="PNG", quality=None):
    arr = np.random.default_rng(0).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(p, format=fmt, **({"quality": quality} if quality else {}))
    return str(p)


def test_thumbnail_features_have_the_declared_width(tmp_path):
    paths = [_write(tmp_path / f"{i}.png", (200, 200)) for i in range(3)]
    f = thumbnail_features(paths, size=16)
    assert f.shape == (3, 16 * 16 * 3) and f.dtype == np.float32


def test_metadata_features_capture_size_and_quality(tmp_path):
    a = _write(tmp_path / "a.jpg", (640, 480), "JPEG", 40)
    b = _write(tmp_path / "b.png", (1024, 1024))
    f = metadata_features([a, b])
    assert f.shape == (2, 4)
    assert f[1, 0] > f[0, 0]          # width
    assert f[0, 3] < f[1, 3]          # estimated quality lower for the q40 JPEG


def test_control_detects_a_deliberately_broken_dataset(tmp_path):
    """Reals 640x480 JPEG, fakes 1024x1024 PNG: metadata alone must separate them."""
    paths, labels = [], []
    for i in range(30):
        paths.append(_write(tmp_path / f"r{i}.jpg", (640, 480), "JPEG", 40)); labels.append(0)
    for i in range(30):
        paths.append(_write(tmp_path / f"f{i}.png", (1024, 1024))); labels.append(1)
    res = content_blind_auc(metadata_features(paths), np.array(labels))
    assert res["auc"] > VERDICT_THRESHOLDS["broken"]
    assert res["verdict"] == "broken"


def test_control_reports_clean_when_classes_are_indistinguishable(tmp_path):
    paths, labels = [], []
    for i in range(40):
        paths.append(_write(tmp_path / f"x{i}.png", (512, 512)))
        labels.append(i % 2)          # label uncorrelated with anything visible
    res = content_blind_auc(metadata_features(paths), np.array(labels))
    assert res["verdict"] == "clean"
    assert res["auc"] < VERDICT_THRESHOLDS["suspect"]


def test_result_includes_a_confidence_interval(tmp_path):
    paths = [_write(tmp_path / f"y{i}.png", (512, 512)) for i in range(40)]
    labels = np.array([i % 2 for i in range(40)])
    res = content_blind_auc(metadata_features(paths), labels)
    lo, hi = res["auc_ci"]
    assert lo <= res["auc"] <= hi
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_controls.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.eval.controls'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/eval/controls.py
"""Content-blind control (spec §4.2, §6.5).

Train a classifier that CANNOT see content — 16x16 thumbnails, or file
metadata alone — and report its AUC. A high score means the dataset is
separable without looking at the image, so every headline number is suspect.
A near-chance score is positive evidence that the real model's signal is
content. This is run on our own splits AND on the official demo set.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from aigcdet.eval.metrics import bootstrap_ci, roc_auc
from aigcdet.features.proxies import estimate_jpeg_quality

VERDICT_THRESHOLDS = {"broken": 0.85, "suspect": 0.70}


def thumbnail_features(paths: list[str], size: int = 16) -> np.ndarray:
    out = []
    for p in paths:
        with Image.open(p) as im:
            t = im.convert("RGB").resize((size, size), Image.BILINEAR)
        out.append(np.asarray(t, dtype=np.float32).reshape(-1) / 255.0)
    return np.stack(out).astype(np.float32)


def metadata_features(paths: list[str]) -> np.ndarray:
    out = []
    for p in paths:
        with Image.open(p) as im:
            w, h = im.size
            q = estimate_jpeg_quality(np.asarray(im.convert("RGB"), dtype=np.uint8), p)
        out.append([float(w), float(h), float(w) / max(1.0, h), float(q)])
    return np.asarray(out, dtype=np.float32)


def content_blind_auc(features: np.ndarray, labels: np.ndarray,
                      seed: int = 20260827, n_splits: int = 5) -> dict:
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, random_state=seed))
    proba = cross_val_predict(clf, features, labels, cv=n_splits,
                              method="predict_proba")[:, 1]
    auc = roc_auc(labels, proba)
    ci = bootstrap_ci(roc_auc, labels, proba, n=500, seed=seed)
    if auc > VERDICT_THRESHOLDS["broken"]:
        verdict = "broken"
    elif auc > VERDICT_THRESHOLDS["suspect"]:
        verdict = "suspect"
    else:
        verdict = "clean"
    return {"auc": auc, "auc_ci": ci, "verdict": verdict}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/eval/test_controls.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/aigcdet/eval/controls.py tests/eval/test_controls.py
git commit -m "feat(eval): content-blind control experiment with verdict thresholds"
```

---

### Task 5: Baselines

Spec §6.3, in priority order. UnivFD is free (rung A0 on the CLIP bank); AEROBLADE is free (branch `r` alone, unthresholded). Only NPR needs new code.

**Files:**
- Create: `src/aigcdet/baselines/__init__.py`, `src/aigcdet/baselines/npr.py`, `src/aigcdet/baselines/aeroblade.py`, `tests/baselines/test_npr.py`, `tests/baselines/test_aeroblade.py`

**Interfaces:**
- Consumes: `features.recon.recon_features`
- Produces:
  - `npr_feature(img: np.ndarray, stride: int = 2) -> np.ndarray` — `(4,)` neighbouring-pixel-relationship statistics
  - `NPRDetector.fit(features, labels) -> self`; `.score(features) -> np.ndarray`
  - `aeroblade_score(recon_vec: np.ndarray) -> float` — negative L1 reconstruction error, so higher means more likely AI-generated (training-free)

- [ ] **Step 1: Write the failing tests**

```python
# tests/baselines/test_npr.py
import numpy as np

from aigcdet.augment import ops
from aigcdet.baselines.npr import NPRDetector, npr_feature


def _upsampled(seed):
    """Mimics a generator's up-sampling: build small, then scale up."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    return ops.resize_roundtrip(np.repeat(np.repeat(small, 4, 0), 4, 1), 1.0)


def _natural(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)


def test_npr_feature_shape_and_finiteness():
    f = npr_feature(_natural(0))
    assert f.shape == (4,) and np.isfinite(f).all()


def test_npr_separates_upsampled_from_natural_images():
    up = np.stack([npr_feature(_upsampled(i)) for i in range(20)])
    nat = np.stack([npr_feature(_natural(100 + i)) for i in range(20)])
    X = np.concatenate([nat, up]); y = np.array([0] * 20 + [1] * 20)
    from aigcdet.eval.metrics import roc_auc
    d = NPRDetector().fit(X, y)
    assert roc_auc(y, d.score(X)) > 0.9


def test_npr_signal_degrades_under_resize_as_expected():
    """The documented failure mode: resampling destroys up-sampling artifacts.
    This test asserts the failure, because the plot of it is a deliverable."""
    from aigcdet.eval.metrics import roc_auc
    up = [_upsampled(i) for i in range(20)]
    nat = [_natural(100 + i) for i in range(20)]
    y = np.array([0] * 20 + [1] * 20)

    X_clean = np.stack([npr_feature(i) for i in nat + up])
    X_deg = np.stack([npr_feature(ops.resize_roundtrip(i, 0.25)) for i in nat + up])
    d = NPRDetector().fit(X_clean, y)
    assert roc_auc(y, d.score(X_clean)) > roc_auc(y, d.score(X_deg))
```

```python
# tests/baselines/test_aeroblade.py
import numpy as np

from aigcdet.baselines.aeroblade import aeroblade_score
from aigcdet.features.recon import RECON_FEATURE_NAMES


def test_lower_reconstruction_error_yields_a_higher_aigc_score():
    low = np.zeros(len(RECON_FEATURE_NAMES), dtype=np.float32)
    low[0] = 0.01                      # tiny L1 -> looks like a VAE round-trip
    high = np.zeros(len(RECON_FEATURE_NAMES), dtype=np.float32)
    high[0] = 0.20
    assert aeroblade_score(low) > aeroblade_score(high)


def test_score_is_finite_for_a_zero_vector():
    assert np.isfinite(aeroblade_score(np.zeros(len(RECON_FEATURE_NAMES), np.float32)))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/baselines -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.baselines'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/baselines/npr.py
"""NPR-style baseline (spec §6.3).

Captures up-sampling artifacts through neighbouring-pixel relationships.
Near-free to implement and expected to COLLAPSE under resize and blur, which
is why it is included: that failure is the most informative row in the
robustness table.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def npr_feature(img: np.ndarray, stride: int = 2) -> np.ndarray:
    """Compare within-cell and across-cell neighbour differences.

    A generator's transposed-convolution up-sampling makes pixels inside a
    stride-sized cell more similar to each other than the cell grid would
    otherwise predict.
    """
    g = img.astype(np.float32).mean(axis=2)
    h, w = g.shape
    h, w = h - h % stride, w - w % stride
    g = g[:h, :w]
    dh = np.abs(np.diff(g, axis=1))
    dv = np.abs(np.diff(g, axis=0))
    cols, rows = np.arange(dh.shape[1]), np.arange(dv.shape[0])
    within_h = dh[:, cols % stride != (stride - 1)].mean()
    across_h = dh[:, cols % stride == (stride - 1)].mean()
    within_v = dv[rows % stride != (stride - 1), :].mean()
    across_v = dv[rows % stride == (stride - 1), :].mean()
    eps = 1e-6
    return np.array([within_h, across_h,
                     within_h / (across_h + eps),
                     within_v / (across_v + eps)], dtype=np.float32)


class NPRDetector:
    def __init__(self, seed: int = 20260827):
        self.clf = make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=2000, random_state=seed))

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "NPRDetector":
        self.clf.fit(features, labels)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(features)[:, 1]
```

```python
# src/aigcdet/baselines/aeroblade.py
"""AEROBLADE baseline (spec §6.3): training-free, from branch `r` alone.

Latent-diffusion images round-trip through their own autoencoder with low
error, so the AIGC score is simply the negated L1 reconstruction error.
No training, which makes it the cheapest baseline in the set.
"""
from __future__ import annotations

import numpy as np

from aigcdet.features.recon import RECON_FEATURE_NAMES

_L1_INDEX = RECON_FEATURE_NAMES.index("l1")


def aeroblade_score(recon_vec: np.ndarray) -> float:
    return float(-recon_vec[_L1_INDEX])
```

Also create empty `src/aigcdet/baselines/__init__.py` and `tests/baselines/__init__.py`.

**UnivFD needs no module:** it is rung A0 trained on the `clipl` bank. Record it in the results table under its published name with a footnote saying so.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/baselines -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/aigcdet/baselines tests/baselines
git commit -m "feat(baselines): NPR and training-free AEROBLADE"
```

---

### Task 6: Robustness table, heatmap, and degradation-head validation

Spec §6.1, §6.4, and the §3.4 requirement that `D` be checked against the proxies on day 4.

**Files:**
- Create: `src/aigcdet/eval/report.py`, `tests/eval/test_report.py`

**Interfaces:**
- Consumes: `eval.metrics.*`, `augment.scenarios.HELDOUT_SEVERITY_CONDITIONS`
- Produces:
  - `condition_metrics(scores_df, probs=None, clean_threshold=None, seed=...) -> pandas.DataFrame` — one row per condition with `auc, auc_lo, auc_hi, tpr_at_1pct, acc_oracle, acc_fixed, ece, n, heldout_severity`
  - `robustness_table(per_rung: dict[str, pandas.DataFrame], tier: str) -> pandas.DataFrame` — rungs × conditions, plus a `robust_auc` mean column
  - `save_heatmap(table, path) -> None`
  - `validate_degradation_head(pred_severity, proxies, families) -> pandas.DataFrame` — Spearman correlation per family against the matching proxy
  - `to_markdown(table, tier, path) -> None` — writes the table with its tier label

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_report.py
import numpy as np
import pandas as pd

from aigcdet.augment.scenarios import EVAL_GRID, HELDOUT_SEVERITY_CONDITIONS
from aigcdet.eval.report import (
    condition_metrics, robustness_table, to_markdown, validate_degradation_head,
)


def _scores(n=400, seed=0, sep=2.0):
    rng = np.random.default_rng(seed)
    rows = []
    for cond in EVAL_GRID:
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        # Harsher conditions get less separation, as reality does.
        s = rng.normal(y * sep * (0.4 if "jpeg_q30" in cond else 1.0), 1.0)
        rows.append(pd.DataFrame({"condition": cond, "image_idx": np.arange(n),
                                  "label": y, "generator": "g", "source": "s",
                                  "score": s}))
    return pd.concat(rows, ignore_index=True)


def test_condition_metrics_has_a_row_per_condition_with_cis():
    m = condition_metrics(_scores(), seed=0)
    assert len(m) == len(EVAL_GRID)
    assert (m["auc_lo"] <= m["auc"]).all() and (m["auc"] <= m["auc_hi"]).all()
    assert {"tpr_at_1pct", "acc_oracle", "acc_fixed", "n"} <= set(m.columns)


def test_heldout_severity_conditions_are_flagged_in_the_table():
    m = condition_metrics(_scores(), seed=0).set_index("condition")
    for c in HELDOUT_SEVERITY_CONDITIONS:
        assert bool(m.loc[c, "heldout_severity"]) is True
    assert bool(m.loc["jpeg_q30", "heldout_severity"]) is False


def test_fixed_threshold_accuracy_never_exceeds_oracle():
    m = condition_metrics(_scores(), seed=0)
    assert (m["acc_fixed"] <= m["acc_oracle"] + 1e-9).all()


def test_harsher_condition_scores_lower_auc():
    m = condition_metrics(_scores(), seed=0).set_index("condition")
    assert m.loc["jpeg_q30", "auc"] < m.loc["clean", "auc"]


def test_robustness_table_rows_are_rungs_and_has_robust_auc():
    t = robustness_table({"a0": _scores(seed=1), "a3": _scores(seed=2)}, tier="ablation")
    assert set(t.index) == {"a0", "a3"}
    assert "robust_auc" in t.columns and "clean" in t.columns
    # robust_auc excludes the clean column by construction
    assert t.loc["a0", "robust_auc"] <= 1.0


def test_markdown_output_names_its_tier(tmp_path):
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation")
    p = tmp_path / "t.md"
    to_markdown(t, tier="ablation", path=str(p))
    text = p.read_text()
    assert "ablation" in text and "a0" in text


def test_degradation_head_validation_finds_a_planted_correlation():
    rng = np.random.default_rng(0)
    n = 500
    true_jpeg_sev = rng.uniform(0, 1, n)
    pred = np.zeros((n, 6), np.float32)
    pred[:, 0] = true_jpeg_sev + rng.normal(0, 0.05, n)
    # Proxy jpeg_quality falls as severity rises, so the correlation is negative
    proxies = np.stack([100 - true_jpeg_sev * 70,
                        rng.normal(size=n), rng.normal(size=n)], axis=1).astype(np.float32)
    out = validate_degradation_head(pred, proxies, families=("jpeg",))
    assert abs(out.loc[0, "spearman"]) > 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.eval.report'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/eval/report.py
"""Robustness table and diagnostics (spec §6.1, §6.4).

Two accuracy columns are reported deliberately. `acc_oracle` re-tunes the
threshold per condition, which most papers do and which implicitly assumes
test-time knowledge of the degradation. `acc_fixed` uses one threshold chosen
on clean validation and frozen, which is the deployment condition. The gap
between them is score drift under degradation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from aigcdet.augment.recipes import FAMILIES
from aigcdet.augment.scenarios import HELDOUT_SEVERITY_CONDITIONS
from aigcdet.eval.metrics import (
    accuracy_at_threshold, bootstrap_ci, expected_calibration_error, roc_auc,
    threshold_at_fpr, tpr_at_fpr,
)

# Which proxy column corresponds to which degradation family (spec §3.4).
_PROXY_FOR_FAMILY = {"jpeg": 0, "blur": 1, "noise": 2}


def _best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    order = np.unique(s)
    if len(order) > 512:
        order = np.quantile(s, np.linspace(0, 1, 512))
    accs = [accuracy_at_threshold(y, s, t) for t in order]
    return float(order[int(np.argmax(accs))])


def condition_metrics(scores_df: pd.DataFrame, probs: pd.Series | None = None,
                      clean_threshold: float | None = None,
                      seed: int = 20260827, n_boot: int = 1000) -> pd.DataFrame:
    df = scores_df.copy()
    if probs is not None:
        df["prob"] = probs.to_numpy()

    clean = df[df["condition"] == "clean"]
    if clean_threshold is None:
        clean_threshold = _best_threshold(clean["label"].to_numpy(),
                                          clean["score"].to_numpy())

    rows = []
    for cond, g in df.groupby("condition", sort=False):
        y, s = g["label"].to_numpy(), g["score"].to_numpy()
        lo, hi = bootstrap_ci(roc_auc, y, s, n=n_boot, seed=seed)
        row = {
            "condition": cond,
            "auc": roc_auc(y, s), "auc_lo": lo, "auc_hi": hi,
            "tpr_at_1pct": tpr_at_fpr(y, s, 0.01),
            "acc_oracle": accuracy_at_threshold(y, s, _best_threshold(y, s)),
            "acc_fixed": accuracy_at_threshold(y, s, clean_threshold),
            "n": len(y),
            "heldout_severity": cond in HELDOUT_SEVERITY_CONDITIONS,
        }
        row["ece"] = (expected_calibration_error(y, g["prob"].to_numpy())
                      if "prob" in g else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def robustness_table(per_rung: dict[str, pd.DataFrame], tier: str,
                     metric: str = "auc", seed: int = 20260827) -> pd.DataFrame:
    out = {}
    for rung, scores in per_rung.items():
        m = condition_metrics(scores, seed=seed).set_index("condition")
        row = m[metric].to_dict()
        row["robust_auc"] = float(m.drop(index="clean")[metric].mean())
        out[rung] = row
    t = pd.DataFrame(out).T
    t.attrs["tier"] = tier
    return t


def save_heatmap(table: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [c for c in table.columns if c != "robust_auc"]
    fig, ax = plt.subplots(figsize=(1 + 0.45 * len(cols), 1 + 0.45 * len(table)))
    im = ax.imshow(table[cols].to_numpy(dtype=float), aspect="auto",
                   vmin=0.5, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(table)), table.index, fontsize=8)
    ax.set_title(f"ROC-AUC by condition (tier: {table.attrs.get('tier', 'unspecified')})")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def to_markdown(table: pd.DataFrame, tier: str, path: str) -> None:
    with open(path, "w") as f:
        f.write(f"# Robustness table\n\n**Evaluation tier:** {tier}\n\n")
        f.write("Rows marked in `heldout_severity` use a severity the training "
                "sampler never drew (spec §4.6).\n\n")
        f.write(table.round(4).to_markdown())
        f.write("\n")


def validate_degradation_head(pred_severity: np.ndarray, proxies: np.ndarray,
                              families: tuple[str, ...] = ("jpeg", "blur", "noise")
                              ) -> pd.DataFrame:
    """Spearman correlation between the learned severity and the model-free
    proxy for the same family (spec §3.4). Weak correlation means the dashboard
    readout is not trustworthy — and it is better to discover that on day 4."""
    rows = []
    for fam in families:
        if fam not in _PROXY_FOR_FAMILY:
            continue
        i = FAMILIES.index(fam)
        rho, p = spearmanr(pred_severity[:, i], proxies[:, _PROXY_FOR_FAMILY[fam]])
        rows.append({"family": fam, "spearman": float(rho), "p_value": float(p)})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/eval/test_report.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/aigcdet/eval/report.py tests/eval/test_report.py
git commit -m "feat(eval): robustness table, heatmap, degradation-head validation"
```

---

### Task 7: Ablation orchestrator and error-analysis sheets

**Files:**
- Create: `scripts/run_ablation.py`, `scripts/make_error_sheet.py`, `src/aigcdet/eval/errors.py`, `tests/eval/test_errors.py`

**Interfaces:**
- Consumes: everything above
- Produces:
  - `select_headline(results: dict[str, dict]) -> str` — implements the §6.4 rule: highest robust TPR@1%FPR on internal validation, held-out generators
  - `top_errors(scores_df, k=24, kind="fp"|"fn") -> pandas.DataFrame`
  - `fp_rate_by_source(scores_df, threshold) -> pandas.DataFrame`
  - `contact_sheet(rows, image_paths, out_path, annotations) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_errors.py
import numpy as np
import pandas as pd
import pytest

from aigcdet.eval.errors import fp_rate_by_source, select_headline, top_errors


def _scores():
    rng = np.random.default_rng(0)
    n = 200
    y = np.array([0] * 100 + [1] * 100)
    s = np.concatenate([rng.normal(-1, 1, 100), rng.normal(1, 1, 100)])
    return pd.DataFrame({"condition": "clean", "image_idx": np.arange(n),
                         "label": y, "score": s, "generator": "g",
                         "source": ["a"] * 100 + ["b"] * 100,
                         "path": [f"/img/{i}.png" for i in range(n)]})


def test_top_false_positives_are_authentic_images_with_high_scores():
    fp = top_errors(_scores(), k=10, kind="fp")
    assert len(fp) == 10
    assert (fp["label"] == 0).all()
    assert fp["score"].is_monotonic_decreasing


def test_top_false_negatives_are_generated_images_with_low_scores():
    fn = top_errors(_scores(), k=10, kind="fn")
    assert (fn["label"] == 1).all()
    assert fn["score"].is_monotonic_increasing


def test_fp_rate_by_source_reports_every_source():
    out = fp_rate_by_source(_scores(), threshold=0.0)
    assert set(out["source"]) == {"a", "b"}
    assert ((out["fp_rate"] >= 0) & (out["fp_rate"] <= 1)).all()


def test_select_headline_uses_robust_tpr_on_heldout_generators():
    results = {
        "a3": {"heldout_robust_tpr_at_1pct": 0.71, "clean_auc": 0.99},
        "a4": {"heldout_robust_tpr_at_1pct": 0.66, "clean_auc": 0.999},  # better clean
        "a5": {"heldout_robust_tpr_at_1pct": 0.74, "clean_auc": 0.95},
    }
    assert select_headline(results) == "a5"


def test_select_headline_ignores_rungs_outside_a3_to_a6():
    results = {
        "a0": {"heldout_robust_tpr_at_1pct": 0.99, "clean_auc": 0.99},
        "a3": {"heldout_robust_tpr_at_1pct": 0.60, "clean_auc": 0.90},
    }
    assert select_headline(results) == "a3"


def test_select_headline_raises_when_no_eligible_rung_present():
    with pytest.raises(ValueError, match="no eligible"):
        select_headline({"a0": {"heldout_robust_tpr_at_1pct": 0.9, "clean_auc": 0.9}})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_errors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.eval.errors'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/eval/errors.py
"""Error analysis and model selection (spec §6.4, §6.6)."""
from __future__ import annotations

import numpy as np
import pandas as pd

#: The selection rule is fixed before results exist, so the choice cannot be
#: accused of being fitted to the demo set (spec §6.4).
ELIGIBLE_RUNGS = ("a3", "a4", "a5", "a6")
SELECTION_METRIC = "heldout_robust_tpr_at_1pct"


def select_headline(results: dict[str, dict]) -> str:
    eligible = {k: v for k, v in results.items() if k in ELIGIBLE_RUNGS}
    if not eligible:
        raise ValueError(f"no eligible rung in {sorted(results)}; expected one of {ELIGIBLE_RUNGS}")
    return max(eligible, key=lambda k: eligible[k][SELECTION_METRIC])


def top_errors(scores_df: pd.DataFrame, k: int = 24, kind: str = "fp") -> pd.DataFrame:
    if kind == "fp":
        pool = scores_df[scores_df["label"] == 0]
        return pool.nlargest(k, "score").reset_index(drop=True)
    if kind == "fn":
        pool = scores_df[scores_df["label"] == 1]
        return pool.nsmallest(k, "score").reset_index(drop=True)
    raise ValueError("kind must be 'fp' or 'fn'")


def fp_rate_by_source(scores_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """False positives concentrated in one dataset indicate a confound, not a
    detector weakness (spec §6.6)."""
    real = scores_df[scores_df["label"] == 0]
    out = (real.assign(fp=(real["score"] >= threshold).astype(float))
               .groupby("source", as_index=False)
               .agg(n=("fp", "size"), fp_rate=("fp", "mean")))
    return out


def contact_sheet(rows: pd.DataFrame, out_path: str, annotations: list[str] | None = None,
                  cols: int = 6, thumb: int = 180) -> None:
    """Grid of the worst errors, each annotated with its score and diagnostics."""
    from PIL import Image, ImageDraw
    n = len(rows)
    if n == 0:
        raise ValueError("nothing to render")
    r = (n + cols - 1) // cols
    pad = 26
    sheet = Image.new("RGB", (cols * thumb, r * (thumb + pad)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows.itertuples()):
        with Image.open(row.path) as im:
            t = im.convert("RGB").resize((thumb, thumb), Image.BILINEAR)
        x, y = (i % cols) * thumb, (i // cols) * (thumb + pad)
        sheet.paste(t, (x, y))
        label = annotations[i] if annotations else f"score={row.score:.3f}"
        draw.text((x + 3, y + thumb + 6), label[:34], fill="black")
    sheet.save(out_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/eval/test_errors.py -v`
Expected: 6 passed

- [ ] **Step 5: Write the orchestrators**

```python
# scripts/run_ablation.py
"""Train and evaluate every rung, then emit the robustness table (spec §6.4).

    python scripts/run_ablation.py --bank banks/dinov3l --eval-bank banks/eval_dinov3l \
        --rungs configs/rungs/a0.yaml configs/rungs/a1.yaml configs/rungs/a2.yaml \
                configs/rungs/a3.yaml configs/rungs/a4.yaml \
        --tier ablation --out docs/robustness_table.md
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import yaml

from aigcdet.eval.errors import select_headline
from aigcdet.eval.grid import score_grid
from aigcdet.eval.metrics import tpr_at_fpr
from aigcdet.eval.report import robustness_table, save_heatmap, to_markdown
from aigcdet.features.bank import FeatureBank
from aigcdet.train.train_head import RungConfig, load_detector, train_rung


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--eval-bank", required=True)
    ap.add_argument("--rungs", nargs="+", required=True)
    ap.add_argument("--tier", required=True, choices=["ablation", "final"])
    ap.add_argument("--out", default="docs/robustness_table.md")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    eb = FeatureBank.open(a.eval_bank)
    heldout = eb.meta["split"].to_numpy() == "heldout_generator"

    per_rung, summary = {}, {}
    for cfg_path in a.rungs:
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
        cfg = RungConfig(bank_dir=a.bank, device=a.device, **raw)
        res = train_rung(cfg)
        model, _ = load_detector(res["checkpoint"], device=a.device)
        scores = score_grid(model, eb, use_recon=cfg.use_recon, device=a.device)
        per_rung[cfg.name] = scores

        # Selection metric: robust TPR@1%FPR on held-out generators (spec §6.4).
        transformed = scores[scores["condition"] != "clean"]
        vals = []
        for _, g in transformed.groupby("condition"):
            m = heldout[g["image_idx"].to_numpy()]
            sub = g[m | (g["label"].to_numpy() == 0)]   # held-out fakes vs all reals
            if sub["label"].nunique() == 2:
                vals.append(tpr_at_fpr(sub["label"].to_numpy(), sub["score"].to_numpy(), 0.01))
        summary[cfg.name] = {
            "heldout_robust_tpr_at_1pct": float(np.mean(vals)) if vals else 0.0,
            "clean_auc": float(res["val_auc"]),
        }
        print(f"{cfg.name}: {summary[cfg.name]}")

    table = robustness_table(per_rung, tier=a.tier)
    to_markdown(table, tier=a.tier, path=a.out)
    save_heatmap(table, a.out.replace(".md", ".png"))
    try:
        headline = select_headline(summary)
    except ValueError as e:
        headline = None
        print(f"headline not selected: {e}")
    with open("docs/selection.json", "w") as f:
        json.dump({"summary": summary, "headline": headline,
                   "rule": "max heldout_robust_tpr_at_1pct over rungs a3-a6"}, f, indent=2)
    print(f"headline model: {headline}")


if __name__ == "__main__":
    main()
```

```python
# scripts/make_error_sheet.py
"""Contact sheets of representative false positives and negatives (spec §6.6).

    python scripts/make_error_sheet.py --scores outputs/scores.parquet \
        --eval-bank banks/eval_dinov3l --condition clean --out docs/errors
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from aigcdet.eval.errors import contact_sheet, fp_rate_by_source, top_errors
from aigcdet.features.bank import FeatureBank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--eval-bank", required=True)
    ap.add_argument("--condition", default="clean")
    ap.add_argument("--out", default="docs/errors")
    ap.add_argument("--k", type=int, default=24)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    scores = pd.read_parquet(a.scores)
    scores = scores[scores["condition"] == a.condition]
    meta = FeatureBank.open(a.eval_bank).meta
    scores = scores.merge(meta[["path"]], left_on="image_idx", right_index=True)

    for kind in ("fp", "fn"):
        rows = top_errors(scores, k=a.k, kind=kind)
        ann = [f"{r.score:+.2f} {r.generator or 'real'}" for r in rows.itertuples()]
        contact_sheet(rows, os.path.join(a.out, f"{a.condition}_{kind}.png"), ann)

    by_src = fp_rate_by_source(scores, threshold=float(scores["score"].median()))
    with open(os.path.join(a.out, "fp_by_source.md"), "w") as f:
        f.write("# False-positive rate by source\n\n")
        f.write("Concentration in one source indicates a confound, not a "
                "detector weakness.\n\n")
        f.write(by_src.round(4).to_markdown(index=False) + "\n")
    print(by_src.to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the full suite and commit**

Run: `python -m pytest -v -m "not gpu"`
Expected: all pass

```bash
git add src/aigcdet/eval/errors.py scripts/run_ablation.py scripts/make_error_sheet.py tests/eval/test_errors.py
git commit -m "feat(eval): ablation orchestrator, selection rule, error-analysis sheets"
```

---

### Task 8: A5 ensemble fusion and A6 test-time augmentation

Rungs A5 and A6 appear in the ablation ladder and in the selection rule's
eligible range (`a3`–`a6`), but neither is a training config: A5 fuses two
independently-trained banks at scoring time, and A6 is inference-only. Without
this task the selection rule can only ever choose between A3 and A4.

**Files:**
- Create: `src/aigcdet/eval/fusion.py`, `src/aigcdet/eval/tta.py`, `tests/eval/test_fusion.py`, `tests/eval/test_tta.py`
- Modify: `scripts/run_ablation.py` — add `--fuse` and `--tta` modes

**Interfaces:**
- Consumes: `eval.grid.score_grid`, `augment.ops`, `features.backbones.embed`
- Produces:
  - `zscore_by_condition(df) -> pandas.DataFrame` — standardises `score` within each condition so two backbones' logit scales are comparable before averaging
  - `fuse_scores(dfs: list[pandas.DataFrame], weights: list[float] | None = None) -> pandas.DataFrame`
  - `TTA_VIEWS: tuple[str, ...]` = `("identity", "hflip", "scale_0.75", "scale_1.25", "jpeg_95", "blur_0.3", "hflip_scale_0.75", "hflip_jpeg_95")`
  - `apply_tta_view(img: np.ndarray, view: str) -> np.ndarray`
  - `tta_logit(backbone, spec, model, img, device, views=TTA_VIEWS, recon_fn=None) -> float` — mean logit across views

- [ ] **Step 1: Write the failing tests**

```python
# tests/eval/test_fusion.py
import numpy as np
import pandas as pd
import pytest

from aigcdet.eval.fusion import fuse_scores, zscore_by_condition
from aigcdet.eval.metrics import roc_auc


def _df(seed, scale, n=200):
    rng = np.random.default_rng(seed)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    rows = []
    for cond in ("clean", "jpeg_q30"):
        rows.append(pd.DataFrame({
            "condition": cond, "image_idx": np.arange(n), "label": y,
            "generator": "g", "source": "s",
            "score": (rng.normal(y * 1.0, 1.0)) * scale,
        }))
    return pd.concat(rows, ignore_index=True)


def test_zscore_makes_each_condition_zero_mean_unit_variance():
    z = zscore_by_condition(_df(0, scale=50.0))
    for _, g in z.groupby("condition"):
        assert abs(g["score"].mean()) < 1e-6
        assert abs(g["score"].std(ddof=0) - 1.0) < 1e-6


def test_fusion_is_invariant_to_one_backbones_logit_scale():
    """Without z-scoring, a backbone with 50x larger logits would dominate."""
    a, b = _df(0, scale=1.0), _df(1, scale=50.0)
    fused = fuse_scores([a, b])
    a2, b2 = _df(0, scale=1.0), _df(1, scale=1.0)
    fused2 = fuse_scores([a2, b2])
    clean = fused[fused["condition"] == "clean"]
    clean2 = fused2[fused2["condition"] == "clean"]
    assert abs(roc_auc(clean["label"], clean["score"])
               - roc_auc(clean2["label"], clean2["score"])) < 1e-9


def test_fusion_of_two_noisy_views_beats_either_alone():
    a, b = _df(0, 1.0), _df(7, 1.0)
    fused = fuse_scores([a, b])
    sel = lambda d: d[d["condition"] == "clean"]
    auc_a = roc_auc(sel(a)["label"], sel(a)["score"])
    auc_b = roc_auc(sel(b)["label"], sel(b)["score"])
    auc_f = roc_auc(sel(fused)["label"], sel(fused)["score"])
    assert auc_f >= min(auc_a, auc_b)


def test_fusion_preserves_row_count_and_keys():
    a, b = _df(0, 1.0), _df(1, 1.0)
    fused = fuse_scores([a, b])
    assert len(fused) == len(a)
    assert set(fused.columns) >= {"condition", "image_idx", "label", "score"}


def test_weights_are_honoured():
    a, b = _df(0, 1.0), _df(1, 1.0)
    only_a = fuse_scores([a, b], weights=[1.0, 0.0])
    za = zscore_by_condition(a).sort_values(["condition", "image_idx"])
    of = only_a.sort_values(["condition", "image_idx"])
    np.testing.assert_allclose(of["score"].to_numpy(), za["score"].to_numpy(), atol=1e-9)


def test_mismatched_frames_are_rejected():
    a = _df(0, 1.0)
    b = _df(1, 1.0).iloc[:10]
    with pytest.raises(ValueError, match="same rows"):
        fuse_scores([a, b])
```

```python
# tests/eval/test_tta.py
import numpy as np
import pytest

from aigcdet.eval.tta import TTA_VIEWS, apply_tta_view


def _img():
    return np.random.default_rng(0).integers(0, 256, (128, 160, 3), dtype=np.uint8)


def test_identity_view_is_the_identity():
    img = _img()
    assert np.array_equal(apply_tta_view(img, "identity"), img)


def test_every_declared_view_applies_and_preserves_shape():
    img = _img()
    for v in TTA_VIEWS:
        out = apply_tta_view(img, v)
        assert out.shape == img.shape, v
        assert out.dtype == np.uint8, v


def test_hflip_is_its_own_inverse():
    img = _img()
    assert np.array_equal(apply_tta_view(apply_tta_view(img, "hflip"), "hflip"), img)


def test_views_are_distinct_from_the_original_except_identity():
    img = _img()
    for v in TTA_VIEWS:
        if v == "identity":
            continue
        assert not np.array_equal(apply_tta_view(img, v), img), v


def test_unknown_view_raises():
    with pytest.raises(KeyError):
        apply_tta_view(_img(), "not_a_view")


def test_tta_logit_averages_over_views():
    from aigcdet.eval.tta import tta_logit

    calls = {"n": 0}

    class StubModel:
        use_recon = False

        def __call__(self, f, r=None):
            import torch
            calls["n"] += 1
            return {"logit": torch.tensor([2.0])}

    def stub_embed(m, s, imgs, device, batch_size=16):
        return np.zeros((len(imgs), 4), np.float32)

    out = tta_logit(None, type("S", (), {"dim": 4, "image_size": 64})(),
                    StubModel(), _img(), device="cpu",
                    views=("identity", "hflip"), embed_fn=stub_embed)
    assert out == pytest.approx(2.0)
    assert calls["n"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/eval/test_fusion.py tests/eval/test_tta.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.eval.fusion'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/eval/fusion.py
"""Rung A5: paradigm-diverse ensemble fusion (spec §6.4).

Two backbones trained independently produce logits on different scales, so a
raw average would let whichever has the larger spread dominate. Standardising
within each condition first makes the average a genuine vote.

Fusing per condition rather than globally is deliberate: score distributions
shift under degradation, and we want the fusion to be fair at every operating
point, not just on clean data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_KEYS = ["condition", "image_idx"]


def zscore_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("condition")["score"]
    out["score"] = (out["score"] - g.transform("mean")) / g.transform("std").replace(0, 1.0)
    return out


def fuse_scores(dfs: list[pd.DataFrame], weights: list[float] | None = None) -> pd.DataFrame:
    if not dfs:
        raise ValueError("nothing to fuse")
    base = dfs[0].sort_values(_KEYS).reset_index(drop=True)
    normed = [zscore_by_condition(d).sort_values(_KEYS).reset_index(drop=True) for d in dfs]
    for n in normed[1:]:
        if len(n) != len(normed[0]) or not n[_KEYS].equals(normed[0][_KEYS]):
            raise ValueError("all frames must cover the same rows (condition, image_idx)")

    w = np.array(weights if weights is not None else [1.0] * len(normed), dtype=float)
    if len(w) != len(normed):
        raise ValueError("weights must match the number of frames")
    w = w / w.sum()

    stacked = np.stack([n["score"].to_numpy() for n in normed])
    fused = base.copy()
    fused["score"] = (stacked * w[:, None]).sum(axis=0)
    return fused
```

```python
# src/aigcdet/eval/tta.py
"""Rung A6: degradation-aware test-time augmentation (spec §6.4).

Eight views mixing geometric and degradation transforms, following the top
NTIRE 2026 entries. Logits are averaged, not probabilities, because averaging
in logit space is less dominated by any single saturated view.

Cost note: TTA multiplies inference by len(views). It is therefore evaluated
from images rather than from the cached eval bank, and only on the ablation
tier's 5k+5k subsample (spec §4.4a). Record that cap in the report.
"""
from __future__ import annotations

import numpy as np
import torch

from aigcdet.augment import ops

TTA_VIEWS: tuple[str, ...] = (
    "identity", "hflip", "scale_0.75", "scale_1.25",
    "jpeg_95", "blur_0.3", "hflip_scale_0.75", "hflip_jpeg_95",
)


def _scale(img: np.ndarray, factor: float) -> np.ndarray:
    """Rescale then restore the original size, so views stay shape-compatible."""
    import cv2
    h, w = img.shape[:2]
    sh, sw = max(1, int(h * factor)), max(1, int(w * factor))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


_VIEW_FUNCS = {
    "identity": lambda i: i.copy(),
    "hflip": lambda i: np.ascontiguousarray(i[:, ::-1]),
    "scale_0.75": lambda i: _scale(i, 0.75),
    "scale_1.25": lambda i: _scale(i, 1.25),
    "jpeg_95": lambda i: ops.jpeg(i, quality=95),
    "blur_0.3": lambda i: ops.blur(i, sigma=0.3),
    "hflip_scale_0.75": lambda i: _scale(np.ascontiguousarray(i[:, ::-1]), 0.75),
    "hflip_jpeg_95": lambda i: ops.jpeg(np.ascontiguousarray(i[:, ::-1]), quality=95),
}


def apply_tta_view(img: np.ndarray, view: str) -> np.ndarray:
    if view not in _VIEW_FUNCS:
        raise KeyError(f"unknown TTA view {view!r}; expected one of {TTA_VIEWS}")
    return _VIEW_FUNCS[view](img).astype(np.uint8)


@torch.no_grad()
def tta_logit(backbone, spec, model, img: np.ndarray, device: str = "cuda",
              views: tuple[str, ...] = TTA_VIEWS, recon_fn=None,
              embed_fn=None) -> float:
    """Mean logit across views. `embed_fn` is injectable so this is testable
    without loading a real backbone."""
    from aigcdet.features.backbones import embed as default_embed
    embed_fn = embed_fn or default_embed

    logits = []
    for v in views:
        view = apply_tta_view(img, v)
        f = embed_fn(backbone, spec, [view], device=device, batch_size=1)
        r = None
        if getattr(model, "use_recon", False):
            if recon_fn is None:
                raise ValueError("model uses the recon branch; pass recon_fn")
            r = torch.from_numpy(recon_fn(view)[None]).to(device)
        out = model(torch.from_numpy(f).to(device), r)
        logits.append(float(out["logit"].reshape(-1)[0]))
    return float(np.mean(logits))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/eval/test_fusion.py tests/eval/test_tta.py -v`
Expected: 12 passed

- [ ] **Step 5: Wire A5 and A6 into the orchestrator**

Add to `scripts/run_ablation.py`, after the per-rung loop:

```python
    # Rung A5: fuse the A3 scores from two independently-trained banks.
    if a.fuse_bank:
        from aigcdet.eval.fusion import fuse_scores
        eb2 = FeatureBank.open(a.fuse_eval_bank)
        cfg2 = RungConfig(name="a3_second", bank_dir=a.fuse_bank,
                          device=a.device, **yaml.safe_load(open("configs/rungs/a3.yaml")))
        cfg2.name = "a3_second"
        res2 = train_rung(cfg2)
        model2, _ = load_detector(res2["checkpoint"], device=a.device)
        per_rung["a5"] = fuse_scores([per_rung["a3"],
                                      score_grid(model2, eb2, use_recon=False,
                                                 device=a.device)])
        summary["a5"] = _selection_metric(per_rung["a5"], heldout)

    # Rung A6: TTA on the headline candidate, over the ablation-tier subsample.
    if a.tta:
        from aigcdet.eval.tta import TTA_VIEWS
        print(f"A6: TTA with {len(TTA_VIEWS)} views multiplies inference cost by "
              f"{len(TTA_VIEWS)}x; evaluated on the ablation-tier subsample only.")
```

Refactor the selection-metric computation inside the loop into a
`_selection_metric(scores, heldout)` helper first, so A5 reuses it rather than
duplicating the logic. Add the CLI flags `--fuse-bank`, `--fuse-eval-bank`,
and `--tta`.

- [ ] **Step 6: Commit**

```bash
git add src/aigcdet/eval/fusion.py src/aigcdet/eval/tta.py \
        tests/eval/test_fusion.py tests/eval/test_tta.py scripts/run_ablation.py
git commit -m "feat(eval): A5 ensemble fusion and A6 test-time augmentation"
```

---

## Plan 3 Completion Criteria

- [ ] `python -m pytest -v -m "not gpu"` passes
- [ ] `docs/robustness_table.md` exists, names its **tier**, and marks `heldout_severity` rows
- [ ] `docs/robustness_table.png` heatmap exists
- [ ] `docs/selection.json` records the summary, the chosen headline rung, and the selection rule
- [ ] The **A2→A3 delta with CIs** is recorded — if the intervals overlap, the README says so plainly
- [ ] The **A3→A4 delta on held-out generators** is recorded, and the kill decision on `R` is made and written down either way
- [ ] The content-blind control has been run on **both** our splits and the official demo set, with both AUCs reported
- [ ] `validate_degradation_head` Spearman correlations are recorded; if any family is below ~0.5, the dashboard readout is labelled unreliable for that family
- [ ] `docs/errors/` holds FP and FN contact sheets plus `fp_by_source.md`
- [ ] Baselines UnivFD, NPR, and AEROBLADE appear as rows in the robustness table
- [ ] Rungs **A5 (fusion)** and **A6 (TTA)** produce rows, so the selection rule's eligible range `a3`–`a6` is actually populated; if either was skipped for time, `docs/selection.json` records that it was skipped rather than leaving it silently absent
- [ ] The TTA cost multiplier and the tier it was evaluated on are stated in the report — no silent caps
