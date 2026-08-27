# Plan 4 — Delivery

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the trained system into the submission: a bulletproof `predict.py`, an explainable Gradio dashboard with live transform sliders, and the README, Devpost text, and video script.

**Architecture:** A single `Predictor` bundles backbone + detector + calibrator + policy behind one method, so `predict.py` and the dashboard cannot drift apart. The two-stage training split disappears here — inference is one path: image → backbone → (recon) → heads → calibrated probability, EQI, decision.

**Tech Stack:** Everything from Plans 1–3, plus Gradio and joblib.

**Spec:** `docs/superpowers/specs/2026-08-27-robust-aigc-detection-design-v2.md` (v2.1)

**Depends on:** Plans 1–3.

## Global Constraints

- **`predict.py` default output contains exactly `image_path` and `pred`, nothing else.** Extra keys are how a submission fails on a technicality. A schema test enforces this.
- `pred` is the **calibrated probability** in [0, 1], so 0.9 means roughly 90%.
- Never raise on a corrupt or non-image file: warn, score it `0.5`, keep going.
- CPU fallback when no GPU is present; fixed seed; deterministic given the weights.
- Demo gallery uses licensed images only; no third-party logos or copyrighted content anywhere in the UI or video (spec §4.5).
- Target: a 5k-image directory in a couple of minutes on the A4500.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/aigcdet/infer.py` | `Predictor` — the single inference path, used by both CLI and dashboard |
| `src/aigcdet/explain/patch_heatmap.py` | Per-patch AIGC map and VAE error map |
| `scripts/predict.py` | ★ The required deliverable |
| `scripts/export_bundle.py` | Freeze checkpoint + calibrator + policy into a release bundle |
| `app/dashboard.py` | Gradio UI with live transform sliders |
| `README.md` | Project overview, setup, reproduction, limitations, contributions |
| `docs/devpost.md` | Devpost description draft |
| `docs/video_script.md` | Shot-by-shot script |

---

### Task 1: Release bundle and the Predictor

**Files:**
- Create: `src/aigcdet/infer.py`, `scripts/export_bundle.py`, `tests/test_infer.py`

**Interfaces:**
- Consumes: `train.train_head.load_detector`, `features.backbones`, `features.recon`, `calibrate.*`
- Produces:
  - `export_bundle(checkpoint, calibrator, eqi, policy, out_dir, backbone_name, use_recon, dim_feat) -> str`
  - `Predictor.load(bundle_dir, device="auto") -> Predictor`
  - `Predictor.predict_array(img: np.ndarray, path: str | None = None) -> dict` with keys `pred, logit, eqi, decision, severity, presence, proxies, deg_embedding`
  - `Predictor.predict_paths(paths: list[str], batch_size: int = 16) -> list[dict]` — never raises on a bad file; failures get `pred=0.5` and `error` set
  - `RESULT_KEYS_MINIMAL = ("image_path", "pred")`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_infer.py
import json

import numpy as np
from PIL import Image

from aigcdet.infer import RESULT_KEYS_MINIMAL


def test_minimal_keys_are_exactly_the_two_required():
    assert RESULT_KEYS_MINIMAL == ("image_path", "pred")


def _fake_bundle(tmp_path):
    """Bundle with a tiny detector and an identity calibrator, no real backbone."""
    import joblib
    import torch
    from aigcdet.models.heads import Detector
    d = Detector(dim_feat=8, use_recon=False)
    out = tmp_path / "bundle"
    out.mkdir()
    torch.save({"state_dict": d.state_dict(),
                "config": {"use_recon": False, "use_film": False},
                "dim_feat": 8, "backbone": "fake"}, out / "checkpoint.pt")
    joblib.dump({"kind": "global", "temperature": 1.0}, out / "calibrator.joblib")
    joblib.dump(None, out / "eqi.joblib")
    with open(out / "policy.json", "w") as f:
        json.dump({"flag_threshold": 0.8, "clear_threshold": 0.2,
                   "eqi_threshold": 0.3}, f)
    with open(out / "config.json", "w") as f:
        json.dump({"backbone": "fake", "use_recon": False, "dim_feat": 8}, f)
    return str(out)


def _patch_backbone(monkeypatch, dim=8, fn=None):
    from aigcdet import infer
    from aigcdet.features.backbones import BackboneSpec
    spec = BackboneSpec("fake", "none", 64, dim, 1, 0)
    monkeypatch.setattr(infer, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(
        infer, "embed",
        fn or (lambda m, s, imgs, device, batch_size=16:
               np.zeros((len(imgs), s.dim), np.float32)))
    return infer


def test_predictor_loads_and_scores(tmp_path, monkeypatch):
    infer = _patch_backbone(monkeypatch)
    p = infer.Predictor.load(_fake_bundle(tmp_path), device="cpu")
    img = np.random.default_rng(0).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    out = p.predict_array(img)
    assert 0.0 <= out["pred"] <= 1.0
    assert out["decision"] in ("clear", "review", "flag")
    assert len(out["severity"]) == 6


def test_predict_paths_survives_a_corrupt_file(tmp_path, monkeypatch):
    infer = _patch_backbone(monkeypatch)
    good = tmp_path / "good.png"
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(good)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"this is not a png")

    p = infer.Predictor.load(_fake_bundle(tmp_path), device="cpu")
    res = p.predict_paths([str(good), str(bad)])
    assert len(res) == 2
    assert res[1]["pred"] == 0.5 and res[1].get("error")
    assert res[0].get("error") is None


def test_predictions_are_deterministic(tmp_path, monkeypatch):
    infer = _patch_backbone(
        monkeypatch,
        fn=lambda m, s, imgs, device, batch_size=16:
            np.stack([np.full(s.dim, float(i.mean()), np.float32) for i in imgs]))
    p = infer.Predictor.load(_fake_bundle(tmp_path), device="cpu")
    img = np.random.default_rng(1).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    assert p.predict_array(img)["pred"] == p.predict_array(img)["pred"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_infer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.infer'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/infer.py
"""The single inference path (spec §3.1).

The two-stage training split disappears here: one image in, one calibrated
probability out. Both predict.py and the dashboard go through this class, so
they cannot drift apart.
"""
from __future__ import annotations

import json
import os
import shutil
import warnings

import joblib
import numpy as np
import torch
from PIL import Image

from aigcdet.calibrate.policy import Policy, decide
from aigcdet.features.backbones import embed, load_backbone
from aigcdet.features.proxies import proxy_vector
from aigcdet.train.train_head import load_detector

RESULT_KEYS_MINIMAL = ("image_path", "pred")


def export_bundle(checkpoint: str, calibrator, eqi, policy: Policy, out_dir: str,
                  backbone_name: str, use_recon: bool, dim_feat: int) -> str:
    """Freeze everything inference needs into one directory."""
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(checkpoint, os.path.join(out_dir, "checkpoint.pt"))
    joblib.dump(calibrator, os.path.join(out_dir, "calibrator.joblib"))
    joblib.dump(eqi, os.path.join(out_dir, "eqi.joblib"))
    with open(os.path.join(out_dir, "policy.json"), "w") as f:
        json.dump(policy.__dict__, f, indent=2)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"backbone": backbone_name, "use_recon": use_recon,
                   "dim_feat": dim_feat}, f, indent=2)
    return out_dir


class Predictor:
    def __init__(self, model, backbone, spec, calibrator, eqi, policy: Policy,
                 use_recon: bool, device: str):
        self.model, self.backbone, self.spec = model, backbone, spec
        self.calibrator, self.eqi, self.policy = calibrator, eqi, policy
        self.use_recon, self.device = use_recon, device
        self._vae = self._lpips = None

    @classmethod
    def load(cls, bundle_dir: str, device: str = "auto") -> "Predictor":
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = json.load(open(os.path.join(bundle_dir, "config.json")))
        model, _ = load_detector(os.path.join(bundle_dir, "checkpoint.pt"), device=device)
        backbone, spec = load_backbone(cfg["backbone"], device=device)
        calibrator = joblib.load(os.path.join(bundle_dir, "calibrator.joblib"))
        eqi = joblib.load(os.path.join(bundle_dir, "eqi.joblib"))
        policy = Policy(**json.load(open(os.path.join(bundle_dir, "policy.json"))))
        return cls(model, backbone, spec, calibrator, eqi, policy,
                   bool(cfg["use_recon"]), device)

    def _recon(self, img: np.ndarray) -> np.ndarray:
        from aigcdet.features.recon import load_recon_models, recon_features
        if self._vae is None:
            self._vae, self._lpips = load_recon_models(self.device)
        return recon_features(img, self._vae, self._lpips, self.device)

    def _calibrate(self, logit: float, cond: np.ndarray) -> float:
        c = self.calibrator
        if isinstance(c, dict):                       # identity / global dict form
            return float(1.0 / (1.0 + np.exp(-logit / c.get("temperature", 1.0))))
        if hasattr(c, "transform"):
            try:
                return float(c.transform(np.array([logit]), cond[None])[0])
            except TypeError:                          # global temperature form
                return float(c.transform(np.array([logit]))[0])
        return float(1.0 / (1.0 + np.exp(-logit)))

    @torch.no_grad()
    def predict_array(self, img: np.ndarray, path: str | None = None) -> dict:
        f = embed(self.backbone, self.spec, [img], device=self.device, batch_size=1)
        r = self._recon(img)[None] if self.use_recon else None
        out = self.model(
            torch.from_numpy(f).to(self.device),
            torch.from_numpy(r).to(self.device) if r is not None else None)
        logit = float(out["logit"].item())
        severity = out["severity"].cpu().numpy()[0]
        presence = torch.sigmoid(out["presence"]).cpu().numpy()[0]
        deg_emb = out["deg_embedding"].cpu().numpy()[0]
        proxies = proxy_vector(img, path)

        cond = np.concatenate([severity, proxies]).astype(np.float32)
        pred = self._calibrate(logit, cond)
        eqi_val = (float(self.eqi.predict(cond[None])[0])
                   if self.eqi is not None else float(max(pred, 1.0 - pred)))
        decision = decide(np.array([pred]), np.array([eqi_val]), self.policy)[0]
        return {"pred": pred, "logit": logit, "eqi": eqi_val, "decision": decision,
                "severity": severity.tolist(), "presence": presence.tolist(),
                "proxies": proxies.tolist(), "deg_embedding": deg_emb}

    def predict_paths(self, paths: list[str], batch_size: int = 16) -> list[dict]:
        results = []
        for p in paths:
            try:
                with Image.open(p) as im:
                    arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
                res = self.predict_array(arr, path=p)
                res["error"] = None
            except Exception as e:      # never crash on someone else's directory
                warnings.warn(f"skipping {p}: {e}")
                res = {"pred": 0.5, "logit": 0.0, "eqi": 0.0, "decision": "review",
                       "severity": [0.0] * 6, "presence": [0.0] * 6,
                       "proxies": [0.0] * 3, "error": str(e)}
            res["image_path"] = p
            results.append(res)
        return results
```

```python
# scripts/export_bundle.py
"""Freeze the selected headline model into a release bundle.

    python scripts/export_bundle.py --checkpoint outputs/rungs/a3/checkpoint.pt \
        --bank banks/dinov3l --eval-bank banks/eval_dinov3l --out outputs/release

Calibration, EQI, and the decision policy are all fitted on internal
validation ONLY (spec §6.7). The external benchmark is never touched here.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from aigcdet.calibrate.eqi import EQI
from aigcdet.calibrate.policy import fit_policy
from aigcdet.calibrate.temperature import ConditionalTemperature
from aigcdet.eval.grid import score_grid
from aigcdet.features.bank import FeatureBank
from aigcdet.infer import export_bundle
from aigcdet.train.train_head import load_detector


def _condition_vectors(eb: FeatureBank, scores) -> np.ndarray:
    """Assemble [severity | proxies] per scored row, matching its condition."""
    names = list(eb.config["conditions"])
    img_idx = scores["image_idx"].to_numpy()
    cond_idx = np.array([names.index(c) for c in scores["condition"]])
    sev = np.asarray(eb.severity)[img_idx, cond_idx]
    prox = np.asarray(eb.proxies)[img_idx, cond_idx]
    return np.concatenate([sev, prox], axis=1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--eval-bank", required=True)
    ap.add_argument("--out", default="outputs/release")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    model, ck = load_detector(a.checkpoint, device=a.device)
    eb = FeatureBank.open(a.eval_bank)
    val = np.where(eb.meta["split"].to_numpy() == "val_internal")[0]
    if len(val) == 0:
        raise SystemExit("eval bank has no val_internal rows; cannot calibrate")

    scores = score_grid(model, eb, use_recon=ck["config"]["use_recon"], device=a.device)
    scores = scores[scores["image_idx"].isin(val)].reset_index(drop=True)

    cond = _condition_vectors(eb, scores)
    y = scores["label"].to_numpy()
    logits = scores["score"].to_numpy()

    cal = ConditionalTemperature(cond_dim=cond.shape[1]).fit(logits, y, cond)
    p = cal.transform(logits, cond)
    correct = ((p >= 0.5).astype(int) == y).astype(int)
    eqi = EQI().fit(cond, correct)
    policy = fit_policy(p, y, eqi.predict(cond))

    export_bundle(a.checkpoint, cal, eqi, policy, a.out,
                  backbone_name=ck["backbone"], use_recon=ck["config"]["use_recon"],
                  dim_feat=ck["dim_feat"])
    print(json.dumps(policy.__dict__, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_infer.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/aigcdet/infer.py scripts/export_bundle.py tests/test_infer.py
git commit -m "feat(infer): release bundle and single-path Predictor"
```

---

### Task 2: predict.py — the required deliverable

**Files:**
- Create: `scripts/predict.py`, `tests/test_predict_schema.py`

**Interfaces:**
- Consumes: `infer.Predictor`, `infer.RESULT_KEYS_MINIMAL`
- Produces:
  - `find_images(image_dir: str) -> list[str]` — recursive, extension-filtered, sorted
  - `to_records(results: list[dict], rich: bool) -> list[dict]`
  - CLI: `--image-dir`, `--out`, `--bundle`, `--rich`, `--device`, `--batch-size`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_predict_schema.py
import json
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "scripts")


def test_find_images_recurses_and_filters(tmp_path):
    from predict import find_images
    (tmp_path / "sub").mkdir()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / "a.png")
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / "sub" / "b.jpg")
    (tmp_path / "notes.txt").write_text("ignore me")
    found = find_images(str(tmp_path))
    assert len(found) == 2
    assert all(f.lower().endswith((".png", ".jpg")) for f in found)
    assert found == sorted(found)


def test_default_records_have_exactly_the_two_required_keys():
    from predict import to_records
    from aigcdet.infer import RESULT_KEYS_MINIMAL
    raw = [{"image_path": "/a.png", "pred": 0.8, "eqi": 0.9, "decision": "flag",
            "severity": [0] * 6, "error": None}]
    recs = to_records(raw, rich=False)
    assert list(recs[0].keys()) == list(RESULT_KEYS_MINIMAL)
    assert isinstance(recs[0]["pred"], float)


def test_rich_records_add_diagnostics_without_dropping_the_required_keys():
    from predict import to_records
    raw = [{"image_path": "/a.png", "pred": 0.8, "eqi": 0.9, "decision": "flag",
            "severity": [0] * 6, "presence": [0] * 6, "proxies": [0] * 3,
            "error": None}]
    recs = to_records(raw, rich=True)
    assert {"image_path", "pred", "eqi", "decision"} <= set(recs[0])


def test_pred_is_clamped_into_the_unit_interval():
    from predict import to_records
    raw = [{"image_path": "/a.png", "pred": 1.4, "error": None},
           {"image_path": "/b.png", "pred": -0.2, "error": None}]
    recs = to_records(raw, rich=False)
    assert recs[0]["pred"] == 1.0 and recs[1]["pred"] == 0.0


def test_output_json_is_a_list_of_objects_and_parses(tmp_path):
    """The shape a grader's parser will expect."""
    from predict import to_records
    recs = to_records([{"image_path": "/a.png", "pred": 0.5, "error": None}], rich=False)
    p = tmp_path / "out.json"
    p.write_text(json.dumps(recs))
    back = json.loads(p.read_text())
    assert isinstance(back, list) and set(back[0]) == {"image_path", "pred"}


def test_cli_help_runs_without_a_bundle():
    r = subprocess.run([sys.executable, "scripts/predict.py", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "--image-dir" in r.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_predict_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'predict'`

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python
"""Score every image in a directory for the likelihood it is AI-generated.

    python scripts/predict.py --image-dir path/to/images --out preds.json

Output is a JSON list of {"image_path": ..., "pred": <float in [0,1]>}.
`pred` is a calibrated probability, so 0.9 means roughly 90 percent.

Design notes for anyone reading this file first:
  * The default schema is exactly two keys. Diagnostics live behind --rich,
    because extra keys are how a submission fails someone else's parser.
  * Unreadable files are never fatal: they are warned about and scored 0.5.
    Crashing on one truncated JPEG in someone else's directory would lose the
    whole run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from tqdm import tqdm

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def find_images(image_dir: str) -> list[str]:
    out = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                out.append(os.path.join(root, f))
    return sorted(out)


def to_records(results: list[dict], rich: bool) -> list[dict]:
    recs = []
    for r in results:
        pred = min(1.0, max(0.0, float(r["pred"])))
        rec = {"image_path": r["image_path"], "pred": pred}
        if rich:
            rec.update({k: r[k] for k in
                        ("eqi", "decision", "severity", "presence", "proxies", "error")
                        if k in r})
        recs.append(rec)
    return recs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image-dir", required=True, help="directory to scan, recursively")
    ap.add_argument("--out", default="predictions.json")
    ap.add_argument("--bundle", default="outputs/release",
                    help="release bundle directory produced by export_bundle.py")
    ap.add_argument("--rich", action="store_true",
                    help="add EQI, decision, and degradation diagnostics")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()

    paths = find_images(a.image_dir)
    if not paths:
        print(f"no images found under {a.image_dir}", file=sys.stderr)
        return 1
    print(f"found {len(paths)} images")

    from aigcdet.infer import Predictor
    predictor = Predictor.load(a.bundle, device=a.device)

    results = []
    for i in tqdm(range(0, len(paths), a.batch_size), desc="scoring"):
        results.extend(predictor.predict_paths(paths[i:i + a.batch_size],
                                               batch_size=a.batch_size))

    with open(a.out, "w") as f:
        json.dump(to_records(results, rich=a.rich), f, indent=2)
    n_err = sum(1 for r in results if r.get("error"))
    print(f"wrote {a.out} ({len(results)} records, {n_err} unreadable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_predict_schema.py -v`
Expected: 6 passed

- [ ] **Step 5: Smoke-test against a real directory**

```bash
python scripts/predict.py --image-dir data/dummy/img --out /tmp/preds.json \
    --bundle outputs/release
python -c "
import json; d=json.load(open('/tmp/preds.json'))
assert isinstance(d, list) and set(d[0])=={'image_path','pred'}, d[0]
assert all(0.0 <= r['pred'] <= 1.0 for r in d)
print('ok', len(d), 'records')
"
```

Then deliberately break it, because this is what a judge's directory looks like:

```bash
mkdir -p /tmp/messy/sub && cp data/dummy/img/dummy_00000.png /tmp/messy/
printf 'not an image' > /tmp/messy/broken.png
printf 'notes' > /tmp/messy/readme.txt
cp data/dummy/img/dummy_00001.png /tmp/messy/sub/
python scripts/predict.py --image-dir /tmp/messy --out /tmp/messy.json
```

Expected: exits 0, reports 3 records and 1 unreadable, no traceback.

- [ ] **Step 6: Commit**

```bash
git add scripts/predict.py tests/test_predict_schema.py
git commit -m "feat: predict.py directory-to-JSON scorer with strict output schema"
```

---

### Task 3: Explainability maps

**Files:**
- Create: `src/aigcdet/explain/__init__.py`, `src/aigcdet/explain/patch_heatmap.py`, `tests/explain/test_patch_heatmap.py`

**Interfaces:**
- Consumes: `features.backbones`, `models.heads.Detector`, `features.recon.error_map`
- Produces:
  - `patch_scores(backbone, spec, model, img, device) -> np.ndarray` — `(g, g)` per-patch AIGC logits
  - `to_overlay(img, heat, alpha=0.45) -> np.ndarray` — RGB overlay at the original image size
  - `PATCH_HEATMAP_CAVEAT: str` — the honest label the dashboard must display

- [ ] **Step 1: Write the failing test**

```python
# tests/explain/test_patch_heatmap.py
import numpy as np
import pytest

from aigcdet.explain.patch_heatmap import PATCH_HEATMAP_CAVEAT, to_overlay


def test_caveat_text_is_present_and_mentions_the_heuristic():
    assert "heuristic" in PATCH_HEATMAP_CAVEAT.lower()


def test_overlay_matches_the_source_image_shape():
    img = np.random.default_rng(0).integers(0, 256, (120, 200, 3), dtype=np.uint8)
    heat = np.random.default_rng(1).random((8, 8)).astype(np.float32)
    out = to_overlay(img, heat)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_overlay_is_bounded_and_blends():
    img = np.zeros((64, 64, 3), np.uint8)
    heat = np.ones((4, 4), np.float32)
    out = to_overlay(img, heat, alpha=1.0)
    assert out.max() <= 255 and out.min() >= 0
    assert not np.array_equal(out, img)


def test_uniform_heat_produces_a_uniform_overlay():
    img = np.full((32, 32, 3), 100, np.uint8)
    out = to_overlay(img, np.full((4, 4), 0.5, np.float32))
    assert out.std(axis=(0, 1)).max() < 5.0


@pytest.mark.gpu
def test_patch_scores_grid_matches_the_backbone_geometry():
    import torch
    from aigcdet.explain.patch_heatmap import patch_scores
    from aigcdet.features.backbones import load_backbone
    from aigcdet.models.heads import Detector
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    bb, spec = load_backbone("clipl", device="cuda")
    model = Detector(dim_feat=spec.dim, use_recon=False).to("cuda")
    img = np.random.default_rng(0).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    heat = patch_scores(bb, spec, model, img, device="cuda")
    assert heat.ndim == 2 and heat.shape[0] == heat.shape[1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/explain -v -m "not gpu"`
Expected: FAIL with `ModuleNotFoundError: No module named 'aigcdet.explain'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/aigcdet/explain/patch_heatmap.py
"""Per-patch AIGC map (spec §3.8).

Because the head consumes global-average-pooled patch tokens, applying that
same head to each token individually yields a spatial map for free: no
Grad-CAM, no extra training.

It is a heuristic, and labelled as one. The head was trained on the pooled
vector, so per-token outputs are off-distribution. Check spatial coherence on
a handful of images before putting it on the dashboard.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

PATCH_HEATMAP_CAVEAT = (
    "Heuristic: the classifier was trained on pooled features, so per-patch "
    "scores indicate where evidence concentrates, not a calibrated per-region "
    "probability."
)


@torch.no_grad()
def patch_scores(backbone, spec, model, img: np.ndarray, device: str = "cuda") -> np.ndarray:
    from aigcdet.features.backbones import _MEAN, _STD, squish
    if model.use_recon:
        raise ValueError("patch heatmap is only defined for models without the "
                         "recon branch; use the VAE error map instead")
    x = squish(img, spec.image_size).astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    t = torch.from_numpy(x).permute(2, 0, 1)[None].to(device, torch.float16)
    h = backbone(pixel_values=t).last_hidden_state[:, spec.num_prefix_tokens:, :]
    tokens = h[0].float()                                # (T, D)
    logits = model(tokens)["logit"].cpu().numpy()        # (T,)
    g = int(round(np.sqrt(len(logits))))
    return logits[:g * g].reshape(g, g)


def to_overlay(img: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h = heat.astype(np.float32)
    rng = h.max() - h.min()
    h = np.zeros_like(h) if rng < 1e-8 else (h - h.min()) / rng
    h = cv2.resize(h, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    colour = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
    out = (1 - alpha) * img.astype(np.float32) + alpha * colour.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)
```

Also create empty `src/aigcdet/explain/__init__.py` and `tests/explain/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/explain -v -m "not gpu"`
Expected: 4 passed

- [ ] **Step 5: Sanity-check spatial coherence before trusting it**

Render the overlay for six images — three authentic, three generated — and look
at them. If the map is spatially incoherent noise, the dashboard shows the VAE
error map alone and the README says the patch heatmap did not survive
inspection. That is a finding, not a failure.

- [ ] **Step 6: Commit**

```bash
git add src/aigcdet/explain tests/explain
git commit -m "feat(explain): per-patch AIGC heatmap with honest caveat text"
```

---

### Task 4: Gradio dashboard with live transform sliders

**Files:**
- Create: `app/dashboard.py`, `tests/test_dashboard_logic.py`

**Interfaces:**
- Consumes: `infer.Predictor`, `augment.ops`, `augment.recipes.FAMILIES`
- Produces:
  - `apply_sliders(img, jpeg_q, blur_sigma, resize_scale, noise_sigma, jitter_amt, crop_frac, seed=0) -> np.ndarray` — pure function, testable without a UI
  - `format_readout(result: dict) -> str`
  - `sweep_curve(predictor, img, param: str, values: list[float]) -> pandas.DataFrame`
  - `build_ui(predictor) -> gradio.Blocks`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dashboard_logic.py
import sys

import numpy as np

sys.path.insert(0, "app")


def _img():
    return np.random.default_rng(0).integers(0, 256, (128, 128, 3), dtype=np.uint8)


def test_sliders_at_neutral_settings_are_the_identity():
    from dashboard import apply_sliders
    img = _img()
    out = apply_sliders(img, jpeg_q=100, blur_sigma=0.0, resize_scale=1.0,
                        noise_sigma=0.0, jitter_amt=0.0, crop_frac=1.0)
    assert np.array_equal(out, img)


def test_each_slider_changes_the_image():
    from dashboard import apply_sliders
    img = _img()
    base = dict(jpeg_q=100, blur_sigma=0.0, resize_scale=1.0,
                noise_sigma=0.0, jitter_amt=0.0, crop_frac=1.0)
    for k, v in [("jpeg_q", 30), ("blur_sigma", 2.0), ("resize_scale", 0.25),
                 ("noise_sigma", 0.1), ("jitter_amt", 0.2), ("crop_frac", 0.8)]:
        out = apply_sliders(img, **{**base, k: v})
        assert out.shape == img.shape, k
        assert not np.array_equal(out, img), k


def test_sliders_are_deterministic_for_a_fixed_seed():
    from dashboard import apply_sliders
    img = _img()
    a = apply_sliders(img, 100, 0.0, 1.0, 0.05, 0.0, 1.0, seed=3)
    b = apply_sliders(img, 100, 0.0, 1.0, 0.05, 0.0, 1.0, seed=3)
    assert np.array_equal(a, b)


def test_readout_mentions_the_estimated_degradations():
    from dashboard import format_readout
    text = format_readout({"pred": 0.8, "eqi": 0.4, "decision": "review",
                           "severity": [0.6, 0.1, 0.0, 0.0, 0.0, 0.0],
                           "presence": [0.9, 0.2, 0.0, 0.0, 0.0, 0.0],
                           "proxies": [52.0, 120.0, 1.4]})
    assert "jpeg" in text.lower()
    assert "review" in text.lower()
    assert "0.4" in text or "40" in text


def test_sweep_curve_returns_one_row_per_value():
    from dashboard import sweep_curve

    class Stub:
        def predict_array(self, img, path=None):
            return {"pred": float(img.mean()) / 255.0, "eqi": 0.5,
                    "decision": "review", "severity": [0] * 6,
                    "presence": [0] * 6, "proxies": [0] * 3}

    df = sweep_curve(Stub(), _img(), "jpeg_q", [90, 70, 50, 30])
    assert len(df) == 4 and {"value", "pred", "eqi"} <= set(df.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dashboard_logic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/dashboard.py
"""Live robustness dashboard (spec §8).

The key interaction: drag JPEG quality from 90 to 30 and watch the degradation
readout track it, EQI fall, and the decision flip Flag -> Review while the
underlying score stays roughly right. That shows the degradation head, the
calibration, and the abstention policy in one gesture.

All slider logic lives in pure functions so it can be tested without a browser.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from aigcdet.augment import ops
from aigcdet.augment.recipes import FAMILIES


def apply_sliders(img: np.ndarray, jpeg_q: float, blur_sigma: float,
                  resize_scale: float, noise_sigma: float, jitter_amt: float,
                  crop_frac: float, seed: int = 0) -> np.ndarray:
    """Applied in redistribution order: geometry, then optics, then encoding."""
    out = img
    if crop_frac < 1.0:
        out = ops.center_crop(out, frac=float(crop_frac))
    if resize_scale < 1.0:
        out = ops.resize_roundtrip(out, scale=float(resize_scale))
    if blur_sigma > 0:
        out = ops.blur(out, sigma=float(blur_sigma))
    if jitter_amt != 0:
        out = ops.jitter(out, float(jitter_amt), float(jitter_amt), float(jitter_amt))
    if noise_sigma > 0:
        out = ops.noise(out, sigma=float(noise_sigma), rng=np.random.default_rng(seed))
    if jpeg_q < 100:
        out = ops.jpeg(out, quality=int(jpeg_q))
    return out


def format_readout(result: dict) -> str:
    sev, pres = result["severity"], result["presence"]
    detected = [f"{FAMILIES[i]} (sev {sev[i]:.2f})"
                for i in range(len(FAMILIES)) if pres[i] > 0.5]
    q, lap, nf = result["proxies"]
    return (
        f"**Decision:** {result['decision'].upper()}\n\n"
        f"**AIGC probability:** {result['pred']:.3f}\n\n"
        f"**Evidence Quality Index:** {result['eqi']:.2f} "
        f"({result['eqi'] * 100:.0f}% of usable evidence retained)\n\n"
        f"**Detected degradation:** {', '.join(detected) if detected else 'none'}\n\n"
        f"**Model-free proxies:** estimated JPEG q≈{q:.0f}, "
        f"sharpness {lap:.0f}, noise floor {nf:.2f}"
    )


def sweep_curve(predictor, img: np.ndarray, param: str, values: list[float]) -> pd.DataFrame:
    base = dict(jpeg_q=100, blur_sigma=0.0, resize_scale=1.0,
                noise_sigma=0.0, jitter_amt=0.0, crop_frac=1.0)
    rows = []
    for v in values:
        res = predictor.predict_array(apply_sliders(img, **{**base, param: v}))
        rows.append({"value": v, "pred": res["pred"], "eqi": res["eqi"]})
    return pd.DataFrame(rows)


def build_ui(predictor):
    import gradio as gr
    from aigcdet.explain.patch_heatmap import PATCH_HEATMAP_CAVEAT

    def run(img, jpeg_q, blur_sigma, resize_scale, noise_sigma, jitter_amt, crop_frac):
        if img is None:
            return None, "Upload an image to begin.", None
        arr = np.asarray(img, dtype=np.uint8)
        deg = apply_sliders(arr, jpeg_q, blur_sigma, resize_scale,
                            noise_sigma, jitter_amt, crop_frac)
        res = predictor.predict_array(deg)
        curve = sweep_curve(predictor, arr, "jpeg_q",
                            [95, 90, 80, 70, 60, 50, 40, 30])
        return deg, format_readout(res), curve

    with gr.Blocks(title="Robust AIGC Detection") as demo:
        gr.Markdown("# Robust AI-Generated Image Detection\n"
                    "Drag the sliders to simulate redistribution and watch the "
                    "detector's confidence and decision respond.")
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(label="Input", type="numpy")
            with gr.Column(scale=1):
                jpeg_q = gr.Slider(30, 100, 100, step=1, label="JPEG quality")
                blur_sigma = gr.Slider(0.0, 2.0, 0.0, step=0.1, label="Blur sigma")
                resize_scale = gr.Slider(0.25, 1.0, 1.0, step=0.05, label="Resize scale")
                noise_sigma = gr.Slider(0.0, 0.10, 0.0, step=0.005, label="Noise sigma")
                jitter_amt = gr.Slider(-0.2, 0.2, 0.0, step=0.02, label="Colour jitter")
                crop_frac = gr.Slider(0.8, 1.0, 1.0, step=0.02, label="Centre crop")
            with gr.Column(scale=1):
                out_img = gr.Image(label="Transformed")
                readout = gr.Markdown()
        plot = gr.LinePlot(x="value", y="pred", title="Score vs JPEG quality")
        gr.Markdown(f"_{PATCH_HEATMAP_CAVEAT}_")

        controls = [inp, jpeg_q, blur_sigma, resize_scale, noise_sigma,
                    jitter_amt, crop_frac]
        for c in controls:
            c.change(run, controls, [out_img, readout, plot])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="outputs/release")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    from aigcdet.infer import Predictor
    build_ui(Predictor.load(a.bundle, device=a.device)).launch(share=a.share)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dashboard_logic.py -v`
Expected: 5 passed

- [ ] **Step 5: Launch and check the key interaction**

```bash
python app/dashboard.py --bundle outputs/release
```

Confirm by hand: dragging JPEG quality 90 → 30 moves the degradation readout,
lowers EQI, and flips the decision. If it does not, the video's central claim
does not hold and that needs to be known before recording, not during.

- [ ] **Step 6: Commit**

```bash
git add app/dashboard.py tests/test_dashboard_logic.py
git commit -m "feat(app): Gradio dashboard with live transform sliders"
```

---

### Task 5: README, Devpost text, and video script

**Files:**
- Create: `README.md`, `docs/devpost.md`, `docs/video_script.md`

**Interfaces:**
- Consumes: every artifact produced by Plans 1–3 (`docs/data_audit.md`, `docs/robustness_table.md`, `docs/selection.json`, `docs/errors/`, `docs/model_licences.md`)
- Produces: the three written deliverables

- [ ] **Step 1: Write the README**

Fill every bracketed value from the generated artifacts. Do not ship a bracket.

````markdown
# Robust Detection of AI-Generated Images Under Real-World Transformations

Detects AI-generated images and keeps working after the post-processing that
real redistribution applies: JPEG re-encoding, blur, rescaling, noise, colour
adjustment, and cropping.

## What this is

A frozen self-supervised vision backbone with two small trained heads. One
predicts whether the image is AI-generated. The other predicts **what was done
to the image**, and that estimate drives calibration and an abstention policy:
when the forensic evidence has been destroyed, the system says so instead of
guessing.

**Headline results** (tier: [ablation | final], [N] images, 95% bootstrap CIs):

| | Clean ROC-AUC | Robust ROC-AUC | TPR @ 1% FPR |
| --- | --- | --- | --- |
| Ours ([rung]) | [x.xxx] | [x.xxx] | [x.xxx] |
| UniversalFakeDetect (rung A0) | [x.xxx] | [x.xxx] | [x.xxx] |
| NPR | [x.xxx] | [x.xxx] | [x.xxx] |
| AEROBLADE | [x.xxx] | [x.xxx] | [x.xxx] |

Full table: [`docs/robustness_table.md`](docs/robustness_table.md) ·
Heatmap: [`docs/robustness_table.png`](docs/robustness_table.png)

**Reviewer load** — the number this is actually for: under [scenario], the
policy auto-decides **[x]%** of the queue while holding false positives on
authentic images at 1%, deferring the rest to a human.

## Setup

```bash
git clone <repo> && cd aigc-robust-detect
pip install -e ".[dev]"
```

## Scoring a directory

```bash
python scripts/predict.py --image-dir path/to/images --out predictions.json
```

Output is a JSON list of `{"image_path": ..., "pred": <float in [0,1]>}`, where
`pred` is a **calibrated** probability. Add `--rich` for EQI, the
Clear/Review/Flag decision, and the estimated degradation.

## Reproducing the results

```bash
# 1. Data: acquire, audit, normalise, dedupe against the demo set, split
python scripts/acquire_data.py --dataset sid_set --limit 30000 --out data/raw
python scripts/build_dataset.py --raw data/raw --out data/normalized \
    --demo-dir data/raw/demo --manifest data/manifest.parquet

# 2. Stage A: cache frozen features (~1.5 h/backbone; ~8 h for the recon branch)
python scripts/extract_features.py --manifest data/manifest.parquet \
    --backbone dinov3l --out banks/dinov3l

# 3. Stage B + evaluation: every ablation rung, then the robustness table
python scripts/run_ablation.py --bank banks/dinov3l --eval-bank banks/eval_dinov3l \
    --rungs configs/rungs/a0.yaml configs/rungs/a1.yaml configs/rungs/a2.yaml \
            configs/rungs/a3.yaml configs/rungs/a4.yaml --tier ablation

# 4. Freeze the release bundle, then demo
python scripts/export_bundle.py --checkpoint outputs/rungs/[rung]/checkpoint.pt \
    --bank banks/dinov3l --eval-bank banks/eval_dinov3l --out outputs/release
python app/dashboard.py --bundle outputs/release
```

## Evaluation discipline

- **Two tiers, always stated.** Ablation tier: 5k internal validation + a 5k
  stratified benchmark subsample over all 20 conditions. Final tier: the full
  13.8k benchmark over the 15 core conditions, run once.
- **Model selection was fixed before any results existed:** highest robust
  TPR @ 1% FPR on internal validation, held-out generators. Not clean AUC, not
  the demo set. See [`docs/selection.json`](docs/selection.json).
- **The external benchmark was evaluated once**, at the end.
- **Bootstrap 95% CIs on every AUC.** Where intervals overlap we say the
  difference is unresolved rather than claiming a win.

## Things we checked that most detectors don't

**The dataset confound.** Detection datasets routinely differ by class in
resolution and JPEG history, letting a model score ~99% without looking at
content — then collapse under re-encoding, because the transform erases the
shortcut rather than the signal. We audited before normalising
([`docs/data_audit.md`](docs/data_audit.md)) and ran a **content-blind
control**: a classifier seeing only 16×16 thumbnails or file metadata scores
**[x.xx] AUC** on our splits and **[x.xx]** on the official demo set.
[One sentence interpreting those two numbers.]

**Held-out severities.** Training never drew JPEG q ∈ [65, 75] or blur
σ ∈ [0.85, 1.15], so the brief's q=70 and σ=1.0 conditions are *unseen
severities* at evaluation. Marked in the robustness table.

**Held-out generators and transforms.** Two generator families were excluded
from training entirely; a separate run excluded Gaussian noise as a whole
family. Both are reported.

## Limitations, and what we'd do with more time

1. Frozen backbones cap accuracy relative to full fine-tuning; the trade-off
   buys generalisation and fits a single shared GPU. It also limits how much
   the consistency loss can achieve, since it only shapes a ~2M-parameter head.
2. The fixed augmentation bank (11 views per image) is less diverse than
   on-the-fly augmentation.
3. The degradation head estimates degradation **relative to an unknown
   baseline** — source images already carry prior compression.
4. The reconstruction branch is specific to the SD 1.5 autoencoder and is
   expected to be weak on DALL·E and proprietary generators, including the
   official demo set. [State whether it survived its kill criterion.]
5. Training data covers the generators in WildFake and SID_Set. The 2026
   benchmark literature reports sharp declines on the newest commercial
   generators (Flux, Firefly v4, Midjourney v7), which are not represented.
6. **No adversarial robustness.** An attacker who knows the detector can evade
   it. Only incidental redistribution transforms are modelled.

With another week: LoRA fine-tuning of the last blocks, a second
paradigm-diverse backbone in the ensemble, and generator attribution as a
secondary head.

## Data and model licences

Datasets: `data/raw/LICENCES.json`. Model weights:
[`docs/model_licences.md`](docs/model_licences.md). CIFAKE was deliberately
excluded: at 32×32, "centre crop 80%" and "JPEG-30" are not the problem this
brief describes.

## Team contributions

| Member | Workstream |
| --- | --- |
| [name] | W1 data, audit, augmentation |
| [name] | W2 features, reconstruction branch, training |
| [name] | W3 evaluation, calibration, baselines |
| [name] | W4 demo, packaging, writeup |
````

- [ ] **Step 2: Write the Devpost description**

`docs/devpost.md` must cover the brief's deliverable 1 in full: how the
solution addresses the problem statement, development tools, models/APIs,
libraries and frameworks, datasets and assets. Reuse the README's opening and
results, then add this section verbatim so nothing required is missing:

```markdown
**Development tools:** VS Code, Jupyter, Kaggle Notebooks (free-tier T4), git.

**Models:** DINOv3 ViT-L/16, SigLIP2-L/16, CLIP ViT-L/14, Stable Diffusion 1.5
VAE, LPIPS (AlexNet). Total under 1B parameters; the brief's limit is 2B.
No external APIs are called at inference.

**Libraries and frameworks:** PyTorch, HuggingFace Transformers and Diffusers,
scikit-learn, NumPy, SciPy, OpenCV, Pillow, pandas, Gradio, pytest.

**Datasets and assets:** WildFake, SID_Set, COCO val2017 (evaluation only).
CIFAKE excluded, with reason. All transformed test cases were generated by our
own augmentation module; no third-party imagery appears in the demo or video.
```

- [ ] **Step 3: Write the video script**

```markdown
# Demo video script (~2.5 min)

| Time | Shot | Say |
| --- | --- | --- |
| 0:00–0:15 | Title, then a real photo and an AI image side by side | Synthetic images are easy to make, and they get compressed, cropped, and reposted before anyone checks them. Detection has to survive that. |
| 0:15–0:35 | `docs/data_audit.md` table on screen | The thing most detectors miss: benchmark datasets differ by class in resolution and JPEG history. A content-blind classifier scores [x.xx] on the official demo set. So we normalised, and we report the control. |
| 0:35–1:05 | Architecture diagram | Frozen backbone, two small heads. One says whether it's AI. The other says what was done to the image — and that drives calibration and abstention. |
| 1:05–2:05 | **Live dashboard.** Drag JPEG quality 90 → 30 | Watch the degradation readout track the compression, EQI fall, and the decision flip from Flag to Review — while the score stays roughly right. The system knows when its evidence is gone. |
| 2:05–2:35 | Robustness table, held-out generator column, calibration plot | [Clean AUC] clean, [robust AUC] across 20 conditions, and here's a generator we never trained on. At 1% false positives it auto-decides [x]% of the queue. |
| 2:35–2:50 | Limitations slide | What it can't do: generators newer than our training data, and anyone deliberately attacking it. |

Constraints: no third-party trademarks, logos, or copyrighted imagery. The
sample gallery uses licensed images only. Upload to YouTube as **public** and
link it in the Devpost description.
```

- [ ] **Step 4: Verify no placeholders survive**

```bash
grep -nE "\[x\.x|\[x\]%|\[rung\]|\[name\]|\[N\]|\[scenario\]|\[One sentence|\[State whether|TBD|TODO" \
    README.md docs/devpost.md docs/video_script.md
```

Expected: no output. Every bracket must hold a real value before submission.

- [ ] **Step 5: Final full-suite run and commit**

```bash
python -m pytest -v
git add README.md docs/devpost.md docs/video_script.md
git commit -m "docs: README, Devpost description, and video script"
```

---

## Plan 4 Completion Criteria

- [ ] `python -m pytest -v` passes, GPU-marked tests included, on the A4500
- [ ] `python scripts/predict.py --image-dir <any dir> --out preds.json` produces a JSON list whose objects have **exactly** `image_path` and `pred`
- [ ] `predict.py` survives a directory containing a corrupt file, a text file, and a subdirectory — exits 0, no traceback
- [ ] A 5k-image directory scores in under ~3 minutes on the A4500
- [ ] The dashboard runs, and dragging JPEG quality 90 → 30 visibly moves the readout, EQI, and decision
- [ ] `grep` for placeholders in README / Devpost / script returns nothing
- [ ] The demo video is on YouTube, public, and linked from the Devpost description
- [ ] README states the content-blind control numbers for **both** our splits and the official demo set
- [ ] README states whether the reconstruction branch survived its kill criterion
- [ ] README states the reviewer-load number at 1% FPR
