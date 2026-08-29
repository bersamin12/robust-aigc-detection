"""The single inference path (spec section 3.1).

The two-stage training split disappears here. Stage A cached features for a
million augmented views and Stage B trained a head on them, but inference is
one straight line: image -> canonicalise -> backbone -> (recon) -> head ->
calibrated probability -> decision.

`scripts/predict.py` (the required deliverable) and the dashboard both go
through this class rather than each assembling the chain themselves, because
the two failure modes of a duplicated inference path are both silent. One is
drift -- the demo shows a number the submitted script would not produce. The
other is omission: the first version of `predict.py` returned
`sigmoid(logit)`, which is in [0, 1], varies with the image, and looks exactly
like a calibrated probability while meaning something else.
"""
from __future__ import annotations

import inspect
import json
import os
import shutil
import warnings

import joblib
import numpy as np
import torch
from PIL import Image

from aigcdet.augment.canonical import canonicalise
from aigcdet.calibrate.policy import Policy, decide
from aigcdet.features.backbones import embed, load_backbone
from aigcdet.features.proxies import proxy_vector
from aigcdet.train.train_head import load_detector

#: The two keys the brief requires in the output JSON, and the only two the
#: default `predict.py` emits. Named here rather than in the script so the
#: schema test and the writer read the same tuple.
RESULT_KEYS_MINIMAL = ("image_path", "pred")

#: What a file that could not be read scores. Deliberately the point of maximum
#: uncertainty: an unreadable file is not evidence of anything, and 0.0 would
#: assert authenticity the detector never established.
FAILED_PRED = 0.5

_N_SEVERITY = 6
_N_PROXIES = 3

CHECKPOINT_NAME = "checkpoint.pt"
CALIBRATOR_NAME = "calibrator.joblib"
EQI_NAME = "eqi.joblib"
POLICY_NAME = "policy.json"
CONFIG_NAME = "config.json"


def export_bundle(checkpoint: str, calibrator, eqi, policy: Policy, out_dir: str,
                  backbone_name: str, use_recon: bool, dim_feat: int) -> str:
    """Freeze everything inference needs into one directory, and return it.

    The checkpoint is COPIED rather than referenced: a bundle that points at a
    path in someone's `outputs/` is not a release, and the failure only shows
    up on the machine that does not have that path.
    """
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(checkpoint, os.path.join(out_dir, CHECKPOINT_NAME))
    joblib.dump(calibrator, os.path.join(out_dir, CALIBRATOR_NAME))
    joblib.dump(eqi, os.path.join(out_dir, EQI_NAME))
    with open(os.path.join(out_dir, POLICY_NAME), "w") as f:
        json.dump({"flag_threshold": policy.flag_threshold,
                   "clear_threshold": policy.clear_threshold,
                   "eqi_threshold": policy.eqi_threshold}, f, indent=2)
    with open(os.path.join(out_dir, CONFIG_NAME), "w") as f:
        json.dump({"backbone": backbone_name, "use_recon": bool(use_recon),
                   "dim_feat": int(dim_feat)}, f, indent=2)
    return out_dir


def _calibrator_arity(calibrator) -> int:
    """How many positional arguments this calibrator's `transform` takes.

    Chosen by inspecting the signature rather than by calling with two
    arguments and catching TypeError. The catch-based version cannot tell "this
    calibrator takes one argument" from "this calibrator takes two and raised a
    TypeError inside", and quietly reinterprets the second as the first.
    """
    transform = getattr(calibrator, "transform", None)
    if transform is None:
        raise TypeError(
            f"calibrator {type(calibrator).__name__} has no .transform(); a "
            "bundle must ship a fitted ConditionalTemperature or "
            "GlobalTemperature. Refusing rather than falling back to a raw "
            "sigmoid, which would look like a calibrated probability.")
    params = [p for p in inspect.signature(transform).parameters.values()
              if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(params)


class Predictor:
    """Bundle in, calibrated decisions out."""

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
        with open(os.path.join(bundle_dir, CONFIG_NAME)) as f:
            cfg = json.load(f)
        model, _ck = load_detector(os.path.join(bundle_dir, CHECKPOINT_NAME),
                                   device=device)
        backbone, spec = load_backbone(cfg["backbone"], device=device)
        calibrator = joblib.load(os.path.join(bundle_dir, CALIBRATOR_NAME))
        eqi = joblib.load(os.path.join(bundle_dir, EQI_NAME))
        with open(os.path.join(bundle_dir, POLICY_NAME)) as f:
            policy = Policy(**json.load(f))
        return cls(model, backbone, spec, calibrator, eqi, policy,
                   bool(cfg["use_recon"]), device)

    def _recon_vector(self, img: np.ndarray) -> np.ndarray:
        from aigcdet.features.recon import load_recon_models, recon_features

        if self._vae is None:
            self._vae, self._lpips = load_recon_models(self.device)
        return recon_features(img, self._vae, self._lpips, self.device)

    def _calibrate(self, logit: float, cond: np.ndarray) -> float:
        logits = np.asarray([logit], dtype=np.float64)
        arity = _calibrator_arity(self.calibrator)
        if arity >= 2:
            return float(self.calibrator.transform(logits, cond[None])[0])
        return float(self.calibrator.transform(logits)[0])

    @torch.no_grad()
    def predict_array(self, img: np.ndarray, path: str | None = None) -> dict:
        """Score one decoded RGB uint8 image.

        `img` is canonicalised first, exactly as `features/extract.py`,
        `eval/grid.py` and `features/recon.py` do. Resolution separates this
        project's real and fake pools almost perfectly and transfers
        backwards (docs/resolution_shortcut.md), so an inference path that
        skipped this step would feed the head a distribution it never trained
        on and report the result with full confidence.
        """
        canon = canonicalise(img)
        f = embed(self.backbone, self.spec, [canon], device=self.device,
                  batch_size=1)
        r = self._recon_vector(canon)[None] if self.use_recon else None
        out = self.model(
            torch.from_numpy(np.asarray(f)).to(self.device),
            torch.from_numpy(np.asarray(r)).to(self.device) if r is not None
            else None)

        logit = float(out["logit"].reshape(-1)[0].item())
        severity = out["severity"].cpu().numpy()[0]
        presence = torch.sigmoid(out["presence"]).cpu().numpy()[0]
        deg_emb = out["deg_embedding"].cpu().numpy()[0]
        # Proxies describe the image as it will be judged, so they are measured
        # on the canonicalised pixels the head actually saw -- not on the
        # original file, whose resolution is the thing being neutralised.
        proxies = proxy_vector(canon, path)

        cond = np.concatenate([severity, proxies]).astype(np.float32)
        pred = self._calibrate(logit, cond)
        eqi_val = (float(self.eqi.predict(cond[None])[0])
                   if self.eqi is not None else float(max(pred, 1.0 - pred)))
        decision = str(decide(np.array([pred]), np.array([eqi_val]),
                              self.policy)[0])
        return {"pred": pred, "logit": logit, "eqi": eqi_val,
                "decision": decision, "severity": severity.tolist(),
                "presence": presence.tolist(), "proxies": proxies.tolist(),
                "deg_embedding": deg_emb.tolist(), "error": None}

    def predict_paths(self, paths: list[str], batch_size: int = 16) -> list[dict]:
        """Score every path, in order, one result each.

        Never raises on someone else's directory: a file that cannot be decoded
        gets `pred=FAILED_PRED` and a populated `error`. Each result carries
        its own `image_path`, so a failure cannot shift the scores of the files
        after it -- re-pairing results to inputs positionally in the caller is
        how every score after the first bad file lands on the wrong image while
        every row still looks well-formed.
        """
        results = []
        for p in paths:
            try:
                with Image.open(p) as im:
                    arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
                res = self.predict_array(arr, path=p)
            except Exception as exc:              # noqa: BLE001 -- reported
                warnings.warn(f"skipping {p}: {type(exc).__name__}: {exc}",
                              stacklevel=2)
                res = {"pred": FAILED_PRED, "logit": 0.0, "eqi": 0.0,
                       "decision": "review",
                       "severity": [0.0] * _N_SEVERITY,
                       "presence": [0.0] * _N_SEVERITY,
                       "proxies": [0.0] * _N_PROXIES,
                       "deg_embedding": [],
                       "error": f"{type(exc).__name__}: {exc}"}
            res["image_path"] = p
            results.append(res)
        return results
