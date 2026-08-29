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


def _failed_result(exc: BaseException) -> dict:
    """The row a file that could not be decoded gets."""
    return {"pred": FAILED_PRED, "logit": 0.0, "eqi": 0.0, "decision": "review",
            "severity": [0.0] * _N_SEVERITY, "presence": [0.0] * _N_SEVERITY,
            "proxies": [0.0] * _N_PROXIES, "deg_embedding": [],
            "error": f"{type(exc).__name__}: {exc}"}


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

    def _calibrate(self, logits: np.ndarray, cond: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        if _calibrator_arity(self.calibrator) >= 2:
            return np.asarray(self.calibrator.transform(logits, cond),
                              dtype=np.float64)
        return np.asarray(self.calibrator.transform(logits), dtype=np.float64)

    @torch.no_grad()
    def predict_arrays(self, imgs: list[np.ndarray],
                       paths: list[str | None] | None = None) -> list[dict]:
        """Score decoded RGB uint8 images, one result each, in order.

        Every image is canonicalised first, exactly as `features/extract.py`,
        `eval/grid.py` and `features/recon.py` do. Resolution separates this
        project's real and fake pools almost perfectly and transfers backwards
        (docs/resolution_shortcut.md), so an inference path that skipped this
        step would feed the head a distribution it never trained on and report
        the result with full confidence.

        Canonicalisation gives images of differing shapes, which is fine:
        `embed` squishes each to the backbone's input size, so the whole list
        still goes through the tower as ONE batch. That is the difference
        between a 5k-image directory taking minutes and taking a quarter of an
        hour.
        """
        if not imgs:
            return []
        paths = list(paths) if paths is not None else [None] * len(imgs)
        canon = [canonicalise(i) for i in imgs]

        f = np.asarray(embed(self.backbone, self.spec, canon,
                             device=self.device, batch_size=len(canon)))
        r = (np.stack([self._recon_vector(c) for c in canon])
             if self.use_recon else None)
        out = self.model(
            torch.from_numpy(f).to(self.device),
            torch.from_numpy(np.asarray(r)).to(self.device) if r is not None
            else None)

        logits = out["logit"].reshape(-1).cpu().numpy().astype(np.float64)
        severity = out["severity"].cpu().numpy()
        presence = torch.sigmoid(out["presence"]).cpu().numpy()
        deg_emb = out["deg_embedding"].cpu().numpy()
        # Proxies describe the image as it will be judged, so they are measured
        # on the canonicalised pixels the head actually saw -- not on the
        # original file, whose resolution is the thing being neutralised.
        proxies = np.stack([proxy_vector(c, p) for c, p in zip(canon, paths)])

        cond = np.concatenate([severity, proxies], axis=1).astype(np.float32)
        preds = self._calibrate(logits, cond)
        eqi = (np.asarray(self.eqi.predict(cond), dtype=np.float64)
               if self.eqi is not None
               else np.maximum(preds, 1.0 - preds))
        decisions = decide(preds, eqi, self.policy)

        return [{"pred": float(preds[i]), "logit": float(logits[i]),
                 "eqi": float(eqi[i]), "decision": str(decisions[i]),
                 "severity": severity[i].tolist(),
                 "presence": presence[i].tolist(),
                 "proxies": proxies[i].tolist(),
                 "deg_embedding": deg_emb[i].tolist(), "error": None}
                for i in range(len(canon))]

    def predict_array(self, img: np.ndarray, path: str | None = None) -> dict:
        """Score one decoded RGB uint8 image. A batch of one."""
        return self.predict_arrays([img], [path])[0]

    def predict_paths(self, paths: list[str], batch_size: int = 16) -> list[dict]:
        """Score every path, in order, one result each.

        Never raises on someone else's directory: a file that cannot be decoded
        gets `pred=FAILED_PRED` and a populated `error`. Each result carries
        its own `image_path`, so a failure cannot shift the scores of the files
        after it -- re-pairing results to inputs positionally in the caller is
        how every score after the first bad file lands on the wrong image while
        every row still looks well-formed.

        Decoding is done a batch at a time so the backbone sees `batch_size`
        images per forward pass rather than one.
        """
        results: list[dict] = []
        for start in range(0, len(paths), batch_size):
            chunk = paths[start:start + batch_size]
            imgs, kept, slots = [], [], {}
            for p in chunk:
                try:
                    with Image.open(p) as im:
                        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
                except Exception as exc:          # noqa: BLE001 -- reported
                    warnings.warn(f"skipping {p}: {type(exc).__name__}: {exc}",
                                  stacklevel=2)
                    slots[p] = _failed_result(exc)
                    continue
                imgs.append(arr)
                kept.append(p)
            scored = self.predict_arrays(imgs, kept)
            # Keyed by path, then re-emitted in the CHUNK's order. Appending
            # the scored rows and then the failed ones would reorder the
            # output around every bad file; re-pairing positionally after a
            # drop is how each remaining score lands on the wrong image.
            slots.update(dict(zip(kept, scored)))
            for p in chunk:
                res = dict(slots[p])
                res["image_path"] = p
                results.append(res)
        return results
