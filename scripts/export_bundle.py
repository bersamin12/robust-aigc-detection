"""Freeze the selected headline model into a release bundle.

    python scripts/export_bundle.py --checkpoint outputs/rungs/a3/checkpoint.pt \
        --eval-bank data/banks/eval_dinov3l --out outputs/release

Calibration, the EQI and the decision policy are all fitted on the INTERNAL
validation split only (spec section 6.7). The external benchmark rows are in
the same eval bank -- they have to be, it is one grid -- so this script filters
by split and then hands the split labels themselves to every `fit`. Those
functions demand per-row evidence rather than the caller's promise, which is
what makes a filter bug fail loudly here instead of quietly inflating every
calibration number in the report.

The bundle this writes is the only thing `Predictor.load` reads: a self-
contained directory of checkpoint, calibrator, EQI, policy and config.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from aigcdet.calibrate import INTERNAL_VAL_SPLIT
from aigcdet.calibrate.eqi import EQI
from aigcdet.calibrate.policy import fit_policy, policy_report
from aigcdet.calibrate.temperature import ConditionalTemperature
from aigcdet.eval.grid import score_grid
from aigcdet.features.bank import FeatureBank
from aigcdet.infer import export_bundle
from aigcdet.train.train_head import load_detector


def condition_vectors(bank: FeatureBank, scores) -> np.ndarray:
    """`[severity | proxies]` per scored row, matched to that row's condition.

    Both arrays are indexed `[image, condition]`, and `scores` is one row per
    (condition, image), so the lookup needs BOTH coordinates. Taking the clean
    view for every row instead would hand the calibrator a condition vector
    that says "undegraded" about a JPEG-30 view -- and the conditional
    temperature exists precisely to tell those apart.
    """
    names = list(bank.config["conditions"])
    img_idx = np.asarray(scores["image_idx"])
    cond_idx = np.array([names.index(c) for c in scores["condition"]])
    sev = np.asarray(bank.severity)[img_idx, cond_idx]
    prox = np.asarray(bank.proxies)[img_idx, cond_idx]
    return np.concatenate([sev, prox], axis=1).astype(np.float32)


def split_labels(bank: FeatureBank, scores) -> np.ndarray:
    """The split each scored row came from, read off the bank.

    Read rather than asserted. `["val_internal"] * n` would satisfy the same
    signature and prove nothing; this is the array that makes a wrong filter
    visible to `check_fit_split`.
    """
    return np.asarray(bank.meta["split"])[np.asarray(scores["image_idx"])]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="a trained head, e.g. outputs/rungs/a3/checkpoint.pt")
    ap.add_argument("--eval-bank", required=True,
                    help="a bank written by extract_eval_bank, carrying the "
                         "condition grid and the val_internal rows")
    ap.add_argument("--out", default="outputs/release")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--target-coverage", type=float, default=0.85)
    a = ap.parse_args(argv)

    model, ck = load_detector(a.checkpoint, device=a.device)
    bank = FeatureBank.open(a.eval_bank)

    # A head reads one backbone's feature space. Scoring it against another's
    # bank yields numbers in the right range that mean nothing at all, and
    # nothing downstream can detect it.
    if bank.config["backbone"] != ck["backbone"]:
        raise SystemExit(
            f"the checkpoint was trained on {ck['backbone']!r} features but "
            f"the eval bank at {a.eval_bank} holds {bank.config['backbone']!r}")

    scores = score_grid(model, bank, use_recon=ck["config"]["use_recon"],
                        device=a.device)
    keep = split_labels(bank, scores) == INTERNAL_VAL_SPLIT
    if not keep.any():
        raise SystemExit(
            f"the eval bank at {a.eval_bank} has no {INTERNAL_VAL_SPLIT!r} "
            "rows, so there is nothing this script is permitted to calibrate "
            "on. Rebuild it with build_eval_manifest.py, which joins the "
            "training and benchmark manifests")
    scores = scores[keep].reset_index(drop=True)

    cond = condition_vectors(bank, scores)
    split = split_labels(bank, scores)
    y = np.asarray(scores["label"])
    logits = np.asarray(scores["score"], dtype=np.float64)

    cal = ConditionalTemperature(cond_dim=cond.shape[1]).fit(
        logits, y, cond, split=split)
    p = cal.transform(logits, cond)
    correct = ((p >= 0.5).astype(int) == y).astype(int)
    eqi = EQI().fit(cond, correct, split=split)
    policy = fit_policy(p, y, eqi.predict(cond), target_fpr=a.target_fpr,
                        target_coverage=a.target_coverage, split=split)

    out = export_bundle(a.checkpoint, cal, eqi, policy, a.out,
                        backbone_name=ck["backbone"],
                        use_recon=ck["config"]["use_recon"],
                        dim_feat=ck["dim_feat"])
    print(f"bundle written to {out}  ({len(scores)} {INTERNAL_VAL_SPLIT} rows)")
    print(json.dumps(policy_report(p, y, eqi.predict(cond), policy), indent=2,
                     default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
