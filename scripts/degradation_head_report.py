"""Score A2's degradation head against the eval bank's own targets.

The precondition nobody had checked. A2 is justified by feeding calibration,
EQI and the dashboard rather than the score (see `handoffs/08-ablation-rungs.md`
-- its classifier is provably bit-identical to A1's), so the question "does the
head work" is not answered anywhere in the ladder. It cannot be: every metric
`run_ablation.py` tabulates is rank- or threshold-invariant, and a confidence
readout is invisible to all of them.

Scored the way `degradation_loss` TRAINS it, which is the only defensible
choice: presence as a per-family binary readout over every view, severity
smooth-L1 MASKED to families actually present. An absent family's severity
target is meaningless -- the transform was never applied -- so scoring it there
would manufacture a number out of padding and flatter the head.

Severity is reported beside the MAE of always predicting the family's mean.
Without that column a plausible-looking 0.22 cannot be distinguished from a
head that has learnt nothing, and for `resize` on the DINOv3 bank that is
exactly the distinction that matters: it scores WORSE than the constant.

Reads only the degradation branch, so it needs no eval manifest, no labels and
no classifier -- it is a property of the head and the bank alone.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from aigcdet.augment.recipes import FAMILIES
from aigcdet.models.heads import Detector


def score(bank: str, ckpt: str, chunk: int = 512) -> dict:
    cfg = json.load(open(f"{bank}/config.json"))
    conds = cfg["conditions"]
    feats = np.load(f"{bank}/feats.npy", mmap_mode="r")
    tgt_p = np.asarray(np.load(f"{bank}/presence.npy", mmap_mode="r"))
    tgt_s = np.asarray(np.load(f"{bank}/severity.npy", mmap_mode="r"))
    n_img, n_view, dim = feats.shape

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    # use_film=False regardless of the rung: FiLM only affects the CLASSIFIER's
    # use of the embedding, never the presence/severity readout, and A7's
    # checkpoints carry a `film_norm` the plain Detector would reject.
    model = Detector(dim_feat=ck["dim_feat"], use_recon=False, use_film=False)
    model.load_state_dict(ck["state_dict"], strict=False)
    model.eval()

    P, S = [], []
    with torch.no_grad():
        for i in range(0, n_img, chunk):
            f = torch.from_numpy(np.asarray(feats[i:i + chunk], dtype=np.float32))
            d = model.degradation(f.reshape(-1, dim))
            P.append(torch.sigmoid(d["presence"]).numpy().reshape(-1, n_view, len(FAMILIES)))
            S.append(d["severity"].numpy().reshape(-1, n_view, len(FAMILIES)))
    P, S = np.concatenate(P), np.concatenate(S)

    out = {"bank": bank, "checkpoint": ckpt, "backbone": ck["backbone"],
           "n_views_scored": int(n_img * n_view), "presence": {}, "severity": {},
           "per_condition": {}}
    for k, fam in enumerate(FAMILIES):
        y, p = tgt_p[:, :, k].ravel().astype(int), P[:, :, k].ravel()
        out["presence"][fam] = {"base_rate": float(y.mean()),
                                "roc_auc": float(roc_auc_score(y, p)),
                                "ap": float(average_precision_score(y, p))}
        m = tgt_p[:, :, k].ravel() > 0.5
        if m.sum() < 50:
            out["severity"][fam] = {"n_present": int(m.sum()), "note": "too few present"}
            continue
        t, q = tgt_s[:, :, k].ravel()[m], S[:, :, k].ravel()[m]
        out["severity"][fam] = {
            "n_present": int(m.sum()), "mae": float(np.abs(t - q).mean()),
            "mae_predicting_the_mean": float(np.abs(t - t.mean()).mean()),
            "pearson_r": float(np.corrcoef(t, q)[0, 1]) if t.std() > 0 else None}
    for j, cond in enumerate(conds):
        out["per_condition"][cond] = {
            fam: {"predicted": float(P[:, j, k].mean()),
                  "truly_present": bool(tgt_p[:, j, k].mean() > 0.5)}
            for k, fam in enumerate(FAMILIES)}
    return out


def render(r: dict) -> None:
    print(f"{r['checkpoint']}  backbone={r['backbone']}  "
          f"{r['n_views_scored']:,} views\n")
    print("PRESENCE                       base rate   ROC AUC       AP")
    for fam, d in r["presence"].items():
        print(f"  {fam:26s} {d['base_rate']:9.3f} {d['roc_auc']:9.4f} "
              f"{d['ap']:8.4f}")
    print("\nSEVERITY (masked to present)   n_present       MAE  mean-baseline")
    for fam, d in r["severity"].items():
        if "mae" not in d:
            print(f"  {fam:26s} {d['n_present']:9d}  {d['note']}"); continue
        flag = "  <-- worse than a constant" if d["mae"] > d["mae_predicting_the_mean"] else ""
        print(f"  {fam:26s} {d['n_present']:9d} {d['mae']:9.4f} "
              f"{d['mae_predicting_the_mean']:14.4f}{flag}")
    print("\nPER CONDITION  mean predicted presence; [] marks truly present")
    print(f"  {'condition':18s} " + " ".join(f"{f:>7s}" for f in FAMILIES))
    for cond, fams in r["per_condition"].items():
        cells = [f"[{d['predicted']:.2f}]" if d["truly_present"]
                 else f" {d['predicted']:.2f} " for d in fams.values()]
        print(f"  {cond:18s} " + " ".join(f"{c:>7s}" for c in cells))


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", default="data/banks/eval_dinov3l",
                    help="EVAL bank: it carries presence/severity per condition")
    ap.add_argument("--checkpoint", default="outputs/rungs/a2/checkpoint.pt",
                    help="any rung with use_degradation: true (A2, A3, A4, A7)")
    ap.add_argument("--out", help="write the full report as JSON")
    a = ap.parse_args(argv)
    r = score(a.bank, a.checkpoint)
    render(r)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=2)
        print(f"\nwrote {a.out}")
    return r


if __name__ == "__main__":
    main()
