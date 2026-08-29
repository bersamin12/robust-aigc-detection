"""Resolution-stratified AUC: the confound-free reading of a trained rung.

Overall AUC cannot distinguish "detects generation" from "reads resolution",
because resolution predicts the label across a third of the pool. WITHIN a
resolution stratum it can: every image in the stratum has the same short side,
so short side explains nothing there and whatever separation remains is the
model's own. A rung whose overall AUC collapses inside its strata was reading
the ruler.

Reported alongside the dimensions-only CONTROL for the same rows, which is the
number the stratified figures have to beat to mean anything.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from aigcdet.features.bank import FeatureBank
from aigcdet.train.train_head import load_detector

#: Strata smaller than this cannot carry a believable AUC, and a stratum with
#: one class present has no AUC at all. Both are reported as coverage rather
#: than silently dropped: they are exactly the rows where resolution alone
#: decides the answer.
MIN_STRATUM = 50


def score_bank(model, bank, idx, device="cpu", batch=4096):
    """Clean-view (view 0) logits for `idx`, in order."""
    out = []
    with torch.inference_mode():
        for i in range(0, len(idx), batch):
            f = np.asarray(bank.feats[idx[i:i + batch], 0, :], dtype=np.float32)
            out.append(model(torch.from_numpy(f).to(device))["logit"].cpu().numpy())
    return np.concatenate(out).ravel()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="val_internal")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)

    bank = FeatureBank(a.bank)
    man = pd.read_parquet(a.manifest)
    man = man[man["split"].isin(["train", "val_internal"])].reset_index(drop=True)
    if not (man["rel_path"].values == bank.meta["rel_path"].values).all():
        raise SystemExit("bank and manifest are not positionally aligned")

    idx = np.flatnonzero(bank.meta["split"].to_numpy() == a.split)
    y = bank.meta["label"].to_numpy()[idx]
    short = np.minimum(man["width"].to_numpy(), man["height"].to_numpy())[idx]

    model, ck = load_detector(a.checkpoint, device=a.device)
    s = score_bank(model, bank, idx, a.device)

    print(f"rung {ck['config']['name']}  backbone {ck['backbone']}  "
          f"split {a.split}  n={len(idx)}")
    print(f"overall AUC                 {roc_auc_score(y, s):.4f}")
    ctrl = roc_auc_score(y, short.astype(float))
    print(f"dimensions-only CONTROL     {max(ctrl, 1 - ctrl):.4f}   "
          "(short side alone; the number below must beat this)")

    rows, used, single, small = [], 0, 0, 0
    for v in np.unique(short):
        m = short == v
        n = int(m.sum())
        if len(np.unique(y[m])) < 2:
            single += n
            continue
        if n < MIN_STRATUM:
            small += n
            continue
        rows.append({"short_side": int(v), "n": n,
                     "authentic": int((y[m] == 0).sum()),
                     "generated": int((y[m] == 1).sum()),
                     "auc": roc_auc_score(y[m], s[m])})
        used += n
    df = pd.DataFrame(rows).sort_values("n", ascending=False)
    print("\nper-stratum (short side held constant, so it explains nothing):")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    w = df["n"] / df["n"].sum()
    print(f"\nweighted mean stratified AUC {float((df['auc'] * w).sum()):.4f}"
          f"   over {used} rows ({used/len(idx):.1%})")
    print(f"excluded: {single} rows in single-class strata "
          f"({single/len(idx):.1%}), {small} in strata under {MIN_STRATUM}")
    return {"overall": float(roc_auc_score(y, s)),
            "stratified": float((df["auc"] * w).sum()),
            "control": float(max(ctrl, 1 - ctrl))}


if __name__ == "__main__":
    main()
