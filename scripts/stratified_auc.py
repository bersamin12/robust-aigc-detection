"""Resolution-stratified AUC: the confound-free reading of a trained rung.

Overall AUC cannot distinguish "detects generation" from "reads resolution",
because resolution predicts the label across a third of the pool. WITHIN a
resolution stratum it can: every image in the stratum has the same short side,
so short side explains nothing there and whatever separation remains is the
model's own. A rung whose overall AUC collapses inside its strata was reading
the ruler.

Reported alongside the dimensions-only CONTROL for the same rows, which is the
number the stratified figures have to beat to mean anything.

`--stratify-by source` answers a different question with the same machinery,
and it is the control the `coco_crop` stream exists under. That corpus trains
on COCO train2017 photographs while the organisers' scored benchmark's real
half IS COCO val2017 -- one photographic distribution, which is why
`data/wildfake.py:_COCO_FORBIDDEN` barred COCO from training in the first
place. A model that has memorised "COCO-like means authentic" scores
brilliantly on that benchmark and has learned nothing about generation.

Per-source AUC cannot see it: `coco_train2017` contributes only authentic
rows, so there is no AUC inside it at all. What does see it is the FALSE
POSITIVE RATE per authentic source at one global threshold. A model reading
generation artefacts has roughly the same rate on all three authentic sources;
a model reading "is this a COCO photograph" has a far lower rate on COCO than
on LAION and SID_Set. That gap is the number to publish beside any headline
from this stream.
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


#: Operating point for the per-source breakdown. 1% FPR because that is the
#: rate spec 6.4's selection rule (`heldout_robust_tpr_at_1pct`) is defined
#: at, so the per-source view is read at the same place as the headline.
SOURCE_FPR = 0.01


def _report_by_source(meta, y, s, idx) -> dict:
    """False positive rate per AUTHENTIC source at one global threshold.

    The threshold is set on ALL authentic rows together, then applied
    unchanged to each source. Setting it per source would normalise away
    exactly the difference being measured.
    """
    source = meta["source"].to_numpy()[idx]
    generator = meta["generator"].to_numpy()[idx]
    real = y == 0
    if not real.any():
        raise SystemExit("no authentic rows in this split to set a threshold on")
    # The score above which SOURCE_FPR of authentic rows fall.
    thr = float(np.quantile(s[real], 1.0 - SOURCE_FPR))
    print(f"\nthreshold at {SOURCE_FPR:.0%} FPR over all {int(real.sum())} "
          f"authentic rows: logit {thr:.4f}")

    rows = []
    for src in sorted(set(source[real])):
        m = real & (source == src)
        rows.append({"authentic source": src, "n": int(m.sum()),
                     "FPR": float((s[m] > thr).mean()),
                     "mean logit": float(s[m].mean())})
    df = pd.DataFrame(rows).sort_values("n", ascending=False)
    print("\nper authentic source, at that one threshold:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    spread = float(df["FPR"].max() - df["FPR"].min()) if len(df) > 1 else 0.0
    print(f"\nFPR spread across authentic sources: {spread:.4f}")
    if len(df) > 1:
        print("  A model reading generation artefacts has roughly the same "
              "rate on every\n  authentic source. A large spread means the "
              "headline is partly a source\n  classifier -- read it against "
              "the source whose rate is WORST, not the mean.")

    gen = []
    for g in sorted(set(generator[~real])):
        m = (~real) & (generator == g)
        gen.append({"generator": g, "n": int(m.sum()),
                    "TPR": float((s[m] > thr).mean())})
    gdf = pd.DataFrame(gen).sort_values("TPR")
    print("\nper generator, same threshold (worst first):")
    print(gdf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return {"threshold": thr, "fpr_by_source":
            dict(zip(df["authentic source"], df["FPR"])),
            "fpr_spread": spread,
            "tpr_by_generator": dict(zip(gdf["generator"], gdf["TPR"]))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="val_internal")
    ap.add_argument("--stratify-by", choices=("resolution", "source"),
                    default="resolution",
                    help="'resolution' (default) is the confound-free reading "
                         "of a rung: within a short-side stratum, resolution "
                         "explains nothing. 'source' is the memorisation "
                         "control for a corpus whose real class overlaps the "
                         "benchmark's -- see the module docstring.")
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

    if a.stratify_by == "source":
        out = _report_by_source(bank.meta, y, s, idx)
        out["overall"] = float(roc_auc_score(y, s))
        out["control"] = float(max(ctrl, 1 - ctrl))
        return out

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
