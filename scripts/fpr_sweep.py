#!/usr/bin/env python3
"""Is OV7's 0.3998 a CALIBRATION failure or a low-FPR ROC-SHAPE failure?

`heldout_robust_tpr` slices per condition and calls `tpr_at_fpr`, which builds
the ROC on that condition's OWN rows and takes the lowest threshold with
FPR <= 1%. The threshold is therefore already re-derived per condition, per
population, with full knowledge of the evaluation set's authentic scores.

That makes every strictly monotone rescaling -- temperature, Platt, quantile
normalisation against the authentic class -- an exact no-op on this metric.
This script proves that on the real scores and then reports the curve.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from aigcdet.features.bank import FeatureBank
from aigcdet.train.train_head import load_detector
from aigcdet.eval.grid import score_grid
from aigcdet.eval.metrics import tpr_at_fpr, roc_auc

BANK = "data/banks/eval_ov7_transfer_crop_dinov2regl"
CKPT = "outputs/rungs_local_all/a3/checkpoint.pt"
FPRS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]

bank = FeatureBank(BANK)
model, _ = load_detector(CKPT, device="cpu")
df = score_grid(model, bank, device="cpu")

splits = np.asarray(bank.meta["split"]).astype(str)
row_split = splits[df["image_idx"].to_numpy()]
label = df["label"].to_numpy()
auth = (row_split == "val_internal") & (label == 0)
gen = (row_split == "heldout_generator") & (label == 1)
sub = df[auth | gen]
conds = [c for c in dict.fromkeys(sub["condition"].tolist()) if c != "clean"]

n_a = int(auth.sum() / len(set(df["condition"])))
n_g = int(gen.sum() / len(set(df["condition"])))
print(f"population per condition: {n_a} authentic, {n_g} generated")
print(f"1% of {n_a} authentic = {n_a*0.01:.1f} false positives allowed "
      f"-> the threshold is set by the {int(n_a*0.01)+1}th-highest authentic score\n")

rows = []
for c in ["clean"] + conds:
    g = sub[sub["condition"] == c] if c != "clean" else df[(auth | gen) & (df["condition"] == "clean")]
    y, s = g["label"].to_numpy(), g["score"].to_numpy()
    if len(np.unique(y)) != 2:
        continue
    rows.append([c, roc_auc(y, s)] + [tpr_at_fpr(y, s, f) for f in FPRS])

t = pd.DataFrame(rows, columns=["condition", "auc"] + [f"tpr@{f:.1%}" for f in FPRS])
pd.set_option("display.width", 200, "display.max_columns", 20)
print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

deg = t[t.condition != "clean"]
print(f"\nmean over {len(deg)} degraded conditions:")
for col in t.columns[1:]:
    print(f"  {col:>12} = {deg[col].mean():.4f}")

# --- the no-op proof -------------------------------------------------------
print("\n--- monotone rescaling, per condition, on the metric itself ---")
base = deg["tpr@1.0%"].mean()
for name, fn in [
    ("temperature T=2.5", lambda s: s / 2.5),
    ("temperature T=0.3", lambda s: s / 0.3),
    ("affine 7s - 3",     lambda s: 7 * s - 3),
    ("rank / quantile",   lambda s: pd.Series(s).rank().to_numpy()),
    ("sigmoid",           lambda s: 1 / (1 + np.exp(-s))),
]:
    vals = []
    for c in conds:
        g = sub[sub["condition"] == c]
        vals.append(tpr_at_fpr(g["label"].to_numpy(), fn(g["score"].to_numpy()), 0.01))
    print(f"  {name:<20} {np.mean(vals):.4f}   delta {np.mean(vals)-base:+.6f}")
print(f"  {'baseline':<20} {base:.4f}")
