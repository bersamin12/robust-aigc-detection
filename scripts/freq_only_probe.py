"""Can the frequency block reach aF's score WITHOUT a backbone? (off-ladder)

    python scripts/freq_only_probe.py \
        --bank data/banks/probe_crop_dinov2regl \
        --eval-bank data/banks/eval_probe_crop_dinov2regl \
        --out docs/freq_only_probe.json

WHY THIS GATES aF. `npr_feature` is 4 numbers. `proxy_vector` is 3 -- JPEG
quality, Laplacian variance, noise floor -- and Laplacian variance ALONE
(sharpness) reaches AUC 0.672 on this corpus. A 4-d descriptor of local
neighbour differences could simply relearn sharpness, in which case aF is not
measuring a transposed-convolution fingerprint at all; it is measuring the same
low-level confound the corpus already leaks, and its rung number would be an
artefact dressed as a finding.

WHAT IS MEASURED. The SAME §6.4 metric aF is scored by -- mean TPR @ 1% FPR
over the 19 degraded conditions, val_internal authentic against
heldout_generator generated -- computed from each feature set ALONE, with no
backbone anywhere. The comparison that matters is against aF's own number and
against a3's:

  * frequency-only near chance  -> aF's gain is complementary to the backbone,
    which is the result the rung claims.
  * frequency-only near aF      -> the backbone is contributing little and the
    rung is a low-level shortcut. A high number here is a RED FLAG, not a
    second result.

`proxies` and `laplacian_var` are carried alongside as the reference the claim
is actually about: if frequency-only lands where sharpness-only lands, they are
the same measurement wearing different names.

OFF-LADDER by construction: these arms differ from a rung in having no backbone
at all, not in one flag, so they can never enter selection.json.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")

from aigcdet.eval.errors import SELECTION_TARGET_FPR, heldout_robust_tpr  # noqa: E402
from aigcdet.features.bank import FeatureBank  # noqa: E402
from aigcdet.features.proxies import PROXY_NAMES  # noqa: E402

#: (name, how to pull the (rows, views, d) block off a bank). One entry per
#: feature set the control is run for.
ARMS = {
    "freq": lambda b: np.asarray(b.freq),
    "proxies": lambda b: np.asarray(b.proxies),
    "laplacian_var": lambda b: np.asarray(b.proxies)[:, :, PROXY_NAMES.index("laplacian_var")][..., None],
    "freq_plus_proxies": lambda b: np.concatenate(
        [np.asarray(b.freq), np.asarray(b.proxies)], axis=-1),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--eval-bank", required=True)
    ap.add_argument("--out", default="docs/freq_only_probe.json")
    ap.add_argument("--seed", type=int, default=20260827)
    a = ap.parse_args(argv)

    bank, ebank = FeatureBank.open(a.bank), FeatureBank.open(a.eval_bank)
    if bank.freq is None or ebank.freq is None:
        raise SystemExit("both banks need freq.npy; run the frequency replay first")

    # Train on the TRAIN split only -- val_internal is half the metric's
    # population and must not be fitted on.
    tr = np.flatnonzero((bank.meta["split"] == "train").to_numpy())
    y_tr = bank.meta["label"].to_numpy()[tr]
    names = ebank.config["conditions"]
    splits = ebank.meta.set_index("image_idx")["split"].astype(str)
    meta = ebank.meta

    rows = {}
    for arm, pull in ARMS.items():
        X = pull(bank)[tr]                       # (rows, views, d)
        v, d = X.shape[1], X.shape[2]
        # Every view is a training row: the rung's head sees them all too.
        Xf = X.reshape(-1, d).astype(np.float64)
        yf = np.repeat(y_tr, v)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=4000, random_state=a.seed))
        clf.fit(Xf, yf)

        E = pull(ebank)                          # (n, conditions, d)
        frames = []
        for j, cond in enumerate(names):
            s = clf.decision_function(E[:, j, :].astype(np.float64))
            frames.append(pd.DataFrame({
                "condition": cond,
                "image_idx": meta["image_idx"].to_numpy(),
                "label": meta["label"].to_numpy(),
                "generator": meta["generator"].to_numpy(),
                "source": meta["source"].to_numpy(),
                "score": s,
            }))
        scores = pd.concat(frames, ignore_index=True)
        tpr = float(heldout_robust_tpr(scores, splits, SELECTION_TARGET_FPR))

        # The same population as an AUC, because the concern this control
        # answers was stated in AUC terms ("sharpness alone reaches 0.672").
        # The two can disagree sharply and both be right: a weak-but-real
        # signal lifts AUC well above 0.5 while contributing almost nothing at
        # a 1% FPR operating point, which is where selection actually reads.
        sp = scores["image_idx"].map(splits)
        keep = (((sp == "val_internal") & (scores["label"] == 0))
                | ((sp == "heldout_generator") & (scores["label"] == 1)))
        pop = scores[keep & (scores["condition"] != "clean")]
        aucs = [roc_auc_score(g["label"], g["score"])
                for _, g in pop.groupby("condition") if g["label"].nunique() == 2]
        auc = float(np.mean(aucs)) if aucs else float("nan")

        rows[arm] = {"dim": int(d), "heldout_robust_tpr_at_1pct": tpr,
                     "heldout_robust_auc": auc}
        print(f"{arm:20s} d={d:2d}  tpr@1%fpr={tpr:.4f}  auc={auc:.4f}", flush=True)

    out = {
        "probe": "freq_only",
        "off_ladder": True,
        "not_eligible_reason": "these arms have no backbone at all, so they "
                               "differ from a rung in more than one flag and "
                               "can never enter selection.json",
        "metric": "heldout_robust_tpr_at_1pct",
        "target_fpr": SELECTION_TARGET_FPR,
        "population": "val_internal authentic vs heldout_generator generated, "
                      "19 degraded conditions, clean excluded",
        "trained_on": "split == train only",
        "bank": a.bank, "eval_bank": a.eval_bank,
        "arms": rows,
        "reading": "compare against aF's own rung number. Near chance means "
                   "aF's gain is complementary to the backbone; near aF means "
                   "the rung is a low-level shortcut and the number is an "
                   "artefact, not a finding. TPR@1%FPR is the gate because it "
                   "is the selection metric; the AUC column is carried because "
                   "the concern was raised in AUC terms and the two can "
                   "disagree without either being wrong.",
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
