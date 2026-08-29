"""Remove a confound from the frozen backbone's feature space, then re-probe.

The backbone is frozen, so its features are cached and a confound's direction
inside them can be found and projected out WITHOUT any re-extraction. This is
the cheapest available answer to "how much of the score is the shortcut?":
train a linear probe on the features, remove the directions that predict the
confound, retrain the same probe, and read the difference.

Method is iterative nullspace projection (INLP): fit a linear map from
features to the confound, project the features onto the nullspace of its
weight vector, repeat. Each round removes the single most confound-predictive
direction that remains, so the confound's own predictability is reported
alongside the label's -- a run where the confound stays predictable has not
removed it and its label number means nothing.

The probe is a linear classifier on the clean view, i.e. rung A0's
architecture, so the before/after pair is a like-for-like comparison rather
than a comparison across model classes.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score

from aigcdet.features.bank import FeatureBank
from aigcdet.features.proxies import PROXY_NAMES


def inlp(X_tr, X_va, c_tr, n_dir, seed=20260827):
    """Project both matrices onto the nullspace of `n_dir` successive linear
    predictors of `c_tr`. Directions are fitted on TRAIN only -- fitting on
    validation would remove the confound using labels the probe never saw and
    flatter the result."""
    Xt, Xv = X_tr.copy(), X_va.copy()
    for _ in range(n_dir):
        w = Ridge(alpha=1.0, random_state=seed).fit(Xt, c_tr).coef_.astype(np.float32)
        n = np.linalg.norm(w)
        if n < 1e-8:
            break
        w /= n
        Xt -= np.outer(Xt @ w, w)
        Xv -= np.outer(Xv @ w, w)
    return Xt, Xv


def probe_auc(X_tr, y_tr, X_va, y_va, seed=20260827):
    clf = LogisticRegression(max_iter=300, C=1.0, random_state=seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    return roc_auc_score(y_va, clf.decision_function(X_va))


def confound_r2(X_tr, c_tr, X_va, c_va, seed=20260827):
    m = Ridge(alpha=1.0, random_state=seed).fit(X_tr, c_tr)
    return r2_score(c_va, m.predict(X_va))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--confound", default="short_side",
                    choices=("short_side", *PROXY_NAMES))
    ap.add_argument("--dirs", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--seed", type=int, default=20260827)
    a = ap.parse_args(argv)

    bank = FeatureBank(a.bank)
    man = pd.read_parquet(a.manifest)
    man = man[man["split"].isin(["train", "val_internal"])].reset_index(drop=True)
    if not (man["rel_path"].values == bank.meta["rel_path"].values).all():
        raise SystemExit("bank and manifest are not positionally aligned")

    split = bank.meta["split"].to_numpy()
    tr, va = np.flatnonzero(split == "train"), np.flatnonzero(split == "val_internal")
    y = bank.meta["label"].to_numpy()

    if a.confound == "short_side":
        # log, because the confound is a ratio scale: 200 -> 256 is the same
        # perceptual step as 512 -> 655, and a linear fit on raw pixels would
        # chase the 1746px tail instead of the 200/512 split that matters.
        c = np.log(np.minimum(man["width"].to_numpy(), man["height"].to_numpy()).astype(np.float64))
    else:
        c = np.asarray(bank.proxies[:, 0, PROXY_NAMES.index(a.confound)], dtype=np.float64)

    X_tr = np.asarray(bank.feats[tr, 0, :], dtype=np.float32)
    X_va = np.asarray(bank.feats[va, 0, :], dtype=np.float32)
    y_tr, y_va, c_tr, c_va = y[tr], y[va], c[tr], c[va]
    print(f"bank {a.bank}  confound {a.confound}  "
          f"train {len(tr)}  val {len(va)}  dim {X_tr.shape[1]}")

    base_auc = probe_auc(X_tr, y_tr, X_va, y_va, a.seed)
    base_r2 = confound_r2(X_tr, c_tr, X_va, c_va, a.seed)
    print(f"\n{'directions removed':>19}{'label AUC':>11}{'confound R2':>13}")
    print(f"{0:>19}{base_auc:>11.4f}{base_r2:>13.4f}")
    for k in a.dirs:
        Xt, Xv = inlp(X_tr, X_va, c_tr, k, a.seed)
        print(f"{k:>19}{probe_auc(Xt, y_tr, Xv, y_va, a.seed):>11.4f}"
              f"{confound_r2(Xt, c_tr, Xv, c_va, a.seed):>13.4f}")
    print("\nA large AUC drop means the probe was leaning on the confound; a small\n"
          "one means the signal lives elsewhere. Read it against confound R2 --\n"
          "an AUC that holds while R2 stays high has removed nothing.")


if __name__ == "__main__":
    main()
