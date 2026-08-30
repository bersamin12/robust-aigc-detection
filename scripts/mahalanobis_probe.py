"""Generator novelty from feature density alone -- no second head (off-ladder).

    python scripts/mahalanobis_probe.py \
        --bank data/banks/dinov3l --eval-bank data/banks/eval_dinov3l \
        --out docs/mahalanobis_probe.json

WHY THIS EXISTS. `scripts/family_experts.py` asked whether two family experts
DISAGREE on a generator neither trained on. They do not -- the novelty AUC came
out at 0.3018, below chance and pointing the wrong way, because in a two-expert
partition the images exactly one expert is unfamiliar with are the `val_internal`
fakes, not the held-out ones. The partition, not the novelty, was doing the
talking. This asks the same question without any experts: fit Gaussians to the
TRAINING features and measure how far an image sits from all of them.

Three scores, all on the frozen bank, none of them trained:

  MD    Mahalanobis distance to the nearest training Gaussian, shared
        covariance (Lee et al. 2018). Two groupings -- by class (real/fake) and
        by generator family -- because "unseen generator" is a statement about
        families, and a fake-class Gaussian pooled over 13 families is a wide,
        uninformative blob.
  RMD   Relative Mahalanobis (Ren et al. 2021): MD minus the distance to a
        single background Gaussian fitted over all training rows. On image
        features the raw distance is dominated by generic content -- how
        unusual the PICTURE is -- and subtracting the background removes the
        part of the distance every model agrees on.

WHAT IT IS EVALUATED ON.

  novelty   Among FAKES ONLY: heldout_generator (positive) against
            val_internal (negative). Both classes are generated, so nothing
            here can be won by detecting generation; the only signal is family.
            Directly comparable to the 0.3018 the disagreement probe scored.
  real/fake `heldout_robust_tpr` via distance to the REAL Gaussian, so the
            §6.4 number is comparable to a3's 0.9012. Not expected to compete;
            it is here so an unsupervised floor exists next to the trained one.

TWO THINGS THAT WOULD MAKE THE NUMBER A LIE IF THEY WERE NOT HANDLED.

*The fit uses `split == "train"` rows only.* `val_internal` is the negative
class of the novelty task; fitting on it would make the negatives close to the
Gaussians by construction and the AUC an artefact of the fit.

*Every AUC is computed WITHIN one condition.* A degraded image is far from a
clean-fitted Gaussian for reasons that have nothing to do with its generator,
and a mean over a pooled frame would mostly rank degradations. Within a
condition both classes carry the same damage, so the shift is common-mode. The
clean condition is reported separately from the degraded mean for the same
reason -- if the two disagree, the score is reading damage, not lineage.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from aigcdet.eval.errors import SELECTION_METRIC, heldout_robust_tpr
from aigcdet.eval.metrics import roc_auc
from aigcdet.features.bank import FeatureBank

#: Ridge added to the pooled covariance before inversion. The features are
#: 1024-dim float16 off a frozen backbone and the pooled scatter is not
#: guaranteed full rank; without it the Cholesky can fail on a bank that is
#: perfectly fine, and the failure looks like a data problem.
RIDGE: float = 1e-3


def _fit_gaussians(x: np.ndarray, groups: np.ndarray) -> tuple[dict, np.ndarray]:
    """Per-group means and ONE shared, within-group-pooled precision.

    Shared covariance is the point of the Lee et al. estimator: per-group
    covariances over 1024 dims from ~3k images each are badly conditioned, and
    a per-group covariance also lets a group win by being diffuse rather than
    by being close.
    """
    means, centred = {}, np.empty_like(x)
    for g in np.unique(groups):
        m = groups == g
        mu = x[m].mean(axis=0)
        means[str(g)] = mu
        centred[m] = x[m] - mu
    cov = (centred.T @ centred) / len(centred)
    cov.flat[:: cov.shape[0] + 1] += RIDGE
    # Whitening transform rather than an explicit inverse: distances are then
    # squared euclidean norms in the whitened space, which is what makes the
    # 20 x 25,332 x n_groups evaluation a couple of matmuls.
    whiten = np.linalg.inv(np.linalg.cholesky(cov)).T
    return means, whiten


def _min_sq_distance(x: np.ndarray, means: dict, whiten: np.ndarray) -> np.ndarray:
    xw = x @ whiten
    best = None
    for mu in means.values():
        d = ((xw - (mu @ whiten)) ** 2).sum(axis=1)
        best = d if best is None else np.minimum(best, d)
    return best


def _sq_distance(x: np.ndarray, mu: np.ndarray, whiten: np.ndarray) -> np.ndarray:
    return (((x - mu) @ whiten) ** 2).sum(axis=1)


def novelty_auc(score: np.ndarray, meta: pd.DataFrame,
                conditions: list[str]) -> dict:
    """AUC for heldout_generator fakes against val_internal fakes, per condition.

    `score` is (n_images, n_conditions). Higher must mean "more novel"; the
    caller passes distances, for which that is already true.
    """
    split = meta["split"].to_numpy().astype(str)
    label = meta["label"].to_numpy()
    pos = (split == "heldout_generator") & (label == 1)
    neg = (split == "val_internal") & (label == 1)
    if not pos.any() or not neg.any():
        raise ValueError(
            f"the novelty population is empty: {int(pos.sum())} held-out and "
            f"{int(neg.sum())} val_internal FAKE rows. Both classes of this "
            "task are generated; a bank with fakes in only one split cannot "
            "pose the question.")
    y = np.concatenate([np.zeros(int(neg.sum())), np.ones(int(pos.sum()))])
    out = {}
    for j, cond in enumerate(conditions):
        out[cond] = float(roc_auc(y, np.concatenate([score[neg, j], score[pos, j]])))
    return out


def as_score_grid(score: np.ndarray, meta: pd.DataFrame,
                  conditions: list[str]) -> pd.DataFrame:
    """A `score_grid`-shaped frame, so the §6.4 metric can be read off it.

    heldout_robust_tpr takes exactly this shape and builds its own population;
    handing it a frame is how an unsupervised score gets a number on the same
    axis as a trained rung instead of a number of its own devising.
    """
    frames = []
    for j, cond in enumerate(conditions):
        frames.append(pd.DataFrame({
            "condition": cond,
            "image_idx": meta["image_idx"].to_numpy(),
            "label": meta["label"].to_numpy(),
            "generator": meta["generator"].to_numpy(),
            "source": meta["source"].to_numpy(),
            "score": score[:, j],
        }))
    return pd.concat(frames, ignore_index=True)


def _summarise(per_condition: dict) -> tuple[float, float]:
    """(clean, mean over degraded). Reported apart -- see the module docstring."""
    degraded = [v for k, v in per_condition.items() if k != "clean"]
    return float(per_condition.get("clean", float("nan"))), float(np.mean(degraded))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--eval-bank", required=True)
    ap.add_argument("--out", default="docs/mahalanobis_probe.json")
    ap.add_argument("--fit-view", type=int, default=0,
                    help="training view to fit on; 0 is the clean view by the "
                         "bank's own invariant")
    a = ap.parse_args()

    train = FeatureBank.open(a.bank)
    train.check_invariants()
    ev = FeatureBank.open(a.eval_bank)
    conditions = list(ev.config["conditions"])

    tm = train.meta
    fit_rows = np.where(tm["split"].to_numpy().astype(str) == "train")[0]
    x = np.asarray(train.feats[fit_rows, a.fit_view, :]).astype(np.float32)
    if np.isnan(x).any():
        raise ValueError(
            f"the training bank at {a.bank} has NaN features in view "
            f"{a.fit_view}. A backbone run in the wrong precision produces a "
            "bank of the right shape and the right size and no content; every "
            "Gaussian fitted here would be NaN and every distance with it.")
    label = tm["label"].to_numpy()[fit_rows]
    gen = tm["generator"].to_numpy()[fit_rows].astype(str)
    # Reals carry generator "" -- they are one group, not one group per source.
    family = np.where(label == 0, "__real__", gen)
    print(f"fitting on {len(x)} train rows, view {a.fit_view}: "
          f"{len(np.unique(family))} families, dim {x.shape[1]}")

    groupings = {
        "by_class": np.where(label == 0, "real", "fake"),
        "by_family": family,
    }
    fits = {name: _fit_gaussians(x, g) for name, g in groupings.items()}
    # Background: ONE Gaussian over every training row, its own covariance.
    bg_mean = x.mean(axis=0)
    bg_cov = np.cov(x, rowvar=False)
    bg_cov.flat[:: bg_cov.shape[0] + 1] += RIDGE
    bg_whiten = np.linalg.inv(np.linalg.cholesky(bg_cov)).T
    real_mean = x[label == 0].mean(axis=0)

    n = len(ev.meta)
    scores = {k: np.empty((n, len(conditions)), dtype=np.float64)
              for k in ("md_class", "md_family", "rmd_class", "rmd_family",
                        "md_real", "rmd_real")}
    for j, cond in enumerate(conditions):
        e = np.asarray(ev.feats[:, j, :]).astype(np.float32)
        bg = _sq_distance(e, bg_mean, bg_whiten)
        for key, fit in (("class", fits["by_class"]), ("family", fits["by_family"])):
            d = _min_sq_distance(e, *fit)
            scores[f"md_{key}"][:, j] = d
            scores[f"rmd_{key}"][:, j] = d - bg
        dr = _sq_distance(e, real_mean, fits["by_class"][1])
        scores["md_real"][:, j] = dr
        scores["rmd_real"][:, j] = dr - bg
        print(f"  scored condition {j + 1}/{len(conditions)}: {cond}", flush=True)

    results = {}
    print(f"\n| score | novelty AUC (clean) | novelty AUC (degraded mean) |")
    print("|---|---|---|")
    for key in ("md_class", "md_family", "rmd_class", "rmd_family"):
        per = novelty_auc(scores[key], ev.meta, conditions)
        clean, deg = _summarise(per)
        results[key] = {"novelty_auc_per_condition": per,
                        "novelty_auc_clean": clean,
                        "novelty_auc_degraded_mean": deg}
        print(f"| {key} | {clean:.4f} | {deg:.4f} |")

    splits = ev.meta["split"].to_numpy()
    print(f"\n| score | {SELECTION_METRIC} |")
    print("|---|---|")
    for key in ("md_real", "rmd_real"):
        frame = as_score_grid(scores[key], ev.meta, conditions)
        v = float(heldout_robust_tpr(frame, splits))
        results[key] = {SELECTION_METRIC: v}
        print(f"| {key} | {v:.4f} |")

    payload = {
        "probe": "mahalanobis",
        "off_ladder": True,
        "trained": False,
        "bank": a.bank, "eval_bank": a.eval_bank,
        "fit_view": a.fit_view,
        "fit_rows": int(len(x)),
        "fit_split": "train",
        "ridge": RIDGE,
        "families": sorted(set(map(str, family))),
        "manifest_sha256": ev.config.get("manifest_sha256"),
        "comparison": {
            "disagreement_novelty_auc_degraded_mean": 0.3018,
            "a3_heldout_robust_tpr_at_1pct": 0.9012,
        },
        "results": results,
    }
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
