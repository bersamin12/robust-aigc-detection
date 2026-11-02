"""Score-level fusion of two separately trained branches, NTIRE protocol.

`eval.fusion.fuse_scores` fuses cached-bank score frames per eval condition
and fits its weight on `val_internal` against the ladder's TPR@1%FPR. The
NTIRE stream scores from pixels, has no condition axis, and is judged by the
challenge's own pair of numbers -- Robust ROC AUC over the distorted half of
a set and Clean ROC AUC over the rest -- so the fit lives here, on the
per-row parquets `scripts/score_plan_splits.py` writes.

Three rules, all of which have been violated before in this project:

* **Fit on the fit rows, apply everywhere.** The z-score parameters and the
  weight come from the selection sets (`ntire_val` + `ntire_val_hard`) and
  are then applied unchanged to `ntire_test`. A parameter refitted on the
  test rows is an oracle bound, not a result.
* **Z-score globally, not per `is_distorted`.** Whether a test image is
  distorted is metadata the detector does not get at inference time, so a
  normalisation that used it could not be served.
* **Fuse logits, not probabilities.** A sigmoid saturates: two branches that
  both emit p = 1.000 carry different evidence in logit space and none in
  probability space. `score_plan_splits` writes both; this module reads the
  logit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from aigcdet.eval.metrics import roc_auc, tpr_at_fpr

#: The same 21-point grid `eval.fusion` sweeps, so a weight quoted here and one
#: quoted there are read on one scale.
WEIGHT_GRID: np.ndarray = np.linspace(0.0, 1.0, 21)

JOIN_KEY = "path"
#: Columns that must agree row for row once the two frames are joined; a
#: disagreement means the two scorers did not see the same manifest.
MUST_AGREE = ("label", "is_distorted")


@dataclass(frozen=True)
class ZScore:
    mu: float
    sigma: float

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mu) / self.sigma


def zscore_fit(x: np.ndarray) -> ZScore:
    x = np.asarray(x, dtype=np.float64)
    sigma = float(x.std())
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("cannot z-score a constant or non-finite score column")
    return ZScore(float(x.mean()), sigma)


def join_branches(da: pd.DataFrame, db: pd.DataFrame,
                  score_col: str = "logit") -> pd.DataFrame:
    """One frame with `score_a`/`score_b`, joined 1:1 on `path`.

    Refuses a partial join and refuses labels or `is_distorted` that disagree:
    either would make the fused number a statement about different rows than
    the solo numbers.
    """
    for name, d in (("a", da), ("b", db)):
        missing = [c for c in (JOIN_KEY, score_col, "label") if c not in d.columns]
        if missing:
            raise ValueError(f"branch {name} frame lacks {missing}")
    keep_b = [JOIN_KEY, score_col] + [c for c in MUST_AGREE if c in db.columns]
    j = da.merge(db[keep_b], on=JOIN_KEY, suffixes=("_a", "_b"), validate="1:1")
    if len(j) != len(da) or len(j) != len(db):
        raise ValueError(f"join is not 1:1 over both frames "
                         f"({len(da)} vs {len(db)} vs {len(j)} joined)")
    for c in MUST_AGREE:
        if f"{c}_a" in j.columns:
            if not (j[f"{c}_a"].to_numpy() == j[f"{c}_b"].to_numpy()).all():
                raise ValueError(f"column {c!r} disagrees between the branches")
            j[c] = j.pop(f"{c}_a")
            j.pop(f"{c}_b")
    j = j.rename(columns={f"{score_col}_a": "score_a", f"{score_col}_b": "score_b"})
    return j


def branch_metrics(y: np.ndarray, s: np.ndarray,
                   is_distorted: np.ndarray | None = None) -> dict:
    """The leaderboard's pair, plus the project's own operating point.

    `robust_auc` is over rows with `is_distorted == 1`, `clean_auc` over the
    rest -- exactly the challenge's definition. Either is None when its half
    lacks a class, never silently computed over the other half.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    out = {"n": int(len(y)),
           "auc": roc_auc(y, s) if len(set(y.tolist())) == 2 else None,
           "tpr_at_1pct_fpr": tpr_at_fpr(y, s) if len(set(y.tolist())) == 2 else None}
    if is_distorted is None:
        out.update(robust_auc=None, clean_auc=None, n_robust=0, n_clean=0)
        return out
    d = np.asarray(is_distorted).astype(int)
    for key, mask in (("robust", d == 1), ("clean", d == 0)):
        out[f"n_{key}"] = int(mask.sum())
        both = mask.any() and len(set(y[mask].tolist())) == 2
        out[f"{key}_auc"] = roc_auc(y[mask], s[mask]) if both else None
    return out


def fusion_objective(m: dict) -> float:
    """What the weight is chosen to maximise: the mean of the two leaderboard
    numbers when both exist, else whichever exists, else pooled AUC."""
    vals = [v for v in (m.get("robust_auc"), m.get("clean_auc")) if v is not None]
    if vals:
        return float(np.mean(vals))
    if m.get("auc") is None:
        raise ValueError("no AUC is defined on the fit rows")
    return float(m["auc"])


def fused(za: np.ndarray, zb: np.ndarray, w: float) -> np.ndarray:
    return w * np.asarray(za, dtype=np.float64) + (1.0 - w) * np.asarray(zb, dtype=np.float64)


def fit_weight(za: np.ndarray, zb: np.ndarray, y: np.ndarray,
               is_distorted: np.ndarray | None,
               grid: np.ndarray = WEIGHT_GRID) -> tuple[float, dict[float, float]]:
    """Sweep `w` on z-scored branch scores; ties go to 0.5.

    The tie rule is `eval.fusion.fit_fusion_weight`'s: when the sweep is flat,
    equal weighting is the answer that assumes least, and it is also what the
    Wave-2 measurement found the fitted weight bought almost nothing over.
    """
    sweep = {}
    for w in grid:
        sweep[round(float(w), 4)] = fusion_objective(
            branch_metrics(y, fused(za, zb, float(w)), is_distorted))
    best = max(sweep.values())
    winners = [w for w, v in sweep.items() if v == best]
    w = 0.5 if 0.5 in winners else min(winners, key=lambda x: abs(x - 0.5))
    return float(w), sweep


@dataclass(frozen=True)
class Fusion:
    """Everything needed to reproduce the fused score on new rows."""
    weight: float
    z_a: ZScore
    z_b: ZScore
    fit_population: str

    def apply(self, score_a: np.ndarray, score_b: np.ndarray) -> np.ndarray:
        return fused(self.z_a.apply(score_a), self.z_b.apply(score_b), self.weight)

    def as_record(self) -> dict:
        return {"weight": self.weight, "z_a": asdict(self.z_a),
                "z_b": asdict(self.z_b), "fit_population": self.fit_population}

    @classmethod
    def from_record(cls, rec: dict) -> "Fusion":
        return cls(float(rec["weight"]), ZScore(**rec["z_a"]), ZScore(**rec["z_b"]),
                   str(rec["fit_population"]))


def fit_fusion(fit: pd.DataFrame, population: str,
               grid: np.ndarray = WEIGHT_GRID) -> tuple[Fusion, dict[float, float]]:
    """Fit z-scores and the weight on `fit` (a `join_branches` frame)."""
    za_p, zb_p = zscore_fit(fit["score_a"]), zscore_fit(fit["score_b"])
    dist = fit["is_distorted"].to_numpy() if "is_distorted" in fit.columns else None
    w, sweep = fit_weight(za_p.apply(fit["score_a"]), zb_p.apply(fit["score_b"]),
                          fit["label"].to_numpy(), dist, grid)
    return Fusion(w, za_p, zb_p, population), sweep


def evaluate(j: pd.DataFrame, fusion: Fusion) -> dict:
    """Solo, equal-weight and fitted-weight metrics on one joined frame."""
    y = j["label"].to_numpy()
    dist = j["is_distorted"].to_numpy() if "is_distorted" in j.columns else None
    za, zb = fusion.z_a.apply(j["score_a"]), fusion.z_b.apply(j["score_b"])
    return {"a": branch_metrics(y, j["score_a"].to_numpy(), dist),
            "b": branch_metrics(y, j["score_b"].to_numpy(), dist),
            "fused_equal": branch_metrics(y, fused(za, zb, 0.5), dist),
            "fused": branch_metrics(y, fused(za, zb, fusion.weight), dist)}
