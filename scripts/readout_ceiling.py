"""Does the trained head earn its keep, or do the features decide? (off-ladder)

    python scripts/readout_ceiling.py --out docs/readout_ceiling.json

THE QUESTION. On DINOv3, a closed-form Gaussian discriminant over the frozen
features scores 0.9005 on the §6.4 metric and the trained A3 head scores
0.9012 -- a paired tie. Two incompatible readings fit that single point:

  (a) the readout is irrelevant because these features are so good that
      anything reads them equally well; or
  (b) the readout is irrelevant because these features have a CEILING near
      0.90 that no readout can pass.

They matter because they point opposite ways. Under (a), effort belongs
anywhere but the backbone. Under (b), the feature space is the ONLY remaining
lever and fine-tuning a backbone is the justified next step.

One point cannot separate them. Three can. This runs the same comparison at
three very different levels of frozen-feature quality -- the trained ladder
spans 0.9012 (DINOv3) to 0.4882 (ConvNeXt) on the same data, same head, same
recipe -- and asks how the closed-form/trained GAP moves:

  gap stays ~0 as the backbone weakens
      -> the readout never matters, whatever the features. Performance is a
         pure function of the feature space, and (b) is the live reading.
  gap OPENS as the backbone weakens
      -> training does real work when the features are poor, and the DINOv3
         tie is a fact about DINOv3 being unusually linearly separable for
         this task, not a general claim that training is worthless. (a).

Either answer is worth having before anyone spends 40 GPU-hours fine-tuning.

WHAT IS COMPARED. `trained` is the A3 checkpoint already on disk for that
backbone -- not retrained here, so nothing about this script can change the
ladder's numbers. `closed_form` is `mahalanobis_probe`'s `rmd_real`: distance
to the training-set REAL Gaussian minus distance to a background Gaussian,
fitted on `split == "train"` clean-view rows and nothing else. It has no
gradient descent, no epochs and no hyperparameters beyond a ridge.

The verdict on each pair comes from `family_experts.bootstrap_panel`, i.e. a
PAIRED bootstrap over images: both scores are read on the same resample, since
overlapping marginal intervals are not a test of a difference.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import numpy as np

from aigcdet.eval.errors import SELECTION_METRIC
from aigcdet.eval.grid import score_grid
from aigcdet.features.bank import FeatureBank
from aigcdet.train.train_head import load_detector

_ROOT = pathlib.Path(__file__).resolve().parent

#: (name, training bank, eval bank, the A3 checkpoint already trained on it).
#: Ordered strongest-first by the trained ladder, because the whole read of
#: this table is how the gap moves DOWN the column.
BACKBONES: tuple[tuple[str, str, str, str], ...] = (
    ("dinov3l",  "data/banks/dinov3l",  "data/banks/eval_dinov3l",
     "outputs/rungs/a3/checkpoint.pt"),
    ("siglip2l", "data/banks/siglip2l", "data/banks/eval_siglip2l",
     "outputs/rungs_siglip2l/a3/checkpoint.pt"),
    ("convnextt", "data/banks/convnextt", "data/banks/eval_convnextt",
     "outputs/rungs_convnextt/a3/checkpoint.pt"),
)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", _ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _background(x: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean and whitening transform of ONE Gaussian over every training row.

    Computed as a centred Gram rather than through `np.cov`, which promotes to
    float64 and materialises a transposed copy -- 4.3 GB at ConvNeXt's 2304
    dims, for a matrix that is 42 MB once formed.
    """
    mu = x.mean(axis=0)
    c = x - mu
    cov = (c.T @ c) / len(c)
    cov.flat[:: cov.shape[0] + 1] += ridge
    return mu, np.linalg.inv(np.linalg.cholesky(cov)).T


def closed_form_scores(train_bank: FeatureBank, eval_bank: FeatureBank,
                       mp, fit_view: int = 0) -> "object":
    """`rmd_real` for one backbone: a score_grid-shaped frame, nothing trained."""
    tm = train_bank.meta
    rows = np.where(tm["split"].to_numpy().astype(str) == "train")[0]
    x = np.asarray(train_bank.feats[rows, fit_view, :]).astype(np.float32)
    if np.isnan(x).any():
        raise ValueError(
            f"the bank at {train_bank.path} has NaN features in view {fit_view}; "
            "a backbone run in the wrong precision yields the right shape and "
            "no content, and every Gaussian fitted here would be NaN")
    label = tm["label"].to_numpy()[rows]
    _, whiten = mp._fit_gaussians(x, np.where(label == 0, "real", "fake"))
    real_mean = x[label == 0].mean(axis=0)
    bg_mean, bg_whiten = _background(x, mp.RIDGE)
    del x

    conditions = list(eval_bank.config["conditions"])
    s = np.empty((len(eval_bank.meta), len(conditions)), dtype=np.float64)
    for j in range(len(conditions)):
        e = np.asarray(eval_bank.feats[:, j, :]).astype(np.float32)
        s[:, j] = (mp._sq_distance(e, real_mean, whiten)
                   - mp._sq_distance(e, bg_mean, bg_whiten))
    return mp.as_score_grid(s, eval_bank.meta, conditions)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/readout_ceiling.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--boot-n", type=int, default=1000)
    a = ap.parse_args()

    fe, mp = _load("family_experts"), _load("mahalanobis_probe")
    results = {}
    for name, bank_dir, eval_dir, ckpt in BACKBONES:
        print(f"\n=== {name}", flush=True)
        tb, eb = FeatureBank.open(bank_dir), FeatureBank.open(eval_dir)
        splits = eb.meta["split"].to_numpy()
        model, ck = load_detector(ckpt, device=a.device)
        if str(ck["config"].get("bank_dir")) != bank_dir:
            raise ValueError(
                f"{ckpt} was trained on {ck['config'].get('bank_dir')!r}, not "
                f"{bank_dir!r}. Scoring a head against a bank of a different "
                "backbone reads another model's feature space through this "
                "one's rows -- and the widths can even match (dinov3l and "
                "siglip2l are both 1024), so it would not fail, just lie.")
        trained = score_grid(model, eb, use_recon=bool(ck["config"]["use_recon"]),
                             device=a.device)
        print("  scored the trained head; fitting the closed form", flush=True)
        closed = closed_form_scores(tb, eb, mp)
        panel = fe.bootstrap_panel({"trained": trained, "closed_form": closed},
                                   splits, baseline="trained", n_boot=a.boot_n)
        d = panel["closed_form"]["vs_baseline"]
        results[name] = {
            "dim": int(tb.config["dim"]),
            "checkpoint": ckpt,
            "trained": panel["trained"]["point"],
            "closed_form": panel["closed_form"]["point"],
            "gap_closed_minus_trained": d["delta"],
            "paired_ci95": d["paired_ci95"],
            "p_closed_better": d["p_better"],
            "verdict": ("tie" if d["paired_ci95"][0] <= 0 <= d["paired_ci95"][1]
                        else "trained wins" if d["paired_ci95"][1] < 0
                        else "closed form wins"),
        }
        r = results[name]
        print(f"  trained {r['trained']:.4f}  closed-form {r['closed_form']:.4f}  "
              f"gap {r['gap_closed_minus_trained']:+.4f}  {r['verdict']}", flush=True)

    print(f"\n| backbone | dim | trained a3 | closed form | gap | paired 95% CI | verdict |")
    print("|---|---|---|---|---|---|---|")
    for name, r in results.items():
        lo, hi = r["paired_ci95"]
        print(f"| {name} | {r['dim']} | {r['trained']:.4f} | {r['closed_form']:.4f} "
              f"| {r['gap_closed_minus_trained']:+.4f} | [{lo:+.4f}, {hi:+.4f}] "
              f"| {r['verdict']} |")

    payload = {
        "probe": "readout_ceiling", "off_ladder": True,
        "metric": SELECTION_METRIC,
        "question": "does the closed-form/trained gap open as the backbone weakens?",
        "boot_n": a.boot_n, "results": results,
    }
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
