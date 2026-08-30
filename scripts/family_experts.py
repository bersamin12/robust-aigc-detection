"""Two generator-family experts over ONE frozen bank (off-ladder probe).

    python scripts/family_experts.py \
        --bank data/banks/dinov3l --eval-bank data/banks/eval_dinov3l \
        --gan  outputs/rungs_family/a3_gan/checkpoint.pt \
        --diff outputs/rungs_family/a3_diff/checkpoint.pt \
        --baseline outputs/rungs/a3/checkpoint.pt \
        --out docs/family_experts.json --device cuda

**This produces no rung and cannot take the headline.** A rung differs from
its parent in exactly one FLAG (`tests/test_rung_ladder.py` enforces it); these
two heads differ from a3 in their training ROWS, which is not a flag and is not
on the ladder. §6.4 chooses among a3-a6 and this is none of them. The number
below goes in the table as a probe, beside A1's ineligible win, and nothing
about it promotes anything.

WHAT IT ANSWERS. Three questions, in the order that makes the third one worth
asking:

1. Does partitioning the fakes by family beat pooling them? Fused experts
   against pooled a3, same bank, same features, same optimisation budget.
2. Does a family expert transfer to a family it never saw? Each expert's own
   score on each held-out generator. If a GAN-only head and a diffusion-only
   head score alike on `SDwithAdaptor_controlnet`, the frozen features are
   doing the generalising and the family label is decoration -- which is the
   result that kills the two-fine-tuned-backbones version of this idea before
   it costs 40 GPU-hours.
3. Does DISAGREEMENT between the experts detect unseen generators better than
   either expert's own logit? This is the only part of the idea that is not
   already published mixture-of-experts ensembling.

TWO DECLARATIONS, BOTH MADE BEFORE ANY NUMBER EXISTED.

*The fusion weight is chosen on `val_internal` alone.* `heldout_robust_tpr` is
read on val_internal authentic against heldout_generator generated; a weight
swept against that is a weight fitted on the rows being reported. The sweep
therefore maximises `val_robust_tpr` below -- same metric shape, both classes
drawn from val_internal -- and the held-out number is read exactly once, after
w is fixed.

*The z-score population is `val_internal` only, and that differs from A5.*
`fuse_scores` standardises each parent per condition, and the ratio of the two
sigmas IS the ratio of the two votes, so which rows set them is a decision.
run_ablation scores A5 with `FIT_SPLITS_FOR_SELECTION`, which is the §6.4
population and therefore includes heldout_generator rows. That is a declared
and defensible choice for A5; it is the wrong one here, because this script
also SELECTS a weight and a sign, and a selection that has seen the held-out
rows through the standardisation is not a clean held-out number. The
`FIT_SPLITS_FOR_SELECTION` variant is computed too and reported beside it --
as the A5-comparable cross-check, never as the headline -- so the gap between
the two standardisations is visible rather than buried in a choice of constant.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd

from aigcdet.eval.errors import (
    SELECTION_METRIC, SELECTION_TARGET_FPR, check_selection_population,
    heldout_robust_tpr,
)
from aigcdet.eval.fusion import (
    FIT_SPLITS_FOR_SELECTION, POPULATION_COLUMN, fuse_scores,
)
from aigcdet.eval.grid import assert_heldout_not_trained, score_grid
from aigcdet.eval.metrics import tpr_at_fpr
from aigcdet.eval.report import DEFAULT_BOOT_SEED, DEFAULT_N_BOOT
from aigcdet.features.bank import FeatureBank
from aigcdet.train.train_head import load_detector

_ROOT = pathlib.Path(__file__).resolve().parent

#: The z-score population for every number this script calls a result. See the
#: second declaration in the module docstring for why it is not
#: FIT_SPLITS_FOR_SELECTION.
PRIMARY_FIT_SPLITS: tuple[str, ...] = ("val_internal",)

#: Weight on the GAN expert. The diffusion expert gets 1 - w. 21 points is a
#: 0.05 grid; finer buys nothing when the objective is a mean of TPRs over
#: ~6.5k validation images and moves in steps of 1/n.
WEIGHT_GRID: np.ndarray = np.linspace(0.0, 1.0, 21)


def _selection_summary_fn():
    """`run_ablation._selection_summary`, imported rather than re-typed.

    The §6.4 metric travels with the declarations that make it checkable
    (`population`, `splits`, `target_fpr`) and `errors._check_provenance` can
    only refuse a contaminated result that carries them. Copying those four
    lines into this file is how the copy drifts from the original and starts
    declaring a population it is not computing. `tests/scripts/` loads script
    modules exactly this way.
    """
    path = _ROOT / "run_ablation.py"
    spec = importlib.util.spec_from_file_location("run_ablation_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._selection_summary


# --- the val-only objective the weight and the sign are chosen on -----------

def val_robust_tpr(scores_df: pd.DataFrame, splits, 
                   target_fpr: float = SELECTION_TARGET_FPR) -> float:
    """Mean TPR @ `target_fpr` over the DEGRADED conditions, val_internal only.

    The same shape as `heldout_robust_tpr` -- mean over the degraded grid, same
    operating point -- with both classes drawn from `val_internal` instead of
    authentic-from-val against generated-from-heldout. It exists so the fusion
    weight and the disagreement sign can be chosen without the held-out rows
    entering the choice.

    It is NOT a substitute for the selection metric and must never be reported
    as one: its positives come from families both experts trained on, so it
    measures fit, not generalisation. It is a knob-setter.
    """
    row_split = np.asarray(splits).astype(str)[scores_df["image_idx"].to_numpy()]
    sub = scores_df[row_split == "val_internal"]
    sub = sub[sub["condition"] != "clean"]
    if sub.empty:
        raise ValueError(
            "no val_internal rows in a degraded condition, so the weight "
            "objective is empty; the eval bank must carry val_internal rows "
            "over the degraded grid for a weight to be fitted off the "
            "held-out families at all")
    values = []
    for cond, g in sub.groupby("condition", sort=False):
        y = g["label"].to_numpy()
        if len(np.unique(y)) != 2:
            raise ValueError(
                f"condition {cond!r} has only class {sorted(set(y.tolist()))} "
                "among val_internal rows, so its TPR@FPR is undefined. "
                "Averaging what survived would be a mean over an unstated "
                "subset of the grid -- the same refusal heldout_robust_tpr "
                "makes for the same reason.")
        values.append(tpr_at_fpr(y, g["score"].to_numpy(), target_fpr))
    return float(np.mean(values))


def _selection_grid(scores_df: pd.DataFrame, splits) -> tuple:
    """The §6.4 population as an (image x degraded condition) score matrix.

    One row per image, one column per degraded condition, plus the label
    vector. Resampling images is then a row index, which is what makes a
    1000-resample paired bootstrap cheap enough to run on every head.
    """
    row_split = np.asarray(splits).astype(str)[scores_df["image_idx"].to_numpy()]
    label = scores_df["label"].to_numpy()
    keep = (((row_split == "val_internal") & (label == 0))
            | ((row_split == "heldout_generator") & (label == 1)))
    sub = scores_df[keep & (scores_df["condition"] != "clean").to_numpy()]
    wide = sub.pivot_table(index="image_idx", columns="condition", values="score")
    y = sub.groupby("image_idx")["label"].first().reindex(wide.index).to_numpy()
    scores = wide.to_numpy()
    if np.isnan(scores).any():
        raise ValueError(
            "the selection population is not a full image x condition grid "
            f"({int(np.isnan(scores).sum())} missing cell(s)); an image scored "
            "under only some conditions would enter the resample with a "
            "different metric than the images beside it")
    return wide.index.to_numpy(), y, scores


def _robust_tpr_of(y: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    return float(np.mean([tpr_at_fpr(y, scores[:, j], target_fpr)
                          for j in range(scores.shape[1])]))


def bootstrap_panel(frames: dict, splits, baseline: str,
                    target_fpr: float = SELECTION_TARGET_FPR,
                    n_boot: int = DEFAULT_N_BOOT,
                    seed: int = DEFAULT_BOOT_SEED) -> dict:
    """Marginal CIs for every head AND paired CIs on each head minus `baseline`.

    Resampled over IMAGES, and over the SAME images for every head in one
    iteration. Both properties are load-bearing:

    *Images, not rows.* `metrics.bootstrap_ci` resamples rows of one (y, s)
    pair, which is right for a per-condition metric and wrong for this one:
    each image contributes a row to all 19 degraded conditions and the metric
    averages over them, so the rows are not independent and the unit of
    resampling is the image.

    *Paired, not marginal.* Two heads score the same images, so the honest
    question "is this head worse than a3" is a statement about the DIFFERENCE,
    and the difference has its own, much narrower interval. Reading it off two
    marginal intervals instead -- "they overlap, so it is a tie" -- is not a
    test, and on this probe it gets the answer wrong: the fused head's marginal
    interval overlaps a3's while the paired interval on the gap excludes zero.
    Marginal intervals are still reported, because the handoff's reporting rule
    asks for the metric with its CI; they are just not what the comparison is
    read from.
    """
    grids = {}
    ref_idx = None
    for name, frame in frames.items():
        idx, y, scores = _selection_grid(frame, splits)
        if ref_idx is None:
            ref_idx, ref_y = idx, y
        elif not (np.array_equal(idx, ref_idx) and np.array_equal(y, ref_y)):
            raise ValueError(
                f"head {name!r} covers a different selection population than "
                f"{list(frames)[0]!r}; a paired resample needs the same images "
                "in the same order for every head")
        grids[name] = scores
    if baseline not in grids:
        raise ValueError(f"baseline {baseline!r} is not among the heads {list(grids)}")

    point = {n: _robust_tpr_of(ref_y, g, target_fpr) for n, g in grids.items()}
    draws = {n: [] for n in grids}
    diffs = {n: [] for n in grids}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        sel = rng.integers(0, len(ref_y), len(ref_y))
        yb = ref_y[sel]
        if len(np.unique(yb)) < 2:
            continue
        vals = {n: _robust_tpr_of(yb, g[sel], target_fpr) for n, g in grids.items()}
        for n, v in vals.items():
            draws[n].append(v)
            diffs[n].append(v - vals[baseline])
    if not draws[baseline]:
        raise ValueError("no valid bootstrap resample kept both classes")

    out = {}
    for name in grids:
        d = np.asarray(diffs[name])
        v = np.asarray(draws[name])
        out[name] = {
            "point": point[name],
            "ci95": [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))],
            "vs_baseline": {
                "baseline": baseline,
                "delta": point[name] - point[baseline],
                "paired_ci95": [float(np.quantile(d, 0.025)),
                                float(np.quantile(d, 0.975))],
                "p_better": float((d > 0).mean()),
            },
            "boot_n": n_boot,
            "boot_seed": seed,
        }
    return out


# --- per-generator transfer: question 2 ------------------------------------

def per_generator_robust_tpr(scores_df: pd.DataFrame, splits,
                             target_fpr: float = SELECTION_TARGET_FPR) -> dict:
    """`heldout_robust_tpr` restricted to ONE held-out generator at a time.

    Same negatives every time -- val_internal authentic -- so the columns are
    comparable across generators and across experts. This is the table that
    answers "did the expert learn its family, or did the frozen feature learn
    everything": a GAN-only head and a diffusion-only head that score alike on
    the same unseen generator did not learn a family.
    """
    row_split = np.asarray(splits).astype(str)[scores_df["image_idx"].to_numpy()]
    label = scores_df["label"].to_numpy()
    authentic = (row_split == "val_internal") & (label == 0)
    held = (row_split == "heldout_generator") & (label == 1)
    neg = scores_df[authentic & (scores_df["condition"] != "clean").to_numpy()]
    out = {}
    for gen in sorted(set(scores_df.loc[held, "generator"].astype(str))):
        pos = scores_df[held & (scores_df["generator"].astype(str) == gen).to_numpy()
                        & (scores_df["condition"] != "clean").to_numpy()]
        values = []
        for cond in dict.fromkeys(pos["condition"].tolist()):
            p = pos[pos["condition"] == cond]
            n = neg[neg["condition"] == cond]
            y = np.concatenate([np.zeros(len(n)), np.ones(len(p))])
            s = np.concatenate([n["score"].to_numpy(), p["score"].to_numpy()])
            values.append(tpr_at_fpr(y, s, target_fpr))
        out[gen] = float(np.mean(values))
    return out


# --- provenance refusals ----------------------------------------------------

def _excluded(ck: dict) -> list[str]:
    """A checkpoint's excluded families, as a list, whether or not it has any.

    `outputs/rungs/a3/checkpoint.pt` predates the field entirely, so the key is
    absent there; a checkpoint written with it explicitly null would return
    None from a bare `.get`. Both mean "excluded nothing" and both must survive
    being iterated.
    """
    return [str(x) for x in (ck["config"].get("train_exclude_generators") or [])]


def assert_complementary(gan_ck: dict, diff_ck: dict, bank_dir: str) -> None:
    """Refuse a pair that is not actually two experts over the same features.

    Every failure here produces a number that looks exactly like a result. Two
    checkpoints trained on the same rows fuse to a self-ensemble and report a
    fusion win that is a variance reduction; two heads over different banks are
    rung A5, not this probe, and would attribute a backbone difference to the
    family split; an empty exclusion list is the pooled model wearing an
    expert's name.
    """
    g, d = _excluded(gan_ck), _excluded(diff_ck)
    if not g or not d:
        raise ValueError(
            f"an expert must exclude something: --gan excludes {g or 'nothing'} "
            f"and --diff excludes {d or 'nothing'}. A checkpoint with an empty "
            "train_exclude_generators is the pooled a3 head, and fusing it with "
            "itself measures nothing about generator families.")
    if set(g) == set(d):
        raise ValueError(
            f"both checkpoints exclude the same families ({sorted(set(g))}), so "
            "they trained on identical rows and their 'fusion' is a "
            "self-ensemble. Two experts must partition the fakes.")
    for name, ck in (("--gan", gan_ck), ("--diff", diff_ck)):
        got = str(ck["config"].get("bank_dir"))
        if pathlib.Path(got).resolve() != pathlib.Path(bank_dir).resolve():
            raise ValueError(
                f"{name} was trained on bank {got!r}, not {bank_dir!r}. The "
                "whole claim of this probe is that the two experts share a "
                "frozen feature space and differ only in training rows; two "
                "banks is rung A5, and any difference would be the backbone.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True, help="the TRAINING bank both experts used")
    ap.add_argument("--eval-bank", required=True)
    ap.add_argument("--gan", required=True, help="checkpoint.pt of the GAN-family expert")
    ap.add_argument("--diff", required=True, help="checkpoint.pt of the diffusion expert")
    ap.add_argument("--baseline", required=True,
                    help="checkpoint.pt of pooled a3 -- the thing to beat")
    ap.add_argument("--out", default="docs/family_experts.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--boot-n", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--boot-seed", type=int, default=DEFAULT_BOOT_SEED)
    a = ap.parse_args()

    selection_summary = _selection_summary_fn()

    eval_bank = FeatureBank.open(a.eval_bank)
    splits = eval_bank.meta["split"].to_numpy()
    check_selection_population(splits)
    train_bank = FeatureBank.open(a.bank)

    models, cks = {}, {}
    for name, path in (("gan", a.gan), ("diff", a.diff), ("a3", a.baseline)):
        models[name], cks[name] = load_detector(path, device=a.device)
    assert_complementary(cks["gan"], cks["diff"], a.bank)
    # The same check run_ablation makes before the first rung trains: a family
    # scored as held out that is also in the training split turns the headline
    # into a score on families the head saw. Run per expert, because each one
    # excludes a different set and the guard is about what THIS head trained on.
    for name in ("gan", "diff", "a3"):
        assert_heldout_not_trained(train_bank, eval_bank, _excluded(cks[name]))

    scored = {name: score_grid(models[name], eval_bank,
                               use_recon=bool(cks[name]["config"]["use_recon"]),
                               device=a.device)
              for name in ("gan", "diff", "a3")}

    # --- question 1: the weight, chosen on val_internal, then read once -----
    sweep = []
    for w in WEIGHT_GRID:
        fused = fuse_scores([scored["gan"], scored["diff"]], weights=[float(w), 1.0 - float(w)],
                            splits=splits, fit_splits=PRIMARY_FIT_SPLITS)
        sweep.append({"w_gan": float(w), "val_robust_tpr": val_robust_tpr(fused, splits)})
    best = max(sweep, key=lambda r: r["val_robust_tpr"])
    w_gan = best["w_gan"]
    print(f"weight chosen on val_internal alone: w_gan={w_gan:.2f} "
          f"(val_robust_tpr={best['val_robust_tpr']:.4f})")

    scored["fused"] = fuse_scores(
        [scored["gan"], scored["diff"]], weights=[w_gan, 1.0 - w_gan],
        splits=splits, fit_splits=PRIMARY_FIT_SPLITS)
    # The A5-comparable cross-check, never the headline: same weight, the
    # standardisation run_ablation uses for the fused rung. Reported so the
    # gap between the two populations is a number rather than a constant.
    scored["fused_a5_population"] = fuse_scores(
        [scored["gan"], scored["diff"]], weights=[w_gan, 1.0 - w_gan],
        splits=splits, fit_splits=FIT_SPLITS_FOR_SELECTION)

    # --- question 3: disagreement, with its sign chosen on val too ---------
    # weights=[1,0] normalises to [1,0], so the fused score IS parent 0's
    # z-score on the shared rows -- the standardisation is reused rather than
    # reimplemented, which keeps the disagreement on the same scale as the
    # fusion it is compared against.
    z_gan = fuse_scores([scored["gan"], scored["diff"]], weights=[1.0, 0.0],
                        splits=splits, fit_splits=PRIMARY_FIT_SPLITS)
    z_diff = fuse_scores([scored["diff"], scored["gan"]], weights=[1.0, 0.0],
                         splits=splits, fit_splits=PRIMARY_FIT_SPLITS)
    if not np.array_equal(z_gan["image_idx"].to_numpy(), z_diff["image_idx"].to_numpy()):
        raise ValueError(
            "the two z-scored frames are not row-aligned, so their difference "
            "is not a per-image disagreement")
    gap = np.abs(z_gan["score"].to_numpy() - z_diff["score"].to_numpy())
    disagree = z_gan.copy()
    disagree["score"] = gap
    flipped = z_gan.copy()
    flipped["score"] = -gap
    # One bit -- "does disagreement mean fake, or does agreement" -- is still a
    # choice, and a choice made by reading both held-out numbers and keeping
    # the better one is not a held-out number. Decided on val_internal.
    sign = 1.0 if val_robust_tpr(disagree, splits) >= val_robust_tpr(flipped, splits) else -1.0
    scored["disagreement"] = disagree if sign > 0 else flipped
    print(f"disagreement sign chosen on val_internal: "
          f"{'|z_gan - z_diff|' if sign > 0 else '-|z_gan - z_diff|'}")

    # --- the numbers -------------------------------------------------------
    panel = bootstrap_panel(scored, splits, baseline="a3",
                            n_boot=a.boot_n, seed=a.boot_seed)
    results = {}
    for name, frame in scored.items():
        summary = selection_summary(frame, splits)
        results[name] = summary | panel[name] | {
            "val_robust_tpr_knob_setter_not_a_result": val_robust_tpr(frame, splits),
            "per_generator": per_generator_robust_tpr(frame, splits),
            POPULATION_COLUMN: (str(frame[POPULATION_COLUMN].iloc[0])
                                if POPULATION_COLUMN in frame else "single_head"),
        }

    print()
    print(f"| head | {SELECTION_METRIC} | 95% CI | vs a3 | paired 95% CI on the gap | P(>a3) |")
    print("|---|---|---|---|---|---|")
    for name in ("a3", "gan", "diff", "fused", "fused_a5_population", "disagreement"):
        r = results[name]
        lo, hi = r["ci95"]
        d = r["vs_baseline"]
        dlo, dhi = d["paired_ci95"]
        # The verdict comes from the PAIRED interval: a tie is a gap whose own
        # interval contains zero, not two marginal intervals that overlap.
        mark = "" if name == "a3" else (
            "  tie" if dlo <= 0.0 <= dhi else "  WORSE" if dhi < 0 else "  BETTER")
        print(f"| {name} | {r[SELECTION_METRIC]:.4f} | [{lo:.4f}, {hi:.4f}] | "
              f"{d['delta']:+.4f} | [{dlo:+.4f}, {dhi:+.4f}] | "
              f"{d['p_better']:.3f}{mark} |")
    print()
    print("per held-out generator, TPR@1%FPR vs val_internal authentic:")
    gens = sorted(results["a3"]["per_generator"])
    print("| head | " + " | ".join(gens) + " |")
    print("|---" * (len(gens) + 1) + "|")
    for name in ("a3", "gan", "diff", "fused", "disagreement"):
        row = results[name]["per_generator"]
        print(f"| {name} | " + " | ".join(f"{row[g]:.4f}" for g in gens) + " |")

    payload = {
        "probe": "family_experts",
        "off_ladder": True,
        "not_eligible_reason":
            "differs from a3 in training ROWS, not in a flag; §6.4 chooses "
            "among a3-a6 and this is none of them",
        "eval_bank": a.eval_bank,
        "train_bank": a.bank,
        "manifest_sha256": eval_bank.config.get("manifest_sha256"),
        "backbone": train_bank.config.get("backbone"),
        "checkpoints": {"gan": a.gan, "diff": a.diff, "a3": a.baseline},
        "exclusions": {n: _excluded(cks[n]) for n in ("gan", "diff", "a3")},
        "epochs": {n: cks[n]["config"].get("epochs") for n in ("gan", "diff", "a3")},
        "primary_zscore_population": list(PRIMARY_FIT_SPLITS),
        "weight_selected_on": "val_internal",
        "w_gan": w_gan,
        "weight_sweep": sweep,
        "disagreement_sign": ("abs" if sign > 0 else "neg_abs"),
        "results": results,
    }
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
