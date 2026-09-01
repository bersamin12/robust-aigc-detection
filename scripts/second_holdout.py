"""Does the winner survive a DIFFERENT held-out family? (off-ladder)

    python scripts/second_holdout.py \
        --arm band=data/banks/probe_band_dinov2regl:data/banks/eval_probe_band_dinov2regl \
        --arm crop=data/banks/probe_crop_dinov2regl:data/banks/eval_probe_crop_dinov2regl \
        --holdout-generators sid_set \
        --out docs/second_holdout_sid_set.json

WHY. Every number this repo has selected on rests on ONE held-out split of TWO
generators, `SDwithAdaptor_controlnet` (766) and `VQGAN` (734) -- and both are
wildfake. The plan has carried this as a known weakness since it was written:

    "The selection metric rests on two generators, both wildfake. A second
     held-out split is Stage B only and is the cheapest insurance on the most
     consequential decision."

That decision is now load-bearing. `dinov2regl` band+crop fused at 0.8714 beat
the barred `dinov3l` reference at 0.8667 by 0.0047 -- a margin far smaller than
anything two generators can be trusted to resolve. If the ORDERING of the arms
flips when the held-out family changes, the headline is a property of those two
generators and not of the arms.

WHAT IS MEASURED. The ordering, not the level. The second split's absolute
number is NOT comparable with 0.8714: a different family at a different sample
size is a different population, and the primary split's 1500 rows dwarf
anything left in `val_internal`. What transfers is the RANK -- does crop still
beat band, and does the fusion still beat both parents?

HOW A SECOND SPLIT IS BUILT WITHOUT RE-EXTRACTING ANYTHING. A lineage holdout
is two halves that live in different files (`grid.assert_heldout_not_trained`):
the eval manifest promotes a family into `heldout_generator` so there is
something to score, and `RungConfig.train_exclude_generators` drops it from the
training rows so the score means anything. Doing only the first inflates the
number silently. This script does BOTH, and does the first in memory:

  * TRAIN side, real: every arm is retrained with `train_exclude_generators`
    set to the chosen family. This is Stage B on a cached bank -- no image is
    read and no backbone runs.
  * EVAL side, in memory: `errors.heldout_robust_tpr(scores_df, splits)` takes
    `splits` as a caller-supplied array and, by its own docstring, "the
    population is built HERE, from `splits`, rather than being trusted from the
    caller". So relabelling the chosen family's rows to `heldout_generator` in
    that array is the supported way to name a different population, not a
    workaround.

WHERE EACH ROW GOES, AND WHY ONLY TWO THINGS MOVE.

    authentic        val_internal, label 0      unchanged from the primary split
    generated        the chosen family          relabelled: was val_internal
    unused           primary held-out pair      dropped from BOTH populations
    weight-fit only  other val_internal fakes   left exactly where they were

The primary held-out generators go to `unused` rather than being kept. They are
still unseen by every arm here, so keeping them would only re-measure the split
this script exists to double-check, diluted.

The remaining `val_internal` fakes STAY `val_internal`, which looks wrong at
first glance -- the heads trained on those families. They stay because
`heldout_robust_tpr` builds its generated population from `heldout_generator`
alone, so they cannot reach the reported number; their only role is to give the
fusion weight an objective. `fusion.val_robust_tpr` is explicit that this is
what trained-on families are for: "its positives come from families the heads
trained on, so it measures fit, not generalisation. It is a knob-setter."
Sending them to `unused` instead leaves `val_internal` with no positive class
at all, and the weight sweep then raises rather than fitting on an unstated
subset -- which is exactly what the first run of this script did.

THE WEIGHT FIT STAYS HONEST FOR FREE. `fusion.val_robust_tpr` selects rows
where the split is literally `val_internal`. The chosen family is relabelled
`heldout_generator` before any weight is swept, so it leaves the weight-fitting
population by construction -- the fitted w cannot see the rows it is reported
on. That is the same discipline as `family_experts`, obtained by the relabel
rather than restated.

OFF-LADDER. This produces no rung. Every arm is a3; they differ in training
ROWS, which is not a flag, and `tests/test_rung_ladder.py` enforces that a rung
differs from its base by exactly one flag. §6.4 chooses among a3-a6 on the
primary split. A disagreement here is a finding about confidence in that
choice, never a substitute for it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from aigcdet.eval.errors import SELECTION_TARGET_FPR, heldout_robust_tpr
from aigcdet.eval.fusion import (
    FIT_SPLITS_WHEN_FITTING_WEIGHT, fit_fusion_weight, fuse_scores,
    val_robust_tpr,
)
from aigcdet.eval.grid import score_grid
from aigcdet.features.bank import FeatureBank
from aigcdet.train.train_head import load_detector, train_rung

sys.path.insert(0, "scripts")
from run_ablation import load_rung_config  # noqa: E402

#: Rows that belong to neither population. Any string that is not
#: `val_internal` or `heldout_generator` works; a named constant is used so the
#: JSON can say what happened to them rather than leaving a reader to infer it
#: from an absence.
UNUSED = "unused_second_holdout"


def relabelled_splits(eval_bank: FeatureBank, families: list[str]) -> pd.Series:
    """The eval bank's split column, with `families` promoted to held out.

    Returns a Series indexed by `image_idx`, which is the indexing
    `score_grid` emits and `heldout_robust_tpr` expects.
    """
    meta = eval_bank.meta
    for col in ("split", "generator", "label", "image_idx"):
        if col not in meta.columns:
            raise SystemExit(
                f"the eval bank has no {col!r} column, so a second holdout "
                "cannot be named from it. Re-extract with a current BankWriter.")

    split = meta.set_index("image_idx")["split"].astype(str).copy()
    gen = meta.set_index("image_idx")["generator"].astype(str)
    label = meta.set_index("image_idx")["label"].astype(int)

    chosen = gen.isin(families)
    if not chosen.any():
        raise SystemExit(
            f"no eval row has generator in {families}; the second holdout "
            f"would be empty. Present: {sorted(gen.unique().tolist())}")
    if (label[chosen] == 0).any():
        raise SystemExit(
            f"{int((label[chosen] == 0).sum())} row(s) in {families} are "
            "authentic (label 0). A held-out GENERATOR family is generated by "
            "definition; scoring reals as the positive class would invert the "
            "metric rather than move its population.")

    # ONLY two things move.
    #
    # The PRIMARY held-out families leave both populations: they are what this
    # script exists to double-check, so keeping them would re-measure the split
    # under test, diluted with the new one.
    #
    # The other `val_internal` fakes STAY `val_internal`, and that is
    # deliberate. An earlier version of this function sent every generated row
    # that was not the chosen family to `UNUSED`, which emptied `val_internal`
    # of positives and made `fusion.val_robust_tpr` raise -- correctly: with no
    # positive class its TPR is undefined. But those rows were never meant to
    # leave. `heldout_robust_tpr` builds its generated population from
    # `heldout_generator` alone, so they cannot reach the reported number;
    # their only job is to give the weight sweep an objective, and
    # `val_robust_tpr` is explicit that this is the job trained-on families are
    # for: "its positives come from families the heads trained on, so it
    # measures fit, not generalisation. It is a knob-setter."
    out = split.copy()
    out[split == "heldout_generator"] = UNUSED
    out[chosen] = "heldout_generator"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    metavar="NAME=BANK:EVAL_BANK",
                    help="repeatable; fused pairwise when exactly two are given")
    ap.add_argument("--holdout-generators", required=True,
                    help="comma-separated generator names to hold out")
    ap.add_argument("--config", default="configs/rungs/a3.yaml")
    ap.add_argument("--out-dir", default="outputs/second_holdout")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    families = [g.strip() for g in a.holdout_generators.split(",") if g.strip()]
    tag = "_".join(families)[:40]

    arms = {}
    for spec in a.arm:
        name, _, paths = spec.partition("=")
        bank_dir, _, eval_dir = paths.partition(":")
        if not (name and bank_dir and eval_dir):
            raise SystemExit(f"--arm wants NAME=BANK:EVAL_BANK, got {spec!r}")
        arms[name] = (bank_dir, eval_dir)

    rows, scored, splits_by_arm = [], {}, {}
    for name, (bank_dir, eval_dir) in arms.items():
        rung_name = os.path.splitext(os.path.basename(a.config))[0]
        print(f"\n=== {name}: retraining {rung_name} without {families} ===",
              flush=True)
        eval_bank = FeatureBank(eval_dir)
        splits = relabelled_splits(eval_bank, families)
        splits_by_arm[name] = splits

        cfg = load_rung_config(a.config, bank_dir, f"{a.out_dir}/{tag}/{name}",
                               a.device, train_exclude_generators=families)
        # The RUNG this actually trained, not a hardcoded "a3": this script now
        # gets pointed at a4 too, and a JSON that says a3 while holding an a4
        # number is worse than no JSON.
        cfg.name = f"{rung_name}_no_{tag}"
        result = train_rung(cfg)
        model, _ = load_detector(result["checkpoint"], device=a.device)
        df = score_grid(model, eval_bank, use_recon=cfg.use_recon,
                        device=a.device)
        scored[name] = df

        second = heldout_robust_tpr(df, splits, SELECTION_TARGET_FPR)
        rows.append({
            "arm": name, "bank": bank_dir, "eval_bank": eval_dir,
            "second_holdout_robust_tpr_at_1pct": float(second),
            "val_robust_tpr": float(val_robust_tpr(df, splits,
                                                   SELECTION_TARGET_FPR)),
            "val_auc_clean_view_only": result.get("val_auc"),
        })
        print(f"{name}: second-holdout robust TPR@1%FPR = {second:.4f}",
              flush=True)

    fusion = None
    if len(arms) == 2:
        names = list(arms)
        # Both arms share a manifest, so either arm's relabelled splits name the
        # same rows; assert it rather than trusting it, because a fused frame
        # has no single owning bank (errors.heldout_robust_tpr, "NOTE FOR TASK 8").
        s0, s1 = (splits_by_arm[n] for n in names)
        if not s0.equals(s1):
            raise SystemExit(
                "the two arms' eval banks disagree on the relabelled split "
                "column, so the fused frame has no defined population.")
        dfs = [scored[n] for n in names]
        (w0, w1), sweep = fit_fusion_weight(
            dfs, s0, fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT)
        equal = fuse_scores(dfs, splits=s0,
                            fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT)
        fitted = fuse_scores(dfs, weights=[w0, w1], splits=s0,
                             fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT)
        fusion = {
            "parents": names, "w": [float(w0), float(w1)],
            "equal_weight_second_holdout": float(
                heldout_robust_tpr(equal, s0, SELECTION_TARGET_FPR)),
            "fitted_weight_second_holdout": float(
                heldout_robust_tpr(fitted, s0, SELECTION_TARGET_FPR)),
            "zscore_fit_splits": list(FIT_SPLITS_WHEN_FITTING_WEIGHT),
            "weight_sweep": sweep,
        }
        print(f"\nfused {names[0]}+{names[1]}  equal="
              f"{fusion['equal_weight_second_holdout']:.4f}  "
              f"fitted(w={w0:.2f})={fusion['fitted_weight_second_holdout']:.4f}",
              flush=True)

    n_pos = int((splits_by_arm[list(arms)[0]] == "heldout_generator").sum())
    print(f"\n{'arm':>10s} {'2nd holdout':>12s} {'val_robust':>11s}")
    for r in rows:
        print(f"{r['arm']:>10s} {r['second_holdout_robust_tpr_at_1pct']:12.4f} "
              f"{r['val_robust_tpr']:11.4f}")
    print(f"\nheld-out positives: {n_pos} images x conditions "
          f"(primary split has 1500 images) -- read the ORDER, not the level")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({
            "probe": "second_holdout", "off_ladder": True,
            "not_eligible_reason": "arms differ in training ROWS, not in a "
                                   "rung flag; §6.4 selects on the primary "
                                   "held-out split",
            "metric": "heldout_robust_tpr_at_1pct on a relabelled population",
            "holdout_generators": families,
            "n_heldout_images": n_pos,
            "unused_label": UNUSED,
            "comparability": "absolute values are NOT comparable with the "
                             "primary split (different family, different n); "
                             "the ordering of the arms is what transfers",
            "base_config": a.config,
            "arms": rows, "fusion": fusion,
        }, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
