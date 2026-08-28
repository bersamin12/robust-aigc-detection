"""Train and evaluate every rung, then emit the robustness table (spec §6.4).

    python scripts/run_ablation.py --bank banks/dinov3l --eval-bank banks/eval_dinov3l \
        --rungs configs/rungs/a0.yaml configs/rungs/a1.yaml configs/rungs/a2.yaml \
                configs/rungs/a3.yaml configs/rungs/a4.yaml \
        --tier ablation --out docs/robustness_table.md

Three properties this script is built around:

**It is resumable, and never silently reuses a checkpoint.** A rung whose
checkpoint already exists is NOT retrained -- an ablation ladder is many hours
of GPU and a killed run must be able to continue -- but every skip is printed,
recorded in `selection.json`, and gated on the stored config matching the one
asked for. Reusing a checkpoint trained under a different config would put a
row in the table describing a model that is no longer on disk, which is worse
than retraining.

**It refuses to compare incomparable rungs.** The rungs' banks are passed to
`robustness_table`, which routes them through `eval.grid.assert_banks_comparable`
and additionally requires a manifest fingerprint. Differing view coverage
between compared rungs measures augmentation budgets rather than models.

**The headline is chosen by the §6.4 rule and by nothing else.** The selection
metric is `eval.errors.heldout_robust_tpr` -- mean TPR@1%FPR over the degraded
conditions, on internal-validation authentic images against held-out-generator
fakes. `val_auc` from `train_rung` is the CLEAN VIEW ONLY and is recorded here
under a name that says so, so it cannot be mistaken for the selection metric.
The population is written into `selection.json` next to the choice.

This script emits no content-blind figure. If one is ever added, it must come
from `eval.controls.metadata_control` and never from
`content_blind_auc(metadata_features(paths), y)` (ruling R37): only the former
carries the estimator-branch confound check.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import torch
import yaml

from aigcdet.baselines import (
    BASELINE_ROW_FOOTNOTES, BASELINE_ROW_LABELS, RUNG_IS_A_BASELINE,
)
from aigcdet.eval.errors import (
    SELECTION_METRIC, SELECTION_POPULATION, SELECTION_SPLITS,
    heldout_robust_tpr, selection_report,
)
from aigcdet.eval.grid import score_grid
from aigcdet.eval.report import (
    DEFAULT_BOOT_SEED, DEFAULT_N_BOOT, TIER_CONDITIONS, robustness_table,
    save_heatmap, to_markdown,
)
from aigcdet.features.bank import FeatureBank
from aigcdet.train.train_head import RungConfig, load_detector, train_rung

CHECKPOINT_NAME = "checkpoint.pt"
RESULT_NAME = "result.json"

#: Config fields that may differ between the checkpoint on disk and this run
#: without making the checkpoint a different experiment. `device` changes
#: nothing about the trained weights; `out_dir` is where they were put;
#: `manifest_path` is a check performed at training time, not a model property.
RESUME_IGNORED_KEYS: tuple[str, ...] = ("device", "out_dir", "manifest_path")


def rung_paths(out_dir: str, name: str) -> tuple[str, str]:
    d = os.path.join(out_dir, name)
    return os.path.join(d, CHECKPOINT_NAME), os.path.join(d, RESULT_NAME)


def load_rung_config(config_path: str, bank_dir: str, out_dir: str, device: str,
                     manifest_path: str | None = None) -> RungConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return RungConfig(bank_dir=bank_dir, out_dir=out_dir, device=device,
                      manifest_path=manifest_path, **raw)


def config_differences(on_disk: dict, cfg: RungConfig) -> dict:
    """Fields where a stored checkpoint's config disagrees with this run's."""
    requested = asdict(cfg)
    keys = (set(on_disk) | set(requested)) - set(RESUME_IGNORED_KEYS)
    missing = object()
    return {k: (on_disk.get(k, missing), requested.get(k, missing))
            for k in sorted(keys)
            if on_disk.get(k, missing) != requested.get(k, missing)}


def train_or_resume(cfg: RungConfig, force: bool = False) -> tuple[dict, bool]:
    """`(result, resumed)`. Trains unless a matching checkpoint already exists.

    A checkpoint with no `result.json` beside it is refused rather than either
    retrained (which would overwrite hours of GPU) or reused (its val AUCs were
    never recorded): that pair is written at the end of `train_rung`, so only a
    run killed between the two lines produces it, and the human should decide.
    """
    ckpt, result_json = rung_paths(cfg.out_dir, cfg.name)
    if force or not os.path.exists(ckpt):
        return train_rung(cfg), False

    if not os.path.exists(result_json):
        raise FileExistsError(
            f"rung {cfg.name!r} has a checkpoint at {ckpt} but no {result_json}: "
            "the previous run was killed between the two writes, so its "
            "validation AUCs were never recorded. Delete the checkpoint or pass "
            "--force-retrain; this script will not overwrite a checkpoint it "
            "cannot verify.")

    stored = torch.load(ckpt, map_location="cpu", weights_only=True)["config"]
    differences = config_differences(stored, cfg)
    if differences:
        raise ValueError(
            f"refusing to resume rung {cfg.name!r}: the checkpoint at {ckpt} was "
            f"trained with a different configuration -- {differences} (on disk, "
            "requested). Reusing it would put a row in the table describing a "
            "model that no longer matches its config file. Delete it or pass "
            "--force-retrain.")

    with open(result_json) as f:
        result = json.load(f)
    result["checkpoint"] = ckpt
    return result, True


def baseline_footnotes(row_names) -> str:
    """Ruling R38/I3: a §6.3 row must not be labelled with a short name that
    understates the published method it is compared against.

    Returns markdown naming the required wording for whichever baseline rows
    this table actually contains, or an empty string when it contains none.
    """
    lines = []
    for name in row_names:
        key = str(name).strip().lower()
        baseline = key if key in BASELINE_ROW_LABELS else RUNG_IS_A_BASELINE.get(key)
        if baseline is None:
            continue
        lines.append(f"- Row `{name}` quoted as a baseline must be labelled "
                     f"**{BASELINE_ROW_LABELS[baseline]}**: "
                     f"{BASELINE_ROW_FOOTNOTES[baseline]}")
    if not lines:
        return ""
    return ("\n## Baseline row labels (spec §6.3, ruling R38)\n\n"
            + "\n".join(lines) + "\n")


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True, help="training feature bank")
    ap.add_argument("--eval-bank", required=True,
                    help="bank written by eval.grid.extract_eval_bank")
    ap.add_argument("--rungs", nargs="+", required=True, help="rung config YAMLs")
    ap.add_argument("--tier", required=True, choices=sorted(TIER_CONDITIONS))
    ap.add_argument("--out", default="docs/robustness_table.md")
    ap.add_argument("--heatmap", default=None,
                    help="default: --out with a .png suffix")
    ap.add_argument("--selection", default="docs/selection.json")
    ap.add_argument("--out-dir", default="outputs/rungs",
                    help="where each rung's checkpoint lives (and is resumed from)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--metric", default="auc",
                    help="metric tabulated per condition; note that the HEADLINE "
                         "is always selected on " + SELECTION_METRIC)
    ap.add_argument("--boot-seed", type=int, default=DEFAULT_BOOT_SEED)
    ap.add_argument("--boot-n", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--manifest", default=None,
                    help="frozen manifest.parquet; when given, each rung verifies "
                         "the training bank is still positionally aligned with it")
    ap.add_argument("--force-retrain", action="store_true",
                    help="retrain every rung even if a checkpoint exists")
    return ap


def main(argv=None) -> dict:
    a = build_parser().parse_args(argv)
    heatmap_path = a.heatmap or (a.out[:-3] + ".png" if a.out.endswith(".md")
                                 else a.out + ".png")

    eval_bank = FeatureBank.open(a.eval_bank)
    # The split column is read from the EVAL BANK, never from a caller-supplied
    # list, so the §6.4 population is built from what was actually scored.
    splits = eval_bank.meta["split"].to_numpy()

    per_rung, summary = {}, {}
    for config_path in a.rungs:
        cfg = load_rung_config(config_path, a.bank, a.out_dir, a.device, a.manifest)
        result, resumed = train_or_resume(cfg, force=a.force_retrain)
        if resumed:
            print(f"SKIP {cfg.name}: reusing the existing checkpoint at "
                  f"{result['checkpoint']} -- NOT retrained. Pass --force-retrain "
                  "to train it again.")
        model, _ = load_detector(result["checkpoint"], device=a.device)
        scores = score_grid(model, eval_bank, use_recon=cfg.use_recon,
                            device=a.device)
        per_rung[cfg.name] = scores
        summary[cfg.name] = {
            SELECTION_METRIC: heldout_robust_tpr(scores, splits),
            "population": SELECTION_POPULATION,
            "splits": list(SELECTION_SPLITS),
            # NOT the selection metric, and named so nobody can read it as one:
            # `val_auc` from train_rung is view 0, the clean view, alone.
            "val_auc_clean_view_only": float(result["val_auc"]),
            "val_auc_mean_views": float(result["val_auc_mean_views"]),
            "resumed_from_checkpoint": resumed,
            "config": config_path,
        }
        row = summary[cfg.name]
        print(f"{cfg.name}: {SELECTION_METRIC}={row[SELECTION_METRIC]:.4f} "
              f"val_auc(clean view only)="
              f"{row['val_auc_clean_view_only']:.4f}")

    # One bank per rung, so `robustness_table` routes them through
    # `assert_banks_comparable` and rejects an eval bank with no manifest
    # fingerprint. Every rung is scored on the SAME eval bank here; passing the
    # mapping anyway is what makes that a checked fact rather than an assumption.
    banks = {rung: eval_bank for rung in per_rung}
    table = robustness_table(per_rung, tier=a.tier, metric=a.metric,
                             seed=a.boot_seed, n_boot=a.boot_n, banks=banks)

    _ensure_parent(a.out)
    to_markdown(table, tier=a.tier, path=a.out)
    footnotes = baseline_footnotes(table.index)
    if footnotes:
        with open(a.out, "a") as f:
            f.write(footnotes)
    _ensure_parent(heatmap_path)
    save_heatmap(table, heatmap_path)

    report = selection_report(summary)
    report["table"] = os.path.abspath(a.out)
    report["heatmap"] = os.path.abspath(heatmap_path)
    report["tier"] = a.tier
    # The table tabulates `--metric` per condition; the HEADLINE is selected on
    # SELECTION_METRIC regardless. Recording both stops a reader of the table
    # from inferring which rule chose the model from which column it can see.
    report["table_metric"] = a.metric
    _ensure_parent(a.selection)
    with open(a.selection, "w") as f:
        json.dump(report, f, indent=2)

    print(f"headline model: {report['headline']} "
          f"(rule: {SELECTION_METRIC} over {SELECTION_POPULATION})")
    if report["headline_error"]:
        print(f"headline not selected: {report['headline_error']}")
    return report


if __name__ == "__main__":
    main()
