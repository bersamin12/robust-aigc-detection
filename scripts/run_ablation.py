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
metric is `eval.errors.heldout_robust_tpr` -- mean TPR at the project operating
point over the degraded conditions, on internal-validation authentic images
against held-out-generator fakes. `val_auc` from `train_rung` is the CLEAN VIEW
ONLY and is recorded here under a name that says so, so it cannot be mistaken
for the selection metric. The population is written into `selection.json` next
to the choice -- and the metric itself is written into the ROBUSTNESS TABLE, as
its own column, because `--metric` tabulates a §6.1 reporting metric over every
scored row and that column ranks the rungs differently. A table whose best
`robust_tpr_at_1pct` and a `selection.json` whose headline name different rungs
is the failure the extra column exists to prevent.

**A5 and A6 are not training configs, and are wired in accordingly.** `--fuse-bank
BANK --fuse-eval-bank EVALBANK` trains a second A3 head on an independently
extracted bank and fuses its grid scores with this run's A3 row
(`eval.fusion.fuse_scores`), producing the `a5` row. The fused row registers a
`FusedEvalBank` over BOTH parents, never one of them, and its selection metric
is computed on the split column the two parents share -- which exists only when
they agree on the frozen manifest, the condition axis and their split and label
columns. `--tta` records rung A6's cost multiplier and the tier it applies to;
A6 is inference-only and is scored from images rather than from the cached eval
bank, so this script produces no `a6` row and says so in `selection.json`
rather than leaving the rung silently absent.

This script emits no content-blind figure. If one is ever added, it must come
from `eval.controls.metadata_control` and never from
`content_blind_auc(metadata_features(paths), y)` (ruling R37): only the former
carries the estimator-branch confound check.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

import torch
import yaml

from aigcdet.baselines import (
    BASELINE_ROW_FOOTNOTES, BASELINE_ROW_LABELS, RUNG_IS_A_BASELINE,
)
from aigcdet.eval.errors import (
    SELECTION_METRIC, SELECTION_POPULATION, SELECTION_SPLITS,
    SELECTION_TARGET_FPR, check_selection_population, heldout_robust_tpr,
    selection_report,
)
from aigcdet.eval.fusion import (
    FIT_SPLITS_FOR_SELECTION, POPULATION_COLUMN,
    FusedEvalBank, assert_fusion_parents, fuse_scores, fused_splits,
)
from aigcdet.eval.grid import score_grid
from aigcdet.eval.report import (
    DEFAULT_BOOT_SEED, DEFAULT_N_BOOT, METRIC_COLUMNS, PROBABILITY_METRICS,
    THRESHOLD_METRICS, TIER_CONDITIONS, clean_validation_threshold,
    robustness_table, save_heatmap, to_markdown,
)
from aigcdet.eval.tta import TTA_VIEWS
from aigcdet.features.bank import FeatureBank
from aigcdet.operating_point import fpr_label
from aigcdet.train.train_head import RungConfig, load_detector, train_rung

CHECKPOINT_NAME = "checkpoint.pt"
RESULT_NAME = "result.json"

#: Rung A5 is not a training config: it fuses the A3 head on this run's bank
#: with an A3 head trained on a second, independently-extracted bank. The
#: partner is trained from A3's OWN config file rather than from a config of
#: its own, so the two halves cannot drift apart into a comparison of training
#: recipes when the point of A5 is the second backbone.
FUSION_RUNG = "a5"
FUSION_BASE_RUNG = "a3"
FUSION_PARTNER_NAME = "a5_partner"

#: Rung A6 is inference-only. It is scored from images, not from the cached
#: eval bank, so this script records its cost and its tier and produces no row.
TTA_RUNG = "a6"

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
    with open(config_path, encoding="utf-8") as f:
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

    with open(result_json, encoding="utf-8") as f:
        result = json.load(f)
    result["checkpoint"] = ckpt
    return result, True


#: Backbone a rung must have been trained on for `RUNG_IS_A_BASELINE` to apply.
#: UnivFD is a linear probe on frozen CLIP features; rung A0 on any other bank
#: is our own rung A0 and nothing else.
UNIVFD_BACKBONE = "clipl"


def baseline_footnotes(row_names, backbone: str | None = None) -> str:
    """Ruling R38/I3: a §6.3 row must not be labelled with a short name that
    understates the published method it is compared against.

    Returns markdown naming the required wording for whichever baseline rows
    this table actually contains, or an empty string when it contains none.

    `backbone` is the training bank's own recorded backbone, and it GATES the
    rung-to-baseline mapping. Emitting "rung A0 on the `clipl` bank" for a row
    named `a0` that was trained on anything else writes a false sentence into
    the results file a report writer copies numbers out of -- R38/I3 inverted,
    manufacturing the mislabelling it exists to prevent. A0 on another bank is
    rung A0, not UnivFD, and gets no footnote.
    """
    lines = []
    for name in row_names:
        key = str(name).strip().lower()
        baseline = key if key in BASELINE_ROW_LABELS else None
        if baseline is None and RUNG_IS_A_BASELINE.get(key) is not None:
            if backbone != UNIVFD_BACKBONE:
                continue
            baseline = RUNG_IS_A_BASELINE[key]
        if baseline is None:
            continue
        lines.append(f"- Row `{name}` quoted as a baseline must be labelled "
                     f"**{BASELINE_ROW_LABELS[baseline]}**: "
                     f"{BASELINE_ROW_FOOTNOTES[baseline]}")
    if not lines:
        return ""
    return ("\n## Baseline row labels (spec §6.3, ruling R38)\n\n"
            + "\n".join(lines) + "\n")


def append_footnotes(path: str, text: str) -> None:
    """Append `text` to `path`, as UTF-8.

    `encoding="utf-8"` is not decoration: the footnote body contains `spec
    §6.3`, and a bare `open(path, "a")` encodes through the locale codec --
    under LC_ALL=C (the default in many container and CI images, and Kaggle is
    in this project's critical path) that is ANSI_X3.4-1968 and the append dies
    with UnicodeEncodeError, after the table has already been written.
    """
    if not text:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def _selection_summary(scores, splits) -> dict:
    """The §6.4 selection metric, with the declarations that make it checkable.

    The number and its provenance travel together: `errors._check_provenance`
    can only refuse a contaminated result that SAYS what population, split set
    and operating point it came from. Every rung goes through here -- the
    trained ones in the loop below and the fused A5 row after it -- so a rung
    that is not a training config cannot end up in `selection.json` carrying
    the metric without the declarations.

    (The brief called this `_selection_metric`; it returns the whole declared
    block rather than the bare float, because a bare float assigned into
    `summary[rung]` is exactly the shape `select_headline` refuses.)
    """
    return {
        # target_fpr passed EXPLICITLY. At the default it is the same call;
        # written out, a change to the operating point is visible in the
        # diff and is contradicted by the declaration two lines below.
        SELECTION_METRIC: heldout_robust_tpr(
            scores, splits, target_fpr=SELECTION_TARGET_FPR),
        "target_fpr": SELECTION_TARGET_FPR,
        "population": SELECTION_POPULATION,
        "splits": list(SELECTION_SPLITS),
    }


def fusion_base_config(config_paths, base_rung: str = FUSION_BASE_RUNG) -> str:
    """The rung config the fusion partner is trained from.

    A5 is "A3 + a second backbone", so its two halves must be the same training
    recipe on two different banks. Reading A3's own config file rather than
    accepting a separate one for the partner makes that structural: there is no
    second file to drift.
    """
    for path in config_paths:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if str(raw.get("name", "")).strip().lower() == base_rung:
            return path
    raise ValueError(
        f"--fuse-bank asks for rung {FUSION_RUNG.upper()}, which fuses two "
        f"{base_rung.upper()} heads, but none of the --rungs configs "
        f"{list(config_paths)} is named {base_rung!r}. Add it, or drop the "
        "fusion flags.")


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
    # `ece` is excluded at PARSE time, not left to fail after the ladder has
    # trained: this script produces scores, never calibrated probabilities, so
    # the run could only end in an all-NaN table -- hours later.
    ap.add_argument("--metric", default="auc",
                    choices=sorted(set(METRIC_COLUMNS) - PROBABILITY_METRICS),
                    help="§6.1 metric tabulated per condition, over EVERY "
                         "scored row; the HEADLINE is selected on "
                         + SELECTION_METRIC + ", which this script computes "
                         "separately and writes into the table as its own "
                         "column. `ece` needs calibrated probabilities and is "
                         "refused here: this script produces scores, not "
                         "probabilities.")
    ap.add_argument("--boot-seed", type=int, default=DEFAULT_BOOT_SEED)
    ap.add_argument("--boot-n", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--manifest", default=None,
                    help="frozen manifest.parquet; when given, each rung verifies "
                         "the training bank is still positionally aligned with it")
    ap.add_argument("--force-retrain", action="store_true",
                    help="retrain every rung even if a checkpoint exists")
    ap.add_argument("--fuse-bank", default=None,
                    help=f"rung {FUSION_RUNG.upper()}: a second, independently "
                         f"extracted TRAINING bank. Its "
                         f"{FUSION_BASE_RUNG.upper()} head is trained from the "
                         f"same config as {FUSION_BASE_RUNG.upper()} and fused "
                         "with it. Requires --fuse-eval-bank.")
    ap.add_argument("--fuse-eval-bank", default=None,
                    help="the eval bank matching --fuse-bank, over the same "
                         "frozen manifest and the same condition axis as "
                         "--eval-bank")
    ap.add_argument("--tta", action="store_true",
                    help=f"record rung {TTA_RUNG.upper()}'s test-time "
                         f"augmentation cost ({len(TTA_VIEWS)} views) and the "
                         "tier it applies to; A6 is inference-only and is not "
                         "scored from the cached eval bank")
    return ap


def _make_stdout_encoding_safe() -> None:
    """Never let a log line kill a run that has already done the work.

    Several messages below quote spec sections (`§`), and error strings from
    `eval.errors` do too. Python encodes stdout with the LOCALE codec and
    `errors="strict"`; under LC_ALL=C -- the default in many container and CI
    images -- that is ASCII, and a single `print` raises UnicodeEncodeError
    after the artefacts are already on disk. stderr already defaults to
    `backslashreplace` for exactly this reason; this gives stdout the same
    treatment rather than making the messages illegible to avoid the codec.
    """
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, OSError):      # not a reconfigurable stream
        pass


def main(argv=None) -> dict:
    a = build_parser().parse_args(argv)
    _make_stdout_encoding_safe()
    heatmap_path = a.heatmap or (a.out[:-3] + ".png" if a.out.endswith(".md")
                                 else a.out + ".png")

    eval_bank = FeatureBank.open(a.eval_bank)
    # The split column is read from the EVAL BANK, never from a caller-supplied
    # list, so the §6.4 population is built from what was actually scored.
    splits = eval_bank.meta["split"].to_numpy()
    # Checked BEFORE the first rung trains. Split coverage is a property of the
    # bank alone, so an operator who points the ladder at a benchmark-only bank
    # should be told in the first millisecond, not after hours of GPU.
    check_selection_population(splits)
    backbone = FeatureBank.open(a.bank).config.get("backbone")

    # Everything the fusion needs is validated BEFORE the first rung trains,
    # for the same reason the split coverage is: a second eval bank over
    # another manifest, or a rung list with no A3 in it, is knowable now and
    # costs hours of GPU to discover afterwards.
    fuse_eval_bank, fuse_base_config = None, None
    if bool(a.fuse_bank) != bool(a.fuse_eval_bank):
        raise ValueError(
            "--fuse-bank and --fuse-eval-bank go together: the partner head is "
            "trained on the training bank and scored on the eval bank, and one "
            f"without the other cannot produce a {FUSION_RUNG} row (got "
            f"--fuse-bank={a.fuse_bank!r}, --fuse-eval-bank={a.fuse_eval_bank!r})")
    if a.fuse_bank:
        fuse_base_config = fusion_base_config(a.rungs)
        fuse_eval_bank = FeatureBank.open(a.fuse_eval_bank)
        # Checks the manifest, the condition axis and the split/label columns
        # -- NOT the backbone, which is what rung A5 varies on purpose.
        assert_fusion_parents([eval_bank, fuse_eval_bank])

    # `rung_splits` is the split column each rung's scores were built against
    # -- this run's eval bank for the trained rungs, the column the two parents
    # SHARE for the fused A5 row. It is what both §6.4's population and the
    # frozen `acc_fixed` threshold are built from, so it is recorded per rung
    # beside the scores rather than assumed to be the same list for all of them.
    per_rung, summary, banks, rung_splits = {}, {}, {}, {}
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
        # Registered per rung, not built as a comprehension after the loop.
        # The A5 block below is scored on a SECOND eval bank as well and
        # registers a `FusedEvalBank` over both; a comprehension over
        # `per_rung` would map it to this bank instead -- a false statement
        # that makes assert_banks_comparable pass on a row it never covered.
        banks[cfg.name] = eval_bank
        rung_splits[cfg.name] = splits
        summary[cfg.name] = _selection_summary(scores, splits) | {
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

    # Rung A5: fuse this run's A3 scores with an A3 head trained on a second,
    # independently extracted bank. Not a training config, so it is built here
    # rather than in the loop -- but it reaches `summary` through the same
    # `_selection_summary`, so its number carries the same declarations.
    fusion_record = {
        "rung": FUSION_RUNG, "run": False,
        "reason": "--fuse-bank/--fuse-eval-bank were not given, so rung "
                  f"{FUSION_RUNG.upper()} was not evaluated in this run",
    }
    if fuse_eval_bank is not None:
        cfg2 = load_rung_config(fuse_base_config, a.fuse_bank, a.out_dir,
                                a.device, a.manifest)
        cfg2.name = FUSION_PARTNER_NAME
        partner, partner_resumed = train_or_resume(cfg2, force=a.force_retrain)
        if partner_resumed:
            print(f"SKIP {cfg2.name}: reusing the existing checkpoint at "
                  f"{partner['checkpoint']} -- NOT retrained. Pass "
                  "--force-retrain to train it again.")
        model2, _ = load_detector(partner["checkpoint"], device=a.device)
        partner_scores = score_grid(model2, fuse_eval_bank,
                                    use_recon=cfg2.use_recon, device=a.device)
        # Matched case-insensitively, the way `errors._normalise` matches rung
        # eligibility: a config named "A3" is still the A3 this fuses with.
        base_key = next(k for k in per_rung
                        if str(k).strip().lower() == FUSION_BASE_RUNG)
        # Computed ONCE, and BEFORE the fusion, because it is two things at
        # once: the population the per-condition z-score parameters are fitted
        # on, and the population the selection metric is then read on. The
        # splits are the ones the two parents SHARE -- a fused frame has no
        # single owning bank, so `fused_splits` refuses to answer at all unless
        # the parents agree on the manifest, the rows and the splits.
        rung_splits[FUSION_RUNG] = fused_splits([eval_bank, fuse_eval_bank])
        # Fitting the z-scores over the whole frame would let the ablation
        # bank's 5k benchmark subsample (spec 4.4a) set half of each parent's
        # sigma -- and sigma_1/sigma_2 IS how much each backbone votes -- so
        # the organisers' demo set would be choosing A5's fusion weights and
        # moving its 6.4 candidacy. `heldout_robust_tpr` refuses benchmark rows
        # at its own door; this is the same refusal one step earlier.
        per_rung[FUSION_RUNG] = fuse_scores(
            [per_rung[base_key], partner_scores],
            splits=rung_splits[FUSION_RUNG],
            fit_splits=FIT_SPLITS_FOR_SELECTION)
        # BOTH parents, never one of them. Registering the first bank would
        # make `assert_banks_comparable` pass on a row it never covered, which
        # is the augmentation-budget confound the check exists to prevent.
        banks[FUSION_RUNG] = FusedEvalBank([eval_bank, fuse_eval_bank])
        summary[FUSION_RUNG] = _selection_summary(
            per_rung[FUSION_RUNG], rung_splits[FUSION_RUNG]
        ) | {
            "zscore_population": per_rung[FUSION_RUNG][POPULATION_COLUMN].iloc[0],
            "fused_from": [base_key, FUSION_PARTNER_NAME],
            "partner_bank": a.fuse_bank,
            "partner_eval_bank": a.fuse_eval_bank,
            "partner_config": fuse_base_config,
            "resumed_from_checkpoint": partner_resumed,
        }
        fusion_record = {
            "rung": FUSION_RUNG, "run": True,
            "base_rung": base_key,
            "partner_bank": a.fuse_bank,
            "partner_eval_bank": a.fuse_eval_bank,
            "partner_config": fuse_base_config,
            "partner_checkpoint": partner["checkpoint"],
            "eval_banks": list(banks[FUSION_RUNG].config["fused_from"]),
            "backbones": list(banks[FUSION_RUNG].config["fused_backbones"]),
            # The label AND the splits behind it: a fused score is not a fixed
            # function of its two parents, so a reader cannot reconstruct what
            # this row means without knowing what the z-scores were fitted on.
            "zscore_population": per_rung[FUSION_RUNG][POPULATION_COLUMN].iloc[0],
            "zscore_fit_splits": list(FIT_SPLITS_FOR_SELECTION),
        }
        row = summary[FUSION_RUNG]
        print(f"{FUSION_RUNG}: {SELECTION_METRIC}={row[SELECTION_METRIC]:.4f} "
              f"(fused {base_key} with a head trained on "
              f"{a.fuse_bank})")

    # Rung A6: inference-only, so no row is produced here. What IS recorded is
    # the cost multiplier and the tier, because a cap nobody wrote down is
    # indistinguishable from a result.
    tta_record = {
        "rung": TTA_RUNG, "requested": bool(a.tta), "scored_here": False,
        "views": list(TTA_VIEWS), "cost_multiplier": len(TTA_VIEWS),
        "tier": a.tier,
        "reason": f"rung {TTA_RUNG.upper()} is test-time augmentation: it is "
                  "applied to images at inference, not to the cached eval "
                  "bank, so this script records its cost and tier and produces "
                  "no row",
    }
    if a.tta:
        print(f"{TTA_RUNG.upper()}: TTA with {len(TTA_VIEWS)} views multiplies "
              f"inference cost by {len(TTA_VIEWS)}x; evaluated on the "
              f"{a.tier}-tier subsample only, and scored from images rather "
              "than from the cached eval bank, so it has no row in this table.")

    # One bank per rung, so `robustness_table` routes them through
    # `assert_banks_comparable` and rejects an eval bank with no manifest
    # fingerprint. Each entry was filled beside the scores it describes -- in
    # the loop for the trained rungs, in the A5 block for the fused one -- so
    # it stays true now that a rung IS scored on another bank.
    # `acc_fixed` is the only tabulated metric that needs a threshold, and it
    # must come from clean VALIDATION rows -- not from the clean rows of the
    # frame being scored, which at the final-report tier is the benchmark.
    # Computed only when it is actually tabulated, so a bank with no
    # val_internal generated rows can still produce an AUC table.
    thresholds = None
    if a.metric in THRESHOLD_METRICS:
        thresholds = {rung: clean_validation_threshold(scores, rung_splits[rung])
                      for rung, scores in per_rung.items()}

    # The §6.4 number travels IN the table, under the rule's own name. The
    # per-condition columns are §6.1 reporting metrics over every scored row
    # and rank the rungs differently; without this column a reader picks the
    # headline off `robust_<metric>` and contradicts `selection.json`.
    table = robustness_table(per_rung, tier=a.tier, metric=a.metric,
                             seed=a.boot_seed, n_boot=a.boot_n, banks=banks,
                             selection={rung: summary[rung][SELECTION_METRIC]
                                        for rung in per_rung},
                             clean_threshold=thresholds)

    _ensure_parent(a.out)
    to_markdown(table, tier=a.tier, path=a.out)
    append_footnotes(a.out, baseline_footnotes(table.index, backbone=backbone))
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
    # §6.4's eligible range is a3-a6. A rung that was not run is recorded as
    # not run, so an absent A5 or A6 row is never read as an A5 or A6 that lost.
    report["fusion"] = fusion_record
    report["tta"] = tta_record
    _ensure_parent(a.selection)
    with open(a.selection, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"headline model: {report['headline']} "
          f"(rule: {SELECTION_METRIC} @ {fpr_label(SELECTION_TARGET_FPR)} FPR "
          f"over {SELECTION_POPULATION})")
    # On stdout, beside the headline. The IneligibleRungWarning goes to stderr
    # and is easy to lose in a multi-hour log; the exclusion belongs where the
    # choice it constrains is read.
    if report["excluded_as_ineligible"]:
        print(f"excluded as ineligible under §6.4 (candidates are "
              f"{report['eligible_rungs']}): "
              f"{report['excluded_as_ineligible']}")
    if report["headline_error"]:
        print(f"headline not selected: {report['headline_error']}")
    return report


if __name__ == "__main__":
    main()
