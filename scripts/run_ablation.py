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

import numpy as np
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
    FIT_SPLITS_FOR_SELECTION, FIT_SPLITS_WHEN_FITTING_WEIGHT, POPULATION_COLUMN,
    FusedEvalBank, assert_fusion_parents, fit_fusion_weight, fuse_scores,
    fused_splits,
)
from aigcdet.eval.grid import (
    TtaEvalBank, assert_heldout_not_trained, assert_tta_bank_matches,
    score_grid, score_grid_tta, tta_axis,
)
from aigcdet.eval.report import (
    DEFAULT_BOOT_SEED, DEFAULT_N_BOOT, METRIC_COLUMNS, PROBABILITY_METRICS,
    THRESHOLD_METRICS, TIER_CONDITIONS, clean_validation_threshold,
    CHALLENGE_ROBUST_CONDITIONS, CHALLENGE_WEIGHTS, challenge_score,
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
#: Config keys a stored checkpoint may disagree with this run about without
#: the checkpoint being a different model. The first three are where the
#: artefact lives and what hardware wrote it. `tta` is here on a stronger
#: claim: it is applied at inference and `train_rung` never reads it, so a
#: checkpoint trained before the field existed is the same model as one
#: trained after -- and refusing to resume over it would orphan every head on
#: disk for a flag that provably did not touch their weights.
#: Which rung each TTA rung is the inference-time variant OF. A6 is its base
#: rung's head scored differently, so the base is what its score must be read
#: against and what its WEIGHTS must equal. Kept here rather than inferred from
#: the config files: two rungs can differ by one flag without one being the
#: other's TTA variant, and guessing wrong would print a reassuring zero for a
#: comparison nobody asked for.
TTA_BASE_RUNG: dict[str, str] = {"a6": "a3"}


def _fit_tta_temperature(scores, splits) -> dict:
    """Refit a global temperature on one rung's logits over `val_internal`.

    Returns the fitted `T`, the logit spread it was fitted on, and the row
    count -- or `T: None` and the error, because a failed calibration fit must
    not take down a ladder whose scores are already computed and whose
    selection metric does not depend on it.
    """
    from aigcdet.calibrate import INTERNAL_VAL_SPLIT
    from aigcdet.calibrate.temperature import GlobalTemperature

    # The row's OWN split label, carried through the mask rather than
    # reconstructed as `np.full(n, "val_internal")`. `check_fit_split` exists
    # to refuse a caller's promise in place of evidence, and handing it a
    # constant array I just built would satisfy it with exactly the promise it
    # is there to reject.
    split_of = np.asarray(splits)[scores["image_idx"].to_numpy()]
    keep = split_of == INTERNAL_VAL_SPLIT
    rows, fit_split = scores[keep], split_of[keep]
    lg = rows["score"].to_numpy().astype(np.float64)
    out = {"fit_split": INTERNAL_VAL_SPLIT, "n_rows": int(len(rows)),
           "logit_sd": float(lg.std()) if len(lg) else 0.0}
    if len(rows) == 0 or len(np.unique(rows["label"].to_numpy())) < 2:
        return out | {"T": None, "error": "no two-class val_internal rows"}
    try:
        t = GlobalTemperature().fit(
            lg, rows["label"].to_numpy().astype(np.float64), split=fit_split)
        return out | {"T": float(t.temperature)}
    except Exception as exc:                       # calibration is a diagnostic
        return out | {"T": None, "error": f"{type(exc).__name__}: {exc}"}


RESUME_IGNORED_KEYS: tuple[str, ...] = ("device", "out_dir", "manifest_path", "tta")


def rung_paths(out_dir: str, name: str) -> tuple[str, str]:
    d = os.path.join(out_dir, name)
    return os.path.join(d, CHECKPOINT_NAME), os.path.join(d, RESULT_NAME)


def load_rung_config(config_path: str, bank_dir: str, out_dir: str, device: str,
                     manifest_path: str | None = None,
                     train_exclude_generators=None) -> RungConfig:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # A lineage holdout is a property of the RUN, not of any one rung: every
    # rung in the table must train on the same rows or the table compares
    # corpora rather than rungs. So the CLI value overrides whatever a rung
    # file says, and a rung file that disagrees is reported rather than
    # silently losing.
    if train_exclude_generators:
        was = raw.get("train_exclude_generators")
        if was and sorted(map(str, was)) != sorted(map(str, train_exclude_generators)):
            raise SystemExit(
                f"{config_path} sets train_exclude_generators={list(was)} but "
                f"the run passed {list(train_exclude_generators)}. One table "
                "cannot hold rungs trained on different corpora.")
        raw["train_exclude_generators"] = list(train_exclude_generators)
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
    ap.add_argument("--train-exclude-generators", default="",
                    help="comma-separated generator families to drop from "
                         "EVERY rung's training rows. Use with "
                         "build_eval_manifest --extra-heldout-generators to "
                         "score a whole lineage as unseen; the two must name "
                         "the same families or the run is refused.")
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
    ap.add_argument("--fit-fuse-weight", action="store_true",
                    help=f"rung {FUSION_RUNG.upper()}: choose the two parents' "
                         "weights by maximising val_robust_tpr on val_internal "
                         "alone, instead of weighting them equally. The "
                         "held-out number is still read once, after the weight "
                         "is fixed. Records the whole sweep in --selection.")
    ap.add_argument("--tta-eval-bank", default=None,
                    help=f"rung {TTA_RUNG.upper()}: an eval bank extracted with "
                         f"`--tta-views` over the SAME manifest and condition "
                         f"axis as --eval-bank, whose view axis is "
                         f"condition x tta_view. Required by any rung config "
                         f"with `tta: true`; its {len(TTA_VIEWS)} views are "
                         "averaged in logit space to one score per condition.")
    ap.add_argument("--tta", action="store_true",
                    help=f"record rung {TTA_RUNG.upper()}'s inference cost "
                         f"({len(TTA_VIEWS)} views) and the tier it applies "
                         "to, without scoring it. Kept for a run that wants "
                         "the cost footnote and no A6 row; pass "
                         "--tta-eval-bank instead to actually score A6.")
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
    train_bank = FeatureBank.open(a.bank)
    backbone = train_bank.config.get("backbone")

    # The other half of a lineage holdout, checked before the first rung
    # trains. `build_eval_manifest --extra-heldout-generators` can promote a
    # family into the eval bank's `heldout_generator` split so there is
    # something to score; that only MEANS anything if the same family was kept
    # out of training. Do one without the other and the headline is a score on
    # families the head trained on, in range and plausible and inflated.
    exclude = [g.strip() for g in a.train_exclude_generators.split(",")
               if g.strip()]
    assert_heldout_not_trained(train_bank, eval_bank, exclude)
    if exclude:
        print(f"excluding {exclude} from every rung's training rows")

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
    # Opened before the first rung trains, for the same reason the fusion bank
    # is: a missing or mis-axed TTA bank is knowable now and costs a full
    # ladder of GPU to discover afterwards.
    tta_eval_bank = None
    if a.tta_eval_bank:
        tta_eval_bank = FeatureBank.open(a.tta_eval_bank)
        assert_tta_bank_matches(eval_bank, tta_eval_bank)

    per_rung, summary, banks, rung_splits = {}, {}, {}, {}
    # Checkpoint paths per rung, so the A6 block below can state -- from the
    # weights themselves rather than from a comment -- that its head is its
    # base rung's head.
    checkpoints, tta_scored = {}, []
    for config_path in a.rungs:
        cfg = load_rung_config(config_path, a.bank, a.out_dir, a.device,
                               a.manifest, exclude)
        result, resumed = train_or_resume(cfg, force=a.force_retrain)
        if resumed:
            print(f"SKIP {cfg.name}: reusing the existing checkpoint at "
                  f"{result['checkpoint']} -- NOT retrained. Pass --force-retrain "
                  "to train it again.")
        model, _ = load_detector(result["checkpoint"], device=a.device)
        # A6 is scored on its OWN bank, whose view axis is condition x
        # tta_view, and the eight views are collapsed in logit space before a
        # number leaves `score_grid_tta`. Everything after this line sees a
        # frame of exactly the shape `score_grid` returns.
        if cfg.tta:
            if tta_eval_bank is None:
                raise SystemExit(
                    f"rung {cfg.name!r} sets tta: true but no --tta-eval-bank "
                    "was given. A6 cannot be scored from the plain eval bank: "
                    "that bank holds one view per condition, and averaging "
                    "over the axis it does have would average over CONDITIONS "
                    "-- a number that looks like a robustness score and is not "
                    "one. Extract the TTA bank with "
                    "`scripts/extract_eval_bank.py --tta-views`.")
            scores = score_grid_tta(model, tta_eval_bank,
                                    use_recon=cfg.use_recon,
                                    use_recon_vq=cfg.use_recon_vq,
                                    use_freq=cfg.use_freq, device=a.device)
            tta_scored.append(cfg.name)
        else:
            scores = score_grid(model, eval_bank, use_recon=cfg.use_recon,
                                use_recon_vq=cfg.use_recon_vq,
                                use_freq=cfg.use_freq, device=a.device)
        per_rung[cfg.name] = scores
        # Registered per rung, not built as a comprehension after the loop.
        # The A5 block below is scored on a SECOND eval bank as well and
        # registers a `FusedEvalBank` over both; a comprehension over
        # `per_rung` would map it to this bank instead -- a false statement
        # that makes assert_banks_comparable pass on a row it never covered.
        banks[cfg.name] = TtaEvalBank(tta_eval_bank) if cfg.tta else eval_bank
        checkpoints[cfg.name] = result["checkpoint"]
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
        # The partner head trains on a SECOND bank, so it needs the same
        # exclusion and the same check. A5 fusing a clean head with a
        # contaminated one is still contaminated, and the fused number carries
        # no mark of which half did it.
        assert_heldout_not_trained(FeatureBank.open(a.fuse_bank),
                                   fuse_eval_bank, exclude)
        cfg2 = load_rung_config(fuse_base_config, a.fuse_bank, a.out_dir,
                                a.device, a.manifest, exclude)
        cfg2.name = FUSION_PARTNER_NAME
        partner, partner_resumed = train_or_resume(cfg2, force=a.force_retrain)
        if partner_resumed:
            print(f"SKIP {cfg2.name}: reusing the existing checkpoint at "
                  f"{partner['checkpoint']} -- NOT retrained. Pass "
                  "--force-retrain to train it again.")
        model2, _ = load_detector(partner["checkpoint"], device=a.device)
        partner_scores = score_grid(model2, fuse_eval_bank,
                                    use_recon=cfg2.use_recon,
                                    use_recon_vq=cfg2.use_recon_vq,
                                    use_freq=cfg2.use_freq,
                                    device=a.device)
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
        # THE WEIGHT, and the default is not innocent.
        #
        # Measured on the frozen corpus (docs/selection_a5*.json): dinov3l
        # alone takes a3 0.9012, and fusing it 50/50 with siglip2l -- which
        # alone takes 0.2893 -- gives a5 0.8773. WORSE than the strong parent
        # by itself. convnextt, alone 0.4882, costs only 0.0152. The penalty
        # tracks how weak the partner is, which is what equal weighting does to
        # unequal parents and is a property of the constant rather than a
        # finding about ensembling.
        #
        # --fit-fuse-weight lets a weak partner fall to a small weight and
        # contribute only where it actually helps. `fit_fusion_weight` chooses
        # it on val_internal alone, so the held-out number below is still read
        # exactly once, after w is fixed.
        # THE Z-SCORE POPULATION MOVES WITH THE WEIGHT, and it has to.
        # FIT_SPLITS_FOR_SELECTION includes heldout_generator rows, which is
        # defensible while the weights are a fixed constant -- the
        # standardisation sees those rows but nothing is CHOSEN from what it
        # sees. Fitting a weight breaks that, because the sweep's objective is
        # computed on scores whose scale the held-out rows helped set. So a
        # fitted run standardises on val_internal alone.
        fuse_weights, fuse_sweep = None, None
        fuse_fit_splits = FIT_SPLITS_FOR_SELECTION
        if a.fit_fuse_weight:
            fuse_fit_splits = FIT_SPLITS_WHEN_FITTING_WEIGHT
            fuse_weights, fuse_sweep = fit_fusion_weight(
                [per_rung[base_key], partner_scores],
                rung_splits[FUSION_RUNG],
                fit_splits=fuse_fit_splits)
            print(f"{FUSION_RUNG.upper()}: weight fitted on "
                  f"{'+'.join(fuse_fit_splits)} alone -- "
                  f"w({base_key})={fuse_weights[0]:.2f} "
                  f"w({FUSION_PARTNER_NAME})={fuse_weights[1]:.2f}")
        per_rung[FUSION_RUNG] = fuse_scores(
            [per_rung[base_key], partner_scores],
            weights=fuse_weights,
            splits=rung_splits[FUSION_RUNG],
            fit_splits=fuse_fit_splits)
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
            # "equal" is recorded explicitly rather than left absent: a reader
            # comparing two a5 rows has to be able to tell a fitted 0.50 from
            # the default, and an absent key reads as neither.
            "fusion_weights": list(fuse_weights) if fuse_weights else "equal",
            "weight_fitted_on": (list(FIT_SPLITS_WHEN_FITTING_WEIGHT)
                                 if fuse_weights else None),
            # The whole sweep, not just the argmax: a flat objective and a
            # sharply peaked one justify very different confidence in the
            # same w, and only the sweep distinguishes them.
            "weight_sweep": fuse_sweep,
        }
        # THE CONTROL THAT MAKES THE COMPARISON READABLE.
        #
        # A fitted run differs from the recorded a5 baselines
        # (docs/selection_a5*.json, both standardised on
        # heldout_generator+val_internal) in TWO ways: the weight, and the
        # z-score population -- and the population HAD to move, for the leakage
        # reason above. Reporting only the fitted number would leave a reader
        # unable to say which change did the work.
        #
        # So when fitting, also score the EQUAL-weight fusion under the SAME
        # standardisation. The gap from that to the fitted number is the
        # weight's contribution with nothing else moving. It goes in the
        # fusion record rather than in `summary` because it is a diagnostic,
        # not a rung, and anything in `summary` is a §6.4 selection candidate.
        equal_weight_control = None
        if a.fit_fuse_weight:
            equal_weight_control = _selection_summary(
                fuse_scores([per_rung[base_key], partner_scores],
                            splits=rung_splits[FUSION_RUNG],
                            fit_splits=fuse_fit_splits),
                rung_splits[FUSION_RUNG])

        fusion_record = {
            "rung": FUSION_RUNG, "run": True,
            "equal_weight_control": equal_weight_control,
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
            # The splits ACTUALLY used, not the default constant: under
            # --fit-fuse-weight this is FIT_SPLITS_WHEN_FITTING_WEIGHT, and
            # recording the constant here contradicted `zscore_population`
            # two lines above -- the one field a reader checks to rule out
            # the very leak that flag exists to prevent.
            "zscore_fit_splits": list(fuse_fit_splits),
        }
        row = summary[FUSION_RUNG]
        print(f"{FUSION_RUNG}: {SELECTION_METRIC}={row[SELECTION_METRIC]:.4f} "
              f"(fused {base_key} with a head trained on "
              f"{a.fuse_bank})")

    # Rung A6: inference-only, so no row is produced here. What IS recorded is
    # the cost multiplier and the tier, because a cap nobody wrote down is
    # indistinguishable from a result.
    tta_views_used = tta_axis(tta_eval_bank) if tta_eval_bank is not None else []
    tta_record = {
        "rung": TTA_RUNG,
        "requested": bool(a.tta) or bool(tta_scored),
        "scored_here": bool(tta_scored),
        "scored_rungs": list(tta_scored),
        "views": list(tta_views_used or TTA_VIEWS),
        "cost_multiplier": len(tta_views_used or TTA_VIEWS),
        "tier": a.tier,
        "eval_bank": a.tta_eval_bank,
    }
    if not tta_scored:
        tta_record["reason"] = (
            f"no rung in this run set `tta: true`, so rung {TTA_RUNG.upper()} "
            "has no row here. Its cost is recorded because a cap nobody wrote "
            "down is indistinguishable from a result.")
        if a.tta:
            print(f"{TTA_RUNG.upper()}: TTA with {len(TTA_VIEWS)} views "
                  f"multiplies inference cost by {len(TTA_VIEWS)}x on the "
                  f"{a.tier}-tier subsample. No A6 row: pass --tta-eval-bank "
                  "and a rung config with `tta: true` to score it.")
    else:
        tta_record["reason"] = (
            f"scored from {a.tta_eval_bank}, whose view axis is "
            "condition x tta_view; the views are averaged in LOGIT space to "
            "one score per condition, so the row covers the same conditions "
            "over the same images as every other rung, at "
            f"{len(tta_views_used)}x the inference cost.")

        # The one-flag claim, verified on the WEIGHTS rather than on the two
        # config files. `tta` is inference-only, so an A6 head trained from the
        # same seed on the same rows must be its base rung's head exactly; if
        # it is not, something in training read the flag after all and A6's
        # score is not a measurement of TTA.
        deltas = {}
        for name in tta_scored:
            base = TTA_BASE_RUNG.get(name)
            if base is None or base not in checkpoints:
                continue
            wa = torch.load(checkpoints[name], map_location="cpu",
                            weights_only=True)["state_dict"]
            wb = torch.load(checkpoints[base], map_location="cpu",
                            weights_only=True)["state_dict"]
            if set(wa) != set(wb):
                deltas[f"{name}_vs_{base}"] = "different parameter sets"
                continue
            d = max(float((wa[k].double() - wb[k].double()).abs().max())
                    for k in wa) if wa else 0.0
            deltas[f"{name}_vs_{base}"] = d
            # Stated as a measurement plus its two readings, not as a verdict.
            # Zero is unambiguous. Non-zero has a second explanation besides
            # "training read the flag" -- GPU training is not bit-reproducible
            # in general -- and the two are told apart by MAGNITUDE, not by
            # this line: a nondeterministic replay differs in the last bits,
            # while a head that actually trained differently diverges by
            # something you can see in the third decimal.
            if d == 0.0:
                verdict = "identical head; TTA is the only difference"
            elif d < 1e-5:
                verdict = ("last-bit difference, consistent with "
                           "nondeterministic training rather than the flag")
            else:
                verdict = ("NOT the same head -- too large for training "
                           "nondeterminism, so A6's score is not a measurement "
                           "of TTA alone. Check whether training reads `tta`.")
            print(f"{name}: max|w_{name} - w_{base}| = {d:.3e} ({verdict})")
        tta_record["weight_delta_vs_base"] = deltas

        # THE TEMPERATURE TRAP (`aigcdet.eval.tta` module docstring), discharged
        # with a number rather than deferred to whoever ships A6.
        #
        # A mean of eight correlated logits has a materially narrower spread
        # than the single-view logits a temperature was fitted on, so reusing
        # that T leaves A6 systematically under-confident -- and the inference
        # clamp cannot catch it, because the clamp is keyed on the CONDITION
        # and the condition has not moved, only the spread.
        #
        # Recorded, NOT applied. This script produces scores and never
        # calibrated probabilities, and selection is invariant to it either
        # way: TPR at a fixed FPR is computed within a condition and a
        # temperature is a monotone rescale. What is at stake is the shipped
        # bundle's calibration, EQI and dashboard, and those read this number.
        for name in tta_scored:
            base = TTA_BASE_RUNG.get(name)
            fitted = _fit_tta_temperature(per_rung[name], rung_splits[name])
            entry = dict(fitted)
            if base in per_rung:
                single = _fit_tta_temperature(per_rung[base], rung_splits[base])
                entry["base_rung"] = base
                entry["base_T_single_view"] = single["T"]
                entry["base_logit_sd"] = single["logit_sd"]
                if single["logit_sd"]:
                    entry["spread_ratio_vs_base"] = (
                        fitted["logit_sd"] / single["logit_sd"])
            tta_record.setdefault("temperature", {})[name] = entry
            if entry.get("T") is not None:
                against = ""
                if entry.get("base_T_single_view") is not None:
                    ratio = entry.get("spread_ratio_vs_base", float("nan"))
                    against = (f" (base {base} single-view "
                               f"T={entry['base_T_single_view']:.4f}, logit sd "
                               f"ratio {ratio:.3f})")
                print(f"{name}: temperature REFITTED on TTA-averaged logits "
                      f"over val_internal, T={entry['T']:.4f}{against} -- "
                      "recorded, not applied to any score in this table.")
            else:
                print(f"{name}: temperature refit did not converge "
                      f"({entry.get('error')}); recorded as null.")

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
    # The organisers' announced score (28 Aug webinar):
    # `0.50 x AUC_clean + 0.50 x AUC_robust`. It is a REPORTING number and is
    # recorded beside the §6.4 headline, never in place of it -- the two are
    # computed over different populations and rank the rungs differently, and
    # a disagreement between them is a finding worth reading on the day rather
    # than on submission night. `challenge_score` refuses a non-AUC table, so
    # the guard here is the same condition stated once.
    if a.metric == "auc":
        cs = challenge_score(table)
        report["challenge_score"] = {
            "weights": {"clean": CHALLENGE_WEIGHTS[0],
                        "robust": CHALLENGE_WEIGHTS[1]},
            # Written out because the robust half is the brief's required
            # transforms only -- NOT the table's `robust_auc`, which at this
            # tier also averages the five composed scenarios this project
            # invented. A reader cannot check that from the number alone.
            "robust_conditions": list(CHALLENGE_ROBUST_CONDITIONS),
            "per_rung": {str(r): {k: float(v) for k, v in cs.loc[r].items()}
                         for r in cs.index},
            "best": str(cs["challenge_score"].idxmax()),
        }
    else:
        report["challenge_score"] = {
            "computed": False,
            "reason": f"the table holds {a.metric!r}; the announced score is "
                      "defined on ROC AUC. Re-run with --metric auc.",
        }
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
    cs_record = report["challenge_score"]
    if cs_record.get("per_rung"):
        best = cs_record["best"]
        print(f"challenge score (0.50 clean + 0.50 robust, reporting only): "
              f"best {best} at "
              f"{cs_record['per_rung'][best]['challenge_score']:.4f}")
        if report["headline"] and best != report["headline"]:
            # Printed, not resolved. The two rules answer different questions
            # and the project reports both; silently preferring either is how
            # a submission ships a model chosen by a rule nobody wrote down.
            print(f"NOTE: the announced score prefers {best}, the §6.4 rule "
                  f"chose {report['headline']}. Both are recorded in "
                  f"{a.selection}.")
    if report["headline_error"]:
        print(f"headline not selected: {report['headline_error']}")
    return report


if __name__ == "__main__":
    main()
