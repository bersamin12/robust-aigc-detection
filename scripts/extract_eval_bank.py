"""Build the evaluation bank the robustness table is scored from (spec §4.4a, §6.1).

`scripts/run_ablation.py --eval-bank` has always documented "a bank written by
`eval.grid.extract_eval_bank`", and until this script existed nothing could
produce one. This is that producer.

    # ablation / model-selection tier: 20 conditions, 5k benchmark subsample
    python scripts/extract_eval_bank.py --manifest data/manifest.parquet \
        --backbone dinov3l --out banks/eval_dinov3l --tier ablation

    # final-report tier: the complete benchmark, 15 core conditions, day 6
    python scripts/extract_eval_bank.py --manifest data/manifest.parquet \
        --backbone dinov3l --out banks/eval_final_dinov3l --tier final_report

**The tier is the argument, and it decides three things at once** (§4.4a's
two-tier cap): the condition axis (`report.TIER_CONDITIONS`), the splits the
bank must carry, and the row budget. Choosing them separately is how a
selection-tier number gets quoted as a final-report one, so `--tier` sets all
three and records itself in `config.json` -- `report._check_banks` then checks
the bank's condition list against the table's tier instead of trusting a
human.

**The subsample is applied to the MANIFEST, before extraction, and only to the
benchmark rows.** Subsampling afterwards would pay the full ~13,800-image
extraction the cap exists to avoid. Subsampling the whole frame instead of the
benchmark rows would let the ablation tier's cap eat into the held-out
generator families, and `eval.errors.heldout_robust_tpr` then refuses the
entire rung. The `(budgets, seed)` actually applied are written into the
bank's config, because the subsample changes `manifest_fingerprint` -- so
`report._check_banks` will correctly refuse to compare a subsampled ablation
bank against a full final-report one, but nothing else records WHICH
subsample was taken.

**Portability, for the Kaggle fleet.** `--root` (or `$AIGCDET_DATA_ROOT`)
rebases the frozen manifest onto wherever the Dataset is actually attached,
and the bank records `manifest_root` so each row keeps the portable identity
that lets a shard extracted on Kaggle merge with one extracted anywhere else.
Before paying for the extraction, run the pre-flight -- this script runs the
presence half of `aigcdet.data.verify.verify_images` itself and aborts if the
mount is wrong; the content half is `--verify`, or, standalone:

    python -m aigcdet.data.verify --manifest data/manifest.parquet \
        --root /kaggle/input/<slug>/data --sample 2000

**Sharding across a five-person fleet.** `--shard I/N` takes a contiguous,
disjoint block of the (already subsampled) manifest. Every view's pixels
depend only on `(seed, row_id, view_idx)`, so the shards are byte-identical to
one uninterrupted run and `scripts/merge_banks.py` puts them back together.
All five teammates must pass the same `--tier`/`--subsample`/`--seed`;
`merge_banks` refuses shards whose config disagrees. `--shard` and `--limit`
are deliberately NOT recorded in the bank config: they are per-session facts,
and recording them would make `merge_banks` refuse the very shards it exists
to combine.

    # teammate k of 5, resumable across Kaggle session timeouts
    python scripts/extract_eval_bank.py --manifest /kaggle/input/aigc/manifest.parquet \
        --root /kaggle/input/aigc/data --backbone dinov3l \
        --out /kaggle/working/eval_shard0 --tier ablation --shard 0/5 --resume
    python scripts/merge_banks.py --out banks/eval_dinov3l \
        banks/eval_shard0 banks/eval_shard1 banks/eval_shard2 \
        banks/eval_shard3 banks/eval_shard4
"""
from __future__ import annotations

import argparse
from collections import namedtuple

import numpy as np
import pandas as pd

from aigcdet.augment.canonical import (
    CANON_CROP_SIDE, MODE_BAND, MODES, CanonPolicy)
from aigcdet.augment.scenarios import EVAL_GRID
from aigcdet.data.manifest import SPLITS, read_manifest
from aigcdet.data.verify import verify_images
from aigcdet.eval.grid import BENCHMARK_SEED, extract_eval_bank, stratified_subsample
from aigcdet.eval.report import TIER_CONDITIONS
from aigcdet.eval.tta import TTA_VIEWS
from aigcdet.features.bank import CHECKPOINT_EVERY, FeatureBank
from aigcdet.features.extract import shard_bounds

#: §4.4a's benchmark cap for the ablation/selection tier.
BENCHMARK_SUBSAMPLE_N = 5000


#: The splits and row budgets a tier is defined over.
#:
#: The condition axis is NOT here: it is `report.TIER_CONDITIONS`, the mapping
#: `report._check_banks` and the render-time tier gate already read.
#: Duplicating it would let this script write a bank whose condition list no
#: rendering of that tier could accept.
#:
#: A `collections.namedtuple` rather than a dataclass: `tests/scripts` loads
#: these scripts through `importlib` without registering them in
#: `sys.modules`, and `@dataclass` resolves its field annotations through
#: `sys.modules[cls.__module__]` -- which is None for such a module.
TierPlan = namedtuple("TierPlan", "splits subsample")


#: What each tier means in rows. The ablation tier carries `val_internal` AND
#: `heldout_generator` because §6.4's selection population is built from both
#: (`errors.SELECTION_SPLITS`); `run_ablation.py` calls
#: `check_selection_population` on the bank's split column in its first
#: millisecond, so a bank missing either is refused before a rung trains.
#:
#: `heldout_generator` is deliberately UNCAPPED. It is the small held-out
#: family pool, it is one half of the selection population, and starving it is
#: how a rung comparison becomes "whichever rung had enough held-out fakes
#: left to score". The benchmark side is the one §4.4a caps, and the one whose
#: 13.8k rows are the cost.
TIER_PLANS: dict[str, TierPlan] = {
    "ablation": TierPlan(
        splits=("val_internal", "heldout_generator", "benchmark"),
        subsample={"benchmark": BENCHMARK_SUBSAMPLE_N}),
    "final_report": TierPlan(splits=("benchmark",), subsample={}),
    # `smoke` has no fixed coverage in TIER_CONDITIONS either -- it is the
    # three-condition run, and its rows are whatever fixture is at hand.
    "smoke": TierPlan(splits=(), subsample={}),
}


def resolve_conditions(tier: str, names_arg: str = "") -> dict:
    """The ordered condition mapping this tier's bank must be extracted over.

    `report._check_banks` compares the bank's recorded condition list against
    the table's tier coverage with LIST equality, so a bank whose conditions
    merely overlap the tier's is unrenderable as that tier. Rejecting that
    here costs a second; discovering it after the extraction costs the
    extraction.
    """
    if tier not in TIER_CONDITIONS:
        raise ValueError(f"--tier must be one of {sorted(TIER_CONDITIONS)}, "
                         f"got {tier!r}")
    coverage = TIER_CONDITIONS[tier]
    wanted = [c.strip() for c in str(names_arg).split(",") if c.strip()]
    if not wanted:
        if coverage is None:
            raise ValueError(
                f"--tier {tier} defines no fixed condition coverage "
                "(TIER_CONDITIONS maps it to None, meaning 'any subset'), so "
                "--conditions must name them explicitly, e.g. "
                "--conditions clean,jpeg_q70,blur_s1.0")
        wanted = list(coverage)
    unknown = [c for c in wanted if c not in EVAL_GRID]
    if unknown:
        raise ValueError(
            f"--conditions names {unknown}, which are not in the evaluation "
            f"grid; it holds {list(EVAL_GRID)}")
    if coverage is not None and wanted != list(coverage):
        raise ValueError(
            f"--tier {tier} is defined over the conditions {list(coverage)} "
            f"but --conditions asks for {wanted}. report._check_banks compares "
            "the bank's condition list against the tier's coverage for list "
            "equality, so this bank could never be rendered as a "
            f"{tier} table -- and a table that relabels it as another tier is "
            "exactly the failure the tier vocabulary exists to prevent.")
    return {c: EVAL_GRID[c] for c in wanted}


def select_splits(df: pd.DataFrame, splits_arg: str) -> pd.DataFrame:
    """Filter `df` to a comma-separated list of manifest splits.

    An unknown split name is a typo that would otherwise produce an empty or
    wrong bank after the extraction has been paid for, so it raises here.

    No `.reset_index()`: `extract_eval_bank` keys each view's RNG on the frozen
    manifest's index LABEL (see `aigcdet.features.extract`'s module docstring),
    so a filtered frame must keep those labels or an independently-extracted
    shard draws different pixels for the same physical image.
    """
    wanted = [s.strip() for s in str(splits_arg).split(",") if s.strip()]
    if not wanted:
        return df
    present = sorted(set(str(s) for s in df["split"].unique()))
    unknown = [s for s in wanted if s not in present]
    if unknown:
        raise ValueError(
            f"--split names {unknown}, which the manifest does not contain; "
            f"its splits are {present}")
    return df[df["split"].astype(str).isin(wanted)]


def parse_budgets(entries) -> dict[str, int]:
    """`["benchmark=5000"]` -> `{"benchmark": 5000}`."""
    out: dict[str, int] = {}
    for entry in entries or ():
        name, sep, value = str(entry).partition("=")
        name = name.strip()
        if not sep or not name:
            raise ValueError(
                f"--subsample takes SPLIT=N, e.g. --subsample benchmark=5000; "
                f"got {entry!r}")
        try:
            n = int(value)
        except ValueError:
            raise ValueError(
                f"--subsample {entry!r}: N must be an integer") from None
        if n <= 0:
            raise ValueError(
                f"--subsample {entry!r}: N must be positive (pass "
                "--no-subsample to take every row)")
        if name not in SPLITS:
            raise ValueError(
                f"--subsample names split {name!r}, which is not a manifest "
                f"split; they are {list(SPLITS)}")
        out[name] = n
    return out


def subsample_manifest(df: pd.DataFrame, budgets: dict[str, int],
                       seed: int = BENCHMARK_SEED) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply §4.4a's stratified cap to the named splits, PER SPLIT.

    Three things about this are load-bearing.

    It runs on the manifest, BEFORE extraction. Extracting 13.8k benchmark
    images and subsampling the resulting bank pays the exact cost the cap
    exists to avoid.

    Each budget is a separate `stratified_subsample` call over that split's
    rows alone. One call over the whole frame balances class x generator x
    source across everything at once, so the two held-out generator families
    -- ~2 strata against the benchmark's many -- lose most of their rows to
    the cap, and `errors.heldout_robust_tpr` then refuses the whole rung for
    having a condition with only one class left.

    The result is `df.iloc[...]` with NO `reset_index()`. `extract_eval_bank`
    keys each row's per-view RNG on the frozen manifest's index label; a reset
    index would renumber every row from 0, so a subsampled bank's pixels would
    stop matching the full run's and two independently-extracted shards of the
    same subsample would disagree with each other.

    Returns the frame in the manifest's own row order, and how many rows each
    budget actually kept (a split smaller than its budget keeps all of it).
    """
    if not budgets:
        return df, {}
    splits = df["split"].astype(str).to_numpy()
    absent = [s for s in sorted(budgets) if not (splits == s).any()]
    if absent:
        raise ValueError(
            f"--subsample names split(s) {absent}, which the selected rows do "
            f"not contain (they hold {sorted(set(splits.tolist()))}); a budget "
            "that matches nothing silently caps nothing")

    keep = [np.where(~np.isin(splits, sorted(budgets)))[0]]
    kept: dict[str, int] = {}
    for name in sorted(budgets):
        pos = np.where(splits == name)[0]
        # Positional WITHIN this split's sub-frame; mapped back through `pos`.
        picked = stratified_subsample(df.iloc[pos], int(budgets[name]), seed=seed)
        keep.append(pos[np.asarray(picked, dtype=np.int64)])
        kept[name] = int(len(picked))
    order = np.sort(np.concatenate(keep))
    return df.iloc[order], kept


def shard_frame(df: pd.DataFrame, spec: str | None) -> pd.DataFrame:
    """`--shard I/N` -> the I-th of N contiguous, disjoint, exhaustive blocks.

    Contiguous `.iloc[a:b]`, which preserves the frozen manifest's index
    labels -- the RNG key. Applied AFTER the subsample, so all N sessions
    shard the same subsampled manifest and their row_id sets do not overlap
    (`merge_banks` refuses them if they do).

    The block boundaries come from `features.extract.shard_bounds`, the one
    partition rule in the project, rather than a local `np.linspace`. Both are
    contiguous, disjoint and exhaustive -- so every property asserted here
    held under either -- but they are not the same partition: at 120,001 rows
    over 5 shards every boundary lands one row apart, and at 1 row over 2
    shards they disagree about which shard gets the row. Nothing merges an
    eval bank with a training bank today, and nothing enforced that either.
    """
    if not spec:
        return df
    parts = str(spec).split("/")
    try:
        if len(parts) != 2:
            raise ValueError
        i, n = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(
            f"--shard takes I/N with 0 <= I < N, e.g. 0/5; got {spec!r}") from None
    if n < 1 or not (0 <= i < n):
        raise ValueError(
            f"--shard takes I/N with 0 <= I < N, e.g. 0/5; got {spec!r}")
    lo, hi = shard_bounds(len(df), n)[i]
    return df.iloc[lo:hi]


def preflight(df: pd.DataFrame, digest: str | None = None,
              sample: int | None = None, workers: int = 8):
    """`verify_images` over the rows this session is about to extract.

    The default is presence only -- one `os.path.isfile` per row, effectively
    free -- because the failure it catches is the one that actually happens on
    Kaggle: the Dataset is attached under a different slug, every path is
    wrong, and the run dies (or worse, half-succeeds) an hour in. `--verify`
    escalates to the manifest's digests, which is the check that also catches
    a re-encoded copy.
    """
    report = verify_images(df, digest=digest, sample=sample, check_extra=False,
                           workers=workers)
    print(report.describe())
    report.raise_for_status()
    return report


def describe_plan(tier: str, conditions, df: pd.DataFrame,
                  budgets: dict[str, int], kept: dict[str, int],
                  shard: str | None) -> str:
    """The row/condition plan, printed before anything expensive happens."""
    per_split = df["split"].astype(str).value_counts().sort_index()
    lines = [
        f"tier               {tier}",
        f"conditions         {len(conditions)}: {list(conditions)}",
        f"rows               {len(df)}"
        + (f"  (shard {shard})" if shard else ""),
        f"  by split         {dict(per_split)}",
        f"subsample          {budgets or 'none'}"
        + (f" -> kept {kept}" if kept else ""),
        f"forwards           {len(df) * len(conditions)}",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--out", required=True, help="directory for the eval bank")
    ap.add_argument("--tier", required=True, choices=sorted(TIER_CONDITIONS),
                    help="§4.4a evaluation tier; sets the condition axis "
                         "(report.TIER_CONDITIONS), the default splits and the "
                         "default row budget, and is recorded in the bank")
    ap.add_argument("--root", default=None,
                    help="where the dataset actually is on this machine "
                         "(Kaggle: /kaggle/input/<slug>/...); defaults to "
                         "$AIGCDET_DATA_ROOT, then to the manifest's own paths")
    ap.add_argument("--conditions", default="",
                    help="comma-separated condition names, in view order; "
                         "required for --tier smoke, and for the other tiers "
                         "must equal that tier's coverage")
    ap.add_argument("--split", default="",
                    help="comma-separated manifest splits; defaults to the "
                         "tier's own (ablation: "
                         + ",".join(TIER_PLANS["ablation"].splits) + ")")
    ap.add_argument("--subsample", action="append", metavar="SPLIT=N",
                    help="stratified cap on one split, applied to the manifest "
                         "before extraction; repeatable. Replaces the tier "
                         "default rather than adding to it.")
    ap.add_argument("--no-subsample", action="store_true",
                    help="extract every selected row, ignoring the tier's cap")
    ap.add_argument("--subsample-seed", type=int, default=BENCHMARK_SEED,
                    help="fixed and committed (spec §4.4a), so the tier is "
                         "reproducible from the constant alone")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="extract the I-th of N contiguous blocks of the "
                         "subsampled manifest; recombine with merge_banks.py")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N rows after the subsample (smoke runs only)")
    ap.add_argument("--canon-mode", choices=MODES, default=MODE_BAND,
                    help="resolution standardisation. MUST match the training "
                         "bank the rung was trained on: a robustness curve "
                         "measured on pixels the head never saw is a curve "
                         "for a different model. Deterministic here whatever "
                         "the mode -- crop takes the CENTRE window and the "
                         "band is unjittered -- so the grid measures the "
                         "condition and not a different picture.")
    ap.add_argument("--crop-side", type=int, default=CANON_CROP_SIDE,
                    help="window size for --canon-mode crop; must match the "
                         "training bank's")
    ap.add_argument("--tta-views", default="", metavar="A,B,C",
                    help="rung A6: also apply these TTA views on top of every "
                         "condition, making the view axis condition x tta_view "
                         "(flattened j*len(views)+k). Pass 'all' for the full "
                         f"set {','.join(TTA_VIEWS)}. Costs one forward per "
                         "(image, condition, view) -- 8x the plain bank -- and "
                         "produces a bank that only `score_grid_tta` can read.")
    ap.add_argument("--tower-checkpoint", default=None,
                    help="unfreeze ladder (D1..D4): load this rung's finetuned "
                         "tower before extracting. A tower whose weights moved "
                         "does not produce the features in the frozen bank, so "
                         "each depth needs its own. The weights' sha256 is "
                         "written into config.json, which is what stops two "
                         "depths' shards merging into one bank.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=BENCHMARK_SEED,
                    help="per-view RNG seed; must match across shards")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted extraction into --out, "
                         "skipping the rows already written (the same manifest, "
                         "tier, subsample, backbone and seed must be given)")
    ap.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
                    help="flush meta/views parquet every N images; this is also "
                         "how much work a session timeout can cost")
    ap.add_argument("--verify", action="store_true",
                    help="escalate the pre-flight from presence to the "
                         "manifest's content digests")
    ap.add_argument("--verify-sample", type=int, default=None,
                    help="with --verify: digest only this many evenly spaced "
                         "rows (a clean sample is evidence, not proof)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the pre-flight entirely (not recommended: the "
                         "presence check is one stat call per row)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the row/condition plan and exit, without "
                         "loading a backbone or touching --out")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)

    # Everything knowable without the data comes first: a bad --tier/--conditions
    # pairing must not cost a manifest read, let alone a backbone load.
    conditions = resolve_conditions(a.tier, a.conditions)
    plan = TIER_PLANS[a.tier]
    if a.subsample and a.no_subsample:
        raise ValueError("--subsample and --no-subsample contradict each other")
    budgets = ({} if a.no_subsample
               else parse_budgets(a.subsample) if a.subsample
               else dict(plan.subsample))

    df = read_manifest(a.manifest, root=a.root)
    df = select_splits(df, a.split or ",".join(plan.splits))
    df, kept = subsample_manifest(df, budgets, seed=a.subsample_seed)
    if a.limit is not None:
        df = df.iloc[:a.limit]
    df = shard_frame(df, a.shard)

    print(describe_plan(a.tier, conditions, df, budgets, kept, a.shard))
    if a.dry_run:
        return 0
    if not a.no_verify:
        preflight(df, digest="auto" if a.verify else None,
                  sample=a.verify_sample)

    # 'all' rather than requiring the eight names to be typed, because a
    # partial list is a real and useful thing to pass (a cheap 2-view probe)
    # and a MISTYPED one would otherwise silently produce a narrower average.
    # `extract_eval_bank` rejects an unknown name before decoding anything.
    tta_views = None
    if a.tta_views:
        tta_views = (list(TTA_VIEWS) if a.tta_views.strip() == "all"
                     else [v.strip() for v in a.tta_views.split(",") if v.strip()])

    extract_eval_bank(
        df, a.backbone, a.out, conditions=conditions, device=a.device,
        tta_views=tta_views,
        seed=a.seed, batch_size=a.batch_size, resume=a.resume,
        checkpoint_every=a.checkpoint_every,
        policy=CanonPolicy(mode=a.canon_mode, crop_side=a.crop_side),
        tower_checkpoint=a.tower_checkpoint,
        # The subsample changes `manifest_fingerprint`, which is what makes
        # `report._check_banks` refuse to compare a subsampled ablation bank
        # against a full final-report one. That refusal is right but mute: it
        # says the two banks are different, not what the difference WAS. This
        # is the record. `--shard`/`--limit` are excluded on purpose -- they
        # differ between shards of one bank, and `merge_banks` requires every
        # unrecognised config key to agree.
        extra_config={
            "tier": a.tier,
            "subsample": {"seed": int(a.subsample_seed),
                          "budgets": {k: int(v) for k, v in sorted(budgets.items())},
                          "kept": {k: int(v) for k, v in sorted(kept.items())}},
        })

    bank = FeatureBank.open(a.out)
    axis = (f"{len(conditions)} conditions x {len(tta_views)} TTA views"
            if tta_views else f"{bank.config['n_views']} conditions")
    print(f"wrote {a.out}: {len(bank.meta)} rows x {axis} "
          f"= {bank.config['n_views']} views, backbone "
          f"{bank.config['backbone']}, tier {bank.config['tier']}")
    if a.shard:
        print(f"shard {a.shard} done -- recombine every shard with "
              "scripts/merge_banks.py before scoring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
