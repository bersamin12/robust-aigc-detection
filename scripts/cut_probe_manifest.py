"""Cut a *probe manifest*: a stratified subsample of a frozen manifest.

Why this exists
---------------
Deciding between two extraction POLICIES -- band-limit standardisation against
random cropping -- requires both arms to run over the SAME rows, differing in
exactly one flag. The `coco_crop` corpus is 175,150 train+val rows, roughly
7 h of GPU per arm; two arms is a day and a half. A 20,000-row probe answers
the same question in ~45 min per arm, and both arms can run concurrently in
two Kaggle sessions.

It is deliberately a THROWAWAY identity
---------------------------------------
A probe manifest fingerprints differently from its parent -- `rel_path` in row
order is the identity `manifest_sha256` is taken over -- so a probe bank cannot
verify against, merge with, resume from, or fuse against a full-run bank. That
is the intended behaviour and not a limitation: a probe is evidence about a
POLICY, never a component of the shipping system, and the machinery that would
otherwise let one be mistaken for the other is exactly what `features/bank.py`
exists to enforce. Every refusal you meet after using one of these is correct.

The selection rule is borrowed, not reimplemented
-------------------------------------------------
`extract_eval_bank.subsample_manifest` is loaded from that script by path
rather than copied here. The eval bank already caps its splits with a
stratified, per-split, index-label-preserving sampler; the training side must
choose rows by the SAME rule or the two halves of one probe are drawn
differently. A second copy of a stratified sampler is a copy that drifts, and
the drift would be invisible -- both would look like plausible subsamples.

Usage
-----
    python scripts/cut_probe_manifest.py \\
        --manifest data/manifest_coco_crop.parquet \\
        --out data/probe/manifest_coco_crop_probe.parquet \\
        --budget train=16000 --budget val_internal=4000 \\
        --split train,val_internal
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import sys

import numpy as np
import pandas as pd

from aigcdet.data.manifest import MANIFEST_COLUMNS, MANIFEST_IDENTITY_COLUMNS
from aigcdet.features.bank import manifest_fingerprint

_SCRIPTS = pathlib.Path(__file__).resolve().parent


def _eval_bank_script():
    """`scripts/extract_eval_bank.py` as a module.

    `scripts/` is not a package, so a plain import cannot reach it. Loading by
    path is what the tests already do (`tests/scripts/*::_load_script`) and it
    is preferred here over a copy of `subsample_manifest`: the two callers must
    select rows identically, and importing is the only way to guarantee that
    without a test standing in for a guarantee.
    """
    path = _SCRIPTS / "extract_eval_bank.py"
    spec = importlib.util.spec_from_file_location("extract_eval_bank_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def select_splits(df: pd.DataFrame, splits_arg: str) -> pd.DataFrame:
    """Filter to a comma-separated split list, refusing one that is absent.

    A split that matches nothing must raise rather than silently contribute no
    rows: `--split train,val_internl` would otherwise write a manifest holding
    only `train`, and Stage B's "bank has no val_internal rows" would surface
    it an hour of GPU later.
    """
    if not splits_arg:
        return df
    wanted = [s.strip() for s in splits_arg.split(",") if s.strip()]
    present = sorted(set(df["split"].astype(str)))
    unknown = [s for s in wanted if s not in present]
    if unknown:
        raise ValueError(
            f"--split names {unknown}, which the manifest does not contain; "
            f"its splits are {present}")
    return df[df["split"].astype(str).isin(wanted)]


SAMPLERS = ("uniform", "stratified")


def _uniform_positions(sub: pd.DataFrame, n: int, seed: int):
    """`n` positions drawn uniformly at random from `sub`, sorted.

    The right sampler for a TRAINING-side probe, and the default here. A probe
    exists to predict what the full run will do, so it should be a scale model
    of the corpus -- and `coco_crop`'s parent is already 91,032 authentic
    against 91,118 generated, so a uniform draw preserves class balance without
    being asked to.

    `stratified_subsample` is the other option and is the right one for an eval
    bank, where rare generator families must not be starved. It is the wrong
    one here, measured on this corpus: it splits each class's quota evenly over
    (generator, source) strata, so WildFake's many generator families
    collectively outdraw SID_Set's single one, and the 32/68 sid:wildfake ratio
    of the parent's generated half came back 6/94. Both arms of a policy A/B
    read the same rows either way, so that distortion cannot flip the sign of
    the result -- but it makes the probe's MAGNITUDE a poor predictor of the
    full run's, which is most of what a probe is for.
    """
    rng = np.random.default_rng(seed)
    pos = np.arange(len(sub))
    if n >= len(pos):
        return pos
    return np.sort(rng.choice(pos, size=n, replace=False))


def cut(df: pd.DataFrame, budgets: dict[str, int], splits_arg: str,
        seed: int, sampler: str = "uniform") -> tuple[pd.DataFrame, dict[str, int]]:
    """The probe frame, in the parent manifest's own row order.

    NO `reset_index()`, under either sampler, so a caller that uses the frame
    directly keeps the parent's index labels -- which are the per-view RNG key
    (`features/extract.py`).

    The FILE is a different matter and worth being precise about: it is written
    `index=False`, the same convention as `write_manifest`, so a probe read back
    from disk carries a fresh 0..N-1 numbering. Its views are therefore NOT the
    full run's views for the same image. That is harmless for what a probe is
    for -- both arms of an A/B read the same file and so draw the same pixels --
    but it does mean a probe bank can never stand in for part of a full one, on
    top of already fingerprinting differently.
    """
    if sampler not in SAMPLERS:
        raise ValueError(f"--sampler must be one of {list(SAMPLERS)}, got {sampler!r}")
    selected = select_splits(df, splits_arg)
    if sampler == "stratified":
        eb = _eval_bank_script()
        return eb.subsample_manifest(selected, budgets, seed=seed)

    splits = selected["split"].astype(str).to_numpy()
    absent = [s for s in sorted(budgets) if not (splits == s).any()]
    if absent:
        raise ValueError(
            f"--budget names split(s) {absent}, which the selected rows do not "
            f"contain (they hold {sorted(set(splits.tolist()))}); a budget that "
            "matches nothing silently caps nothing")
    keep = [np.where(~np.isin(splits, sorted(budgets)))[0]]
    kept: dict[str, int] = {}
    for name in sorted(budgets):
        pos = np.where(splits == name)[0]
        picked = _uniform_positions(selected.iloc[pos], int(budgets[name]), seed)
        keep.append(pos[np.asarray(picked, dtype=np.int64)])
        kept[name] = int(len(picked))
    order = np.sort(np.concatenate(keep))
    return selected.iloc[order], kept


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="the frozen parent manifest")
    ap.add_argument("--out", required=True, help="where to write the probe manifest")
    ap.add_argument("--budget", action="append", metavar="SPLIT=N", required=True,
                    help="stratified cap on one split, balanced across "
                         "class x generator x source; repeatable")
    ap.add_argument("--split", default="",
                    help="comma-separated splits to keep at all, applied BEFORE "
                         "the budgets. Omit to keep every split, capping only "
                         "the ones named by --budget.")
    ap.add_argument("--sampler", choices=SAMPLERS, default="uniform",
                    help="'uniform' (default) makes the probe a scale model of "
                         "the corpus, which is what a training-side probe is "
                         "for. 'stratified' reuses the eval bank's sampler, "
                         "which balances across generator x source and is the "
                         "right choice only when rare families must not be "
                         "starved.")
    ap.add_argument("--seed", type=int, default=None,
                    help="subsample seed; defaults to the project's committed "
                         "BENCHMARK_SEED so a probe is reproducible from the "
                         "command line alone")
    return ap


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    eb = _eval_bank_script()
    seed = args.seed if args.seed is not None else eb.BENCHMARK_SEED
    budgets = eb.parse_budgets(args.budget)

    df = pd.read_parquet(args.manifest)
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest at {args.manifest} missing columns: {missing}")

    probe, kept = cut(df, budgets, args.split, seed, sampler=args.sampler)

    # Identity columns are CARRIED OVER, never recomputed: `content_sha256` and
    # `pixel_sha256` describe the bytes at `rel_path`, which the subsample does
    # not touch. Re-deriving them would re-hash 20,000 images to reproduce
    # values already frozen, and any difference would mean the parent manifest
    # was already stale -- a separate problem that `data.verify` is for.
    cols = MANIFEST_COLUMNS + [c for c in MANIFEST_IDENTITY_COLUMNS
                               if c in probe.columns]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    probe[cols].to_parquet(args.out, index=False)

    fp = manifest_fingerprint(probe)
    print(f"parent : {args.manifest}  ({len(df):,} rows, "
          f"fingerprint {manifest_fingerprint(df)[:16]}...)")
    print(f"probe  : {args.out}  ({len(probe):,} rows)")
    print(f"  sampler {args.sampler}, seed {seed}, budgets {budgets}, kept {kept}")
    print(f"  manifest_sha256 {fp}")
    print("  by split:")
    for split, n in probe["split"].value_counts().sort_index().items():
        print(f"    {split:20s} {n:7,}")
    print("  by source x label:")
    for (src, lab), n in probe.groupby(["source", "label"]).size().items():
        print(f"    {src:20s} label={lab}  {n:7,}")
    # The two arms of a policy A/B must read the SAME file. Print the digest of
    # the bytes, not just of the row identity, so "we both used the probe" can
    # be checked rather than assumed.
    import hashlib
    with open(args.out, "rb") as fh:
        print(f"  file sha256 {hashlib.sha256(fh.read()).hexdigest()}")
    return {"n_rows": len(probe), "kept": kept, "manifest_sha256": fp}


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
