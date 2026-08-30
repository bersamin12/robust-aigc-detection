"""Join the training and benchmark manifests into the one the eval bank needs.

`extract_eval_bank.py --tier ablation` is defined over three splits --
`val_internal`, `heldout_generator` and `benchmark` -- but the first two live
in the training manifest and the third in the benchmark manifest, and
`--manifest` takes one file. Nothing joined them, so the ablation tier could
not be extracted at all.

    python scripts/build_eval_manifest.py \
        --manifest data/normalized/manifest.parquet \
        --benchmark-manifest data/demo/benchmark_manifest.parquet \
        --out data/eval_manifest.parquet

This is not `pd.concat`. Two things break silently first, and a third loudly:

**Two dataset roots.** The training tree is `data/normalized`, the benchmark
tree is `data/demo`. `dataset_root` refuses a frame implying both -- correctly,
since a manifest describes one tree that can be rebased onto one mount point --
and `extract_eval_bank` calls it *after* loading the backbone. So the combined
frame is re-rooted onto the two trees' common ancestor, which rewrites every
`rel_path`. That is a new identity, deliberately: this is a new manifest, not a
view of either input, and its fingerprint should not match either.

**Colliding index labels.** Both inputs are indexed 0..N. The index label is
the per-view RNG key (`extract_bank`), so a naive concat would hand two
different images the same key. Freezing through `write_manifest`, which writes
`index=False`, is what makes the labels unique -- and it must happen exactly
once, here, at freeze time. Every later consumer slices with `.iloc` and must
never call `reset_index`.

**Digests are recomputed, not copied.** The eval bank is about to be extracted
from these exact files. Re-reading them proves they still are what each source
manifest was frozen against; a divergence is reported by name and refused,
because features extracted from a changed file do not correspond to its
recorded label and nothing downstream could tell.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from aigcdet.data.manifest import (
    MANIFEST_COLUMNS,
    derive_root,
    read_manifest,
    write_manifest,
)
from aigcdet.data.sources import is_heldout_eligible

#: The splits the ablation tier is defined over, minus `benchmark`, which comes
#: from the other file. Kept in step with `extract_eval_bank.TIERS["ablation"]`.
DEFAULT_SPLITS = ("val_internal", "heldout_generator")


def select(df: pd.DataFrame, splits: tuple[str, ...], where: str) -> pd.DataFrame:
    """`df` restricted to `splits`, in the frame's own row order.

    An unknown split is a typo that would otherwise produce a manifest quietly
    missing a whole population -- and the eval bank built from it would look
    fine until `run_ablation` refused the selection population three hours
    later.
    """
    present = sorted(set(df["split"]))
    missing = [s for s in splits if s not in present]
    if missing:
        raise SystemExit(
            f"{where} has no rows for split(s) {missing}; it contains {present}")
    return df[df["split"].isin(splits)]


def check_no_overlap(train: pd.DataFrame, bench: pd.DataFrame) -> None:
    """Refuse an image that appears on both sides.

    COCO val2017 is benchmark data and may never be trained on (spec §4.1);
    `build_dataset` enforces that by source. This is the second net, on
    content: if a row did reach both frames, the eval bank would score the
    same image twice under two different splits, and every per-split metric
    would be computed over a population that overlaps itself.
    """
    both = set(train["content_sha256"]) & set(bench["content_sha256"])
    both.discard("")
    if both:
        example = sorted(both)[0]
        row = train[train["content_sha256"] == example].iloc[0]
        raise SystemExit(
            f"{len(both)} image(s) appear in both manifests, e.g. "
            f"{row['path']} (split {row['split']!r}). The benchmark must not "
            "overlap the training tree -- check the --raw/--demo-dir split in "
            "scripts/build_dataset.py.")


def check_digests(frozen: pd.DataFrame, rebuilt: pd.DataFrame) -> None:
    """Every file must still hash to what its source manifest recorded."""
    was = dict(zip(frozen["path"], frozen["content_sha256"]))
    changed = [p for p, d in zip(rebuilt["path"], rebuilt["content_sha256"])
               if was.get(p) and was[p] != d]
    if changed:
        raise SystemExit(
            f"{len(changed)} file(s) differ from the digest their manifest was "
            f"frozen against, e.g.\n  {changed[0]}\n"
            "Features extracted from a changed file do not correspond to its "
            "recorded label, and nothing downstream can detect that. Restore "
            "the files, or re-freeze the source manifest deliberately.")


def promote_to_heldout(train_full: pd.DataFrame, families: tuple[str, ...]):
    """Boolean mask over `train_full`: the `train` rows of `families`.

    WHY THIS EXISTS. The manifest's held-out pair holds out
    `SDwithAdaptor_controlnet` while `SDwithAdaptor_lora` and
    `SDwithAdaptor_lycris` stay in TRAINING, and `VQGAN` while `VQVAE` and
    `vqdm` stay in. An adapter changes the conditioning, not the decoder that
    leaves the forensic trace, so the pinned pair asks an easier question than
    its name suggests. Asking the harder one -- hold out the whole lineage --
    needs the siblings on BOTH sides: excluded from training
    (`RungConfig.train_exclude_generators`) and present in the eval manifest so
    there is something to score. They are in `train`, so nothing puts them here
    by default.

    WHY IT IS SAFE TO RELABEL THEM. The eval manifest is already a NEW
    manifest with its own identity, not a view of either input (see the module
    docstring), so its split column is its own. What it is NOT safe to do is
    score these rows against a rung that trained on them -- and nothing in
    this file can see a rung. That coupling is checked at scoring time by
    `eval.grid.assert_heldout_not_trained`, which refuses a table whose eval
    bank holds a family the training bank also trained on.

    A family that matches nothing raises. It would contribute zero rows, and
    the resulting number would still be reported as a lineage-holdout score.
    """
    if not families:
        return pd.Series(False, index=train_full.index)
    ineligible = sorted(f for f in families if not is_heldout_eligible(f))
    if ineligible:
        raise SystemExit(
            f"--extra-heldout-generators names {ineligible}, which are "
            "dataset-level pseudo-generators. Holding one out removes an "
            "entire source, so the score would measure dataset shift rather "
            "than an unseen generator family (spec 4.6).")
    already = sorted(set(families) & set(
        train_full.loc[train_full["split"] == "heldout_generator", "generator"]))
    if already:
        raise SystemExit(
            f"--extra-heldout-generators names {already}, which the manifest "
            "ALREADY holds out. Naming them here would say the eval manifest "
            "and the training manifest disagree about the split, when they "
            "agree. Name only the siblings that are still in `train`.")
    mask = ((train_full["split"] == "train")
            & train_full["generator"].isin(list(families)))
    missing = sorted(set(families) - set(train_full.loc[mask, "generator"]))
    if missing:
        present = sorted(set(train_full.loc[train_full["split"] == "train",
                                            "generator"]) - {""})
        raise SystemExit(
            f"--extra-heldout-generators names {missing}, which no `train` row "
            f"carries. The manifest's train split holds {present}. A name that "
            "matches nothing contributes no rows, and the number would still "
            "be reported as a lineage-holdout score.")
    return mask


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Combine the training and benchmark manifests for the "
                    "ablation-tier eval bank.")
    ap.add_argument("--manifest", required=True, help="the training manifest")
    ap.add_argument("--benchmark-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default=",".join(DEFAULT_SPLITS),
                    help="comma-separated splits to take from --manifest; "
                         "`benchmark` always comes from --benchmark-manifest")
    ap.add_argument("--extra-heldout-generators", default="",
                    help="comma-separated generator families to PROMOTE from "
                         "the training manifest's `train` split into this "
                         "manifest's `heldout_generator` split, so a whole "
                         "lineage can be scored as unseen. The same families "
                         "must be passed to the rung as "
                         "train_exclude_generators, or the score is of a "
                         "family the head trained on; "
                         "eval.grid.assert_heldout_not_trained enforces that "
                         "at scoring time.")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)

    splits = tuple(s.strip() for s in a.splits.split(",") if s.strip())
    train = read_manifest(a.manifest)
    bench = read_manifest(a.benchmark_manifest)

    stray = sorted(set(bench["split"]) - {"benchmark"})
    if stray:
        raise SystemExit(
            f"--benchmark-manifest holds non-benchmark split(s) {stray}. That "
            "is almost certainly the training manifest passed twice, which "
            "would double-count its rows in the eval bank.")

    extra = tuple(g.strip() for g in a.extra_heldout_generators.split(",")
                  if g.strip())
    # One mask over the WHOLE training manifest rather than a concat of two
    # selections, so the promoted rows keep the manifest's own row order --
    # row order is the key space, and it must be a function of the inputs
    # alone.
    promote = promote_to_heldout(train, extra)
    # `select` still owns the unknown-split guard; the promoted rows are added
    # by INDEX and the result re-sorted, because a frozen manifest's index is
    # its row order and row order is the key space. A concat of two selections
    # would put the promoted rows after the benchmark boundary instead.
    keep = sorted(select(train, splits, "--manifest").index
                  .union(train.index[promote]))
    promoted_here = train.index[promote]
    train = train.loc[keep].copy()
    if extra:
        train.loc[promoted_here, "split"] = "heldout_generator"
        print(f"promoted {len(promoted_here)} row(s) of {list(extra)} from "
              "train to heldout_generator")
    check_no_overlap(train, bench)

    # Fixed order: the eval splits in the training manifest's own order, then
    # the benchmark rows in theirs. Deterministic given the inputs, and the
    # thing that must never change afterwards -- row order IS the key space.
    combined = pd.concat([train, bench], ignore_index=True)[MANIFEST_COLUMNS]

    # The root is derived from the IMAGE paths, by `write_manifest`'s own
    # `derive_root` -- the deepest directory containing every image, which for
    # `data/normalized/...` plus `data/demo/...` is `data`. Deriving it from
    # the two manifest FILES instead would land a level too shallow whenever a
    # manifest sits outside the tree it describes, and every `rel_path` would
    # carry a spurious leading component.
    root = derive_root(combined["path"])
    print(f"{len(train)} eval rows + {len(bench)} benchmark rows -> {len(combined)}")
    print(f"derived root: {root}")

    write_manifest(combined, a.out, root=root, digests="bytes",
                   workers=a.workers)
    check_digests(pd.concat([read_manifest(a.manifest),
                             read_manifest(a.benchmark_manifest)]), combined)

    print(combined["split"].value_counts().to_string())
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
