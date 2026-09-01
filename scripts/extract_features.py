"""Stage A CLI.

`--split` takes a COMMA-SEPARATED list, and the training splits below are the
combination that actually works end to end. Stage B's `train_rung` evaluates
on the bank's own `val_internal` rows, so a bank extracted with `--split train`
alone is rejected -- after the full extraction has already been paid for.

    python scripts/extract_features.py --manifest data/manifest.parquet \
        --backbone dinov3l --out banks/dinov3l --split train,val_internal
    # leave-one-transform-out bank for the A3-LOTO run:
    python scripts/extract_features.py --manifest data/manifest.parquet \
        --backbone dinov3l --out banks/dinov3l_loto --split train,val_internal \
        --exclude noise
    # attach the reconstruction branch (spec section 3.3) to an existing
    # bank, for ALL of its already-cached views -- --split must select the
    # same rows the bank was originally built from, or attach_recon_to_bank
    # raises rather than silently misaligning:
    python scripts/extract_features.py --manifest data/manifest.parquet \
        --out banks/dinov3l --split train,val_internal --recon

Stage A takes 8-13 h per bank, against Kaggle's 30 h/week free tier, so it is
built to be interrupted. `--resume` continues into the same --out, skipping
rows already written; `--workers N` runs the CPU stage (decode, augment,
proxies) in N subprocesses while this process feeds the GPU.

    python scripts/extract_features.py --manifest data/manifest.parquet \
        --backbone dinov3l --out banks/dinov3l --split train,val_internal \
        --resume --workers 4

**Sharding across a five-person fleet.** `--shard I/N` takes the I-th of N
contiguous, disjoint, exhaustive blocks of the rows this run would otherwise
have extracted, so five teammates on five free Kaggle accounts each pay for a
fifth of the 8-13 h and `scripts/merge_banks.py` puts the result back
together. Because every view's pixels depend only on (seed, row_id,
view_idx), the merged bank is identical to one extracted in a single run.

    # teammate k of 5 (k = 0..4), resumable across session timeouts
    python scripts/extract_features.py --manifest data/manifest.parquet \
        --backbone dinov3l --out banks/dinov3l_shard0 \
        --split train,val_internal --shard 0/5 --resume --workers 4
    python scripts/merge_banks.py --out banks/dinov3l \
        banks/dinov3l_shard0 banks/dinov3l_shard1 banks/dinov3l_shard2 \
        banks/dinov3l_shard3 banks/dinov3l_shard4

Two properties make that promise true, and both fail SILENTLY if broken, so
they are stated here rather than left to the reader:

* The blocks are CONTIGUOUS, never strided. `merge_banks` concatenates shards
  in the order given and re-fingerprints over that concatenation, and every
  downstream reader indexes a bank positionally against the manifest. A
  strided split (`iloc[k::5]`) preserves index labels and so produces
  byte-identical pixels, while producing a merged bank whose rows run
  0,5,10,...,1,6,11,... -- which merges without complaint and then mismatches
  the manifest.
* The shard frame keeps the frozen manifest's INDEX LABELS. Those labels are
  the per-view RNG key; a `reset_index()` would restart every shard's keys at
  0, five shards would collide, and the same physical image would carry
  different pixels depending on who extracted it.

**Order of operations.** Row selection happens in exactly this order, and the
order is part of the contract because the alternatives are all plausible and
all different:

    read_manifest -> --split -> --limit -> --shard

`--split` first, so the N shards partition the rows that will actually be
extracted; sharding the whole manifest first would hand shard 0 nothing but
`train` rows and shard 4 nothing but `benchmark`. `--limit` before `--shard`,
so the shards still tile one contiguous prefix and still merge into a
coherent (if truncated) bank -- note this is the opposite of
`notebooks/run_shard.py`, whose `--limit` deliberately truncates each shard
individually and produces a smoke bank that must never be merged.

`--exclude` is NOT a row filter and takes no part in that order: it removes a
transform family from the sampled recipe POOL (the A3-LOTO run, spec 4.6).
It must be identical across every shard of one bank -- the bank's config does
not record it, so `merge_banks` cannot catch a shard that disagrees.

`--shard` also applies to `--recon`, and must be given again (with the same
`--split`/`--limit`) when attaching the reconstruction branch to a shard's
own bank, since `attach_recon_to_bank` requires exactly the rows the bank was
built from.

`notebooks/run_shard.py` reaches the same partition through the Kaggle
bootstrap, spelled `--shard 0 --n-shards 5`. It carries its own copy of the
block arithmetic (`notebooks/kaggle_bootstrap.shard_bounds`), and
`tests/scripts/test_extract_features_cli.py` asserts the two agree row for
row, so a fleet may mix the two entry points without shards overlapping.
`scripts/extract_eval_bank.py` shares the `I/N` spelling but uses a different
remainder rule; that is safe only because an eval bank never merges with a
training one.
"""
from __future__ import annotations

import argparse

from aigcdet.data.manifest import read_manifest
from aigcdet.features.bank import CHECKPOINT_EVERY
from aigcdet.augment.canonical import (
    CANON_CROP_SIDE, MODE_BAND, MODES, CanonPolicy)
from aigcdet.features.extract import extract_bank, shard_frame


def select_splits(df, splits_arg: str):
    """Filter `df` to a comma-separated list of manifest splits.

    A list, not a single value: Stage B's `train_rung` evaluates on the bank's
    own `val_internal` rows, so a bank must carry the training AND the
    internal-validation split or it is unusable. An unknown split name is a
    typo that would otherwise produce an empty (or wrong) bank after hours of
    extraction, so it raises here instead.
    """
    wanted = [s.strip() for s in splits_arg.split(",") if s.strip()]
    if not wanted:
        return df
    present = sorted(set(df["split"].unique()))
    unknown = [s for s in wanted if s not in present]
    if unknown:
        raise ValueError(
            f"--split names {unknown}, which the manifest does not contain; "
            f"its splits are {present}")
    # No .reset_index(): extract_bank keys its per-image RNG on this index
    # label (see aigcdet.features.extract's module docstring), and
    # attach_recon_to_bank replays those views from the row_id the bank
    # stores, so the filtered frame must keep the frozen manifest's labels.
    return df[df["split"].isin(wanted)]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--backbone", required=False,
                     help="required unless --recon (an existing bank already names its backbone)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="",
                     help="comma-separated manifest splits to include, e.g. "
                          "'train,val_internal' (Stage B needs both in one bank)")
    ap.add_argument("--limit", type=int, default=None,
                     help="first N rows of the --split selection, BEFORE "
                          "--shard, so the shards still tile one contiguous "
                          "prefix (smoke runs only)")
    ap.add_argument("--shard", default=None, metavar="I/N",
                     help="extract the I-th of N contiguous, disjoint, "
                          "exhaustive blocks of the selected rows, and nothing "
                          "else; recombine every shard with merge_banks.py. "
                          "Applied last: --split, then --limit, then --shard. "
                          "Every shard of one bank must be run with the same "
                          "--manifest, --split, --limit, --exclude, --backbone "
                          "and seed")
    ap.add_argument("--exclude", default="",
                     help="comma-separated families to remove from the sampled "
                          "recipe pool (spec 4.6 LOTO). NOT a row filter, and "
                          "not recorded in the bank, so every shard of one "
                          "bank must pass the same value")
    ap.add_argument("--canon-mode", choices=MODES, default=MODE_BAND,
                    help="resolution standardisation. 'band' (default) is the "
                         "frozen stream's policy: downscale to a common "
                         "bandwidth ceiling, then upscale. 'crop' takes a "
                         "random square window at NATIVE resolution instead, "
                         "one per view, so a generator's high-frequency "
                         "signature survives inside the window. The choice is "
                         "recorded in the bank config, so a crop bank can "
                         "never be resumed from, merged with or fused against "
                         "a band one.")
    ap.add_argument("--crop-side", type=int, default=CANON_CROP_SIDE,
                    help="window size for --canon-mode crop. The corpus "
                         "preset's `min_short_side` must equal this, or images "
                         "too small for the window reach extraction and raise.")
    ap.add_argument("--geometric", action="store_true",
                    help="dihedral augmentation: a random flip and "
                         "90-degree rotation per view, applied after "
                         "standardisation and before the recipe. Needs "
                         "--canon-mode crop, because a 90-degree rotation "
                         "transposes a non-square image. No interpolation, so "
                         "it moves no pixel value and no proxy.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--resume", action="store_true",
                     help="continue an interrupted extraction into --out, skipping "
                          "the rows already written (the same manifest, split, "
                          "backbone and seed must be given)")
    ap.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
                     help="flush meta/views parquet every N images; this is also "
                          "how much work a session timeout can cost")
    ap.add_argument("--workers", type=int, default=0,
                     help="subprocesses running the CPU stage (decode, augment, "
                          "proxies) while this process feeds the GPU; 0 = inline")
    ap.add_argument("--recon", action="store_true",
                     help="attach the reconstruction branch (spec section 3.3) to the "
                          "existing bank at --out for all of its cached views, instead "
                          "of building a new bank")
    ap.add_argument("--block-shard", "--recon-shard", dest="recon_shard",
                     default=None, metavar="I/N",
                     help="with --recon: compute only contiguous row block I of N "
                          "of THIS bank and write it to recon_part_IofN.npy without "
                          "attaching. Unlike --shard, which cuts the MANIFEST into "
                          "separate banks, this cuts one bank's rows across "
                          "processes so four GPUs can replay it at once. Finish "
                          "with --recon-merge N.")
    ap.add_argument("--block", default=None,
                     choices=("recon", "recon_vq", "freq"),
                     help="attach ONE auxiliary block to the existing bank at "
                          "--out instead of building a new bank: recon (SD 1.5 "
                          "KL VAE round-trip, 12-d), recon_vq (a "
                          "vector-quantised autoencoder, 12-d, its own file) "
                          "or freq (NPR's periodic-upsampling descriptor, 4-d, "
                          "crop banks only, no GPU). Supersedes "
                          "--recon/--recon-kind, which remain as aliases.")
    ap.add_argument("--allow-band-freq", action="store_true",
                     help="--block freq refuses a band-canonicalised bank, "
                          "where the descriptor measures the resampler rather "
                          "than the generator and leaks. Pass this only to "
                          "reproduce that negative result deliberately.")
    ap.add_argument("--recon-kind", default="kl", choices=("kl", "vq"),
                     help="with --recon: WHICH autoencoder round-trips the "
                          "pixels, and therefore which block is written -- "
                          "kl -> recon.npy (SD 1.5 KL VAE), vq -> "
                          "recon_vq.npy (a vector-quantised autoencoder). Two "
                          "12-d blocks, never one wider one: the width is "
                          "pinned by attach_recon and the ladder allows one "
                          "flag per rung.")
    ap.add_argument("--block-merge", "--recon-merge", dest="recon_merge",
                     type=int, default=None, metavar="N",
                     help="with --recon: assemble the N recon_part_*.npy shards, "
                          "verify they cover every row exactly once and are "
                          "finite, attach the result and delete the parts.")
    return ap



def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)

    df = read_manifest(a.manifest)
    # Row selection, in the one order the module docstring documents:
    # --split, then --limit, then --shard. `--limit` is applied HERE rather
    # than handed to `extract_bank` (which would apply it to whatever it was
    # given) precisely so that it lands BEFORE the shard split -- otherwise
    # each of the N shards would take its own first N rows and their union
    # would be neither the first `limit` rows nor a contiguous block.
    df = select_splits(df, a.split)
    if a.limit is not None:
        df = df.iloc[:a.limit]
    n_selected = len(df)
    df = shard_frame(df, a.shard)
    if a.shard and not len(df):
        # A zero-row shard merges silently and contributes nothing, so the
        # session that would have produced it should not start.
        ap.error(f"--shard {a.shard} selects no rows: N exceeds the "
                 f"{n_selected} rows --split/--limit select. Use fewer shards.")

    if a.recon or a.block:
        import functools
        import glob
        import os

        import numpy as np

        from aigcdet.features.bank import AUX_BLOCKS, FeatureBank
        from aigcdet.features.recon import RECON_KINDS, attach_recon_to_bank
        from aigcdet.features.replay import merge_blocks, shard_bounds

        # `--recon [--recon-kind vq]` is the older spelling of `--block`.
        block = a.block or RECON_KINDS[a.recon_kind]
        dim = dict((n, d) for _, n, d in AUX_BLOCKS)[block]
        if block == "freq":
            from aigcdet.features.freq import attach_freq_to_bank
            compute = functools.partial(attach_freq_to_bank,
                                        allow_band=a.allow_band_freq)
        else:
            kind = "kl" if block == "recon" else "vq"
            compute = functools.partial(attach_recon_to_bank,
                                        device=a.device, kind=kind)
        recon_bounds = shard_bounds

        bank = FeatureBank.open(a.out)
        if a.recon_shard and a.recon_merge:
            ap.error("--block-shard computes one block; --block-merge "
                     "assembles them. Give one or the other.")

        if a.recon_merge:
            n_shards = a.recon_merge
            n = len(bank.meta)
            parts = []
            for i in range(n_shards):
                p = os.path.join(a.out, f"part_{block}_{i}of{n_shards}.npy")
                if not os.path.exists(p):
                    ap.error(f"missing shard {p}. Every block must be present: "
                             "a merge over the shards that happen to exist "
                             "would attach zeros for the rest.")
                start, stop = recon_bounds(n, i, n_shards)
                parts.append((start, stop, np.load(p)))
            merge_blocks(bank, parts, block, dim)
            for p in glob.glob(os.path.join(a.out, f"part_{block}_*of*.npy")):
                os.remove(p)
            bank.check_invariants()
            print(f"merged {n_shards} shards -> {a.out}/{block}.npy")
            return 0

        if a.recon_shard:
            try:
                i_s, n_s = (int(x) for x in a.recon_shard.split("/"))
            except ValueError:
                ap.error(f"--recon-shard {a.recon_shard!r} is not I/N")
            start, stop = recon_bounds(len(bank.meta), i_s, n_s)
            print(f"{block} shard {i_s}/{n_s}: rows [{start}, {stop}) of "
                  f"{len(bank.meta)}")
            arr = compute(bank, df, start=start, stop=stop, attach=False)
            out_p = os.path.join(a.out, f"part_{block}_{i_s}of{n_s}.npy")
            np.save(out_p, arr)
            print(f"wrote {out_p} {arr.shape}")
            return 0

        compute(bank, df)
        # Post-condition on a multi-hour job: recon.npy must cover every view
        # of every row, or A3-vs-A4 compares different augmentation budgets.
        bank.check_invariants()
        return 0

    if not a.backbone:
        ap.error("--backbone is required unless --recon/--block is given")
    if a.shard:
        print(f"shard {a.shard}: {len(df)} of {n_selected} selected rows, "
              f"row_id {int(df.index[0])}..{int(df.index[-1])}")
    # No `limit=`: it was already applied above, before the shard split. The
    # bank must fingerprint exactly the rows it contains, and handing
    # `extract_bank` a limit as well would truncate the shard a second time.
    extract_bank(df, a.backbone, a.out, seed=20260827, device=a.device,
                 batch_size=a.batch_size,
                 exclude_families=tuple(f for f in a.exclude.split(",") if f),
                 policy=CanonPolicy(mode=a.canon_mode, crop_side=a.crop_side),
                 geometric=a.geometric,
                 resume=a.resume, checkpoint_every=a.checkpoint_every,
                 workers=a.workers)
    if a.shard:
        print(f"shard {a.shard} done -- recombine every shard with "
              "scripts/merge_banks.py before training")
    return 0


if __name__ == "__main__":
    # The guard is load-bearing, not conventional: `--workers > 0` spawns
    # subprocesses that re-import this module as __main__, and without it each
    # one would start its own extraction.
    raise SystemExit(main())
