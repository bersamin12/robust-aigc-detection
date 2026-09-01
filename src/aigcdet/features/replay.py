"""Recompute a bank's cached views, to attach a feature block after the fact.

Stage A writes embeddings and throws the pixels away. A block added later --
the reconstruction branch (§3.3), the frequency descriptor -- has to reproduce
the EXACT pixels each view was embedded from, or it is describing different
images than the row it is stored beside.

That reproduction is the dangerous part, and it is the same for every block,
so it lives here once rather than in each extractor. The block-specific part
is a single `per_view(uint8 HWC) -> (dim,)` callable.

Nothing here reads a model or a GPU: `per_view` may use one (the
reconstruction branch does) or be pure numpy (the frequency descriptor is).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from aigcdet.augment.canonical import (
    DEFAULT_POLICY, MODE_CROP, CanonPolicy, canonical_rng, canonicalise)
from aigcdet.augment.geometric import dihedral, geometric_rng, sample_dihedral


def canon_base(decoded: np.ndarray, policy: CanonPolicy,
               is_eval: bool) -> np.ndarray | None:
    """The standardised image every view of one row starts from, or `None`
    when each view must re-derive its own.

    THREE production sites canonicalise before replaying a recipe, and they do
    NOT all agree -- which is the whole reason this decision is one function:

      `features/extract`   base = None for crop, so each view draws its OWN
                           window from `canonical_rng(seed, row_id, view)`.
      `features/replay`    must mirror whichever site wrote the bank it is
                           replaying, and that is what `is_eval` selects.
      `eval/grid`          ONE `canonicalise(base, policy=policy)` with NO rng,
                           shared by all 20 conditions. For crop that is the
                           CENTRE window (`_random_square_crop`: "rng is None
                           gives the CENTRE window"), held fixed so a
                           condition's measured effect is not confounded with
                           "a different picture"; for band it is the
                           unjittered ceiling.

    Replaying an eval bank down the training branch would therefore take a
    fresh RANDOM crop window per condition and compute the block on pixels
    that bank never contained -- with no shape error and no warning, only
    wrong numbers. That is the failure this function exists to prevent.
    """
    if is_eval:
        return canonicalise(decoded, policy=policy)
    return None if policy.mode == MODE_CROP else canonicalise(
        decoded, policy=policy)


def replay_views(bank, manifest_df, per_view: Callable[[np.ndarray], np.ndarray],
                 dim: int, *, seed: int = 20260827, start: int = 0,
                 stop: int | None = None, allow_partial: bool = False,
                 desc: str = "replay") -> np.ndarray:
    """`(stop - start, n_views, dim)` for one contiguous block of rows.

    Checks the bank is still positionally aligned with `manifest_df` before
    computing anything -- a re-split manifest would otherwise silently attach
    a block to the wrong rows (project-constraints.md's "manifest is frozen
    once written" rule). `manifest_df` is used ONLY for that check: the replay
    key comes from `bank.row_ids`, which `extract_bank` stores in
    `meta.parquet`. Recovering it from `manifest_df.index` instead made the
    caller's index a load-bearing input that nothing verified -- a
    `reset_index()`ed frame passes `verify_against_manifest` (it compares
    paths positionally) and then replays every noise-containing view against
    different pixels.

    Reproduces each view's exact cached pixels, not just an equally-valid
    resampling of the same recipe: `extract_bank` derives a fresh
    `np.random.default_rng([seed, row_id, view_idx])` to APPLY each view (a
    separate generator from the one used to SAMPLE its recipe), so the only
    randomness an already-known recipe's replay needs -- the `noise` op's
    realisation, the one op that reads from the generator -- is reproduced by
    re-deriving that same key here. This holds regardless of view order or how
    many draws the sampling step consumed, because that step used a different
    generator instance entirely.
    """
    from PIL import Image
    from tqdm import tqdm

    from aigcdet.augment.recipes import Recipe

    bank.verify_against_manifest(manifest_df)

    # A bank written before these keys existed has neither, and that can only
    # mean the band policy with no geometry -- there was nothing else.
    policy = CanonPolicy.from_record(bank.config["canon_policy"]) \
        if "canon_policy" in bank.config else DEFAULT_POLICY
    geometric = bool(bank.config.get("geometric"))
    # An EVAL bank's view axis is the CONDITION axis, and `eval/grid` builds it
    # differently ON PURPOSE (see `canon_base`). `extract_eval_bank` writes
    # `conditions`; `extract_bank` never does, so the config says which kind of
    # artefact this is without the caller having to be trusted about it.
    is_eval = "conditions" in bank.config
    if is_eval and geometric:
        # grid.py applies no dihedral, deliberately: evaluating on a random
        # orientation would measure orientation invariance by accident. A bank
        # claiming otherwise means grid.py changed and this replay no longer
        # mirrors it.
        raise ValueError(
            f"the eval bank at {bank.path} records geometric={geometric!r}, but "
            "`eval/grid` applies no dihedral -- this replay cannot reproduce "
            "its pixels. Re-check eval/grid.py against this function.")

    n, v = len(bank.meta), bank.config["n_views"]
    stop = n if stop is None else min(int(stop), n)
    start = max(0, int(start))
    if start >= stop:
        raise ValueError(f"empty row range [{start}, {stop}) over {n} rows")
    if not allow_partial and (start, stop) != (0, n):
        # Attaching a partial block would pin `(N, V, dim)` full of zeros for
        # every row this shard did not compute, and nothing downstream reads
        # zeros as "missing".
        raise ValueError(
            f"refusing to attach rows [{start}, {stop}) of {n}: attach the "
            "merged block, not a shard (see `merge_blocks`)")

    row_ids = bank.row_ids
    out = np.zeros((stop - start, v, dim), dtype=np.float32)
    for i in tqdm(range(start, stop), desc=desc):
        with Image.open(bank.meta.iloc[i]["path"]) as im:
            decoded = np.asarray(im.convert("RGB"), dtype=np.uint8)
        base = canon_base(decoded, policy, is_eval)
        rid = int(row_ids[i])
        for j in range(v):
            # Same three-step reconstruction as extract._prepare_image, in the
            # same order, from the same keys: standardise, orient, degrade.
            std = base if base is not None else canonicalise(
                decoded, policy=policy, rng=canonical_rng(seed, rid, j))
            if geometric:
                std = dihedral(std, sample_dihedral(geometric_rng(seed, rid, j)))
            apply_rng = np.random.default_rng([seed, rid, j])
            view = Recipe.from_json(bank.recipe_json(i, j)).apply(std, apply_rng)
            out[i - start, j] = per_view(view)
    return out


def merge_blocks(bank, parts, name: str, dim: int) -> np.ndarray:
    """Assemble contiguous `(start, stop, block)` shards and attach them.

    Refuses anything but an exact, gap-free, overlap-free cover. A block is
    indexed positionally by every consumer, so a missing row is not a smaller
    array -- it is `dim` zeros that read as a real measurement.
    """
    n, v = len(bank.meta), bank.config["n_views"]
    parts = sorted(parts, key=lambda p: p[0])
    cursor = 0
    for start, stop, block in parts:
        if start != cursor:
            raise ValueError(
                f"shard cover is not contiguous: expected a shard starting at "
                f"{cursor}, got one starting at {start}")
        if block.shape != (stop - start, v, dim):
            raise ValueError(
                f"shard [{start}, {stop}) has shape {block.shape}, expected "
                f"{(stop - start, v, dim)}")
        cursor = stop
    if cursor != n:
        raise ValueError(
            f"shards cover {cursor} of {n} rows; refusing to attach a block "
            "whose tail would be zeros")
    out = np.concatenate([b for _, _, b in parts], axis=0)
    if not np.isfinite(out).all():
        raise ValueError(
            f"{int((~np.isfinite(out)).sum())} non-finite values in {name}; "
            "a NaN here is a silent feature, not a crash")
    bank.attach_block(out, name)
    return out


def shard_bounds(n: int, i: int, n_shards: int) -> tuple[int, int]:
    """Contiguous block `i` of `n_shards` over `n` rows.

    The boundaries are a pure function of `(n, n_shards)`, so a part file
    needs no sidecar recording which rows it holds, and a merge run with the
    wrong N cannot silently reinterpret one. The remainder goes to the first
    `n % n_shards` blocks.
    """
    if not 0 <= i < n_shards:
        raise ValueError(f"shard {i} out of range for {n_shards} shards")
    base, rem = divmod(n, n_shards)
    start = i * base + min(i, rem)
    return start, start + base + (1 if i < rem else 0)
