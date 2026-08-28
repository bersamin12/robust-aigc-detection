"""Stage A (spec §3.1): images x (1 clean + K augmented views) -> feature bank.

Runs once per backbone. Everything downstream trains on the output in minutes.
Each view's RNG is derived from (seed, the row's manifest index label, the
view index) -- not from a running stream advanced per image or from this
call's local loop position -- see the comment in the loop below for why, and
what that requires of a caller that slices `manifest_df`.

Deriving a fresh generator per (image, view) -- and, within a view, a
separate fresh generator for sampling the recipe than for applying it --
means a view's pixels depend only on its own key, never on how many random
draws some other view or phase happened to consume first. That is what
lets `aigcdet.features.recon.attach_recon_to_bank` replay a cached view's
exact pixels later from nothing but its stored recipe: it does not need to
replay the sampling step at all, only re-derive the same apply-time
generator from the same key.

That same property is what makes this stage survivable on a session-limited
machine, which it has to be: 8-13 h per bank against Kaggle's 30 h/week free
tier. Three mechanisms use it, and none of them changes a single pixel:
`resume=True` continues an interrupted run into the same directory (the
metadata is checkpointed as it goes -- see `bank.BankWriter`); `workers > 0`
runs the CPU stage in a process pool while this process feeds the GPU; and
`bank.merge_banks` concatenates independently-extracted shards afterwards.

`shard_frame`/`shard_bounds` are how a caller cuts those shards. They live
here, next to the RNG-key rule they exist to protect, rather than in one of
the entry points: `scripts/extract_features.py` and
`notebooks/kaggle_bootstrap.py` both hand a slice of one frozen manifest to
`extract_bank`, and the two must agree on where the block boundaries fall or
their shards overlap and `merge_banks` refuses the merged bank.
"""
from __future__ import annotations

import collections
import itertools
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from aigcdet.augment.recipes import FAMILIES, Recipe, sample_training_recipe
from aigcdet.data.manifest import dataset_root
from aigcdet.features.backbones import embed, load_backbone
from aigcdet.features.bank import (
    CHECKPOINT_EVERY,
    N_VIEWS,
    BankWriter,
    FeatureBank,
    manifest_fingerprint,
)
from aigcdet.features.proxies import proxy_vector


def _sample_recipe_excluding(rng: np.random.Generator, exclude: tuple[str, ...]) -> Recipe:
    """Sample a recipe drawn only from the families not in `exclude`.

    Used for the leave-one-transform-out run (spec §4.6): the excluded family
    must be entirely absent from training so evaluation on it measures
    generalisation to an unanticipated degradation.

    This restricts the family POOL rather than rejection-sampling whole
    recipes. Rejection sampling was confounding the one comparison this
    function exists to enable: an excluded family is likelier to appear in a
    3-op chain than in a 1-op chain, so rejecting whole recipes discards long
    ones disproportionately. Measured over 30,000 draws, no exclusion gave a
    mean of 2.005 ops and `exclude=("noise",)` gave 1.834 -- the LOTO bank
    trained on ~8.5% lighter augmentation overall, not just one fewer family,
    violating the project's "identical view coverage across compared rungs"
    hard constraint.
    """
    if not exclude:
        return sample_training_recipe(rng)
    kept = tuple(f for f in FAMILIES if f not in exclude)
    if not kept:
        raise ValueError(f"excluding {exclude} leaves no transform families to sample")
    return sample_training_recipe(rng, families=kept)


def shard_bounds(n: int, n_shards: int) -> list[tuple[int, int]]:
    """`n_shards` contiguous half-open row ranges covering `range(n)` exactly.

    CONTIGUOUS, never strided (`iloc[k::n]`), and that is not a style choice.
    `bank.merge_banks` concatenates shards in the order it is handed them and
    re-fingerprints the result over the concatenated identity list, and every
    downstream reader indexes a bank POSITIONALLY against the manifest. Only
    contiguous ascending blocks, merged in ascending shard order, reconstruct
    the frozen manifest's row order. A strided split preserves index labels
    and therefore produces byte-identical PIXELS -- so no pixel-level check
    can see it -- while producing a bank whose rows run 0,5,10,...,1,6,11,...
    That merges without complaint (no `row_id` overlap) and then fails
    `FeatureBank.verify_against_manifest`, or worse, passes a check nobody ran
    and trains a head against permuted labels.

    The remainder goes to the FIRST `n % n_shards` shards, so the blocks are
    balanced to within one row and the partition is exhaustive: every row is
    extracted exactly once. Dropping the remainder (`n // n_shards` rows each)
    would leave up to `n_shards - 1` images out of the merged bank, which
    nothing raises on -- `merge_banks` checks for overlap, not for coverage.

    This remainder rule is shared with `notebooks/kaggle_bootstrap.shard_bounds`
    ON PURPOSE and must stay identical: `notebooks/run_shard.py` and
    `scripts/extract_features.py` build shards of the SAME training bank, and
    the two get merged. The two rules that look equally reasonable in
    isolation are not interchangeable here -- at 120,001 rows over 5 shards,
    "remainder first" and `np.linspace(...).astype(int)` (which
    `scripts/extract_eval_bank.py` uses for the separate EVAL bank) put every
    boundary one row apart, so mixing the entry points makes shard k and
    shard k+1 overlap on one image and `merge_banks` refuses the lot after
    five people have each paid for a session.
    `tests/scripts/test_extract_features_cli.py` pins the agreement.
    """
    n, n_shards = int(n), int(n_shards)
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    base, rem = divmod(n, n_shards)
    bounds, start = [], 0
    for k in range(n_shards):
        stop = start + base + (1 if k < rem else 0)
        bounds.append((start, stop))
        start = stop
    return bounds


def shard_frame(df: pd.DataFrame, spec: str | None) -> pd.DataFrame:
    """`--shard I/N` -> the I-th of N contiguous, disjoint, exhaustive blocks.

    A plain `df.iloc[a:b]`, which keeps the frozen manifest's index LABELS.
    There is no `reset_index` here and there must never be one: `extract_bank`
    derives every view's RNG from `(seed, row_id, view_idx)` where `row_id` is
    that index label, so a reset would restart every shard's key space at 0.
    Five shards would then collide in RNG-key space and the same physical
    image would carry different pixels depending on who extracted it --
    silently. `extract_bank` raises on a duplicated index within one call, but
    it cannot see that two SEPARATE sessions produced overlapping keys.

    `spec` of `None` or `""` returns `df` unchanged, so the flag is optional.
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
    start, stop = shard_bounds(len(df), n)[i]
    return df.iloc[start:stop]


#: One image's CPU work, as a picklable argument tuple for `_prepare_image`.
PrepareTask = tuple  # (write_idx, row_id, path, n_views, seed, exclude_families)


def _prepare_image(task: PrepareTask) -> dict:
    """Everything for one image that runs on the CPU, before the GPU forward.

    Decoding, recipe sampling, augmentation and the handcrafted proxies -- the
    measured ~199 ms/image that dominates Stage A and is currently serialised
    behind the GPU. This is a module-level function of a plain tuple so it can
    be dispatched to a process pool: a view's pixels depend ONLY on
    `(seed, row_id, view_idx)` and its own image file, and nothing here is
    shared or mutated, so running it in another process is safe by
    construction and bit-identical to running it inline.
    """
    write_idx, row_id, path, n_views, seed, exclude_families = task
    with Image.open(path) as im:
        base = np.asarray(im.convert("RGB"), dtype=np.uint8)

    # View 0 is always the clean view (bank invariant) -- no sampling.
    # Views 1..n_views-1 each get their own fresh generator, keyed on
    # (seed, row_id, view index), to pick their recipe. This is a SEPARATE
    # generator instance from the one used to apply that same view below --
    # both are re-derived from the same key rather than one shared stream
    # threaded through both steps, so how many draws the sampling step
    # happens to consume never shifts where the apply step's own draws (the
    # noise op's realisation, the only op that reads `rng`) land. That is
    # what makes a view's pixels reproducible from nothing but
    # (seed, row_id, view index) and its own stored recipe -- see the module
    # docstring, and `recon.attach_recon_to_bank`, which relies on exactly
    # this to replay cached views without re-sampling them.
    recipes = [Recipe(())]
    for v in range(1, n_views):
        sample_rng = np.random.default_rng([seed, row_id, v])
        recipes.append(_sample_recipe_excluding(sample_rng, exclude_families))

    views = [r.apply(base, np.random.default_rng([seed, row_id, v]))
             for v, r in enumerate(recipes)]
    labels = [r.labels() for r in recipes]
    return {
        "write_idx": write_idx,
        "row_id": row_id,
        "views": views,
        "recipes": [r.to_json() for r in recipes],
        "presence": np.stack([l["presence"] for l in labels]),
        "severity": np.stack([l["severity"] for l in labels]),
        "proxies": np.stack([proxy_vector(view_img) for view_img in views]),
    }


def _iter_prepared(tasks: Iterable[PrepareTask], workers: int) -> Iterator[dict]:
    """Yield `_prepare_image(task)` in task order, optionally in parallel.

    `workers <= 1` runs inline, which is the default and is exactly the
    previous behaviour. Otherwise a process pool is kept at most `2 * workers`
    images in flight: `Executor.map` would submit every task at once, and one
    image's views are ~8.6 MB at short-side 512, so an unbounded queue would
    hold the whole dataset in memory. (`Executor.map`'s `buffersize` argument
    that would do this for us is Python 3.14+; Kaggle runs 3.11.)

    The pool uses the "spawn" start method, which re-imports the caller's
    `__main__`. A script that calls `extract_bank` at module level therefore
    has to put it behind `if __name__ == "__main__":` (as
    `scripts/extract_features.py` does); otherwise use `workers=0`.
    """
    if workers <= 1:
        for task in tasks:
            yield _prepare_image(task)
        return

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    # "spawn", not the Linux default "fork": this process holds a CUDA context
    # (the backbone is already loaded) and is multi-threaded, and forking
    # either of those is a documented deadlock/corruption hazard. Spawned
    # children re-import this module and call `_prepare_image`, which needs no
    # inherited state -- that is exactly why it takes a plain tuple.
    pending: collections.deque = collections.deque()
    remaining = iter(tasks)
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        try:
            for task in itertools.islice(remaining, 2 * workers):
                pending.append(pool.submit(_prepare_image, task))
        except RuntimeError as exc:                     # pragma: no cover
            raise RuntimeError(
                "could not start the worker pool. workers > 0 uses the 'spawn' "
                "start method (this process holds a CUDA context, which must "
                "never be forked), and spawn re-imports the caller's __main__ "
                "module -- so the call to extract_bank must sit behind "
                '`if __name__ == "__main__":`. Otherwise pass workers=0.'
            ) from exc
        while pending:
            done = pending.popleft()
            nxt = next(remaining, None)
            if nxt is not None:
                pending.append(pool.submit(_prepare_image, nxt))
            yield done.result()


def extract_bank(
    manifest_df: pd.DataFrame,
    backbone_name: str,
    out_dir: str,
    n_views: int = N_VIEWS,
    seed: int = 20260827,
    device: str = "cuda",
    limit: int | None = None,
    exclude_families: tuple[str, ...] = (),
    batch_size: int = 16,
    resume: bool = False,
    checkpoint_every: int = CHECKPOINT_EVERY,
    workers: int = 0,
) -> str:
    """Extract a feature bank from `manifest_df` and write it to `out_dir`.

    View 0 is always the clean, unmodified image with an empty recipe
    (`FeatureBank.check_invariants` enforces this on the output). Views 1..
    n_views-1 are sampled augmented recipes, one recipe drawn per view from a
    generator keyed on (seed, each row's manifest index label, view index)
    (see the loop below for why, and what that requires of a caller that
    slices `manifest_df`).

    `limit` truncates `manifest_df` to its first `limit` rows before anything
    else. A caller that also shards must apply its own limit BEFORE calling
    `shard_frame` and leave this one at None -- passing both would truncate
    each shard a second time, so the N shards would no longer tile one
    contiguous prefix (`scripts/extract_features.py` does exactly that).

    `exclude_families` forbids an entire transform family (spec FAMILIES
    names) from every sampled recipe in the bank, supporting the A3-LOTO
    ablation run (spec §4.6). It is NOT recorded in the bank's config, so
    `merge_banks` cannot see a shard that was extracted with a different
    value; every shard of one bank must be given the same one.

    `resume=True` continues an extraction into an existing `out_dir`, skipping
    the rows already written (see `BankWriter`). The same `manifest_df`,
    `seed`, `backbone_name` and `n_views` must be given -- `BankWriter`
    refuses a resume whose config disagrees with what is on disk.
    `checkpoint_every` controls how often the metadata is flushed, which is
    also how much work a kill can cost.

    `workers` runs the CPU stage (decode, augment, proxies) in that many
    subprocesses while this process feeds the GPU. It is bit-identical to the
    serial path -- a view's pixels depend only on
    `(seed, row_id, view_idx)` -- and 0 or 1 means inline.
    """
    df = manifest_df if limit is None else manifest_df.iloc[:limit]

    if not df.index.is_unique:
        dupes = df.index[df.index.duplicated()].unique().tolist()
        raise ValueError(
            f"manifest_df index has {len(dupes)} duplicated label(s), e.g. "
            f"{dupes[:3]!r}; extract_bank keys each image's RNG on its index "
            "label, so a duplicate would make two different images silently "
            "draw identical views. Call df.reset_index(drop=True) yourself "
            "only if you accept that as a single, self-contained bank (never "
            "on a slice of a larger manifest you intend to compare or merge "
            "against other shards) -- otherwise deduplicate the index first.")

    model, spec = load_backbone(backbone_name, device=device)
    # `manifest_root` is where this session's copy of the dataset is mounted;
    # the bank stores each row's path relative to it, so a shard extracted on
    # Kaggle (/kaggle/input/<slug>/...) still fingerprints to what the frozen
    # manifest fingerprints to, and merges with shards extracted elsewhere.
    # None for a frame with no `rel_path` -- an ad-hoc fixture rather than a
    # frozen manifest -- in which case the bank falls back to absolute paths,
    # which are portable enough for a bank that never leaves one machine.
    writer = BankWriter(out_dir, len(df), n_views, spec.dim, backbone_name, seed,
                        manifest_sha256=manifest_fingerprint(df),
                        manifest_root=dataset_root(df),
                        resume=resume, checkpoint_every=checkpoint_every)

    # Each image's RNG is keyed on the row's own index label -- its position
    # in the frozen manifest -- not on write_idx (this call's local array
    # position). A shard/session may be handed a slice of the full manifest
    # (e.g. `full_df.iloc[40000:50000]`, which preserves original index
    # labels); write_idx would then restart at 0 for every shard and collide
    # in RNG-key space with another shard's images, so the same physical
    # image would draw different views depending on which shard processed it.
    # The index label survives that slicing as long as the caller does not
    # reset it, so this stays stable across shards, sessions, and restarts,
    # independent of processing order -- which is also what makes `resume`
    # and `bank.merge_banks` safe. The uniqueness check above is what makes
    # "index label uniquely identifies an image" a checked precondition
    # rather than an assumed one. int(row_id) also assumes an integer-valued
    # index -- true for every manifest this project produces
    # (write_manifest/read_manifest round-trip a plain RangeIndex; a boolean
    # --split filter and sort_values/sample preserve integer labels rather
    # than replacing them) -- and would raise on a string-ID manifest, which
    # is not a schema this project has.
    meta_rows: dict[int, dict] = {}
    tasks: list[PrepareTask] = []
    for write_idx, (row_id, row) in enumerate(df.iterrows()):
        if write_idx in writer.completed:
            continue                     # already written by an earlier session
        meta_rows[write_idx] = {
            "path": row["path"], "label": int(row["label"]),
            "generator": row["generator"], "source": row["source"],
            "split": row["split"]}
        tasks.append((write_idx, int(row_id), row["path"], n_views, seed,
                      exclude_families))

    for prepared in tqdm(_iter_prepared(tasks, workers), total=len(tasks),
                          desc=f"extract:{backbone_name}"):
        write_idx = prepared["write_idx"]
        feats = embed(model, spec, prepared["views"], device=device,
                      batch_size=batch_size)
        writer.write_image(
            write_idx,
            meta_rows[write_idx],
            row_id=prepared["row_id"],
            feats=feats,
            presence=prepared["presence"],
            severity=prepared["severity"],
            proxies=prepared["proxies"],
            recipes=prepared["recipes"],
        )
    writer.close()
    # The cheapest possible post-condition on a job that just ran for hours:
    # re-open what was written and check the bank's own invariants (view 0
    # clean in both presence and recipe encodings, row_ids unique). A bank
    # that violates them is unusable downstream, and finding that out here
    # costs seconds instead of a second extraction.
    FeatureBank.open(out_dir).check_invariants()
    return out_dir
