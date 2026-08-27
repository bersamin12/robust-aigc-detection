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
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from aigcdet.augment.recipes import FAMILIES, Recipe, sample_training_recipe
from aigcdet.features.backbones import embed, load_backbone
from aigcdet.features.bank import (
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
) -> str:
    """Extract a feature bank from `manifest_df` and write it to `out_dir`.

    View 0 is always the clean, unmodified image with an empty recipe
    (`FeatureBank.check_invariants` enforces this on the output). Views 1..
    n_views-1 are sampled augmented recipes, one recipe drawn per view from a
    generator keyed on (seed, each row's manifest index label, view index)
    (see the loop below for why, and what that requires of a caller that
    slices `manifest_df`).

    `exclude_families` forbids an entire transform family (spec FAMILIES
    names) from every sampled recipe in the bank, supporting the A3-LOTO
    ablation run (spec §4.6).
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
    writer = BankWriter(out_dir, len(df), n_views, spec.dim, backbone_name, seed,
                        manifest_sha256=manifest_fingerprint(df))

    rows = enumerate(tqdm(df.iterrows(), total=len(df), desc=f"extract:{backbone_name}"))
    for write_idx, (row_id, row) in rows:
        # Keyed on the row's own index label -- its position in the frozen
        # manifest -- not on write_idx (this call's local array position). A
        # shard/session may be handed a slice of the full manifest (e.g.
        # `full_df.iloc[40000:50000]`, which preserves original index
        # labels); write_idx would then restart at 0 for every shard and
        # collide in RNG-key space with another shard's images, so the same
        # physical image would draw different views depending on which shard
        # processed it. The index label survives that slicing as long as the
        # caller does not reset it, so this stays stable across shards,
        # sessions, and restarts, independent of processing order. The
        # uniqueness check above is what makes "index label uniquely
        # identifies an image" a checked precondition rather than an assumed
        # one. int(row_id) also assumes an integer-valued index -- true for
        # every manifest this project produces (write_manifest/read_manifest
        # round-trip a plain RangeIndex; a boolean --split filter and
        # sort_values/sample preserve integer labels rather than replacing
        # them) -- and would raise on a string-ID manifest, which is not a
        # schema this project has.
        rid = int(row_id)
        with Image.open(row["path"]) as im:
            base = np.asarray(im.convert("RGB"), dtype=np.uint8)

        # View 0 is always the clean view (bank invariant) -- no sampling.
        # Views 1..n_views-1 each get their own fresh generator, keyed on
        # (seed, rid, view index), to pick their recipe. This is a SEPARATE
        # generator instance from the one used to apply that same view below
        # -- both are re-derived from the same key rather than one shared
        # stream threaded through both steps, so how many draws the sampling
        # step happens to consume never shifts where the apply step's own
        # draws (the noise op's realisation, the only op that reads `rng`)
        # land. That is what makes a view's pixels reproducible from nothing
        # but (seed, rid, view index) and its own stored recipe -- see the
        # module docstring, and `recon.attach_recon_to_bank`, which relies on
        # exactly this to replay cached views without re-sampling them.
        recipes = [Recipe(())]
        for v in range(1, n_views):
            sample_rng = np.random.default_rng([seed, rid, v])
            recipes.append(_sample_recipe_excluding(sample_rng, exclude_families))

        views = []
        for v, r in enumerate(recipes):
            apply_rng = np.random.default_rng([seed, rid, v])
            views.append(r.apply(base, apply_rng))

        feats = embed(model, spec, views, device=device, batch_size=batch_size)
        labels = [r.labels() for r in recipes]
        writer.write_image(
            write_idx,
            {"path": row["path"], "label": int(row["label"]),
             "generator": row["generator"], "source": row["source"],
             "split": row["split"]},
            row_id=rid,
            feats=feats,
            presence=np.stack([l["presence"] for l in labels]),
            severity=np.stack([l["severity"] for l in labels]),
            proxies=np.stack([proxy_vector(view_img) for view_img in views]),
            recipes=[r.to_json() for r in recipes],
        )
    writer.close()
    # The cheapest possible post-condition on a job that just ran for hours:
    # re-open what was written and check the bank's own invariants (view 0
    # clean in both presence and recipe encodings, row_ids unique). A bank
    # that violates them is unusable downstream, and finding that out here
    # costs seconds instead of a second extraction.
    FeatureBank.open(out_dir).check_invariants()
    return out_dir
