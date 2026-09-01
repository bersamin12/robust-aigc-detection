"""Evaluation over the fixed condition grid (spec §6.1-6.2).

The eval bank reuses the Stage A layout, but its view axis is the CONDITION
axis: view j is always condition j, identically for every image. That makes
every rung's grid score a single pass over cached vectors.

The grid is deterministic by construction -- conditions are fixed recipes, not
sampled ones -- but `noise` and `jitter` still consume randomness when they are
applied. Each view's generator is therefore derived exactly the way Stage A
derives its own (`aigcdet.features.extract`): a fresh
`np.random.default_rng([seed, row_id, view_idx])` per view, never one stream
advanced in loop order. A view's pixels then depend only on its own key, so the
eval bank is shardable, restartable and reproducible run to run, and a shard of
the manifest yields byte-identical views to the full run.
"""
from __future__ import annotations

import os
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from aigcdet.augment.recipes import Recipe
from aigcdet.augment.scenarios import EVAL_GRID
from aigcdet.data.manifest import dataset_root
from aigcdet.features.backbones import embed, load_backbone
from aigcdet.features.bank import (
    CHECKPOINT_EVERY,
    BankWriter,
    FeatureBank,
    manifest_fingerprint,
)
from aigcdet.features.proxies import proxy_vector
from aigcdet.augment.canonical import DEFAULT_POLICY, CanonPolicy, canonicalise
from aigcdet.eval.tta import TTA_VIEWS, apply_tta_view

#: The benchmark subsample seed, fixed by the plan so the tier is reproducible
#: from nothing but this constant.
BENCHMARK_SEED = 20260827


def _check_condition_order(conditions: dict[str, Recipe]) -> list[str]:
    """Return the condition names, having checked view 0 is the clean view.

    `FeatureBank.check_invariants` requires view 0 to carry zero degradation
    presence AND an empty recipe. `EVAL_GRID` satisfies that because its first
    key is `clean`, but a caller may reorder or filter the mapping, so the
    precondition is checked here rather than assumed -- otherwise the bank is
    written first and rejected hours later.
    """
    names = list(conditions)
    if not names:
        raise ValueError("conditions must not be empty")
    first = names[0]
    if first != "clean" or conditions[first].ops != ():
        raise ValueError(
            f"condition 0 must be the undegraded view, named 'clean' with an "
            f"empty recipe, but it is {first!r} with ops "
            f"{conditions[first].ops!r}. The bank's view-0-is-clean invariant "
            "(FeatureBank.check_invariants) is positional, so reordering or "
            "filtering the grid must keep the clean condition first.")
    return names


def extract_eval_bank(manifest_df: pd.DataFrame, backbone_name: str, out_dir: str,
                      conditions: dict[str, Recipe] | None = None,
                      device: str = "cuda", seed: int = BENCHMARK_SEED,
                      batch_size: int = 16, resume: bool = False,
                      checkpoint_every: int = CHECKPOINT_EVERY,
                      policy: CanonPolicy = DEFAULT_POLICY,
                      extra_config: dict | None = None,
                      tta_views: Sequence[str] | None = None,
                      tower_checkpoint: str | None = None) -> str:
    """Write a bank whose view axis is the fixed evaluation condition axis.

    `config["conditions"]` records the condition names in view order, through
    `BankWriter`'s `extra_config`, so it participates in the resume equality
    check: continuing an extraction against a different condition list is a
    different bank, not a continuation. `extra_config` adds to that record --
    `scripts/extract_eval_bank.py` uses it to name the evaluation tier and the
    (n, seed) of the manifest subsample the bank was extracted from, both of
    which are otherwise unrecoverable from the artefact. Anything put there
    must agree across shards, because `bank.merge_banks` treats every
    unrecognised config key as a must-match extra; per-session facts (which
    shard this is, which mount it read) belong in `manifest_root`/`n_images`,
    which are already per-shard.

    Like `extract_bank`, each row's RNG is keyed on its index label in the
    frozen manifest, never on this call's loop position, so a shard
    (`full_df.iloc[a:b]`, which preserves index labels) reproduces the full
    run's exact pixels. `resume=True` continues an interrupted extraction into
    an existing `out_dir`, skipping the rows already written, and
    `checkpoint_every` is how much work a killed session can cost.

    **`tta_views` (rung A6).** Given a view list, the view axis becomes the
    CROSS PRODUCT `condition x tta_view`, flattened as `j * len(tta_views) + k`,
    and `config["tta_views"]` records the second factor. The order of
    composition is canonicalise -> CONDITION -> TTA view, and it is not
    interchangeable: the condition is what the world did to the image before it
    reached us, and TTA is what the detector chooses to do to the image it was
    handed. Applying TTA first would measure a detector that got to clean the
    picture up before it was degraded, which is not a detector anyone can
    deploy.

    The condition's RNG stays keyed on `(seed, row_id, j)` -- the CONDITION
    index, not the flattened one -- so all `len(tta_views)` views of condition
    `j` are views of the SAME degraded image rather than of `len(tta_views)`
    independent draws. That is what makes the average a test-time average over
    transforms instead of an average over noise realisations, and it has a
    checkable consequence: when `identity` is among the views, its column of
    condition `j` is built from exactly the pixels the plain eval bank's
    column `j` was built from, given the same seed.

    The PIXELS are identical and `tests/eval/test_tta_bank.py` asserts that
    exactly, under a deterministic stub embedder. What the two banks STORE
    agrees to about one float16 ULP rather than exactly, and the reason is
    worth knowing before someone reads a 1-ULP report as a bug: the plain bank
    embeds an image's 20 views in one batch and this one embeds 160, different
    batch shapes take different GEMM reduction orders, and the float16 cast of
    two float32 values that differ in the last bits can land one step apart.
    `scripts/verify_tta_bank.py` measures that distance in ULPs, where a real
    divergence -- a flipped composition order, a re-keyed RNG, a transposed
    flattening -- shows up three to four orders of magnitude above it.
    """
    conditions = EVAL_GRID if conditions is None else conditions
    names = _check_condition_order(conditions)

    df = manifest_df
    if not df.index.is_unique:
        dupes = df.index[df.index.duplicated()].unique().tolist()
        raise ValueError(
            f"manifest_df index has {len(dupes)} duplicated label(s), e.g. "
            f"{dupes[:3]!r}; extract_eval_bank keys each image's RNG on its "
            "index label, so a duplicate would make two different images draw "
            "identical noise. Deduplicate the index first.")

    # Validated before anything is decoded: an unknown view name is a typo that
    # would otherwise surface tens of thousands of forwards into the run.
    if tta_views is not None:
        tta_views = list(tta_views)
        if not tta_views:
            raise ValueError(
                "tta_views=[] would give this bank a zero-width view axis; "
                "pass None for a plain eval bank")
        unknown = [v for v in tta_views if v not in TTA_VIEWS]
        if unknown:
            raise ValueError(
                f"unknown TTA view(s) {unknown}; expected a subset of "
                f"{list(TTA_VIEWS)}")

    extras = {"conditions": names}
    if extra_config:
        clashing = sorted(set(extra_config) & set(extras))
        if clashing:
            raise ValueError(
                f"extra_config may not shadow {clashing}: the condition list is "
                "the bank's view axis and is written from `conditions`, not "
                "from a caller-supplied duplicate that could disagree with it")
        extras.update(extra_config)
    # The standardisation this bank's pixels went through. Same argument as in
    # `extract_bank`: a crop bank and a band bank have identical shapes,
    # dtypes and row counts, so without this nothing on disk distinguishes
    # them -- and an eval bank read against a rung trained under the other
    # policy would report a robustness curve for pixels the head never saw.
    extras["canon_policy"] = policy.as_record()
    # Recorded through `extra_config`, so it joins `conditions` in the resume
    # and merge equality checks: continuing a plain eval extraction into a TTA
    # bank (or the reverse) is a different bank, not a continuation, and their
    # view axes differ by a factor of eight with no shape error to say so.
    if tta_views is not None:
        extras["tta_views"] = list(tta_views)

    model, spec = load_backbone(backbone_name, device=device)
    # The unfreeze ladder (D1..D4) scores each depth on a bank extracted by ITS
    # OWN tower -- a tower whose weights have moved does not produce the
    # features on disk, so there is no way to reuse the frozen bank.
    #
    # The weights therefore become part of the bank's identity, and are
    # recorded as such. `BankWriter` treats every unrecognised config key as
    # must-match, so `tower_sha256` alone stops two depths' shards merging into
    # one bank -- which would otherwise be a bank half-computed by each of two
    # different models, with no shape error to say so. `unfreeze_depth` is
    # recorded beside it because a hash names nothing a human can act on.
    if tower_checkpoint:
        import hashlib

        ck = torch.load(tower_checkpoint, map_location="cpu", weights_only=False)
        if "tower_state_dict" not in ck:
            raise ValueError(
                f"{tower_checkpoint} has no 'tower_state_dict'. A frozen-head "
                "checkpoint cannot re-extract a bank: at depth 0 the tower is "
                "unchanged and the existing bank already holds its features, "
                "and at depth > 0 the head alone does not describe the model.")
        sd = ck["tower_state_dict"]
        model.load_state_dict(sd)
        model.eval()
        h = hashlib.sha256()
        for k in sorted(sd):
            h.update(k.encode())
            h.update(np.ascontiguousarray(
                sd[k].detach().to(torch.float32).cpu().numpy()).tobytes())
        extras["tower_sha256"] = h.hexdigest()
        extras["tower_checkpoint"] = os.path.basename(tower_checkpoint)
        extras["unfreeze_depth"] = int(
            (ck.get("unfrozen") or {}).get("depth", -1))
    # `manifest_root` is where this session's copy of the dataset is mounted.
    # The bank stores each row's path relative to it, so an eval shard
    # extracted on Kaggle (/kaggle/input/<slug>/...) carries the same portable
    # identity as the frozen manifest and merges with shards extracted
    # elsewhere. Without it every row's `rel_path` is an absolute path, and
    # `merge_banks` fingerprints the merged bank over strings no other machine
    # produces -- so `verify_against_manifest` and `report._check_banks` refuse
    # the merged artefact after the whole fleet has paid for it. None for a
    # frame with no `rel_path` (an ad-hoc fixture rather than a frozen
    # manifest), which falls back to absolute paths.
    n_views = len(names) * (1 if tta_views is None else len(tta_views))
    writer = BankWriter(out_dir, len(df), n_views, spec.dim, backbone_name, seed,
                        manifest_sha256=manifest_fingerprint(df),
                        manifest_root=dataset_root(df),
                        resume=resume, checkpoint_every=checkpoint_every,
                        extra_config=extras)

    recipes = [conditions[n] for n in names]
    labels = [r.labels() for r in recipes]
    presence = np.stack([l["presence"] for l in labels])
    severity = np.stack([l["severity"] for l in labels])
    recipe_json = [r.to_json() for r in recipes]
    if tta_views is not None:
        # The degradation labels describe the CONDITION, so each condition's
        # row is repeated once per TTA view rather than recomputed from the
        # view. This is not an approximation being tolerated: `jpeg_95` and
        # `blur_0.3` are things the DETECTOR chose to do to the image it was
        # handed, not things that happened to the image in the world, and the
        # degradation head's target is the latter. Labelling a TTA-blurred
        # view as "blurred" would teach the readout to report the detector's
        # own preprocessing as evidence about the image's history.
        presence = np.repeat(presence, len(tta_views), axis=0)
        severity = np.repeat(severity, len(tta_views), axis=0)
        # The CONDITION's recipe, repeated -- not a composite object naming
        # the TTA view as well. Two reasons, and either alone is decisive.
        # `recipe_json` must stay a parseable `Recipe`: `check_invariants`
        # replays it, so a richer object here fails at the end of every
        # extraction. And the TTA view does not belong in a recipe in the
        # first place, for the same reason it does not appear in `presence` --
        # a recipe records what happened to the image, and `jpeg_95` is what
        # the DETECTOR chose to do to it. Which view a column is remains fully
        # recoverable: `config["tta_views"]` and the flattening give it, and
        # `tta_axis` refuses a bank where those two disagree.
        recipe_json = [recipe_json[j] for j in range(len(names))
                       for _ in tta_views]

    for write_idx, (row_id, row) in enumerate(
            tqdm(df.iterrows(), total=len(df), desc=f"eval:{backbone_name}")):
        if write_idx in writer.completed:
            continue                     # already written by an earlier session
        with Image.open(row["path"]) as im:
            base = np.asarray(im.convert("RGB"), dtype=np.uint8)
        # Resolution canonicalisation, BEFORE any recipe/condition transform
        # (docs/resolution_shortcut.md). Native resolution leaks the label:
        # 29% of the training pool sits at sizes that are 100% generated, and
        # the scored benchmark separates perfectly on dimensions alone. The
        # backbone squishes to a fixed square, so what survives is the
        # resampling signature -- soft-because-upscaled correlates with the
        # label. Applied identically to both classes.
        #
        # This MUST be applied at every decode site or at none: recon.py
        # replays stored recipes to reproduce exact cached pixels, so a site
        # that skips it computes features on different pixels than were
        # cached, silently and with no shape error. Wire exactly once per
        # site -- canonicalise is size-stable but NOT pixel-idempotent.
        #
        # `rng=None`, so under a crop policy this is the CENTRE window and
        # under a band policy the band is unjittered -- deterministic either
        # way, and identical to what `infer.Predictor` does at serving time.
        # Two reasons, and the first is the grid's whole purpose: it measures
        # how far a score falls under `jpeg_q30`, so if the window also moved
        # between conditions that measurement would be confounded with "a
        # different picture". The second is that an eval number is a
        # prediction about serving, and a bank standardised differently from
        # the served path is predicting something else.
        #
        # For the same reason there is no dihedral here. The training bank
        # applies one per view to teach orientation invariance; evaluating on
        # a random orientation would measure that invariance by accident and
        # differently for every image. Averaging a score OVER orientations is
        # a real technique and it is A6's, applied at inference to both the
        # eval set and the served path alike.
        base = canonicalise(base, policy=policy)
        views = [r.apply(base, np.random.default_rng([seed, int(row_id), j]))
                 for j, r in enumerate(recipes)]
        if tta_views is not None:
            # Flattened `j * len(tta_views) + k`, and the condition's own view
            # is computed ONCE and then transformed, so every TTA view of
            # condition j is a view of the same degraded image. Keying the
            # condition RNG on the flattened index instead would draw a fresh
            # noise realisation per TTA view and turn the average over
            # transforms into an average over noise -- a strictly easier
            # problem, and one that would flatter A6 for the wrong reason.
            views = [apply_tta_view(deg, v) for deg in views for v in tta_views]
        writer.write_image(
            write_idx,
            {"path": row["path"], "label": int(row["label"]),
             "generator": row["generator"], "source": row["source"],
             "split": row["split"]},
            feats=embed(model, spec, views, device=device, batch_size=batch_size),
            presence=presence,
            severity=severity,
            proxies=np.stack([proxy_vector(v) for v in views]),
            recipes=recipe_json,
            row_id=int(row_id),
        )
    writer.close()
    FeatureBank.open(out_dir).check_invariants()
    return out_dir


#: Config keys two banks must agree on before their scores may be compared.
#: `conditions` is the view coverage itself -- two banks over the same twenty
#: conditions in a different ORDER agree on `n_views` and are still not
#: comparable, because view j means a different thing in each. `n_views` stays
#: alongside it as the check that also applies to a training bank, which has
#: no `conditions` at all. A mismatch in either turns a rung comparison into a
#: comparison of augmentation budgets (project constraint: "identical view
#: coverage across compared rungs").
_COMPARABLE_KEYS = ("n_views", "conditions", "backbone", "manifest_sha256")

#: The evaluation-axis half of `_COMPARABLE_KEYS`: what was scored, over which
#: views, from which manifest. Every bank in a table must agree on all three,
#: composite or not. `backbone` is the other half and is handled separately,
#: because a declared composite is allowed to disagree on it (R43, below).
_AXIS_KEYS = tuple(k for k in _COMPARABLE_KEYS if k != "backbone")

#: Config keys by which a bank DECLARES itself a composite of other banks
#: (`eval.fusion.FusedEvalBank` writes both).
_COMPOSITE_KEYS = ("fused_from", "fused_backbones")


def _declared_composite(config: Mapping) -> bool:
    """Whether this bank declares itself a composite, and says so consistently.

    Rung A5 as the spec defines it is DINOv3 + SigLIP2 fused (§6.4), so its row
    has two backbones and `_COMPARABLE_KEYS` would refuse it a place in the
    results table beside any single-backbone rung -- meaning the A5 the spec
    describes could not be reported at all. The ruled fix (R43) is that a
    declared composite is comparable when its parents are comparable on every
    OTHER key, which `eval.fusion.assert_fusion_parents` already forces
    (`manifest_sha256`, `n_views`, `conditions`, and element-wise `split` and
    `label`), and which the composite's own `_AXIS_KEYS` therefore state.

    The exemption attaches to the DECLARATION, not to the row. A bank claiming
    it is fused must name its parents and their backbones, and its `backbone`
    must be exactly the composite of those names -- so a row that borrowed one
    parent's name is refused here rather than being let into the table under a
    label that hides half of what produced it (the R24 confound, laundered).
    """
    parents = config.get("fused_from")
    backbones = config.get("fused_backbones")
    if parents is None and backbones is None:
        return False
    if not isinstance(parents, (list, tuple)) or not isinstance(backbones, (list, tuple)):
        raise ValueError(
            f"a bank declares itself fused but does not say what from: "
            f"fused_from={parents!r}, fused_backbones={backbones!r}. Both must "
            "be lists, because the composite's exemption from the backbone "
            "check is granted to what it declares.")
    if len(parents) != len(backbones) or len(parents) < 2:
        raise ValueError(
            f"a bank declares itself fused from {len(parents)} parent(s) with "
            f"{len(backbones)} backbone(s); a fusion has at least two parents "
            "and one recorded backbone per parent.")
    expected = "+".join(dict.fromkeys(str(b) for b in backbones))
    if str(config.get("backbone")) != expected:
        raise ValueError(
            f"a bank declares itself fused from backbones {list(backbones)}, "
            f"whose composite name is {expected!r}, but records "
            f"backbone={config.get('backbone')!r}. A composite row is admitted "
            "beside single-backbone rungs because it SAYS what produced it; one "
            "that borrows a parent's name is the R24 confound wearing a label.")
    return True


def assert_banks_comparable(banks: Sequence[FeatureBank]) -> None:
    """Refuse to compare banks that differ in view coverage, backbone or rows.

    Two rungs scored over different condition sets, different embeddings, or
    different images are not a model comparison at all: the numbers differ
    because the *evaluation* differed, which invalidates the model-selection
    kill criterion this table exists to serve.

    The one exception is the backbone of a DECLARED composite (`_declared_composite`,
    R43): rung A5 IS the two-backbone rung, so its row names both and is not
    refused for doing so. Every bank that is not a composite must still agree
    with every other on the backbone -- the check is over the set of plain
    backbones, not against whichever bank happened to be passed first, so a
    composite in the list cannot become a bridge that lets two single-backbone
    rungs on different embeddings into the same table.
    """
    if len(banks) < 2:
        return
    configs = [getattr(b, "config", {}) for b in banks]
    composite = [_declared_composite(c) for c in configs]
    ref, ref_config = banks[0], configs[0]
    for other, config in zip(banks[1:], configs[1:]):
        differing = {k: (ref_config.get(k), config.get(k))
                     for k in _AXIS_KEYS if ref_config.get(k) != config.get(k)}
        if differing:
            raise ValueError(
                f"banks at {ref.path} and {other.path} are not comparable: they "
                f"disagree on {differing} (first, this one). Differing view "
                "coverage, backbone or row set between compared rungs turns a "
                "rung comparison into a comparison of augmentation budgets, "
                "which invalidates the model-selection kill criterion.")
    plain = {getattr(bank, "path", "?"): str(config.get("backbone"))
             for bank, config, is_composite in zip(banks, configs, composite)
             if not is_composite}
    if len(set(plain.values())) > 1:
        raise ValueError(
            f"banks at {sorted(plain)} are not comparable: they disagree on "
            f"backbone {plain}. Differing view "
            "coverage, backbone or row set between compared rungs turns a rung "
            "comparison into a comparison of augmentation budgets, which "
            "invalidates the model-selection kill criterion. (A rung that "
            "DECLARES itself a fusion of several backbones -- rung A5 -- is "
            "exempt from this key and is not counted here.)")


@torch.no_grad()
def score_grid(model, bank: FeatureBank, use_recon: bool = False,
               device: str = "cpu", batch_size: int = 4096,
               manifest_df: pd.DataFrame | None = None, *,
               use_recon_vq: bool = False,
               use_freq: bool = False) -> pd.DataFrame:
    """One row per (condition, image): the model's logit on that cached view.

    `manifest_df`, when the caller has it, is checked against the bank before
    anything is scored. Rows are positional, so a bank scored against a
    re-split manifest reports labels that belong to other images.
    """
    names = bank.config.get("conditions")
    if names is None:
        raise ValueError(
            f"the bank at {bank.path} has no 'conditions' in its config.json, so "
            "its view axis is not the condition axis -- score_grid needs a bank "
            "written by extract_eval_bank, not a training bank")
    # A TTA bank ALSO has `conditions`, and its column `j` is
    # (condition j // n_tta, view j % n_tta) rather than condition `j`. Nothing
    # about that is visible downstream: the loop below would read twenty of the
    # hundred and sixty columns, label them with the twenty condition names,
    # and return a frame of exactly the right shape in which `jpeg_q90` is
    # really an hflip of `clean`. Refused here, because it is the one caller
    # that cannot detect it for itself.
    if bank.config.get("tta_views"):
        raise ValueError(
            f"the bank at {bank.path} declares tta_views="
            f"{list(bank.config['tta_views'])}, so its view axis is "
            "condition x tta_view, not condition. score_grid would read its "
            f"first {len(names)} columns -- which are all views of the first "
            "few conditions -- and label them with the condition names. Use "
            "score_grid_tta, which averages each condition's views.")
    if manifest_df is not None:
        bank.verify_against_manifest(manifest_df)

    meta = bank.meta
    if len(meta) != bank.feats.shape[0]:
        raise ValueError(
            f"the bank at {bank.path} has {len(meta)} metadata rows but "
            f"{bank.feats.shape[0]} feature rows; it was not written to "
            "completion (resume the extraction before scoring it)")
    blocks = bank.aux_blocks(use_recon, use_recon_vq, use_freq)

    model.eval()
    frames = []
    for j, cond in enumerate(names):
        feats = np.asarray(bank.feats[:, j, :]).astype(np.float32)
        recon = (np.concatenate(
            [np.asarray(b[:, j, :]).astype(np.float32) for b in blocks],
            axis=-1) if blocks else None)
        scores = []
        for s in range(0, len(feats), batch_size):
            f = torch.from_numpy(feats[s:s + batch_size]).to(device)
            r = (torch.from_numpy(recon[s:s + batch_size]).to(device)
                 if recon is not None else None)
            scores.append(model(f, r)["logit"].cpu().numpy())
        frames.append(pd.DataFrame({
            "condition": cond,
            "image_idx": meta["image_idx"].to_numpy(),
            "label": meta["label"].to_numpy(),
            "generator": meta["generator"].to_numpy(),
            "source": meta["source"].to_numpy(),
            "score": np.concatenate(scores),
        }))
    return pd.concat(frames, ignore_index=True)


def tta_axis(bank) -> list[str]:
    """The TTA view list a bank was built over, or `[]` for a plain eval bank.

    Read through this rather than `config["tta_views"]` directly, so the one
    place that decides "is this a TTA bank" is also the place that checks the
    axis it declares is consistent with the width it actually has. A bank whose
    `n_views` is not `len(conditions) * len(tta_views)` was written by a version
    that disagreed with this one about the flattening, and every column index
    computed from it would be off by a silent amount.
    """
    views = list(bank.config.get("tta_views") or [])
    if not views:
        return []
    names = bank.config.get("conditions") or []
    n_views = int(bank.config.get("n_views", bank.feats.shape[1]))
    if n_views != len(names) * len(views):
        raise ValueError(
            f"the bank at {bank.path} declares {len(names)} condition(s) and "
            f"{len(views)} TTA view(s), which is {len(names) * len(views)} "
            f"columns, but its view axis is {n_views} wide. Its flattening does "
            "not match this version's `j * len(tta_views) + k`, so every column "
            "read out of it would be the wrong view.")
    return views


@torch.no_grad()
def score_grid_tta(model, bank: FeatureBank, use_recon: bool = False,
                   device: str = "cpu", batch_size: int = 4096,
                   manifest_df: pd.DataFrame | None = None, *,
                   use_recon_vq: bool = False,
                   use_freq: bool = False) -> pd.DataFrame:
    """Rung A6. One row per (condition, image), scored as the MEAN LOGIT over
    the bank's TTA views of that condition.

    The output frame has exactly the shape `score_grid` returns -- same
    columns, same rows, one score per image per condition -- which is what lets
    A6 be a row in the results table rather than a footnote beside it. The 8x
    is spent inside this function and does not appear on the evaluation axis,
    because after averaging A6 has covered the same twenty conditions over the
    same images with the same backbone as every other rung. What it does appear
    on is inference COST, which `TtaEvalBank` records.

    Logits, not probabilities (`eval.tta` docstring): the mean of a saturated
    and an unsaturated probability is dominated by the saturated one, whereas
    in logit space every view contributes on the same scale.

    The returned scores are means of `len(views)` correlated logits and so have
    a NARROWER spread than the single-view logits any fitted temperature was
    calibrated on. They must not be pushed through a `T` fitted on single-view
    logits; refit it on these. Selection is unaffected either way -- TPR at a
    fixed FPR is computed within a condition and is invariant to a monotone
    rescale -- but calibration, EQI and the dashboard are not.
    """
    views = tta_axis(bank)
    if not views:
        raise ValueError(
            f"the bank at {bank.path} has no 'tta_views' in its config.json, so "
            "it has no TTA axis to average over -- score_grid_tta needs a bank "
            "built by `extract_eval_bank(..., tta_views=...)`. Scoring a plain "
            "eval bank here would average over CONDITIONS instead, which is not "
            "a robustness measurement at all.")
    names = bank.config["conditions"]
    if manifest_df is not None:
        bank.verify_against_manifest(manifest_df)

    meta = bank.meta
    if len(meta) != bank.feats.shape[0]:
        raise ValueError(
            f"the bank at {bank.path} has {len(meta)} metadata rows but "
            f"{bank.feats.shape[0]} feature rows; it was not written to "
            "completion (resume the extraction before scoring it)")
    blocks = bank.aux_blocks(use_recon, use_recon_vq, use_freq)

    model.eval()
    frames = []
    for j, cond in enumerate(names):
        per_view = []
        for k in range(len(views)):
            col = j * len(views) + k
            feats = np.asarray(bank.feats[:, col, :]).astype(np.float32)
            recon = (np.concatenate(
                [np.asarray(b[:, col, :]).astype(np.float32) for b in blocks],
                axis=-1) if blocks else None)
            scores = []
            for st in range(0, len(feats), batch_size):
                f = torch.from_numpy(feats[st:st + batch_size]).to(device)
                r = (torch.from_numpy(recon[st:st + batch_size]).to(device)
                     if recon is not None else None)
                scores.append(model(f, r)["logit"].cpu().numpy())
            per_view.append(np.concatenate(scores))
        frames.append(pd.DataFrame({
            "condition": cond,
            "image_idx": meta["image_idx"].to_numpy(),
            "label": meta["label"].to_numpy(),
            "generator": meta["generator"].to_numpy(),
            "source": meta["source"].to_numpy(),
            "score": np.mean(np.stack(per_view, axis=0), axis=0),
        }))
    return pd.concat(frames, ignore_index=True)


def assert_tta_bank_matches(plain, tta) -> None:
    """Refuse an A6 bank that is not the same evaluation as the rest of the run.

    A6's whole claim is "the same images under the same conditions, scored
    eight ways". Every part of that is a property of the bank, and every part
    of it fails silently: a TTA bank over a different manifest subsample, or
    with the conditions in a different order, or canonicalised under the other
    policy, has the right shape and the wrong meaning, and the score it
    produces lands in the table beside A3 as though the two were comparable.

    Checked BEFORE the ladder trains, because the alternative is discovering it
    after every rung has been fitted.
    """
    if not tta_axis(tta):
        raise ValueError(
            f"the bank at {tta.path} declares no 'tta_views', so it is a plain "
            "eval bank. Averaging over the axis it does have would average "
            "over CONDITIONS, which is not a robustness measurement.")
    for key in ("manifest_sha256", "conditions", "backbone", "canon_policy"):
        want, got = plain.config.get(key), tta.config.get(key)
        if want != got:
            raise ValueError(
                f"the TTA bank at {tta.path} disagrees with the eval bank at "
                f"{plain.path} on {key!r}: {got!r} against {want!r}. A6 must be "
                "the same evaluation as the rungs it is tabulated beside, or "
                "its row compares an evaluation rather than a rung.")
    if len(plain.meta) != len(tta.meta):
        raise ValueError(
            f"the TTA bank at {tta.path} holds {len(tta.meta)} rows and the "
            f"eval bank at {plain.path} holds {len(plain.meta)}. A6 would be "
            "scored on a different subsample of the tier.")
    a = plain.meta["row_id"].to_numpy()
    b = tta.meta["row_id"].to_numpy()
    if not np.array_equal(a, b):
        n = int((a != b).sum()) if a.shape == b.shape else -1
        raise ValueError(
            f"the TTA bank at {tta.path} and the eval bank at {plain.path} hold "
            f"the same number of rows in a different ORDER ({n} position(s) "
            "differ). Scores are joined positionally, so A6's row would carry "
            "another image's label.")


class TtaEvalBank:
    """The evaluation identity of rung A6, as `robustness_table` reads it.

    Duck-types the `path`/`config` surface that `assert_banks_comparable` and
    `report._check_banks` use, exactly as `eval.fusion.FusedEvalBank` does for
    A5, and for the same reason: the row must register the evaluation that
    produced it rather than borrow another one.

    The declared `n_views` is the CONDITION count, not the bank's physical
    width. That is the honest statement and it is worth being explicit about,
    because the alternative reading is available and wrong. `n_views` is in
    `_COMPARABLE_KEYS` to stop a table from comparing augmentation BUDGETS --
    a rung scored over more views of the world than another has been asked an
    easier or a harder question. A6 is not that: after averaging it has been
    asked about the same 4000 images under the same 20 conditions as every
    other rung, and the eight views are eight looks at ONE degraded image,
    collapsed before a single number leaves this bank. Reporting 160 here would
    make A6 non-comparable with its own base rung and so unreportable, which is
    the A5 problem (R43) in a different costume.

    What the 8x really is, is inference cost, and that is recorded -- under its
    own name, next to the view list -- so no reader can mistake A6's score for
    one obtained at the same price as A3's.
    """

    def __init__(self, bank):
        self.bank = bank
        views = tta_axis(bank)
        if not views:
            raise ValueError(
                f"the bank at {getattr(bank, 'path', '?')} declares no TTA "
                "views, so it is not the bank that produced an A6 row; "
                "registering it here would state a cost multiplier of one for "
                "a rung that paid eight")
        names = list(bank.config.get("conditions") or [])
        self.path = f"tta({bank.path})"
        self.config = {
            "n_views": len(names),
            "conditions": names,
            "manifest_sha256": bank.config.get("manifest_sha256"),
            "backbone": bank.config.get("backbone"),
            "n_images": int(len(bank.meta)),
            "tta_views": list(views),
            "tta_cost_multiplier": len(views),
            "physical_n_views": int(bank.config.get("n_views", 0)),
        }

    def __repr__(self) -> str:
        return f"TtaEvalBank({self.path!r})"


def _allocate(capacities: list[int], total: int, rng: np.random.Generator) -> list[int]:
    """Spread `total` draws over bins as evenly as their capacities allow.

    Bins that run out give their share back to the rest, so a small generator
    family caps its own quota instead of leaving the tier short. When the
    remainder is smaller than the number of bins still open, the leftover
    single draws go to a random subset of them -- picked from `rng`, so the
    result is still reproducible from the seed alone.
    """
    alloc = [0] * len(capacities)
    remaining = int(total)
    active = [b for b, c in enumerate(capacities) if c > 0]
    while remaining > 0 and active:
        share = remaining // len(active)
        if share == 0:
            for b in rng.permutation(np.array(active))[:remaining]:
                alloc[int(b)] += 1
            break
        for b in active:
            take = min(share, capacities[b] - alloc[b])
            alloc[b] += take
            remaining -= take
        active = [b for b in active if capacities[b] - alloc[b] > 0]
    return alloc


def stratified_subsample(meta_df: pd.DataFrame, n: int,
                         seed: int = BENCHMARK_SEED) -> np.ndarray:
    """Positional indices balanced across class x generator x source (spec §4.4a).

    A uniform random subsample would under-represent small generator families,
    which are exactly the ones the held-out evaluation cares about. The
    balancing is hierarchical: the classes split `n` evenly first, then each
    class's quota is split evenly over its (generator, source) strata. Flat
    per-stratum balancing does NOT preserve class balance -- with one real
    stratum and two fake ones it returns two-thirds fakes -- and class balance
    is what every metric in this project is read against.

    Reproducible from `seed` alone, and returns sorted positional indices so
    the caller can `meta_df.iloc[...]` regardless of the frame's index labels.
    """
    if n >= len(meta_df):
        return np.arange(len(meta_df))
    rng = np.random.default_rng(seed)

    label = meta_df["label"].to_numpy()
    stratum = (meta_df["generator"].astype(str) + "|"
               + meta_df["source"].astype(str)).to_numpy()

    class_pos = [np.where(label == c)[0] for c in np.unique(label)]
    per_class = _allocate([len(p) for p in class_pos], n, rng)

    picked: list[np.ndarray] = []
    for pos, quota in zip(class_pos, per_class):
        keys = stratum[pos]
        strata = sorted(set(keys.tolist()))
        members = [pos[keys == k] for k in strata]
        for idx, take in zip(members, _allocate([len(m) for m in members], quota, rng)):
            if take:
                picked.append(rng.choice(idx, size=take, replace=False))
    if not picked:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(picked).astype(np.int64))


def assert_heldout_not_trained(train_bank, eval_bank,
                               excluded: Sequence[str] = ()) -> None:
    """Refuse to score a held-out family the training bank actually trained on.

    A lineage holdout is two halves that live in different files, and each is
    silent without the other. `build_eval_manifest --extra-heldout-generators`
    promotes a family's rows into the eval manifest's `heldout_generator`
    split, so there is something to score; `RungConfig.train_exclude_generators`
    drops the same family from the training rows, so the score means anything.
    Do only the first and the head trained on precisely the family the headline
    calls unseen -- and nothing about the resulting number looks wrong. It is
    in range, it is plausible, and it is inflated.

    Nothing else can catch this. `assert_banks_comparable` compares the two
    banks' shape, backbone and manifest, all of which agree here. The metric
    itself (`errors.heldout_robust_tpr`) builds its population from the eval
    bank's split column alone, which is exactly the column that has been told
    the family is held out.

    So the check is a comparison ACROSS the two banks: every generator the eval
    bank scores as held out, that the training bank also has rows for in
    `train`, must appear in `excluded`.
    """
    ev, tr = eval_bank.meta, train_bank.meta
    for name, meta in (("eval", ev), ("train", tr)):
        for col in ("split", "generator"):
            if col not in meta.columns:
                raise ValueError(
                    f"the {name} bank's meta has no {col!r} column, so a "
                    "lineage holdout cannot be checked. Re-extract with a "
                    "current BankWriter.")
    scored = set(map(str, ev.loc[ev["split"] == "heldout_generator", "generator"]))
    trained = set(map(str, tr.loc[tr["split"] == "train", "generator"]))
    leaked = sorted((scored & trained) - {str(e) for e in excluded} - {""})
    if leaked:
        raise ValueError(
            f"generator(s) {leaked} are scored as held out by the eval bank "
            "and are also in the training bank's `train` split, and the rung "
            "did not exclude them. The headline would be a score on families "
            "the head trained on. Pass them as "
            f"train_exclude_generators={tuple(leaked)!r}, or rebuild the eval "
            "manifest without --extra-heldout-generators.")
