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

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from aigcdet.augment.recipes import Recipe
from aigcdet.augment.scenarios import EVAL_GRID
from aigcdet.features.backbones import embed, load_backbone
from aigcdet.features.bank import BankWriter, FeatureBank, manifest_fingerprint
from aigcdet.features.proxies import proxy_vector

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
                      batch_size: int = 16) -> str:
    """Write a bank whose view axis is the fixed evaluation condition axis.

    `config["conditions"]` records the condition names in view order, through
    `BankWriter`'s `extra_config`, so it participates in the resume equality
    check: continuing an extraction against a different condition list is a
    different bank, not a continuation.

    Like `extract_bank`, each row's RNG is keyed on its index label in the
    frozen manifest, never on this call's loop position, so a shard
    (`full_df.iloc[a:b]`, which preserves index labels) reproduces the full
    run's exact pixels.
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

    model, spec = load_backbone(backbone_name, device=device)
    writer = BankWriter(out_dir, len(df), len(names), spec.dim, backbone_name, seed,
                        manifest_sha256=manifest_fingerprint(df),
                        extra_config={"conditions": names})

    recipes = [conditions[n] for n in names]
    labels = [r.labels() for r in recipes]
    presence = np.stack([l["presence"] for l in labels])
    severity = np.stack([l["severity"] for l in labels])
    recipe_json = [r.to_json() for r in recipes]

    for write_idx, (row_id, row) in enumerate(
            tqdm(df.iterrows(), total=len(df), desc=f"eval:{backbone_name}")):
        with Image.open(row["path"]) as im:
            base = np.asarray(im.convert("RGB"), dtype=np.uint8)
        views = [r.apply(base, np.random.default_rng([seed, int(row_id), j]))
                 for j, r in enumerate(recipes)]
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


def assert_banks_comparable(banks: Sequence[FeatureBank]) -> None:
    """Refuse to compare banks that differ in view coverage, backbone or rows.

    Two rungs scored over different condition sets, different embeddings, or
    different images are not a model comparison at all: the numbers differ
    because the *evaluation* differed, which invalidates the model-selection
    kill criterion this table exists to serve.
    """
    if len(banks) < 2:
        return
    ref = banks[0]
    for other in banks[1:]:
        differing = {k: (ref.config.get(k), other.config.get(k))
                     for k in _COMPARABLE_KEYS
                     if ref.config.get(k) != other.config.get(k)}
        if differing:
            raise ValueError(
                f"banks at {ref.path} and {other.path} are not comparable: they "
                f"disagree on {differing} (first, this one). Differing view "
                "coverage, backbone or row set between compared rungs turns a "
                "rung comparison into a comparison of augmentation budgets, "
                "which invalidates the model-selection kill criterion.")


@torch.no_grad()
def score_grid(model, bank: FeatureBank, use_recon: bool = False,
               device: str = "cpu", batch_size: int = 4096,
               manifest_df: pd.DataFrame | None = None) -> pd.DataFrame:
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
    if manifest_df is not None:
        bank.verify_against_manifest(manifest_df)

    meta = bank.meta
    if len(meta) != bank.feats.shape[0]:
        raise ValueError(
            f"the bank at {bank.path} has {len(meta)} metadata rows but "
            f"{bank.feats.shape[0]} feature rows; it was not written to "
            "completion (resume the extraction before scoring it)")
    if use_recon and bank.recon is None:
        raise ValueError(
            f"the bank at {bank.path} has no recon features; run attach_recon "
            "before scoring a use_recon=True model")

    model.eval()
    frames = []
    for j, cond in enumerate(names):
        feats = np.asarray(bank.feats[:, j, :]).astype(np.float32)
        recon = (np.asarray(bank.recon[:, j, :]).astype(np.float32)
                 if use_recon else None)
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
