"""Does the union-trained model transfer to 2024-25 generators? (off-ladder)

    python scripts/ov7_transfer.py \
        --arm crop_dinov2regl=data/banks/eval_ov7_crop_dinov2regl:outputs/.../a3/checkpoint.pt \
        --arm band_siglipso400m=data/banks/eval_ov7_band_siglipso400m:outputs/.../a3/checkpoint.pt \
        --ov7-manifest /workspace/data/ov7/manifest_ov7.parquet \
        --union-manifest /workspace/data/manifest_union.parquet \
        --out docs/ov7_transfer.json

WHAT THIS IS AND IS NOT. No head is trained here. The heads are the full-scale
`a3` checkpoints fitted on the union corpus; AI-OV7 is read once, as an
external test set, and every number is a transfer number. That is the whole
point: `second_holdout_lattice.py` answers "does the ranking survive holding a
family out of OUR corpus", and this answers the different, larger question of
whether it survives a corpus assembled by someone else's pipeline over
generators that postdate ours.

WHY IT MATTERS MORE THAN ANOTHER HELD-OUT FAMILY. Every fake in the union
corpus comes from a 2017-2023 generator. Published results put detection at
~79% on 2020-21 generators against ~38% on 2024 ones, and AI-OV7 is the only
2024-25 material we may redistribute (`docs/ai_ov7_generation.md` §1). A four-
way that holds up here is a four-way with a claim about the shipping regime; a
four-way that collapses has been selected on an era, not on a signal.

TWO READINGS, AND THE FIRST ONE IS THE HONEST ONE.

  A. THE DESIGNED READING. AI-OV7 carries its own splits, drawn by
     `pair_split_by_stem` at exactly 50/50, and its `heldout_generator` is the
     klein4b lineage (`klein4b_t2i` 1,200 + `klein4b_ref_image` 600). Authentic
     from its `val_internal`, generated from its `heldout_generator`: that is
     `heldout_robust_tpr`'s own population rule applied unmodified, so the
     number is constructed exactly like the 0.9247 it is compared against.

     (NOTE FOR A LATER READER: `docs/ai_ov7_generation.md` §10 says this
     held-out family is `flux2_vae`. The frozen manifest disagrees and no
     `flux2_vae` row exists in it. The manifest is what shipped, so the
     manifest is what this script reads, and the doc line is wrong.)

  B. THE PER-FAMILY READING. For each generated family in turn, authentic =
     every OV7 real, generated = that family alone. This is diagnostic, not a
     headline: the families are of very unequal size (`sdxl_img2img` has 94
     rows in `val_internal` against `sdxl_t2i`'s 307), so it says WHERE the
     detector fails, and the small families say it loosely.

NOTHING IS FITTED ON AI-OV7. Weights arrive from the union fit or are equal;
sweeping them here would make this a selection split rather than a test split,
and there is no second AI-OV7 to recover the honest number from afterwards.
The z-score population is `val_internal` -- the authentic class -- exactly as
`FIT_SPLITS_WHEN_FITTING_WEIGHT` specifies, which is unsupervised with respect
to the generated class and so does not spend the test.

THE CONTAMINATION EXCLUSION, AND WHY IT IS A FLOOR. AI-OV7's reals and the
union's `open_images` reals are both drawn from the same 60,000-image Open
Images portrait pool. 71 of AI-OV7's 9,978 reals are byte-identical to a union
real, and 68 of those sit in the union's TRAIN split -- so a union-trained head
has already seen them. They are excluded here.

That exclusion is a floor and must not be read as "now it is clean": AI-OV7's
reals are MCU-aligned CROPS re-saved, so a crop of a photograph the union also
holds has different bytes and `content_sha256` cannot see it. `pixel_sha256`
is "" on every row of both manifests (`manifest.py:74` -- they were frozen with
byte digests), and a decoded-pixel digest would not catch a crop either. The
true photograph-level overlap is UNMEASURED. Treat a strong transfer number as
evidence and a weak one as conclusive, not the other way round.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from aigcdet.eval.errors import SELECTION_TARGET_FPR, heldout_robust_tpr
from aigcdet.eval.fusion import (
    FIT_SPLITS_WHEN_FITTING_WEIGHT, assert_fusion_parents, fuse_scores,
)
from aigcdet.eval.grid import score_grid
from aigcdet.features.bank import FeatureBank
from aigcdet.train.train_head import load_detector



def aux_flags(ck) -> dict:
    """The auxiliary-block flags a checkpoint was TRAINED with.

    `score_grid` takes three, not one. Passing only `use_recon` was silently
    wrong the moment a rung above a4 existed: an `aF` head is built with a
    frequency block on its input, and scoring it with `use_freq=False` feeds
    the head a different vector from the one it was fitted on. Derive all
    three from the checkpoint rather than from the caller, because the
    checkpoint is the only thing that knows.
    """
    cfg = ck["config"]
    return {"use_recon": bool(cfg.get("use_recon", False)),
            "use_recon_vq": bool(cfg.get("use_recon_vq", False)),
            "use_freq": bool(cfg.get("use_freq", False))}

def contaminated_reals(ov7_manifest: str, union_manifest: str) -> set[str]:
    """`rel_path` of AI-OV7 reals whose bytes also appear in the union corpus.

    Byte equality only -- see the module docstring on why this is a floor.

    RETURNS rel_path, NOT content_sha256, and the difference is load-bearing.
    An eval bank's `meta.parquet` carries `image_idx, row_id, path, label,
    generator, source, split, rel_path` and NO digest column, so a set of
    hashes has nothing in the bank to match against: the exclusion would run,
    match zero rows, and report a clean number that had excluded nothing. The
    hashes are therefore resolved to rel_paths HERE, against the manifest that
    does carry both, and `rel_path` is what the bank is filtered on.
    """
    ov = pd.read_parquet(ov7_manifest)
    un = pd.read_parquet(union_manifest)
    ov_real = ov[ov["label"] == 0]
    un_oi = un[un["source"] == "open_images"]
    shared = set(ov_real["content_sha256"].dropna()) & set(un_oi["content_sha256"].dropna())
    shared.discard("")
    return set(ov_real.loc[ov_real["content_sha256"].isin(shared), "rel_path"].astype(str))


def build_splits(meta: pd.DataFrame, *, generated: list[str] | None,
                 drop_sha: set[str]) -> pd.Series:
    """A split column naming the population `heldout_robust_tpr` should build.

    `generated=None` keeps AI-OV7's own splits (reading A). Otherwise every
    real becomes `val_internal`, the named families become `heldout_generator`
    and every other generated row becomes `train`, which the metric's
    population rule ignores (reading B).

    Rows whose `content_sha256` is in `drop_sha` are demoted to `train` rather
    than deleted, so all arms keep identical row sets and a fused frame stays
    well defined.
    """
    idx = meta.set_index("image_idx")
    split = idx["split"].astype(str).copy()
    if generated is not None:
        label = idx["label"].astype(int)
        gen = idx["generator"].astype(str)
        split = pd.Series("train", index=split.index, dtype=object)
        split[label == 0] = "val_internal"
        split[(label == 1) & gen.isin(generated)] = "heldout_generator"
    if drop_sha:
        if "rel_path" not in idx.columns:
            raise SystemExit(
                "the eval bank meta has no rel_path column, so the "
                "contamination exclusion has nothing to match on. Refusing to "
                "report a transfer number that silently excluded nothing.")
        split[idx["rel_path"].astype(str).isin(drop_sha)] = "train"
    return split.astype(str)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    metavar="NAME=EVAL_BANK:CHECKPOINT")
    ap.add_argument("--ov7-manifest", required=True)
    ap.add_argument("--union-manifest", required=True)
    ap.add_argument("--weights", default="",
                    help="comma-separated fusion weights carried in from the "
                         "UNION fit, in --arm order. Empty means equal. "
                         "Nothing is ever fitted on AI-OV7.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arms = []
    for spec in a.arm:
        name, _, rest = spec.partition("=")
        eval_dir, _, ckpt = rest.partition(":")
        if not (name and eval_dir and ckpt):
            raise SystemExit(f"--arm wants NAME=EVAL_BANK:CHECKPOINT, got {spec!r}")
        arms.append((name, eval_dir, ckpt))

    banks = [FeatureBank(d) for _, d, _ in arms]
    assert_fusion_parents(banks)
    meta = banks[0].meta

    drop = contaminated_reals(a.ov7_manifest, a.union_manifest)
    if "rel_path" not in meta.columns:
        raise SystemExit("the eval bank meta has no rel_path column; the "
                         "contamination exclusion cannot be applied.")
    present = int(meta["rel_path"].astype(str).isin(drop).sum())
    print(f"contamination: {len(drop)} AI-OV7 reals share bytes with a union "
          f"open_images real; {present} of them are in this eval bank and are "
          f"excluded (a FLOOR -- crops are invisible to a byte digest)")
    if drop and present == 0:
        raise SystemExit(
            f"{len(drop)} contaminated rel_paths were resolved but NONE matched "
            "a row in the eval bank. That is a join failure, not a clean "
            "corpus -- check that the bank was built from this manifest.")

    frames, singles = {}, {}
    splits_A = build_splits(meta, generated=None, drop_sha=drop)
    for name, eval_dir, ckpt in arms:
        model, ck = load_detector(ckpt, device=a.device)
        bank = banks[[n for n, _, _ in arms].index(name)]
        frames[name] = score_grid(model, bank,
                                  device=a.device, **aux_flags(ck))
        singles[name] = float(heldout_robust_tpr(frames[name], splits_A,
                                                 SELECTION_TARGET_FPR))
        print(f"  single {name:>22s}  {singles[name]:.4f}")

    names = [n for n, _, _ in arms]
    if a.weights:
        w = [float(x) for x in a.weights.split(",")]
        if len(w) != len(names):
            raise SystemExit(f"--weights has {len(w)} values for {len(names)} arms")
    else:
        w = None

    # ---- reading A: AI-OV7's own splits, klein4b held out -------------------
    out: dict = {"probe": "ov7_transfer", "off_ladder": True,
                 "metric": "heldout_robust_tpr_at_1pct",
                 "nothing_fitted_on_ov7": True,
                 "contamination_excluded_rows": present,
                 "contamination_is_a_floor": "byte digests cannot see a crop of "
                                             "a photograph the union also holds",
                 "singles_designed_split": singles}

    combos = {}
    for k in range(2, len(names) + 1):
        for sub in itertools.combinations(names, k):
            fused = fuse_scores([frames[x] for x in sub], splits=splits_A,
                                fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT)
            combos["+".join(sub)] = {
                "arity": k,
                "equal": float(heldout_robust_tpr(fused, splits_A,
                                                  SELECTION_TARGET_FPR))}
    if w is not None and len(names) >= 2:
        fw = fuse_scores([frames[x] for x in names], weights=w, splits=splits_A,
                         fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT)
        out["union_fitted_weights"] = w
        out["all_arms_union_fitted"] = float(
            heldout_robust_tpr(fw, splits_A, SELECTION_TARGET_FPR))
    out["combinations_designed_split"] = combos

    best_single = max(singles.values())
    best_combo = max((v["equal"] for v in combos.values()), default=float("nan"))
    print(f"\n=== READING A: AI-OV7's own splits (held out: klein4b lineage) ===")
    print(f"  best single        {best_single:.4f}")
    print(f"  best combination   {best_combo:.4f}")
    if w is not None:
        print(f"  all arms, union w  {out['all_arms_union_fitted']:.4f}  w={w}")

    # ---- reading B: per generated family ------------------------------------
    fam_counts = (meta[meta["label"] == 1]["generator"].astype(str)
                  .value_counts().to_dict())
    per_family = {}
    print(f"\n=== READING B: per generated family (diagnostic; unequal n) ===")
    print(f"  {'family':>20s} {'n':>6s}  " +
          "  ".join(f"{n[:12]:>12s}" for n in names) + f"  {'fused':>8s}")
    for fam in sorted(fam_counts):
        sp = build_splits(meta, generated=[fam], drop_sha=drop)
        row = {"n_rows": int(fam_counts[fam])}
        try:
            for n in names:
                row[n] = float(heldout_robust_tpr(frames[n], sp, SELECTION_TARGET_FPR))
            fused = fuse_scores([frames[x] for x in names], weights=w, splits=sp,
                                fit_splits=FIT_SPLITS_WHEN_FITTING_WEIGHT)
            row["fused"] = float(heldout_robust_tpr(fused, sp, SELECTION_TARGET_FPR))
        except Exception as exc:                       # too few rows for the FPR
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  {fam:>20s} {row['n_rows']:>6d}  SKIPPED ({row['error'][:60]})")
            per_family[fam] = row
            continue
        per_family[fam] = row
        print(f"  {fam:>20s} {row['n_rows']:>6d}  " +
              "  ".join(f"{row[n]:>12.4f}" for n in names) +
              f"  {row['fused']:>8.4f}")
    out["per_family"] = per_family

    worst = min(((f, r["fused"]) for f, r in per_family.items() if "fused" in r),
                key=lambda t: t[1], default=None)
    if worst:
        print(f"\n  worst family for the fusion: {worst[0]} at {worst[1]:.4f}")
        out["worst_family"] = {"family": worst[0], "fused": worst[1]}

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
