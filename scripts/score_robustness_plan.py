"""Robustness grid for a finetuned checkpoint, from pixels, on a plan split.

Mirrors `eval/grid.py`'s protocol exactly: canonicalise FIRST (centre window,
no rng), then apply each condition's recipe to the canonicalised image with an
rng keyed on (seed, row_index, condition_index). The clean condition is view 0,
and every condition sees the same window, so a fall from clean measures the
degradation and nothing else.

Conditions are the brief's core grid at every severity the report table asks
for: the repo's jpeg q30 AND the table's q10, the combined jitter_20 AND the
three single-channel jitters. One parquet of per-row per-condition
probabilities, one metrics json with AUC / TPR@1%FPR / delta-vs-clean.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from aigcdet.augment.canonical import canonicalise
from aigcdet.augment.recipes import Op, Recipe
from aigcdet.train.finetune import _forward_tower, _windowed

from score_plan_splits import _load_models, _policy_from_config, tpr_at_fpr

CONDITIONS: dict[str, Recipe] = {
    "clean": Recipe(()),
    **{f"jpeg_q{q}": Recipe((Op("jpeg", {"quality": q}),))
       for q in (90, 70, 50, 30, 10)},
    **{f"blur_s{s}": Recipe((Op("blur", {"sigma": s}),)) for s in (0.5, 1.0, 2.0)},
    **{f"resize_{sc}": Recipe((Op("resize", {"scale": sc}),)) for sc in (0.5, 0.25)},
    **{f"noise_s{s}": Recipe((Op("noise", {"sigma": s}),)) for s in (0.02, 0.05, 0.10)},
    "jitter_b20": Recipe((Op("jitter", {"brightness": 0.2, "contrast": 0.0,
                                        "saturation": 0.0}),)),
    "jitter_c20": Recipe((Op("jitter", {"brightness": 0.0, "contrast": 0.2,
                                        "saturation": 0.0}),)),
    "jitter_s20": Recipe((Op("jitter", {"brightness": 0.0, "contrast": 0.0,
                                        "saturation": 0.2}),)),
    "jitter_20": Recipe((Op("jitter", {"brightness": 0.2, "contrast": 0.2,
                                       "saturation": 0.2}),)),
    "crop_80": Recipe((Op("crop", {"frac": 0.8}),)),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="data/manifest_plan.parquet")
    ap.add_argument("--split", default="test_transfer")
    ap.add_argument("--limit", type=int, default=4000,
                    help="rows, stratified by (source,label), random_state=0 "
                         "so every checkpoint scores the SAME rows; 0 = all")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8,
                    help="rows per batch (each row expands to one view per "
                         "condition)")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--swa", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-short-side", type=int, default=200)
    a = ap.parse_args()

    from PIL import Image
    from tqdm import tqdm

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    towers, specs, head, tag = _load_models(ck, a.device, a.swa)
    if head.use_recon:
        raise SystemExit("REFUSING: recon head needs the recon branch wired "
                         "per view; this scorer covers plain heads only")
    policy = _policy_from_config(ck["config"])
    names = list(CONDITIONS)
    recipes = list(CONDITIONS.values())
    print(f"model: {'+'.join(ck.get('backbones', [ck.get('backbone')]))} "
          f"({tag}, epoch {ck.get('epoch')}), policy {policy.as_record()}, "
          f"{len(names)} conditions")

    m = pd.read_parquet(a.manifest)
    d = m[m["split"] == a.split].reset_index(drop=True)
    if len(d) == 0:
        raise SystemExit(f"REFUSING: split {a.split!r} has no rows")
    if a.min_short_side:
        before = len(d)
        d = d[d[["width", "height"]].min(axis=1) >= a.min_short_side]
        d = d.reset_index(drop=True)
        if before - len(d):
            print(f"dropped {before - len(d)} rows with short side < "
                  f"{a.min_short_side} (of {before})")
    if a.limit:
        groups = [g for _, g in d.groupby(["source", "label"])]
        per = max(1, a.limit // len(groups))
        d = pd.concat([g.sample(min(len(g), per), random_state=0)
                       for g in groups]).reset_index(drop=True)
    print(f"{a.split}: {len(d)} rows x {len(names)} conditions "
          f"= {len(d) * len(names):,} forwards")

    def prepare(rows: list[tuple[int, str]]) -> list[np.ndarray]:
        out = []
        for idx, p in rows:
            with Image.open(p) as im:
                base = np.asarray(im.convert("RGB"), dtype=np.uint8)
            base = canonicalise(base, policy=policy)
            out.extend(r.apply(base, np.random.default_rng([a.seed, idx, j]))
                       for j, r in enumerate(recipes))
        return out

    rows = list(enumerate(d["path"].tolist()))
    batches = [rows[i:i + a.batch] for i in range(0, len(rows), a.batch)]
    probs = np.empty((len(rows), len(names)), dtype=np.float64)
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as pool, torch.no_grad():
        for _, fut in tqdm(_windowed(pool, prepare, batches, 2 * a.workers),
                           total=len(batches), desc=a.split):
            imgs = fut.result()
            feats = torch.cat(
                [_forward_tower(t, sp, imgs, a.device, torch.bfloat16, a.chunk)
                 for t, sp in zip(towers, specs)], dim=-1)
            p = torch.sigmoid(head(feats.float(), None)["logit"])
            p = p.view(-1, len(names)).double().cpu().numpy()
            probs[done:done + len(p)] = p
            done += len(p)
    assert done == len(rows)

    wide = d[["path", "image_id", "source", "generator", "label"]].copy()
    for j, n in enumerate(names):
        wide[f"prob_{n}"] = probs[:, j]
    wide.to_parquet(f"{a.out_prefix}.parquet", index=False)

    from sklearn.metrics import roc_auc_score
    y = d["label"].to_numpy()
    conds = {}
    for j, n in enumerate(names):
        conds[n] = {"auc": float(roc_auc_score(y, probs[:, j])),
                    "tpr_at_1pct_fpr": tpr_at_fpr(y, probs[:, j])}
    clean_auc = conds["clean"]["auc"]
    for n in names:
        conds[n]["delta_auc_vs_clean"] = conds[n]["auc"] - clean_auc
    metrics = {"ckpt": os.path.abspath(a.ckpt), "weights": tag,
               "epoch": int(ck.get("epoch", -1)), "split": a.split,
               "n": int(len(d)), "seed": a.seed,
               "policy": policy.as_record(), "conditions": conds}
    with open(f"{a.out_prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    for n in names:
        c = conds[n]
        print(f"{n:>12}  AUC {c['auc']:.4f}  d {c['delta_auc_vs_clean']:+.4f}  "
              f"TPR@1% {c['tpr_at_1pct_fpr']:.4f}")
    print(f"wrote {a.out_prefix}_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
