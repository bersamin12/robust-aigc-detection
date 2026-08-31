"""Score a finetuned checkpoint over plan-manifest splits, from images.

The plan bank holds train and val only, and a finetuned tower invalidates
every cached feature anyway, so this scores straight from pixels: decode,
canonicalise under the EVAL convention (the centre window, no rng -- the same
choice `eval/grid` makes, so a probability here is comparable with one there),
forward, sigmoid.

Handles both checkpoint shapes `_write_ckpt` produces: the single-tower
unfreeze arm (`tower_state_dict`) and the dual arm (`tower_state_dicts`, two
towers concatenated in a FIXED order -- swapping them would feed the head's
first half the second tower's features).

Weights are loaded into fp32 EXACTLY as saved, then cast to bfloat16 for
inference: bf16 shares fp32's exponent range, so a tower whose weights moved
in training cannot overflow the way fp16 famously did for dinov3l.

Outputs one parquet of per-row probabilities per split (the ensemble ablation
consumes these) and one metrics json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from aigcdet.augment.canonical import CanonPolicy, canonicalise
from aigcdet.features.backbones import load_backbone
from aigcdet.models.heads import Detector
from aigcdet.train.finetune import _forward_tower, _windowed


def _policy_from_config(cfg: dict) -> CanonPolicy:
    kw = {}
    if cfg.get("nominal_side") is not None:
        kw["nominal_side"] = int(cfg["nominal_side"])
    return CanonPolicy(mode=cfg["policy_mode"], crop_side=cfg.get("crop_side"),
                       crop_clamp=bool(cfg.get("crop_clamp", False)), **kw)


def _load_models(ck: dict, device: str, use_swa: bool):
    """(towers, specs, head, tag). Dual and single share everything but keys."""
    dual = "tower_state_dicts" in ck
    names = ck["backbones"] if dual else [ck["backbone"]]
    tower_sds = ck["tower_state_dicts"] if dual else [ck["tower_state_dict"]]
    swa_towers = (ck.get("swa_tower_state_dicts") if dual
                  else ([ck["swa_tower_state_dict"]]
                        if ck.get("swa_tower_state_dict") is not None else None))
    head_sd, tag = ck["state_dict"], "final"
    if use_swa:
        if not ck.get("swa_n"):
            sys.exit("REFUSING --swa: this checkpoint holds no SWA state "
                     f"(swa_n={ck.get('swa_n')}); score the final weights or "
                     "a later epoch")
        head_sd, tower_sds, tag = ck["swa_state_dict"], swa_towers, "swa"

    towers, specs = [], []
    for name, sd in zip(names, tower_sds):
        tower, spec = load_backbone(name, device=device)
        # fp32 first so the saved weights land exactly, THEN bf16 for speed.
        tower = tower.to(torch.float32)
        tower.load_state_dict(sd)
        tower = tower.to(torch.bfloat16).eval()
        towers.append(tower)
        specs.append(spec)

    cfg = ck["config"]
    head = Detector(dim_feat=ck["dim_feat"], use_recon=False,
                    use_film=bool(cfg.get("use_film")),
                    hidden=int(cfg.get("head_hidden", 512))).to(device)
    head.load_state_dict(head_sd)
    head.eval()
    return towers, specs, head, tag


def _score_split(df: pd.DataFrame, towers, specs, head, policy: CanonPolicy,
                 device: str, batch: int, workers: int, chunk: int,
                 desc: str) -> np.ndarray:
    from PIL import Image
    from tqdm import tqdm

    def prepare(paths: list[str]) -> list[np.ndarray]:
        out = []
        for p in paths:
            with Image.open(p) as im:
                decoded = np.asarray(im.convert("RGB"), dtype=np.uint8)
            # Eval convention: ONE deterministic canonicalisation, the centre
            # window (`rng is None gives the CENTRE window`). A random crop
            # here would score a different picture per invocation.
            out.append(canonicalise(decoded, policy=policy))
        return out

    paths = df["path"].tolist()
    batches = [paths[i:i + batch] for i in range(0, len(paths), batch)]
    probs = np.empty(len(paths), dtype=np.float64)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool, torch.no_grad():
        for _, fut in tqdm(_windowed(pool, prepare, batches, 2 * workers),
                           total=len(batches), desc=desc):
            imgs = fut.result()
            feats = torch.cat(
                [_forward_tower(t, sp, imgs, device, torch.bfloat16, chunk)
                 for t, sp in zip(towers, specs)], dim=-1)
            logit = head(feats.float())["logit"]
            p = torch.sigmoid(logit).double().cpu().numpy()
            probs[done:done + len(imgs)] = p
            done += len(imgs)
    assert done == len(paths)
    return probs


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr: float = 0.01) -> float:
    neg = np.sort(scores[labels == 0])
    if len(neg) == 0 or (labels == 1).sum() == 0:
        return float("nan")
    thr = neg[min(len(neg) - 1, int(np.ceil((1 - fpr) * len(neg))))]
    return float((scores[labels == 1] > thr).mean())


def split_metrics(d: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score
    y, p = d["label"].to_numpy(), d["prob"].to_numpy()
    out = {"n": int(len(d)),
           "auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
           "tpr_at_1pct_fpr": tpr_at_fpr(y, p),
           "acc_at_0.5": float(((p >= 0.5) == y).mean())}
    by = {}
    for src, g in d.groupby("source"):
        ys, ps = g["label"].to_numpy(), g["prob"].to_numpy()
        by[str(src)] = {
            "n": int(len(g)),
            "auc": float(roc_auc_score(ys, ps)) if len(set(ys)) == 2 else None,
            "acc_at_0.5": float(((ps >= 0.5) == ys).mean())}
    out["by_source"] = by
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="data/manifest_plan.parquet")
    ap.add_argument("--splits", default="test_transfer,demo")
    ap.add_argument("--out-prefix", required=True,
                    help="writes <prefix>_<split>.parquet and <prefix>_metrics.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=8,
                    help="images per tower forward inside a batch")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--swa", action="store_true",
                    help="score the SWA average instead of the final weights")
    ap.add_argument("--limit", type=int, default=0,
                    help="rows per split, stratified by (source,label); 0 = all")
    ap.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW",
                    help="prefix substitution applied to manifest paths, for "
                         "scoring on a box whose corpus lives elsewhere; "
                         "repeatable, matched against the ORIGINAL path only")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="score only contiguous shard I of N per split (for "
                         "one process per GPU); the merge step reassembles")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    towers, specs, head, tag = _load_models(ck, a.device, a.swa)
    policy = _policy_from_config(ck["config"])
    print(f"model: {'+'.join(ck.get('backbones', [ck.get('backbone')]))} "
          f"({tag}, epoch {ck.get('epoch')}), policy {policy.as_record()}")

    m = pd.read_parquet(a.manifest)
    metrics = {"ckpt": os.path.abspath(a.ckpt), "weights": tag,
               "epoch": int(ck.get("epoch", -1)),
               "policy": policy.as_record(), "splits": {}}
    for sp in a.splits.split(","):
        d = m[m["split"] == sp].reset_index(drop=True)
        if len(d) == 0:
            sys.exit(f"REFUSING: split {sp!r} has no rows in {a.manifest}")
        if a.path_map:
            maps = [pm.split("=", 1) for pm in a.path_map]
            src = np.asarray([str(x) for x in d["path"]], dtype=object)
            out, hit = src.copy(), np.zeros(len(src), dtype=bool)
            for old_p, new_p in maps:
                h = np.fromiter((x.startswith(old_p) for x in src),
                                dtype=bool, count=len(src))
                out[h] = [new_p + x[len(old_p):] for x in src[h]]
                hit |= h
            if not hit.all():
                sys.exit(f"REFUSING {sp}: {int((~hit).sum())} paths matched "
                         f"no --path-map prefix, e.g. "
                         f"{src[~hit][:2].tolist()}")
            d = d.assign(path=out)
        if a.shard:
            i, n = (int(x) for x in a.shard.split("/"))
            base, rem = divmod(len(d), n)
            start = i * base + min(i, rem)
            stop = start + base + (1 if i < rem else 0)
            d = d.iloc[start:stop].reset_index(drop=True)
        if a.limit:
            # Explicit loop: pandas 3.0's groupby.apply strips the grouping
            # columns from what it hands the callable, and the sampled frames
            # come back without `source` and `label`.
            groups = [g for _, g in d.groupby(["source", "label"])]
            per = max(1, a.limit // len(groups))
            d = pd.concat([g.sample(min(len(g), per), random_state=0)
                           for g in groups]).reset_index(drop=True)
        miss = [p for p in d["path"].sample(min(200, len(d)), random_state=0)
                if not os.path.exists(p)]
        if miss:
            sys.exit(f"REFUSING {sp}: {len(miss)} sampled paths missing, "
                     f"e.g. {miss[:2]}")
        d["prob"] = _score_split(d, towers, specs, head, policy, a.device,
                                 a.batch, a.workers, a.chunk, desc=sp)
        out = f"{a.out_prefix}_{sp}.parquet"
        d[["path", "image_id", "source", "generator", "label", "prob"]].to_parquet(
            out, index=False)
        metrics["splits"][sp] = split_metrics(d)
        ms = metrics["splits"][sp]
        # A contiguous shard of a source-ordered split can be single-class,
        # so AUC can legitimately be None here; the merge computes the real one.
        auc = "n/a" if ms["auc"] is None else f"{ms['auc']:.4f}"
        print(f"{sp}: n={ms['n']:,} AUC={auc} "
              f"TPR@1%FPR={ms['tpr_at_1pct_fpr']:.4f} acc={ms['acc_at_0.5']:.4f}"
              f" -> {out}")

    with open(f"{a.out_prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote {a.out_prefix}_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
