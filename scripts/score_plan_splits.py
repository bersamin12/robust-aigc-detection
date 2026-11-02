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
from aigcdet.models.pooling import (
    TOKEN_POOL_MEAN, build_pooler, pooled_dim, tower_poolers,
)
from aigcdet.models.heads import Detector
from aigcdet.train.finetune import _forward_tower, _windowed


def _policy_from_config(cfg: dict, crop_clamp: bool | None = None) -> CanonPolicy:
    """The checkpoint's own canonicalisation, optionally with `crop_clamp`
    forced on. The override exists for eval sets that carry images SMALLER
    than the crop window (NTIRE's downscale distortions): without it those
    rows can only be dropped, and dropping rows changes the leaderboard-
    matched number. It is reported beside n by the caller."""
    kw = {}
    if cfg.get("nominal_side") is not None:
        kw["nominal_side"] = int(cfg["nominal_side"])
    clamp = bool(cfg.get("crop_clamp", False)) if crop_clamp is None else bool(crop_clamp)
    return CanonPolicy(mode=cfg["policy_mode"], crop_side=cfg.get("crop_side"),
                       crop_clamp=clamp, **kw)


def _load_models(ck: dict, device: str, use_swa: bool, use_ema: bool = False):
    """(towers, specs, head, tag). Dual and single share everything but keys."""
    if use_swa and use_ema:
        sys.exit("REFUSING: --swa and --ema are two different models; score "
                 "them in two runs")
    dual = "tower_state_dicts" in ck
    names = ck["backbones"] if dual else [ck["backbone"]]
    tower_sds = ck["tower_state_dicts"] if dual else [ck["tower_state_dict"]]
    swa_towers = (ck.get("swa_tower_state_dicts") if dual
                  else ([ck["swa_tower_state_dict"]]
                        if ck.get("swa_tower_state_dict") is not None else None))
    ema_towers = (ck.get("ema_tower_state_dicts") if dual
                  else ([ck["ema_tower_state_dict"]]
                        if ck.get("ema_tower_state_dict") is not None else None))
    head_sd, tag = ck["state_dict"], "final"
    if use_swa:
        if not ck.get("swa_n"):
            sys.exit("REFUSING --swa: this checkpoint holds no SWA state "
                     f"(swa_n={ck.get('swa_n')}); score the final weights or "
                     "a later epoch")
        head_sd, tower_sds, tag = ck["swa_state_dict"], swa_towers, "swa"
    if use_ema:
        if ema_towers is None or ck.get("ema_state_dict") is None:
            sys.exit("REFUSING --ema: this checkpoint holds no EMA teacher "
                     "(the run was not trained with --distil ema)")
        head_sd, tower_sds, tag = ck["ema_state_dict"], ema_towers, "ema"

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
    # A checkpoint from before the readout was a variable carries no
    # `token_pool`; it was trained with the mean, which the default says.
    mode = str(cfg.get("token_pool", TOKEN_POOL_MEAN))
    want = pooled_dim(mode, specs)
    if int(ck["dim_feat"]) != want:
        sys.exit(f"REFUSING: checkpoint dim_feat={ck['dim_feat']} but token_pool="
                 f"{mode!r} over {names} pools to {want}")
    head = Detector(dim_feat=ck["dim_feat"],
                    use_recon=bool(cfg.get("use_recon")),
                    use_film=bool(cfg.get("use_film")),
                    film_shuffle=bool(cfg.get("film_shuffle")),
                    hidden=int(cfg.get("head_hidden", 512)),
                    pooler=build_pooler(mode, specs)).to(device)
    head.load_state_dict(head_sd)
    head.eval()
    return towers, specs, head, tag


def _score_split(df: pd.DataFrame, towers, specs, head, policy: CanonPolicy,
                 device: str, batch: int, workers: int, chunk: int,
                 desc: str, tta: bool = False,
                 grid: dict | None = None,
                 degrade: tuple | None = None,
                 jpeg_q: int | None = None) -> np.ndarray:
    from PIL import Image
    from tqdm import tqdm

    recon = None
    if head.use_recon:
        # The VAE branch scores from the SAME canonicalised image the towers
        # see -- its clean-view feature, computed live (no bank covers the
        # eval splits).
        from aigcdet.features.recon import load_recon_models, recon_features
        recon = load_recon_models(device=device)

    tta_views = None
    if tta:
        from aigcdet.eval.tta import TTA_VIEWS, apply_tta_view
        tta_views = TTA_VIEWS
    conds = list(grid.items()) if grid else None

    def prepare(paths: list[str]) -> list[np.ndarray]:
        out = []
        for p in paths:
            with Image.open(p) as im:
                decoded = np.asarray(im.convert("RGB"), dtype=np.uint8)
            if degrade is not None:
                decoded = degrade_native(decoded, p, *degrade)[0]
            if jpeg_q is not None:
                decoded = reencode_jpeg(decoded, jpeg_q)
            # Eval convention: ONE deterministic canonicalisation, the centre
            # window (`rng is None gives the CENTRE window`). A random crop
            # here would score a different picture per invocation.
            std = canonicalise(decoded, policy=policy)
            if conds is not None:
                # The robustness grid: each condition's recipe applied on top
                # of the SAME centre window, exactly eval/grid's convention --
                # one canonicalisation shared by all conditions so a
                # condition's effect is not confounded with a different
                # picture. The rng key is (seed, row-position, condition), so
                # the one op that draws from it (noise) is reproducible.
                for ci, (_, recipe) in enumerate(conds):
                    rng = np.random.default_rng([20260827, hash(p) % (2**31), ci])
                    out.append(recipe.apply(std, rng))
            elif tta_views is None:
                out.append(std)
            else:
                # TTA composes ON TOP of the canonicalised image, one entry
                # per view, flattened; the consumer reshapes by len(views).
                out.extend(apply_tta_view(std, v) for v in tta_views)
        return out

    paths = df["path"].tolist()
    batches = [paths[i:i + batch] for i in range(0, len(paths), batch)]
    poolers = tower_poolers(head, len(towers))
    width = len(conds) if conds is not None else 1
    # LOGITS, in float64. The sigmoid is taken by the caller: a probability
    # saturates at 1.0 in float32 for any logit past ~17, and two branches
    # that both saturate carry different evidence in logit space and none
    # in probability space -- which is what late fusion reads.
    logits = np.empty(len(paths) * width, dtype=np.float64)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool, torch.no_grad():
        for _, fut in tqdm(_windowed(pool, prepare, batches, 2 * workers),
                           total=len(batches), desc=desc):
            imgs = fut.result()
            feats = torch.cat(
                [_forward_tower(t, sp, imgs, device, torch.bfloat16, chunk,
                                pooler=pl)
                 for t, sp, pl in zip(towers, specs, poolers)], dim=-1)
            r = None
            if recon is not None:
                vae, lpips_fn = recon
                r = torch.from_numpy(np.stack(
                    [recon_features(im, vae, lpips_fn, device=device)
                     for im in imgs])).to(device)
            logit = head(feats.float(), r)["logit"]
            if conds is not None:
                pass  # one prob per (image, condition); no aggregation
            elif tta_views is not None:
                # Mean of per-view LOGITS, then sigmoid -- eval/tta.py's
                # aggregation. Monotone for AUC/TPR either way; acc@0.5 is
                # what logit-space averaging keeps honest.
                logit = logit.view(-1, len(tta_views)).mean(dim=1)
            z = logit.double().cpu().numpy()
            logits[done:done + len(z)] = z
            done += len(z)
    assert done == len(paths) * width
    return logits.reshape(len(paths), width) if conds is not None else logits


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr: float = 0.01) -> float:
    neg = np.sort(scores[labels == 0])
    if len(neg) == 0 or (labels == 1).sum() == 0:
        return float("nan")
    thr = neg[min(len(neg) - 1, int(np.ceil((1 - fpr) * len(neg))))]
    return float((scores[labels == 1] > thr).mean())


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else None


def _distortion_tokens(s) -> list[str]:
    """`distortions` is stored as the challenge csv's python-list string,
    e.g. "['jpeg', 'blur']"; an empty list or a missing value is no token."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    txt = str(s).strip()
    if txt in ("", "[]"):
        return []
    try:
        import ast
        v = ast.literal_eval(txt)
        return [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])]
    except (ValueError, SyntaxError):
        return [txt]


def degrade_key(path: str) -> int:
    """A per-file key that is stable across processes and boxes. Python's
    `hash` is salted per process, so it would hand two scorers of the same
    file different recipes and break the pairing between models."""
    import zlib
    return zlib.crc32(os.path.basename(str(path)).encode())


def degrade_native(decoded: np.ndarray, path: str, seed: int, pool: str,
                   max_ops: int):
    """`(pixels, recipe)`: one seeded degradation chain on the NATIVE image.

    For the `_deg` external splits. The chain is drawn exactly as a training
    view is (`recipe_for_view`, view 1) but keyed on the file name, and it is
    applied to the decoded file BEFORE canonicalisation -- the organisers'
    order, and the only order under which a crop model and a squish model are
    handed the same degraded file and differ only in how they present it.
    This is our synthesis of the NTIRE families, not the organisers' code.
    """
    from aigcdet.features.extract import recipe_for_view

    k = degrade_key(path)
    r = recipe_for_view(1, k, seed, pool=pool, max_ops=max_ops)
    pre, post = r.split()
    out = pre.apply_pre(decoded) if pre.ops else decoded
    out = post.apply_post(out, np.random.default_rng([seed, k, 1]))
    return out, r


def reencode_jpeg(decoded: np.ndarray, quality: int) -> np.ndarray:
    """The `_jpeg95` external splits: every file, real or fake, through one
    JPEG encode at `quality` before canonicalisation, so container format no
    longer separates the classes (AIGIBench CommunityAI ships every real as
    JPEG and every fake as PNG)."""
    import cv2
    ok, buf = cv2.imencode(".jpg", decoded[:, :, ::-1],
                           [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return np.ascontiguousarray(cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1])


def merge_metrics(path, metrics):
    """Keep the splits an earlier invocation wrote to the same prefix.

    The pipelines score one split per call so an interrupted run resumes at
    the split it stopped on, and every call writes the whole file; without
    this the prefix ends up describing only the split that happened to go
    last. Splits are merged only when the file describes the same checkpoint
    scored the same way -- a different policy (say `--crop-clamp` toggled) is
    a different number for the same split name, so that file is replaced.
    """
    if not os.path.exists(path):
        return metrics
    try:
        with open(path) as f:
            old = json.load(f)
    except (OSError, ValueError):
        return metrics
    if any(old.get(k) != metrics[k] for k in ("ckpt", "weights", "policy")):
        return metrics
    merged = dict(old.get("splits", {}))
    merged.update(metrics["splits"])
    metrics["splits"] = merged
    return metrics


def split_metrics(d: pd.DataFrame) -> dict:
    """Pooled AUC / TPR@1%FPR / acc@0.5, per source, and -- when the split
    carries `is_distorted` -- the leaderboard's pair: Robust ROC AUC over the
    distorted rows and Clean ROC AUC over the rest, plus one AUC per
    distortion token (that token's distorted rows against every clean row).
    """
    y, p = d["label"].to_numpy(), d["prob"].to_numpy()
    out = {"n": int(len(d)),
           "auc": _auc(y, p),
           "tpr_at_1pct_fpr": tpr_at_fpr(y, p),
           "acc_at_0.5": float(((p >= 0.5) == y).mean())}
    if "is_distorted" in d.columns:
        dist = d["is_distorted"].to_numpy().astype(int)
        for key, m in (("robust", dist == 1), ("clean", dist == 0)):
            out[f"n_{key}"] = int(m.sum())
            out[f"{key}_auc"] = _auc(y[m], p[m]) if m.any() else None
        if "distortions" in d.columns:
            per = {}
            clean = dist == 0
            toks = [_distortion_tokens(s) for s in d["distortions"]]
            names = sorted({t for ts in toks for t in ts})
            for name in names:
                has = np.fromiter((name in ts for ts in toks), dtype=bool, count=len(toks))
                m = has | clean
                per[name] = {"n": int(has.sum()), "auc": _auc(y[m], p[m])}
            out["by_distortion"] = per
    by = {}
    for src, g in d.groupby("source"):
        ys, ps = g["label"].to_numpy(), g["prob"].to_numpy()
        by[str(src)] = {
            "n": int(len(g)),
            "auc": _auc(ys, ps),
            "acc_at_0.5": float(((ps >= 0.5) == ys).mean())}
    out["by_source"] = by
    return out


#: Columns written to the per-row parquet, in this order, when present.
OUT_COLUMNS = ("path", "image_id", "source", "generator", "label", "split",
               "is_distorted", "distortions", "logit", "prob")


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
    ap.add_argument("--grid", action="store_true",
                    help="score every EVAL_GRID condition (20x the compute): "
                         "canonicalise once (centre window), then apply each "
                         "condition recipe on top -- the robustness table")
    ap.add_argument("--tta", action="store_true",
                    help="8-view test-time augmentation: mean of per-view "
                         "LOGITS (eval/tta.py's aggregation), 8x the compute")
    ap.add_argument("--swa", action="store_true",
                    help="score the SWA average instead of the final weights")
    ap.add_argument("--ema", action="store_true",
                    help="score the EMA teacher (tower and head) a --distil "
                         "ema run kept, instead of the final weights")
    ap.add_argument("--limit", type=int, default=0,
                    help="rows per split, stratified by (source,label); 0 = all")
    ap.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW",
                    help="prefix substitution applied to manifest paths, for "
                         "scoring on a box whose corpus lives elsewhere; "
                         "repeatable, matched against the ORIGINAL path only")
    ap.add_argument("--min-short-side", type=int, default=0,
                    help="drop rows whose min(width,height) is below this "
                         "(crop without clamp cannot score them; the count "
                         "dropped is printed and must be reported beside n)")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="score only contiguous shard I of N per split (for "
                         "one process per GPU); the merge step reassembles")
    ap.add_argument("--degrade-seed", type=int, default=None,
                    help="required for, and only applied to, splits whose name "
                         "ends in _deg: one seeded chain per file on the "
                         "native image (degrade_native)")
    ap.add_argument("--degrade-pool", default="ntire")
    ap.add_argument("--degrade-max-ops", type=int, default=5)
    ap.add_argument("--crop-clamp", action="store_true",
                    help="force CanonPolicy.crop_clamp on, so rows with a "
                         "side below the crop window (NTIRE's downscale "
                         "distortions) are scored instead of refused; the "
                         "count of such rows is printed beside n")
    a = ap.parse_args()

    deg_splits = [sp for sp in a.splits.split(",") if sp.endswith("_deg")]
    if deg_splits and a.degrade_seed is None:
        sys.exit(f"REFUSING {deg_splits}: a _deg split needs --degrade-seed")
    if a.grid and a.tta:
        sys.exit("REFUSING: --grid and --tta are separate measurements")
    grid = None
    if a.grid:
        from aigcdet.augment.scenarios import EVAL_GRID
        grid = dict(EVAL_GRID)

    # mmap: a fat per-epoch checkpoint carries the optimiser moments and the
    # SWA copy, three times the weights, and this reads only the weights.
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
    towers, specs, head, tag = _load_models(ck, a.device, a.swa, a.ema)
    policy = _policy_from_config(ck["config"], True if a.crop_clamp else None)
    print(f"model: {'+'.join(ck.get('backbones', [ck.get('backbone')]))} "
          f"({tag}, epoch {ck.get('epoch')}, token_pool "
          f"{ck['config'].get('token_pool', TOKEN_POOL_MEAN)}), "
          f"policy {policy.as_record()}")

    m = pd.read_parquet(a.manifest)
    metrics = {"ckpt": os.path.abspath(a.ckpt), "weights": tag,
               "epoch": int(ck.get("epoch", -1)),
               "token_pool": str(ck["config"].get("token_pool", TOKEN_POOL_MEAN)),
               "dim_feat": int(ck["dim_feat"]),
               "policy": policy.as_record(), "splits": {}}
    for sp in a.splits.split(","):
        d = m[m["split"] == sp].reset_index(drop=True)
        if len(d) == 0:
            sys.exit(f"REFUSING: split {sp!r} has no rows in {a.manifest}")
        if a.min_short_side:
            before = len(d)
            keep = d[["width", "height"]].min(axis=1) >= a.min_short_side
            d = d[keep].reset_index(drop=True)
            if before - len(d):
                print(f"{sp}: dropped {before - len(d)} rows with short side "
                      f"< {a.min_short_side} (of {before})")
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
        if (policy.mode == "crop" and policy.crop_side
                and {"width", "height"} <= set(d.columns)):
            small = int((d[["width", "height"]].min(axis=1) < policy.crop_side).sum())
            if small:
                print(f"{sp}: {small} of {len(d)} rows have a side below the "
                      f"{policy.crop_side}px crop window"
                      + (" (scored under crop_clamp)" if policy.crop_clamp
                         else " -- canonicalise will REFUSE them; pass "
                              "--crop-clamp or --min-short-side"))
        degrade = None
        if sp.endswith("_deg"):
            if a.degrade_seed is None:
                sys.exit(f"REFUSING {sp}: a _deg split needs --degrade-seed")
            degrade = (a.degrade_seed, a.degrade_pool, a.degrade_max_ops)
            from aigcdet.features.extract import recipe_for_view
            d = d.assign(is_distorted=1, distortions=[
                str([op.name for op in recipe_for_view(
                    1, degrade_key(p_), a.degrade_seed, pool=a.degrade_pool,
                    max_ops=a.degrade_max_ops).ops]) for p_ in d["path"]])
        jpeg_q = None
        if sp.endswith("_jpeg95"):
            jpeg_q = 95
            d = d.assign(distortions="['jpeg_q95_all_rows']")
        scored = _score_split(d, towers, specs, head, policy, a.device,
                              a.batch, a.workers, a.chunk, desc=sp,
                              tta=a.tta, grid=grid, degrade=degrade,
                              jpeg_q=jpeg_q)
        cols = [c for c in OUT_COLUMNS if c in d.columns or c in ("logit", "prob")]
        if grid is not None:
            names = list(grid)
            long = d.loc[d.index.repeat(len(names))].reset_index(drop=True)
            long["condition"] = names * len(d)
            long["logit"] = scored.reshape(-1)
            long["prob"] = 1.0 / (1.0 + np.exp(-long["logit"].to_numpy()))
            out = f"{a.out_prefix}_{sp}_grid.parquet"
            long[[c for c in cols if c != "prob"] + ["condition", "prob"]
                 ].to_parquet(out, index=False)
            per_cond = {}
            for c in names:
                g = long[long["condition"] == c]
                per_cond[c] = split_metrics(g)
                per_cond[c].pop("by_source", None)
            degraded = [v["tpr_at_1pct_fpr"] for k, v in per_cond.items()
                        if k != "clean"]
            metrics["splits"][sp] = {
                "n": int(len(d)), "conditions": per_cond,
                "robust_tpr_at_1pct": float(np.mean(degraded))}
            ms = metrics["splits"][sp]
            print(f"{sp}: n={ms['n']:,} robust TPR@1%FPR (mean over "
                  f"{len(degraded)} degraded) = {ms['robust_tpr_at_1pct']:.4f}"
                  f" -> {out}")
            continue
        d["logit"] = scored
        d["prob"] = 1.0 / (1.0 + np.exp(-scored))
        out = f"{a.out_prefix}_{sp}.parquet"
        d[cols].to_parquet(out, index=False)
        metrics["splits"][sp] = split_metrics(d)
        ms = metrics["splits"][sp]
        # A contiguous shard of a source-ordered split can be single-class,
        # so AUC can legitimately be None here; the merge computes the real one.
        f4 = lambda v: "n/a" if v is None else f"{v:.4f}"  # noqa: E731
        extra = ""
        if "robust_auc" in ms:
            extra = (f" robustAUC={f4(ms['robust_auc'])} (n={ms['n_robust']}) "
                     f"cleanAUC={f4(ms['clean_auc'])} (n={ms['n_clean']})")
        print(f"{sp}: n={ms['n']:,} AUC={f4(ms['auc'])} "
              f"TPR@1%FPR={ms['tpr_at_1pct_fpr']:.4f} acc={ms['acc_at_0.5']:.4f}"
              f"{extra} -> {out}")

    out_path = f"{a.out_prefix}_metrics.json"
    metrics = merge_metrics(out_path, metrics)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote {out_path} ({', '.join(metrics['splits'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
