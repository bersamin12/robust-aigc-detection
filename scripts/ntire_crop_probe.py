"""Does the 200px centre window leave usable evidence on the floor?

WHY
---
`CanonPolicy(mode="crop", crop_side=200)` takes a 200x200 window at NATIVE
resolution and upscales it to `nominal_side`. `augment/canonical.py` states
the trade in prose -- "a whole frame for a 200px image and a detail for a
640px photograph" -- but never costed it. On ntire_val_hard it costs this
much (robust AUC of the frozen fused pair, by native short side):

    80-336px   (window keeps 40.7% of the pixels)   0.9128
    352-512    (16.1%)                              0.9053
    528-736    (7.7%)                               0.8637
    752-4016   (2.9%)                               0.8639

The gradient is 2.8x larger on distorted rows (+0.0400) than clean (+0.0145),
and it survives restricting to the dominant distortion family (downscale-only
rows: small 0.8940 vs big 0.8363), so it is not a distortion-mix artefact.
Every top-4 team on the Open Test feeds the whole image instead (MICV resize
512x512 "to preserve the global spatial context"; vincentlc squish to 384x384
"instead of random resized cropping (which may remove localized forensic
cues)"; INTSIG a 448 branch).

WHAT THIS MEASURES, AND WHAT IT CANNOT
--------------------------------------
This is a TEST-TIME probe on the two frozen round-two checkpoints. It asks
whether evidence outside the centre window is legible to the model we already
have, by averaging logits over several windows. It is NOT the retrain: nine
local decisions averaged is a different estimator from one global decision,
and a model trained on whole images could use context this one never learned
to read. The gate below is written to respect that asymmetry -- a pass is
strong evidence, a null is weak.

ARMS (all on top of the checkpoint's own policy, crop_clamp forced on)
    center  the frozen eval convention, one centre window. Reproduces the
            round-two number, so it is also the probe's own sanity check.
    grid9   nine 200px windows at native resolution, tops and lefts at the
            start / middle / end of the available slack, duplicates dropped
            (a 200px-native image yields exactly one window, i.e. `center`).
            Mean of per-view logits, eval/tta.py's aggregation.
    fov3    downscale the short side to 3*200=600 first (INTER_AREA, the
            kernel canonical.py's step 1 uses), then grid9. Nine windows then
            cover the whole frame at the field of view the centre window has
            on a 600px image -- the regime the small-image bucket sits in and
            the model is measurably best at.
    squish  the whole image at nominal_side x nominal_side, aspect discarded,
            vincentlc's input shape. Two downscales, one kernel each, so no
            image is routed through a different kernel for its size. A severe
            scale mismatch for a crop-trained tower; included to bound the
            question, not as a candidate.
    short_center, letterbox (added 2026-10-12, paper_prereg.md amendment 2)
            aspect-preserving whole-image controls for squish: short side to
            nominal then the centre square, and long side to nominal padded
            black to a square.

GATE (registered 2026-10-07, before any number for any arm exists)
    PRIMARY    Some multi-view arm beats `center` robust AUC on
               ntire_val_hard by more than 1.0 point while giving up no more
               than 0.5 clean point, on at least one of the two branches.
    MECHANISM  That arm's robust gain in the two large-image buckets (short
               side >= 528px) exceeds its gain in the smallest bucket
               (80-336px) by more than 1.0 point. In the smallest bucket the
               centre window already covers 41% of the frame, so extra views
               have little left to add; a gain spread evenly across buckets
               is generic TTA variance reduction, not the crop.
    READING    pass + pass  -> the window is the bottleneck; the whole-image
                              retrain is justified.
               pass + fail  -> take the free TTA points, but the size
                              gradient is unexplained. Not grounds to
                              retrain.
               PRIMARY fail -> INCONCLUSIVE FOR THE RETRAIN, not a
                              refutation: see WHAT THIS CANNOT MEASURE. It
                              does establish that the evidence outside the
                              window is not legible to this model, which
                              argues the retrain is necessary rather than
                              optional. Next cheapest test is then a
                              whole-image proxy arm (1 h on 25% of train),
                              not a 2-day retrain.

Reads ntire_val_hard only -- a fit split, already scored many times. It does
not touch ntire_test, which was read once on 2026-10-04 and is sealed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The private names are imported deliberately rather than reimplemented: this
# probe's `center` arm has to produce the frozen number to the last digit, and
# a second copy of the presentation resize is exactly how that stops being
# true.
from aigcdet.augment.canonical import (  # noqa: E402
    _DOWNSCALE_INTERP, _UPSCALE_INTERP, MODE_BAND, MODE_CROP, MODE_RRC, MODE_SQUISH,
    CanonPolicy,
    _resize_short_side, canonicalise,
)
from aigcdet.models.pooling import tower_poolers  # noqa: E402
from aigcdet.train.finetune import _forward_tower, _windowed  # noqa: E402
from score_plan_splits import (  # noqa: E402
    _auc, _load_models, _policy_from_config, tpr_at_fpr,
)

#: Native-short-side bucket edges, the quartiles of ntire_val_hard that the
#: motivating table above is cut on. Hard-coded so the probe reports against
#: the same cuts rather than re-deriving quartiles from whatever rows it sees.
BUCKETS = ((80, 336), (352, 512), (528, 736), (752, 4016))


def _windows(h: int, w: int, side: int, clamp: bool):
    """`(side, [(top, left), ...])` -- start / middle / end of each axis's
    slack, duplicates dropped. An image with no slack on an axis yields one
    position there, so a 200x200 image yields exactly one window."""
    if clamp:
        side = min(side, h, w)
    if min(h, w) < side:
        raise ValueError(f"cannot take {side}x{side} from {h}x{w}")
    tops = sorted({0, (h - side) // 2, h - side})
    lefts = sorted({0, (w - side) // 2, w - side})
    return side, [(t, l) for t in tops for l in lefts]


#: The `center` presentation for an rrc-trained checkpoint (amendment 5,
#: Test 7): the crop proxies' own eval convention, one 200 px native centre
#: window at the checkpoint's nominal side, clamp on. An rrc policy has no
#: deterministic window of its own (with no rng it is the whole frame, which
#: is the `squish` arm), so `center` has to name one explicitly.
RRC_CENTER_CROP_SIDE = 200


def views_center(img, policy):
    if policy.mode == MODE_RRC:
        centre_policy = CanonPolicy(mode=MODE_CROP, crop_side=RRC_CENTER_CROP_SIDE,
                                    nominal_side=policy.nominal_side,
                                    crop_clamp=True)
        return [canonicalise(img, policy=centre_policy)]
    return [canonicalise(img, policy=policy)]


def views_grid9(img, policy):
    h, w = img.shape[:2]
    side, wins = _windows(h, w, policy.crop_side, policy.crop_clamp)
    return [_resize_short_side(img[t:t + side, l:l + side],
                               policy.nominal_side, _UPSCALE_INTERP)
            for t, l in wins]


def views_fov3(img, policy):
    tgt = 3 * policy.crop_side
    base = (_resize_short_side(img, tgt, _DOWNSCALE_INTERP)
            if min(img.shape[:2]) > tgt else img)
    return views_grid9(base, policy)


def views_squish(img, policy):
    # The trainer's own implementation, so a squish number from this probe
    # and one from a squish-TRAINED checkpoint are the same pixels.
    return [canonicalise(img, policy=CanonPolicy(
        mode=MODE_SQUISH, nominal_side=policy.nominal_side))]


def views_band_squish(img, policy, band: int):
    """Full field of view, with canonical.py's bandwidth ceiling restored.

    The control `squish` is owed. `squish` presents every image at
    nominal_side^2 and so removes canonical.py's LAST-STEP component, but it
    does nothing about the INTRINSIC BAND-LIMIT: a 150px image squished to
    224 arrives soft, a 4016px one arrives sharp. Native short side scores
    AUC 0.6483 against the label in ntire_val_hard's largest bucket and
    0.7450 in ntire_val's, so that residue is a real cue. The within-decile
    table already shows the cue does not EXPLAIN squish's gain (+7.5 points
    inside strata where the resample factor is near-constant, against +7.3
    pooled); this arm asks the forward question instead -- can the gain be
    kept while the cue is closed, which is what decides the canon policy a
    retrain should use.

    `band` must sit below the effective bandwidth of all but a handful of
    rows, so it is a flag and not `policy.band_side`: at nominal_side 224 a
    ceiling of 200 is a 12% trim and close to a no-op.
    """
    sq = views_squish(img, policy)[0]
    return [canonicalise(sq, policy=CanonPolicy(
        mode=MODE_BAND, band_side=band, nominal_side=policy.nominal_side,
        jitter=0.0))]


def views_short_center(img, policy):
    """Aspect-preserving control for `squish` (paper_prereg.md amendment 2).

    Short side to nominal_side, then the centre nominal_side^2 square: the
    usual resize-then-centre-crop. Field of view is the whole short axis and
    the middle of the long one, and nothing is stretched. Uses squish's one
    kernel in both directions, so the two arms differ only in geometry.
    """
    n = policy.nominal_side
    out = img if min(img.shape[:2]) == n else _resize_short_side(img, n, _DOWNSCALE_INTERP)
    h, w = out.shape[:2]
    t, l = (h - n) // 2, (w - n) // 2
    return [np.ascontiguousarray(out[t:t + n, l:l + n])]


def views_letterbox(img, policy):
    """The other aspect-preserving control: the WHOLE frame, long side to
    nominal_side, centred on a black nominal_side^2 square. Full field of
    view, no stretch; the cost is the padded area, identical for both
    classes."""
    n = policy.nominal_side
    h, w = img.shape[:2]
    s = n / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    out = img if (nh, nw) == (h, w) else cv2.resize(
        img, (nw, nh), interpolation=_DOWNSCALE_INTERP)
    canvas = np.zeros((n, n, 3), dtype=img.dtype)
    t, l = (n - nh) // 2, (n - nw) // 2
    canvas[t:t + nh, l:l + nw] = out
    return [canvas]


ARMS = {"center": views_center, "grid9": views_grid9,
        "fov3": views_fov3, "squish": views_squish,
        "band_squish": views_band_squish,
        "short_center": views_short_center, "letterbox": views_letterbox}


def score_arm(df, towers, specs, head, policy, arm, device, batch, workers,
              chunk, band: int = 0):
    from PIL import Image
    from tqdm import tqdm

    build = ARMS[arm]

    def prepare(paths):
        imgs, counts = [], []
        for p in paths:
            with Image.open(p) as im:
                decoded = np.asarray(im.convert("RGB"), dtype=np.uint8)
            vs = (build(decoded, policy, band) if band
                  else build(decoded, policy))
            imgs.extend(vs)
            counts.append(len(vs))
        return imgs, counts

    paths = df["path"].tolist()
    batches = [paths[i:i + batch] for i in range(0, len(paths), batch)]
    poolers = tower_poolers(head, len(towers))
    out = np.empty(len(paths), dtype=np.float64)
    nviews = np.empty(len(paths), dtype=np.int64)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool, torch.no_grad():
        for _, fut in tqdm(_windowed(pool, prepare, batches, 2 * workers),
                           total=len(batches), desc=arm):
            imgs, counts = fut.result()
            feats = torch.cat(
                [_forward_tower(t, sp, imgs, device, torch.bfloat16, chunk,
                                pooler=pl)
                 for t, sp, pl in zip(towers, specs, poolers)], dim=-1)
            z = head(feats.float(), None)["logit"].double().cpu().numpy()
            i = 0
            for c in counts:
                out[done] = z[i:i + c].mean()
                nviews[done] = c
                i += c
                done += 1
            assert i == len(z), (i, len(z))
    assert done == len(paths)
    return out, nviews


def arm_metrics(d: pd.DataFrame, logit: np.ndarray) -> dict:
    y = d["label"].to_numpy()
    dist = d["is_distorted"].to_numpy().astype(int)
    short = d[["width", "height"]].min(axis=1).to_numpy()
    out = {"n": int(len(d)), "auc": _auc(y, logit),
           "tpr_at_1pct_fpr": tpr_at_fpr(y, logit),
           "robust_auc": _auc(y[dist == 1], logit[dist == 1]),
           "clean_auc": _auc(y[dist == 0], logit[dist == 0]),
           "by_bucket": {}}
    for lo, hi in BUCKETS:
        m = (short >= lo) & (short <= hi)
        md, mc = m & (dist == 1), m & (dist == 0)
        out["by_bucket"][f"{lo}-{hi}"] = {
            "n": int(m.sum()),
            "robust_auc": _auc(y[md], logit[md]) if md.sum() else None,
            "clean_auc": _auc(y[mc], logit[mc]) if mc.sum() else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--swa", action="store_true")
    ap.add_argument("--manifest", default="data/manifest_ntire.parquet")
    ap.add_argument("--split", default="ntire_val_hard")
    ap.add_argument("--arms", default="center,grid9,fov3,squish")
    ap.add_argument("--band-side", type=int, default=160,
                    help="bandwidth ceiling for the band_squish arm")
    ap.add_argument("--out", required=True, help="json written here")
    ap.add_argument("--parquet-prefix", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    if a.split == "ntire_test":
        sys.exit("REFUSING: ntire_test was read once on 2026-10-04 and is "
                 "sealed. This probe runs on fit splits only.")

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
    towers, specs, head, tag = _load_models(ck, a.device, a.swa)
    if head.use_recon:
        sys.exit("REFUSING: this head wants recon features; the probe passes "
                 "None for them")
    policy = _policy_from_config(ck["config"], True)
    if policy.mode not in ("crop", "squish", "rrc"):
        sys.exit(f"REFUSING: probe compares crop and squish presentations, "
                 f"policy is {policy.mode!r}")
    # Under a squish checkpoint `center` is that checkpoint's own convention
    # (the whole frame); grid9/fov3 still read crop_side from the policy.
    print(f"model: {'+'.join(ck.get('backbones', [ck.get('backbone')]))} "
          f"({tag}, epoch {ck.get('epoch')}), policy {policy.as_record()}",
          flush=True)

    m = pd.read_parquet(a.manifest)
    d = m[m["split"] == a.split].reset_index(drop=True)
    if len(d) == 0:
        sys.exit(f"REFUSING: split {a.split!r} has no rows")
    short = d[["width", "height"]].min(axis=1)
    print(f"{a.split}: n={len(d)}  short side {int(short.min())}-"
          f"{int(short.max())}  {int((short < policy.crop_side).sum())} rows "
          f"below the {policy.crop_side}px window (crop_clamp)", flush=True)

    res = {"ckpt": os.path.abspath(a.ckpt), "weights": tag,
           "split": a.split, "policy": policy.as_record(),
           "band_side": a.band_side, "arms": {}}
    for arm in a.arms.split(","):
        logit, nviews = score_arm(
            d, towers, specs, head, policy, arm, a.device, a.batch,
            a.workers, a.chunk,
            band=a.band_side if arm == "band_squish" else 0)
        res["arms"][arm] = arm_metrics(d, logit)
        res["arms"][arm]["views_mean"] = float(nviews.mean())
        res["arms"][arm]["views_max"] = int(nviews.max())
        mt = res["arms"][arm]
        def f4(v):
            return "  n/a " if v is None else f"{v:.4f}"
        print(f"{arm:7s} views~{nviews.mean():.2f}  AUC={f4(mt['auc'])}  "
              f"robust={f4(mt['robust_auc'])}  clean={f4(mt['clean_auc'])}  "
              + "  ".join(f"[{k}] {f4(v['robust_auc'])}"
                          for k, v in mt["by_bucket"].items()), flush=True)
        if a.parquet_prefix:
            cols = {"path": d["path"], "label": d["label"],
                    "is_distorted": d["is_distorted"],
                    "distortions": d["distortions"],
                    "width": d["width"], "height": d["height"],
                    "logit": logit, "n_views": nviews}
            pd.DataFrame(cols).to_parquet(
                f"{a.parquet_prefix}_{arm}_{a.split}.parquet", index=False)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
    print(f"wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
