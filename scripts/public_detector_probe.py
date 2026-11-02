"""Field of view on two public detectors (docs/ntire/paper_prereg.md, amendment 3).

WHY
---
The squish result (docs/ntire/crop_probe.md, docs/ntire/paper/) is on our own
two DINOv2 branches. Two published detectors have a train/test field-of-view
mismatch of their own, written into their released code:

    UnivFD  (Ojha et al., CVPR 2023; WisconsinAIVision/UniversalFakeDetect)
            train: short side to 256, random 224 crop -> FOV 224/256 = 0.875
            test (validate.py): CenterCrop(224) on the NATIVE image, no resize
                   -> FOV 224 / native short side
    B-Free  (Guillaro et al., CVPR 2025; grip-unina/B-Free, BFREE_dino2reg4)
            train: reals are the largest central crop resized to 512, 504 crops
                   -> FOV about 504/512
            test (main_bfree.py): the NATIVE image through Wrapper5crops, which
                   averages the centre and four corner 504px windows
                   -> FOV 504 / native short side

Neither was trained on NTIRE data. This script scores each under three
presentations and changes nothing else:

    native     the authors' own test-time pipeline, reproduced
    train_fov  short side resized to the authors' training size first, then
               the authors' own crop: the field of view they trained at
    squish     the whole image at the crop size, aspect discarded (our
               canonicalise MODE_SQUISH, the presentation the paper studies)

Both resized arms use canonical.py's one kernel (INTER_AREA) in both
directions, so the arms differ in geometry, not in interpolation.

FIDELITY
--------
`--selfcheck` runs B-Free `native` on the authors' four demo images and must
reproduce their demo_images/results.csv logits to 1e-3; the launcher runs it
before anything is scored. UnivFD ships no reference outputs. Its CLIP
ViT-L/14 is the Hugging Face port of the OpenAI weights; `image_embeds` is
OpenAI's `encode_image` (CLS token, post-LayerNorm, projection), which is what
UnivFD's `fc` was trained on.

Licences: UnivFD code is MIT-style (see its LICENSE). B-Free weights and code
are for "informational and nonprofit purposes" (its LICENSE.txt). Nothing from
either is copied into this repo; the 5-crop forward below re-implements
B-Free's Wrapper5crops so the native arm matches it, and cites it.

Fit splits and external sets only. ntire_test is refused.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aigcdet.augment.canonical import (  # noqa: E402
    _DOWNSCALE_INTERP, MODE_SQUISH, CanonPolicy, _resize_short_side,
    canonicalise,
)

MODELS_DIR = "/scratch/justin_r13/ext/models"
ARMS = ("native", "train_fov", "squish")
CLIP_NORM = ((0.48145466, 0.4578275, 0.40821073),
             (0.26862954, 0.26130258, 0.27577711))
IMAGENET_NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
DEFAULT_SPLITS = ("ntire_val_hard,ntire_val,"
                  "ext_aigi_socialrf,ext_aigi_socialrf_deg,ext_aigi_socialrf_jpeg95,"
                  "ext_aigi_community,ext_aigi_community_deg,ext_aigi_community_jpeg95,"
                  "ext_synthbuster,ext_synthbuster_deg,ext_synthbuster_jpeg95")


def squish(img: np.ndarray, n: int) -> np.ndarray:
    return canonicalise(img, policy=CanonPolicy(mode=MODE_SQUISH, nominal_side=n))


def short_side(img: np.ndarray, n: int) -> np.ndarray:
    return img if min(img.shape[:2]) == n else _resize_short_side(img, n, _DOWNSCALE_INTERP)


def center_crop_pad(img: np.ndarray, n: int) -> np.ndarray:
    """torchvision CenterCrop(n) on a uint8 HWC array, which is what UnivFD's
    validate.py applies: an image smaller than n is zero-padded first, and
    offsets use torchvision's own rounding."""
    import torchvision.transforms.functional as TF
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
    return np.ascontiguousarray(TF.center_crop(t, [n, n]).permute(1, 2, 0).numpy())


def univfd_view(img: np.ndarray, arm: str) -> np.ndarray:
    if arm == "native":
        return center_crop_pad(img, 224)
    if arm == "train_fov":
        return center_crop_pad(short_side(img, 256), 224)
    if arm == "squish":
        return squish(img, 224)
    raise ValueError(arm)


def bfree_view(img: np.ndarray, arm: str) -> np.ndarray:
    if arm == "native":
        return img
    if arm == "train_fov":
        return short_side(img, 512)
    if arm == "squish":
        return squish(img, 504)
    raise ValueError(arm)


def to_tensor(img: np.ndarray, norm) -> torch.Tensor:
    mean, std = (torch.tensor(v).view(3, 1, 1) for v in norm)
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    return (t - mean) / std


def replicate_wrap(x: torch.Tensor, new_shape) -> torch.Tensor:
    """B-Free's replicate_wrap (code/networks/wrapper5crops.py): tile a token
    grid smaller than the model's grid until it fills it."""
    rh = max(int(np.ceil(new_shape[-2] / x.shape[-2])), 1)
    rw = max(int(np.ceil(new_shape[-1] / x.shape[-1])), 1)
    return x.repeat(1, 1, rh, rw)[..., :new_shape[-2], :new_shape[-1]]


class UnivFD:
    name, norm, batch_fixed = "univfd", CLIP_NORM, True
    view = staticmethod(univfd_view)

    def __init__(self, device: str):
        from transformers import CLIPVisionModelWithProjection
        self.enc = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14", dtype=torch.float32).to(device).eval()
        sd = torch.load(f"{MODELS_DIR}/UniversalFakeDetect/pretrained_weights/fc_weights.pth",
                        map_location="cpu", weights_only=True)
        self.fc = torch.nn.Linear(768, 1)
        self.fc.load_state_dict(sd)
        self.fc = self.fc.to(device).eval()

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.enc(pixel_values=x).image_embeds)[:, 0]


class BFree:
    name, norm, batch_fixed = "bfree", IMAGENET_NORM, False
    view = staticmethod(bfree_view)

    def __init__(self, device: str):
        import timm
        m = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m",
                              num_classes=1, pretrained=False)
        m.set_input_size(img_size=504)
        self.patch_embed = m.patch_embed
        m.patch_embed = torch.nn.Identity()
        sd = torch.load(f"{MODELS_DIR}/bfree_weights/BFREE_dino2reg4/model_epoch_best.pth",
                        map_location="cpu", weights_only=True)["model"]
        self.patch_embed.load_state_dict(
            {k[len("patch_embed."):]: v for k, v in sd.items() if k.startswith("patch_embed.")})
        m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("patch_embed.")})
        self.model = m.to(device).eval()
        self.patch_embed = self.patch_embed.to(device).eval()

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Wrapper5crops.forward: the whole image through the patch
        projection, then the centre and four corner 36x36 token windows,
        each through the ViT, logits averaged."""
        pe = self.patch_embed
        gh, gw = pe.grid_size
        e = pe.proj(x)
        if gh > e.shape[-2] or gw > e.shape[-1]:
            e = replicate_wrap(e, (gh, gw))
        hs = max((e.shape[-2] - gh) // 2, 0)
        ws = max((e.shape[-1] - gw) // 2, 0)
        e = torch.cat((e[:, :, hs:gh + hs, ws:gw + ws], e[:, :, :gh, :gw],
                       e[:, :, -gh:, :gw], e[:, :, -gh:, -gw:],
                       e[:, :, :gh, -gw:]), 0)
        if pe.flatten:
            e = e.flatten(2).transpose(1, 2)
        y = self.model(pe.norm(e))
        return torch.mean(torch.stack(torch.split(y, y.shape[0] // 5, 0), 0), 0)[:, 0]


DETECTORS = {"univfd": UnivFD, "bfree": BFree}


def decode(path: str) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)  # writable: torch.from_numpy warns on read-only


def split_rows(split: str) -> pd.DataFrame:
    manifest = "data/manifest_external.parquet" if split.startswith("ext_") \
        else "data/manifest_ntire.parquet"
    m = pd.read_parquet(manifest)
    d = m[m["split"] == split].reset_index(drop=True)
    if len(d) == 0:
        sys.exit(f"REFUSING: split {split!r} has no rows in {manifest}")
    return d


def score_split(det, d: pd.DataFrame, split: str, device: str, batch: int,
                workers: int, degrade_seed: int) -> dict[str, np.ndarray]:
    from tqdm import tqdm
    from aigcdet.train.finetune import _windowed
    from score_plan_splits import degrade_native, reencode_jpeg

    def prepare(paths):
        out = {a: [] for a in ARMS}
        for p in paths:
            img = decode(p)
            if split.endswith("_deg"):
                img, _ = degrade_native(img, p, degrade_seed, "ntire", 5)
            elif split.endswith("_jpeg95"):
                img = reencode_jpeg(img, 95)
            for a in ARMS:
                out[a].append(to_tensor(det.view(img, a), det.norm))
        return out

    paths = d["path"].tolist()
    step = batch if det.batch_fixed else 1
    chunks = [paths[i:i + step] for i in range(0, len(paths), step)]
    res = {a: [] for a in ARMS}
    # A bounded window, not pool.map: map submits every chunk at once, and
    # B-Free's native arm holds full-resolution float tensors.
    with ThreadPoolExecutor(max_workers=workers) as pool, torch.no_grad():
        for _, fut in tqdm(_windowed(pool, prepare, chunks, 2 * workers),
                           total=len(chunks), desc=f"{det.name} {split}"):
            prepared = fut.result()
            for a in ARMS:
                x = torch.stack(prepared[a]).to(device)
                res[a].append(det.logits(x).double().cpu().numpy())
    return {a: np.concatenate(v) for a, v in res.items()}


def selfcheck(device: str) -> int:
    det = BFree(device)
    demo = f"{MODELS_DIR}/B-Free/code/demo_images"
    ref = pd.read_csv(f"{demo}/results.csv")
    worst = 0.0
    with torch.no_grad():
        for _, r in ref.iterrows():
            img = decode(f"{demo}/{r['filename']}")
            got = float(det.logits(to_tensor(img, det.norm)[None].to(device))[0])
            worst = max(worst, abs(got - float(r["BFREE_dino2reg4"])))
            print(f"{r['filename']}: ours {got:+.5f}  authors {float(r['BFREE_dino2reg4']):+.5f}")
    print(f"selfcheck max |diff| = {worst:.2e}")
    return 0 if worst < 1e-3 else 1


def verdict(boot: dict, metrics: dict) -> dict:
    """Amendment 3's rules, applied as written, for one detector on
    ntire_val_hard. `boot` is a paired_bootstrap result with baseline
    `native`; `metrics` is that split's arm_metrics per arm."""
    best = max(metrics[a]["robust_auc"] for a in ARMS)
    out = {"best_robust_auc": best, "interpretable": best >= 0.60}
    for arm, key in (("squish", "primary"), ("train_fov", "secondary")):
        d = boot["vs_baseline"][arm]["robust_auc"]
        b = {k: {a: metrics[a]["by_bucket"][k]["robust_auc"] for a in ("native", arm)}
             for k in ("80-336", "752-4016")}
        small = b["80-336"][arm] - b["80-336"]["native"]
        large = b["752-4016"][arm] - b["752-4016"]["native"]
        out[key] = {"arm": arm, "diff": d["diff"], "ci": d["ci"],
                    "passed": bool(d["ci"][0] > 0)}
        out[f"mechanism_{arm}"] = {"gain_80_336": small, "gain_752_4016": large,
                                   "passed": bool(large > small)}
    return out


def summarise(out_dir: str, md_path: str, n_boot: int) -> int:
    import paired_bootstrap as pb
    import paper_results as pr

    lines = ["# Public detectors under three presentations", "",
             "Generated by `scripts/public_detector_probe.py --summarise`; rules "
             "registered in `docs/ntire/paper_prereg.md` amendment 3 before any "
             "number existed. `native` = the authors' own test pipeline, "
             "`train_fov` = short side to their training size then their crop, "
             "`squish` = the whole image at the crop size.", ""]
    allres = {}
    for det in DETECTORS:
        mpath = f"docs/ntire/paper/public_{det}.json"
        if not os.path.exists(mpath):
            continue
        metrics = json.load(open(mpath))["splits"]
        allres[det] = {"splits": {}}
        lines += [f"## {det}", ""]
        for sp in DEFAULT_SPLITS.split(","):
            f = f"{out_dir}/{det}_{sp}.parquet"
            if not os.path.exists(f):
                continue
            d = pd.read_parquet(f).rename(columns={f"logit_{a}": a for a in ARMS})
            res = pb.paired_bootstrap(d, list(ARMS), "native", n_boot=n_boot)
            entry = {"boot": res}
            if sp == "ntire_val_hard":
                v = verdict(res, metrics[sp])
                entry["verdict"] = v
                lines += [f"**Verdict ({det}, ntire_val_hard).** Best-arm robust AUC "
                          f"{v['best_robust_auc']:.4f}"
                          + ("" if v["interpretable"] else
                             " -- BELOW THE 0.60 FLOOR: reported, not interpreted") + ".", ""]
                for key in ("primary", "secondary"):
                    p = v[key]
                    m = v[f"mechanism_{p['arm']}"]
                    lines.append(
                        f"- {key} ({p['arm']} - native, robust): {p['diff']:+.4f} "
                        f"[{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}] -> "
                        f"{'PASSED' if p['passed'] else 'FAILED'}; mechanism (gain "
                        f"752-4016px {m['gain_752_4016']:+.4f} vs 80-336px "
                        f"{m['gain_80_336']:+.4f}) -> {'PASSED' if m['passed'] else 'FAILED'}")
                lines.append("")
            lines.append(pb.to_markdown(res, "native", title=f"{det}, {sp}"))
            if sp.startswith("ext_") and not sp.endswith("_deg"):
                paths = {}
                for a in ARMS:
                    p = f"{out_dir}/{det}_{sp}__{a}.parquet"
                    d[["path", "label", a]].rename(columns={a: "logit"}).to_parquet(p, index=False)
                    paths[a] = p
                wd = pr.within_decile_boot(paths, "data/manifest_external.parquet", n_boot)
                entry["within_decile"] = wd
                if wd["n_used"] == 0:
                    lines.append(f"Size-matched: not computable -- {wd['note']}.")
                else:
                    lines.append("Size-matched (within deciles of native short side, "
                                 f"{wd['n_used']} rows, {wd['n_deciles_used']} deciles): "
                                 + "; ".join(f"{a} {wd['vs_baseline'][a]['diff']:+.4f} "
                                             f"[{wd['vs_baseline'][a]['ci'][0]:+.4f}, "
                                             f"{wd['vs_baseline'][a]['ci'][1]:+.4f}]"
                                             for a in ARMS[1:]))
                lines.append("")
            allres[det]["splits"][sp] = entry
    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
    with open(os.path.splitext(md_path)[0] + ".json", "w") as fh:
        json.dump(allres, fh, indent=2)
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines).strip() + "\n")
    print("\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", choices=sorted(DETECTORS))
    ap.add_argument("--splits", default=DEFAULT_SPLITS)
    ap.add_argument("--degrade-seed", type=int, default=20261008,
                    help="the seed paper_pipeline.sh scored the _deg splits with")
    ap.add_argument("--out-dir", default="outputs/scores/paper/public")
    ap.add_argument("--metrics", default="docs/ntire/paper/public_{detector}.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--summarise", metavar="MD",
                    help="write the amendment-3 tables and verdicts here and exit")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    if a.selfcheck:
        return selfcheck(a.device)
    if a.summarise:
        return summarise(a.out_dir, a.summarise, a.n_boot)
    if not a.detector:
        ap.error("--detector is required unless --selfcheck")
    splits = a.splits.split(",")
    if any(s.startswith("ntire_test") for s in splits):
        sys.exit("REFUSING: ntire_test is sealed (read twice, never again).")

    from ntire_crop_probe import arm_metrics
    os.makedirs(a.out_dir, exist_ok=True)
    mpath = a.metrics.format(detector=a.detector)
    res = json.load(open(mpath)) if os.path.exists(mpath) else {"detector": a.detector, "splits": {}}
    det = DETECTORS[a.detector](a.device)
    for sp in splits:
        out = f"{a.out_dir}/{a.detector}_{sp}.parquet"
        if os.path.exists(out) and sp in res["splits"]:
            print(f"skip {sp}: {out} exists", flush=True)
            continue
        d = split_rows(sp)
        if sp.endswith("_deg"):
            d = d.assign(is_distorted=1)
        logits = score_split(det, d, sp, a.device, a.batch, a.workers, a.degrade_seed)
        keep = [c for c in ("path", "label", "is_distorted", "distortions", "width", "height")
                if c in d.columns]
        frame = d[keep].copy()
        for arm, z in logits.items():
            frame[f"logit_{arm}"] = z
        frame.to_parquet(out, index=False)
        res["splits"][sp] = {arm: arm_metrics(d, z) for arm, z in logits.items()}
        line = "  ".join(f"{arm}: robust={res['splits'][sp][arm]['robust_auc']} "
                         f"clean={res['splits'][sp][arm]['clean_auc']} auc={res['splits'][sp][arm]['auc']:.4f}"
                         for arm in ARMS)
        print(f"{a.detector} {sp}: {line}", flush=True)
        with open(mpath, "w") as f:
            json.dump(res, f, indent=2)
    print(f"wrote {mpath}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
