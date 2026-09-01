"""Inference for fine-tuned tower checkpoints, straight from pixels.

The bundle path in `aigcdet.infer` serves the frozen-backbone rungs: cached
features in, calibrated decisions out. The shipping model is not one of those
-- it is two fully fine-tuned DINOv2 towers whose weights moved in training,
which invalidates every cached feature bank by construction. So this module
scores from pixels: decode, canonicalise under the EVAL convention (the
centre window, no rng -- the same choice `eval/grid` and
`scripts/score_plan_splits.py` make, so a probability here is comparable with
one there), forward through each tower, concatenate, head, sigmoid.

Loading mirrors `scripts/score_plan_splits.py` exactly, because that script
produced the reported numbers and two loaders that agree today drift apart
quietly. Both checkpoint shapes `_write_ckpt` produces are handled: the
single-tower unfreeze arm (`tower_state_dict`) and the dual arm
(`tower_state_dicts`, two towers concatenated in a FIXED order -- swapping
them would feed the head's first half the second tower's features).

Weights are loaded into fp32 EXACTLY as saved, then cast to bfloat16 for
inference: bf16 shares fp32's exponent range, so a tower whose weights moved
in training cannot overflow the way fp16 did for dinov3l. A checkpoint
exported by `scripts/export_finetuned.py` stores the tower weights in bf16
already; loading bf16 into an fp32 parameter and casting back to bf16 is the
identity, so the slim file scores identically to the 3x larger training one.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from aigcdet.augment.canonical import CanonPolicy, canonicalise
from aigcdet.features.backbones import load_backbone
from aigcdet.models.heads import Detector
from aigcdet.train.finetune import _forward_tower, _windowed


def policy_from_config(cfg: dict) -> CanonPolicy:
    """The canonicalisation policy the checkpoint was trained under.

    Read off the checkpoint rather than passed by the caller: the served path
    must standardise images the way the towers were taught to see them, and
    the checkpoint's config is the only record of which way that was.
    """
    kw = {}
    if cfg.get("nominal_side") is not None:
        kw["nominal_side"] = int(cfg["nominal_side"])
    return CanonPolicy(mode=cfg["policy_mode"], crop_side=cfg.get("crop_side"),
                       crop_clamp=bool(cfg.get("crop_clamp", False)), **kw)


def load_finetuned(ckpt_path: str, device: str, use_swa: bool = False):
    """(towers, specs, head, policy, ck) from a training or exported checkpoint."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    dual = "tower_state_dicts" in ck
    names = ck["backbones"] if dual else [ck["backbone"]]
    tower_sds = ck["tower_state_dicts"] if dual else [ck["tower_state_dict"]]
    swa_towers = (ck.get("swa_tower_state_dicts") if dual
                  else ([ck["swa_tower_state_dict"]]
                        if ck.get("swa_tower_state_dict") is not None else None))
    head_sd = ck["state_dict"]
    if use_swa:
        if not ck.get("swa_n"):
            raise SystemExit(
                "REFUSING --swa: this checkpoint holds no SWA state "
                f"(swa_n={ck.get('swa_n')}); score the final weights instead")
        head_sd, tower_sds = ck["swa_state_dict"], swa_towers

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
    head = Detector(dim_feat=ck["dim_feat"],
                    use_recon=bool(cfg.get("use_recon")),
                    use_film=bool(cfg.get("use_film")),
                    hidden=int(cfg.get("head_hidden", 512))).to(device)
    head.load_state_dict(head_sd)
    head.eval()
    return towers, specs, head, policy_from_config(cfg), ck


def strip_checkpoint(ck: dict, use_swa: bool = False) -> dict:
    """A training checkpoint reduced to what inference reads.

    Drops the optimizer state (2x the weights), the sampler and torch RNG
    streams (resume-only) and the history. Tower weights go to bf16 -- the
    scoring path casts them to bf16 anyway, so the slim file is bit-identical
    in use and one third the size. The head stays fp32: it RUNS in fp32
    (`head(feats.float())`), and it is a rounding error of the total anyway.

    When `use_swa` is set the SWA average is promoted to be THE weights of the
    exported file, so a consumer cannot accidentally score the wrong ones.
    """
    def bf16(sd: dict) -> dict:
        return {k: v.to(torch.bfloat16) if v.is_floating_point() else v
                for k, v in sd.items()}

    dual = "tower_state_dicts" in ck
    if use_swa:
        if not ck.get("swa_n"):
            raise SystemExit(f"REFUSING --swa: no SWA state (swa_n={ck.get('swa_n')})")
        head_sd = ck["swa_state_dict"]
        tower_sds = (ck["swa_tower_state_dicts"] if dual
                     else [ck["swa_tower_state_dict"]])
    else:
        head_sd = ck["state_dict"]
        tower_sds = ck["tower_state_dicts"] if dual else [ck["tower_state_dict"]]

    slim = {"state_dict": head_sd,
            "epoch": ck.get("epoch"), "config": ck["config"],
            "dim_feat": ck["dim_feat"], "exported": True,
            "exported_weights": "swa" if use_swa else "final",
            # Absent, not None: `load_finetuned` keys on presence.
            "swa_n": 0}
    if dual:
        slim["backbones"] = ck["backbones"]
        slim["tower_state_dicts"] = [bf16(sd) for sd in tower_sds]
    else:
        slim["backbone"] = ck["backbone"]
        slim["tower_state_dict"] = bf16(tower_sds[0])
    return slim


def score_paths(paths: list[str], towers, specs, head, policy: CanonPolicy,
                device: str, batch: int = 16, workers: int = 8,
                chunk: int = 8, tta: bool = False,
                progress: bool = False) -> np.ndarray:
    """P(AI-generated) per path, in order. One deterministic view per image
    (the centre window), or the 8-view TTA mean of per-view logits."""
    recon = None
    if head.use_recon:
        # The VAE branch scores from the SAME canonicalised image the towers
        # see, computed live -- no bank covers loose files.
        from aigcdet.features.recon import load_recon_models, recon_features
        recon = load_recon_models(device=device)

    tta_views = None
    if tta:
        from aigcdet.eval.tta import TTA_VIEWS, apply_tta_view
        tta_views = TTA_VIEWS

    from PIL import Image

    def prepare(batch_paths: list[str]) -> list[np.ndarray]:
        out = []
        for p in batch_paths:
            with Image.open(p) as im:
                decoded = np.asarray(im.convert("RGB"), dtype=np.uint8)
            # Eval convention: ONE deterministic canonicalisation, the centre
            # window (`rng is None` gives the centre window). A random crop
            # here would score a different picture per invocation.
            std = canonicalise(decoded, policy=policy)
            if tta_views is None:
                out.append(std)
            else:
                out.extend(apply_tta_view(std, v) for v in tta_views)
        return out

    batches = [paths[i:i + batch] for i in range(0, len(paths), batch)]
    probs = np.empty(len(paths), dtype=np.float64)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool, torch.no_grad():
        stream = _windowed(pool, prepare, batches, 2 * workers)
        if progress:
            from tqdm import tqdm
            stream = tqdm(stream, total=len(batches), desc="scoring")
        for _, fut in stream:
            imgs = fut.result()
            feats = torch.cat(
                [_forward_tower(t, sp, imgs, device, torch.bfloat16, chunk)
                 for t, sp in zip(towers, specs)], dim=-1)
            r = None
            if recon is not None:
                from aigcdet.features.recon import recon_features
                vae, lpips_fn = recon
                r = torch.from_numpy(np.stack(
                    [recon_features(im, vae, lpips_fn, device=device)
                     for im in imgs])).to(device)
            logit = head(feats.float(), r)["logit"]
            if tta_views is not None:
                # Mean of per-view LOGITS, then sigmoid -- eval/tta.py's
                # aggregation, kept identical here.
                logit = logit.view(-1, len(tta_views)).mean(dim=1)
            p = torch.sigmoid(logit).double().cpu().numpy()
            probs[done:done + len(p)] = p
            done += len(p)
    assert done == len(paths)
    return probs
