"""Stage B (spec §3.1, §6.4): train heads on cached features.

Every rung in the ablation ladder is this function with different flags, so
comparisons differ only in the thing under test.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from aigcdet.data.manifest import read_manifest
from aigcdet.eval.metrics import roc_auc
from aigcdet.features.bank import FeatureBank
from aigcdet.models.heads import Detector
from aigcdet.models.losses import (
    LossWeights,
    classification_loss,
    consistency_loss,
    total_loss,
)
from aigcdet.models.sampler import PairedSampler


@dataclass
class RungConfig:
    name: str
    bank_dir: str
    out_dir: str = "outputs/rungs"
    use_recon: bool = False
    use_film: bool = False
    use_augmented: bool = True      # False = A0, clean views only
    use_consistency: bool = False   # A3+
    use_degradation: bool = False   # A2+
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    n_src: int = 64
    m_deg: int = 2
    seed: int = 20260827
    weights: LossWeights = field(default_factory=LossWeights)
    device: str = "cpu"
    # Optional: when the caller has the manifest available (e.g. the CLI),
    # verify the bank is still positionally aligned with it before training.
    # A bank/manifest misalignment after a re-split otherwise produces a
    # silently worse val_auc that nobody can trace back to its cause.
    manifest_path: str | None = None


def _eval_auc(model, bank, idx, use_recon, device) -> float:
    model.eval()
    f = torch.from_numpy(np.asarray(bank.feats[idx, 0]).astype(np.float32)).to(device)
    r = None
    if use_recon:
        if bank.recon is None:
            raise ValueError(
                "bank has no recon features; run attach_recon before evaluating "
                "a use_recon=True rung (see PairedSampler for the same check)")
        r = torch.from_numpy(np.asarray(bank.recon[idx, 0]).astype(np.float32)).to(device)
    with torch.no_grad():
        s = model(f, r)["logit"].cpu().numpy()
    model.train()
    return roc_auc(bank.meta["label"].to_numpy()[idx], s)


def train_rung(cfg: RungConfig) -> dict:
    bank = FeatureBank.open(cfg.bank_dir)
    bank.check_invariants()
    if cfg.manifest_path is not None:
        bank.verify_against_manifest(read_manifest(cfg.manifest_path))

    split = bank.meta["split"].to_numpy()
    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val_internal")[0]
    if len(val_idx) == 0:
        raise ValueError("bank has no val_internal rows; check the manifest splits")

    # nn.Module.reset_parameters() has no generator parameter, so seeding the
    # global RNG is the ordinary way to make module init reproducible. Contain
    # the leak to just this construction with fork_rng rather than mutating
    # process-global state for the rest of the run (Task 5's pattern for the
    # same problem). devices=[] is required: a bare fork_rng() also saves/
    # restores CUDA RNG state, which touches (and initialises) a CUDA context
    # -- and this project runs against a GPU with well under 1 GB free.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        model = Detector(dim_feat=bank.config["dim"], use_recon=cfg.use_recon,
                         use_film=cfg.use_film).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    rng = np.random.default_rng(cfg.seed)
    sampler = PairedSampler(bank, train_idx, n_src=cfg.n_src, m_deg=cfg.m_deg,
                            rng=rng, use_recon=cfg.use_recon, device=cfg.device)

    history = []
    for _ in range(cfg.epochs):
        for batch in sampler:
            if cfg.use_augmented:
                out_clean = model(batch["f_clean"], batch["r_clean"])
                out_deg = model(batch["f_deg"], batch["r_deg"])
                if cfg.use_degradation:
                    w = cfg.weights if cfg.use_consistency else LossWeights(
                        lambda_deg=cfg.weights.lambda_deg, alpha=0.0, beta=0.0)
                    loss, parts = total_loss(out_clean, out_deg, batch, w)
                else:
                    # A1: augmentation only, no degradation auxiliary task --
                    # computed directly rather than via total_loss (which
                    # always includes l_deg) so no degradation-weighted loss
                    # graph is built and thrown away.
                    loss = (classification_loss(out_clean["logit"], batch["y_clean"])
                            + classification_loss(out_deg["logit"], batch["y_deg"])) * 0.5
                    if cfg.use_consistency:
                        loss = loss + consistency_loss(
                            out_clean["logit"].detach(), out_deg["logit"],
                            out_clean["hidden"].detach(), out_deg["hidden"],
                            cfg.weights.alpha, cfg.weights.beta)
                    loss_value = float(loss.detach())
                    parts = {"cls": loss_value, "deg": 0.0, "con": 0.0, "total": loss_value}
            else:
                # A0: clean views only, plain supervised probe.
                out_clean = model(batch["f_clean"], batch["r_clean"])
                loss = classification_loss(out_clean["logit"], batch["y_clean"])
                loss_value = float(loss.detach())
                parts = {"cls": loss_value, "deg": 0.0, "con": 0.0, "total": loss_value}

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        history.append(parts)

    val_auc = _eval_auc(model, bank, val_idx, cfg.use_recon, cfg.device)
    out_dir = os.path.join(cfg.out_dir, cfg.name)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "checkpoint.pt")
    torch.save({"state_dict": model.state_dict(),
                "config": asdict(cfg),
                "dim_feat": bank.config["dim"],
                "backbone": bank.config["backbone"]}, ckpt)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"val_auc": val_auc, "history": history}, f, indent=2)
    return {"checkpoint": ckpt, "val_auc": val_auc, "history": history}


def load_detector(checkpoint_path: str, device: str = "cpu") -> tuple[Detector, dict]:
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = Detector(dim_feat=ck["dim_feat"], use_recon=cfg["use_recon"],
                     use_film=cfg["use_film"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
