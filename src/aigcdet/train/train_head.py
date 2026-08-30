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
    #: Generator families to drop from the TRAINING rows only.
    #:
    #: Which families are held out is frozen into the manifest's `split`
    #: column, and changing it there means a new manifest, a new fingerprint
    #: and every bank on disk orphaned. That is the right cost for the corpus
    #: and the wrong cost for a question we want to ask several times: the
    #: manifest's pinned pair (`SDwithAdaptor_controlnet`, `VQGAN`) holds out
    #: an ADAPTER while its siblings `SDwithAdaptor_lora` and
    #: `SDwithAdaptor_lycris` stay in training, and an adapter changes the
    #: conditioning, not the decoder that leaves the forensic trace. Asking
    #: the harder question -- hold out the whole lineage -- is then a matter
    #: of not TRAINING on the siblings, which is a row mask over features
    #: already cached.
    #:
    #: Applied to `train` only. `val_internal` is the authentic population the
    #: selection metric is computed against, so thinning it would move the
    #: operating point and make two rungs incomparable for a reason that has
    #: nothing to do with the rung.
    train_exclude_generators: tuple[str, ...] = ()


def manifest_rows_for_bank(bank: FeatureBank, manifest_df):
    """The manifest rows `bank` was extracted from, recovered from the whole.

    A training bank is extracted with `--split train,val_internal`, so it
    covers a SUBSET of the manifest and its `manifest_sha256` fingerprints
    that subset -- not the file on disk. Handing the whole manifest to
    `verify_against_manifest` therefore failed on every honest bank: the
    alignment check rejected exactly the banks it exists to bless, and the
    only way past it was to stop passing --manifest, which turned the check
    off. Reconstructing the selection restores it.

    This mirrors `scripts/extract_features.py:select_splits` exactly --
    `df[df["split"].isin(...)]`, which preserves manifest order and index
    labels -- so the reconstructed frame fingerprints to what Stage A
    recorded. It deliberately does NOT reconstruct `--limit` or `--shard`:
    a bank that covers only part of its splits SHOULD still fail here,
    because training on a partial bank while believing it is whole is the
    error the row-count check catches.
    """
    splits = {str(s) for s in bank.meta["split"].unique()}
    return manifest_df[manifest_df["split"].isin(splits)]


def _eval_auc(model, bank, idx, use_recon, device, view: int = 0) -> float:
    """ROC-AUC over `idx` for ONE cached view. `view=0` is the clean view."""
    model.eval()
    f = torch.from_numpy(np.asarray(bank.feats[idx, view]).astype(np.float32)).to(device)
    r = None
    if use_recon:
        if bank.recon is None:
            raise ValueError(
                "bank has no recon features; run attach_recon before evaluating "
                "a use_recon=True rung (see PairedSampler for the same check)")
        r = torch.from_numpy(
            np.asarray(bank.recon[idx, view]).astype(np.float32)).to(device)
    with torch.no_grad():
        s = model(f, r)["logit"].cpu().numpy()
    model.train()
    return roc_auc(bank.meta["label"].to_numpy()[idx], s)


def _eval_aucs(model, bank, idx, use_recon, device) -> dict[str, float]:
    """Both headline numbers for a rung.

    `val_auc` is the clean-view AUC (view 0). It is kept because Plan 3 and
    this plan's completion criterion both reference it -- but on its own it
    measures the ONE condition where robustness training helps least, so A0
    (clean views only) may well win it while being the weakest rung under
    degradation. `val_auc_mean_views` averages the AUC over every cached view,
    clean and augmented alike, which is the ladder's actual thesis.
    """
    per_view = [_eval_auc(model, bank, idx, use_recon, device, view=v)
                for v in range(bank.config["n_views"])]
    return {"val_auc": per_view[0],
            "val_auc_mean_views": float(np.mean(per_view))}


def _drop_generators(bank, train_idx, families) -> "np.ndarray":
    """`train_idx` without the rows whose generator is in `families`.

    Two things are refused rather than absorbed, because both look like a
    working holdout from the outside and neither is:

    * A family that matches NOTHING. A misspelled or absent name would drop
      zero rows and train on the very families the caller meant to hold out,
      while every downstream number carries on calling itself a held-out
      score. There is no honest way to report that, so it raises.
    * A mask that empties the training set. A rung with no training rows is
      not a rung.
    """
    gen = bank.meta["generator"].to_numpy()
    present = set(map(str, np.unique(gen[train_idx])))
    wanted = [str(f) for f in families]
    absent = sorted(f for f in wanted if f not in present)
    if absent:
        raise ValueError(
            f"train_exclude_generators names {absent}, which no training row "
            f"carries. The bank's train split holds {sorted(present)}. A name "
            "that matches nothing drops nothing, so the rung would train on "
            "the families it claims to hold out and still report a held-out "
            "score.")
    keep = ~np.isin(gen[train_idx], wanted)
    if not keep.any():
        raise ValueError(
            f"train_exclude_generators {wanted} removes every training row.")
    return train_idx[keep]


def train_rung(cfg: RungConfig) -> dict:
    bank = FeatureBank.open(cfg.bank_dir)
    bank.check_invariants()
    if cfg.manifest_path is not None:
        # Restricted to the splits this bank covers -- see manifest_rows_for_bank.
        full = read_manifest(cfg.manifest_path)
        bank.verify_against_manifest(manifest_rows_for_bank(bank, full))

    split = bank.meta["split"].to_numpy()
    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val_internal")[0]
    if cfg.train_exclude_generators:
        train_idx = _drop_generators(bank, train_idx,
                                     cfg.train_exclude_generators)
    if len(val_idx) == 0:
        # Name what the bank DOES contain: this fires after Stage A has
        # already been paid for (8-13 h on Kaggle), so "check the manifest
        # splits" is not enough to act on.
        present = {str(s): int(n) for s, n in
                   zip(*np.unique(split, return_counts=True))}
        raise ValueError(
            "bank has no val_internal rows, so the val AUC every rung reports "
            f"cannot be computed. This bank contains splits {present}. Re-extract "
            "with `--split train,val_internal` -- Stage A must cover both, in one "
            "bank (scripts/extract_features.py)")

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

    aucs = _eval_aucs(model, bank, val_idx, cfg.use_recon, cfg.device)
    out_dir = os.path.join(cfg.out_dir, cfg.name)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, "checkpoint.pt")
    torch.save({"state_dict": model.state_dict(),
                "config": asdict(cfg),
                "dim_feat": bank.config["dim"],
                "backbone": bank.config["backbone"]}, ckpt)
    result = {**aucs, "history": history}
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    return {"checkpoint": ckpt, **result}


def load_detector(checkpoint_path: str, device: str = "cpu") -> tuple[Detector, dict]:
    """Rebuild a trained Detector from its checkpoint.

    `weights_only=True` is deliberate and must stay: Plan 4 ships a checkpoint
    the public downloads, and the permissive loader executes arbitrary pickle
    opcodes. Everything train_rung saves -- tensors, `asdict(cfg)` (plain
    dicts/str/int/float/bool after the nested LossWeights is flattened), and
    two scalars -- is inside the safe allowlist.
    """
    ck = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cfg = ck["config"]
    model = Detector(dim_feat=ck["dim_feat"], use_recon=cfg["use_recon"],
                     use_film=cfg["use_film"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
