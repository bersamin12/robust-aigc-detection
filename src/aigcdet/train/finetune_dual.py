"""Two towers in parallel into one head (experiment 2).

`train_finetune` runs ONE tower: `load_backbone` once, `Detector(dim_feat=
spec.dim)` once. This module runs two, concatenates their pooled vectors and
gives the head a `2 * spec.dim` input. Everything else -- the sampler, the loss,
the DDP reducer, the accumulation scheme -- is imported from `finetune` rather
than restated, so a change to how a pair is built or a gradient is summed lands
in both trainers or neither.

**The configuration is composed, not inherited.** `DualFinetuneConfig` holds a
`FinetuneConfig` rather than subclassing it, for the same reason `FinetuneConfig`
is not a `RungConfig`: a dual config handed to `train_finetune` would train
tower 1, ignore tower 2, and write a checkpoint whose `dim_feat` is half what
its head expects -- a failure with no exception and a plausible-looking
artefact. Composition makes that a type error instead.

**What breaks the symmetry, and how weakly.** Both towers start from the SAME
pretrained checkpoint and see the SAME pixels, so at step 0 their outputs are
identical. They diverge only because the head's first layer has different random
weights over the two halves of the concatenated vector, which makes dL/dw1 !=
dL/dw2. That is a real mechanism but a thin one: nothing else in the setup
distinguishes them (the towers are in eval mode, so dropout is off, and there is
no per-tower augmentation). `perturb_tower2` adds optional Gaussian noise to
tower 2's weights at init as a stronger break; it is 0.0 by default because
turning it on makes the two towers no longer the same pretrained model, which is
a different experiment and should be a deliberate one.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from aigcdet.features.backbones import load_backbone
from aigcdet.features.bank import FeatureBank
from aigcdet.models.heads import Detector
from aigcdet.train.finetune import (
    FinetuneConfig, LiveViewSampler, _dist_init, _forward_tower, _GradReducer,
    _LRSchedule, _owned_chunks, _prepare_batch, _shard_task, _step_loss,
    _stratified_subsample, _WeightAverager, unfreeze_last_n,
)


@dataclass
class DualFinetuneConfig:
    """One two-tower run. `base` carries every field the shared code reads."""
    base: FinetuneConfig
    #: The second tower. Defaults to the first, which is the experiment as
    #: specified: two independently fine-tuned copies of one pretrained model.
    backbone2: str | None = None
    #: Std of Gaussian noise added to tower 2's weights at init. See the module
    #: docstring -- 0.0 keeps both towers the true pretrained model.
    perturb_tower2: float = 0.0


_RESUME_INVARIANT = ("name", "bank_dir", "backbone", "depth", "seed", "lr",
                     "tower_lr", "n_src", "m_deg", "head_hidden")


def _write_ckpt(path, *, head, towers, opt, sampler, epoch, history, cfg,
                dim_feat, names, unfrozen, averager=None) -> None:
    """Atomic per-epoch checkpoint holding BOTH towers.

    Both state dicts are required, and in a fixed order: the eval bank for this
    arm can only be extracted by re-running the same two towers over the same
    concatenation, and swapping them would feed the head's first half the
    second tower's features.
    """
    tmp = path + ".tmp"
    swa = averager.state_dicts() if averager is not None else None
    torch.save({"state_dict": head.state_dict(),
                "tower_state_dicts": [t.state_dict() for t in towers],
                "swa_state_dict": None if swa is None else swa[0],
                "swa_tower_state_dicts": None if swa is None else swa[1:],
                "swa_n": 0 if averager is None else averager.n,
                "optimizer_state_dict": opt.state_dict(),
                "sampler_rng_state": sampler.rng.bit_generator.state,
                "epoch": epoch, "history": history,
                "config": asdict(cfg.base), "dual_config": asdict(cfg),
                "dim_feat": dim_feat, "backbones": list(names),
                "n_towers": len(towers), "unfrozen": unfrozen}, tmp)
    os.replace(tmp, path)


def train_dual(cfg: DualFinetuneConfig) -> dict:
    """Train two towers and one head over their concatenated embeddings."""
    from concurrent.futures import ThreadPoolExecutor

    if isinstance(cfg, FinetuneConfig):
        raise TypeError(
            "train_dual was handed a FinetuneConfig. That config describes a "
            "ONE-tower run; running it here would build a head expecting "
            "2*dim and feed it dim.")
    b = cfg.base
    rank, world = _dist_init(b)
    bank = FeatureBank.open(b.bank_dir)
    name1 = b.backbone or bank.config["backbone"]
    name2 = cfg.backbone2 or name1

    split = bank.meta["split"].to_numpy()
    train_idx = np.where(split == "train")[0]
    if len(train_idx) == 0:
        raise ValueError(f"the bank at {b.bank_dir} has no train rows")
    if b.train_subsample_frac != 1.0:
        train_idx = _stratified_subsample(
            bank, train_idx, b.train_subsample_frac,
            np.random.default_rng(b.seed + 1))

    towers, specs = [], []
    for i, nm in enumerate((name1, name2)):
        t, sp = load_backbone(nm, device=b.device)
        t = t.to(getattr(torch, b.tower_dtype))
        if i == 1 and cfg.perturb_tower2 > 0:
            with torch.no_grad():
                g = torch.Generator(device="cpu").manual_seed(b.seed + 7)
                for p in t.parameters():
                    p.add_(torch.randn(p.shape, generator=g,
                                       dtype=torch.float32).to(p.device, p.dtype)
                           * cfg.perturb_tower2)
        towers.append(t); specs.append(sp)
    if specs[0].dim != specs[1].dim:
        raise ValueError(
            f"the two towers pool to different widths ({specs[0].dim} and "
            f"{specs[1].dim}). The head's input is their concatenation, so a "
            "mismatch is fine arithmetically and meaningless experimentally -- "
            "state which half is which before allowing it.")

    unfrozen = [unfreeze_last_n(t, b.depth) for t in towers]
    for t in towers:
        if b.grad_checkpointing and b.depth and hasattr(
                t, "gradient_checkpointing_enable"):
            t.gradient_checkpointing_enable()
        # NEVER .train(): same reason as the single-tower path.
        t.eval()

    dim_feat = specs[0].dim + specs[1].dim
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(b.seed)
        head = Detector(dim_feat=dim_feat, use_recon=False,
                        use_film=b.use_film, hidden=b.head_hidden).to(b.device)

    tower_params = [[p for p in t.parameters() if p.requires_grad]
                    for t in towers]
    groups = [{"params": list(head.parameters()), "lr": b.lr}]
    for tp in tower_params:
        if tp:
            groups.append({"params": tp, "lr": b.tower_lr})
    opt = torch.optim.AdamW(groups, lr=b.lr, weight_decay=b.weight_decay)
    flat = list(head.parameters()) + [p for tp in tower_params for p in tp]
    reducer = _GradReducer(flat) if world > 1 else None

    sampler = LiveViewSampler(
        bank, train_idx, root=b.root, seed=int(bank.config["seed"]),
        policy=b.policy(), geometric=b.geometric, exclude_families=(),
        n_src=b.n_src, m_deg=b.m_deg,
        rng=np.random.default_rng(b.seed), device=b.device)

    amp = getattr(torch, b.amp_dtype)
    pdtypes = [next(t.parameters()).dtype for t in towers]
    steps_per_epoch = len(sampler)
    sched = _LRSchedule(opt, steps_per_epoch * b.epochs, kind=b.lr_schedule,
                        warmup_frac=b.warmup_frac, min_lr_frac=b.min_lr_frac)
    swa_start = int(b.swa_start_frac * steps_per_epoch * b.epochs)
    # Head first, then the towers in the SAME order the checkpoint stores them.
    averager = _WeightAverager([head, *towers]) if b.swa else None
    out_dir = os.path.join(b.out_dir, b.name)
    ckpt = os.path.join(out_dir, "checkpoint.pt")
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    history, start_epoch = [], 0
    if b.resume and os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location=b.device, weights_only=False)
        stored = ck.get("config") or {}
        diff = {k: (stored.get(k), getattr(b, k)) for k in _RESUME_INVARIANT
                if stored.get(k) != getattr(b, k)}
        if diff:
            raise ValueError(f"refusing to resume {ckpt}: differs on {diff}")
        if ck.get("n_towers") != len(towers):
            raise ValueError(
                f"{ckpt} holds {ck.get('n_towers')} towers, this run has "
                f"{len(towers)}")
        head.load_state_dict(ck["state_dict"])
        for t, sd in zip(towers, ck["tower_state_dicts"]):
            t.load_state_dict(sd)
        opt.load_state_dict(ck["optimizer_state_dict"])
        if ck.get("sampler_rng_state") is not None:
            sampler.rng.bit_generator.state = ck["sampler_rng_state"]
        start_epoch, history = int(ck.get("epoch", 0)), list(ck.get("history") or [])
        print(f"resuming {b.name} from epoch {start_epoch}/{b.epochs}", flush=True)

    gstep = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, b.epochs):
        tasks = sampler.batch_tasks()
        with ThreadPoolExecutor(max_workers=max(1, b.workers)) as pool:
            pending = [pool.submit(_prepare_batch,
                                   _shard_task(t, rank, world, b.src_chunk))
                       for t in tasks]
            for task, fut in zip(tasks, pending):
                prepared = fut.result()
                si, vi = task[7], task[8]
                batch = sampler.targets(si, vi, b.device)
                n_src_all, n_pairs = len(task[0]), len(si)
                opt.zero_grad(set_to_none=True)
                parts = {"cls": 0.0, "deg": 0.0, "con": 0.0, "total": 0.0}
                for g0, g1, l0, l1 in _owned_chunks(n_src_all, b.src_chunk,
                                                    rank, world):
                    srcs = prepared["clean"][l0:l1]
                    degs = prepared["deg"][l0 * b.m_deg:l1 * b.m_deg]
                    p0, p1 = g0 * b.m_deg, g1 * b.m_deg
                    share = len(degs) / n_pairs
                    with torch.autocast(device_type=b.device.split(":")[0],
                                        dtype=amp, enabled=b.device != "cpu"):
                        # Each tower forwards the SAME pixels. The clean image
                        # is still forwarded once per source per tower and its
                        # embedding repeated, exactly as the single-tower path.
                        f_src = torch.cat(
                            [_forward_tower(t, sp, srcs, b.device, dt, b.src_chunk)
                             for t, sp, dt in zip(towers, specs, pdtypes)], dim=-1)
                        f_clean = f_src.repeat_interleave(b.m_deg, dim=0)
                        f_deg = torch.cat(
                            [_forward_tower(t, sp, degs, b.device, dt, b.src_chunk)
                             for t, sp, dt in zip(towers, specs, pdtypes)], dim=-1)
                    micro = {k: (v[p0:p1] if torch.is_tensor(v) else v)
                             for k, v in batch.items()}
                    micro["f_clean"] = f_clean.float()
                    micro["f_deg"] = f_deg.float()
                    loss, part = _step_loss(head, micro, b)
                    (loss * share).backward()
                    for k in parts:
                        parts[k] += part[k] * share
                if reducer is not None:
                    reducer.reduce()
                sched.apply(gstep)
                opt.step()
                if averager is not None and gstep >= swa_start:
                    averager.update()
                gstep += 1
        if world > 1:
            import torch.distributed as dist
            t_ = torch.tensor([parts[k] for k in ("cls", "deg", "con", "total")],
                              device=b.device, dtype=torch.float64)
            dist.all_reduce(t_, op=dist.ReduceOp.SUM)
            parts = dict(zip(("cls", "deg", "con", "total"), t_.tolist()))
        history.append({"epoch": epoch, **parts})
        if rank == 0:
            _write_ckpt(ckpt, head=head, towers=towers, opt=opt,
                        sampler=sampler, epoch=epoch + 1, history=history,
                        cfg=cfg, dim_feat=dim_feat, names=(name1, name2),
                        unfrozen=unfrozen, averager=averager)
        if world > 1:
            import torch.distributed as dist
            dist.barrier()

    if world > 1 and rank != 0:
        import torch.distributed as dist
        dist.barrier(); dist.destroy_process_group()
        return {"checkpoint": ckpt, "unfrozen": unfrozen, "history": history,
                "rank": rank}
    if rank == 0:
        _write_ckpt(ckpt, head=head, towers=towers, opt=opt, sampler=sampler,
                    epoch=b.epochs, history=history, cfg=cfg,
                    dim_feat=dim_feat, names=(name1, name2), unfrozen=unfrozen,
                    averager=averager)
    result = {"unfrozen": unfrozen, "history": history, "world_size": world,
              "backbones": [name1, name2], "dim_feat": dim_feat}
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    if world > 1:
        import torch.distributed as dist
        dist.barrier(); dist.destroy_process_group()
    return {"checkpoint": ckpt, **result}
