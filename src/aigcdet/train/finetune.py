"""Rung ladder D0-D4: how much of the tower to unfreeze.

Every other rung in this project trains on cached embeddings. That is the
Stage A / Stage B split, and it is what makes the ladder cheap: Stage A pays
for the tower once, Stage B fits a ~1M-parameter head on a memmap in minutes
and never sees an image. This module is the one place that breaks it, because
"what if the tower were not frozen?" cannot be asked from a cache -- the whole
question is about gradients the cache was built to avoid.

**D0 is the control, and it is exact rather than approximate.** Depth 0
unfreezes nothing, so its tower is the frozen tower, so the features it
computes are the ones already in the bank -- provided the pixels match. They
do, by construction: `LiveViewSampler` rebuilds view `v` of row `r` through
`features.extract.build_view`, which is the same function the extraction used
and keys everything on `(seed, row_id, view_idx)` alone. And the batch
SEQUENCE matches too, because the sampler inherits `PairedSampler.draw_batch`
rather than reimplementing it. So D0 and the cached rung it baselines are the
same model trained twice, not two similar models, and any gap between them is
a bug in this file rather than a finding about unfreezing.
`tests/train/test_finetune.py` asserts exactly that.

**Why the frozen part stays in eval mode.** The tower is never put in
`train()` mode, at any depth. Dropout and any running statistics would then
differ between D0 and D1 for a reason unrelated to which parameters receive
gradient, and the ladder is supposed to isolate that one thing. Gradients
still flow to whatever `unfreeze_last_n` marked trainable; eval mode governs
stochastic layers, not autograd.

**The cost that is easy to miss.** Unfreezing invalidates every cached bank.
A tower whose weights moved does not produce the features on disk, so each
depth needs its own eval bank re-extracted before it can be scored -- the
scoring path is unchanged, but it cannot read the bank the frozen rungs use.
`scripts/run_unfreeze_ladder.py` does that per depth; nothing here does it
implicitly, because a rung silently scored against another tower's bank is
exactly the class of error this repo keeps finding.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from PIL import Image

from aigcdet.augment.canonical import DEFAULT_POLICY, CanonPolicy
from aigcdet.features.backbones import (
    BackboneSpec, load_backbone, model_inputs, _pool,
)
from aigcdet.features.bank import FeatureBank
from aigcdet.features.extract import build_view
from aigcdet.models.heads import Detector
from aigcdet.models.losses import (
    LossWeights, classification_loss, consistency_loss, total_loss,
)
from aigcdet.models.sampler import PairedSampler

#: Attribute paths that hold a tower's sequential block list, tried in order.
#: HF transformer encoders name it `encoder.layer` (BERT-lineage, which DINOv2
#: and DINOv3 follow) or `encoder.layers` (CLIP/SigLIP lineage); timm ViTs use
#: a bare `blocks`. Convolutional towers are `encoder.stages`/`stages`, and are
#: listed so the error below can tell "this tower has stages, you asked for
#: blocks" apart from "nothing here looks like a tower at all".
_BLOCK_PATHS: tuple[str, ...] = (
    "encoder.layer", "encoder.layers", "blocks", "layers",
    "encoder.stages", "stages",
)


def _resolve(model, path: str):
    obj = model
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def tower_blocks(model) -> tuple[str, torch.nn.ModuleList]:
    """`(path, blocks)` for the tower's sequential block list.

    Raises rather than returning empty. A depth ladder that cannot find the
    blocks would unfreeze nothing at every depth, D1..D4 would all equal D0,
    and the run would report "unfreezing does not help" -- a false conclusion
    that produces a complete, plausible table. There is no safe default here,
    so there is no default.
    """
    for path in _BLOCK_PATHS:
        blocks = _resolve(model, path)
        if isinstance(blocks, (torch.nn.ModuleList, torch.nn.Sequential)) and len(blocks):
            return path, blocks
    have = sorted(n for n, _ in model.named_children())
    raise ValueError(
        f"cannot find a block list on {type(model).__name__}: none of "
        f"{list(_BLOCK_PATHS)} resolves to a non-empty ModuleList. Its top-level "
        f"children are {have}. Add this tower's path to `_BLOCK_PATHS` -- do not "
        "let the ladder run without it, because unfreezing nothing at every "
        "depth reports 'depth does not help' from a table that looks fine.")


def unfreeze_last_n(model, depth: int) -> dict:
    """Mark the last `depth` blocks trainable. Returns what was done.

    A record, not a bool: the number of trainable parameters is the one thing
    that distinguishes the rungs of this ladder, so it travels with the result
    instead of being inferred from the config. `depth=0` is the frozen control
    and must leave the tower exactly as `load_backbone` handed it over.

    The tower's final norm rides with the last block when anything at all is
    unfrozen. It sits after the last block and before the pooling, so freezing
    it while unfreezing the block beneath asks the block to move its output
    distribution underneath a normalisation calibrated for the old one.
    """
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")
    path, blocks = tower_blocks(model)
    if depth > len(blocks):
        raise ValueError(
            f"depth={depth} but this tower has only {len(blocks)} blocks at "
            f"{path!r}. Asking for more is not the same as asking for all of "
            "them, and silently clamping would make two different rungs of the "
            "ladder the same rung under different names.")

    for p in model.parameters():
        p.requires_grad_(False)
    unfrozen: list[str] = []
    if depth:
        for block in blocks[len(blocks) - depth:]:
            for name, p in block.named_parameters():
                p.requires_grad_(True)
                unfrozen.append(name)
        # Whatever normalisation sits between the last block and the pooling.
        for attr in ("layernorm", "norm", "post_layernorm", "ln_post"):
            norm = getattr(model, attr, None)
            if isinstance(norm, torch.nn.Module):
                for p in norm.parameters():
                    p.requires_grad_(True)
                unfrozen.append(attr)
                break

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"depth": depth, "block_path": path, "n_blocks": len(blocks),
            "trainable_params": int(trainable), "tower_params": int(total),
            "unfrozen_modules": len(unfrozen)}


# ---------------------------------------------------------------------------
# The live sampler
# ---------------------------------------------------------------------------

#: One prepared batch's CPU work, as a picklable tuple for a worker process.
BatchTask = tuple


def _prepare_batch(task: BatchTask) -> dict:
    """Decode and augment one batch's images. Pure, so a pool can run it.

    Decoding dominates: ~199 ms/image measured, against a GPU step of well
    under a second, so preparing `n_src` images inline would leave the card
    idle for most of every step. Nothing here is shared or mutated and every
    view is a pure function of `(seed, row_id, view_idx)` and its own file, so
    running it in another process is safe by construction and produces the
    same pixels as running it inline.
    """
    (paths, row_ids, view_ids, seed, policy, geometric, exclude) = task
    clean, deg = [], []
    m_deg = view_ids.shape[1]
    for path, row_id, views in zip(paths, row_ids, view_ids):
        with Image.open(path) as im:
            decoded = np.asarray(im.convert("RGB"), dtype=np.uint8)
        clean.append(build_view(decoded, 0, int(row_id), seed, policy=policy,
                                geometric=geometric, exclude_families=exclude)[0])
        for v in views:
            deg.append(build_view(decoded, int(v), int(row_id), seed,
                                  policy=policy, geometric=geometric,
                                  exclude_families=exclude)[0])
    return {"clean": clean, "deg": deg, "m_deg": m_deg}


class LiveViewSampler(PairedSampler):
    """`PairedSampler`'s draws, with the pixels rebuilt instead of read.

    Inherits the draw so the batch sequence is identical to the cached
    sampler's for the same seed -- see `PairedSampler.draw_batch`. The bank is
    still open and still used, for everything except `feats`: labels,
    generators, splits, row ids and the degradation targets all come from it,
    because they are properties of the corpus and the recipe rather than of the
    tower, and recomputing them here would be a second implementation of
    something already on disk.

    What this yields is IMAGES, not features. The forward is the caller's, so
    the tower stays in the training loop where its gradients are.
    """

    def __init__(self, bank, indices, *, root: str, seed: int,
                 policy: CanonPolicy = DEFAULT_POLICY, geometric: bool = False,
                 exclude_families: tuple[str, ...] = (), **kw):
        super().__init__(bank, indices, **kw)
        self.root, self.seed = root, seed
        self.policy, self.geometric = policy, geometric
        self.exclude_families = tuple(exclude_families)
        col = "rel_path" if "rel_path" in bank.meta.columns else "path"
        self._paths = bank.meta[col].to_numpy()
        self._absolute = col == "path"
        self._row_ids = bank.meta["row_id"].to_numpy()

    def _path(self, i: int) -> str:
        p = str(self._paths[i])
        return p if self._absolute else os.path.join(self.root, p)

    def batch_tasks(self) -> list[BatchTask]:
        """Every task for one epoch, drawn up front.

        Drawn ahead rather than lazily so the epoch can be handed to a worker
        pool in order: the draws are cheap and deterministic, the decoding is
        neither. It also means the RNG is advanced exactly once per epoch, in
        one place, which is what keeps the sequence comparable with the cached
        sampler's.
        """
        tasks = []
        for _ in range(len(self)):
            src, si, vi = self.draw_batch()
            views = vi.reshape(len(src), self.m_deg)
            tasks.append((
                [self._path(i) for i in src],
                self._row_ids[src],
                views,
                self.seed, self.policy, self.geometric, self.exclude_families,
                si, vi,
            ))
        return tasks

    def targets(self, si: np.ndarray, vi: np.ndarray, device: str) -> dict:
        """The parts of a batch that come from the bank, not from the tower."""
        y = self.bank.meta["label"].to_numpy()[si].astype(np.float32)
        return {
            "y_clean": torch.from_numpy(y).to(device),
            "y_deg": torch.from_numpy(y).to(device),
            "presence_deg": torch.from_numpy(
                np.asarray(self.bank.presence[si, vi]).astype(np.float32)).to(device),
            "severity_deg": torch.from_numpy(
                np.asarray(self.bank.severity[si, vi]).astype(np.float32)).to(device),
        }


# ---------------------------------------------------------------------------
# The rung
# ---------------------------------------------------------------------------

@dataclass
class FinetuneConfig:
    """One rung of the unfreeze ladder.

    Deliberately NOT a subclass of `RungConfig`. The two share most of their
    fields, and sharing the class would let a finetune rung be handed to
    `train_rung` (which would silently ignore `depth` and train a frozen head)
    or a cached rung to `train_finetune`. They are different experiments with
    different costs and different artefacts, so they are different types.
    """
    name: str
    bank_dir: str                    # for labels, splits and degradation targets
    root: str                        # where this box mounts the corpus
    #: How many trailing tower blocks receive gradient. 0 is the frozen
    #: control and must reproduce the cached rung of the same configuration.
    depth: int = 0
    backbone: str | None = None      # defaults to the bank's own
    out_dir: str = "outputs/unfreeze"
    use_augmented: bool = True
    use_consistency: bool = True
    use_degradation: bool = True
    use_film: bool = False
    head_hidden: int = 512
    epochs: int = 5
    lr: float = 1e-3                 # the HEAD's learning rate
    #: The tower's, and it is two orders of magnitude smaller on purpose. The
    #: head is random at step 0 and the tower is not: one pretrained-scale step
    #: at the head's rate would undo more than the ladder is trying to measure,
    #: and the first epochs would be spent recovering from it. Recorded per
    #: rung because it is the single most consequential number here after
    #: `depth` itself.
    tower_lr: float = 1e-5
    weight_decay: float = 1e-4
    n_src: int = 64
    m_deg: int = 2
    seed: int = 20260827
    weights: LossWeights = field(default_factory=LossWeights)
    device: str = "cuda"
    workers: int = 16
    #: Source images per micro-batch. The optimiser still steps once per FULL
    #: batch -- this is gradient accumulation, not a smaller batch.
    #:
    #: It exists because the batch does not fit. dinov2regl sees 518px, so one
    #: image is 1374 tokens of width 1024, and holding a 64-source batch's
    #: activations across 24 blocks for the backward pass OOMs a 24 GiB 4090
    #: outright (measured, 2026-08-31). Shrinking `n_src` instead would have
    #: been the easy fix and the wrong one: batch size changes the gradient,
    #: and the ladder is supposed to be comparable with the cached rungs, which
    #: use n_src=64. Accumulating leaves the step mathematically identical --
    #: every loss here is a mean, so a micro-batch's contribution is exact once
    #: weighted by its share of the pairs -- and only changes the peak memory.
    #: 8, measured on a 4090 (2026-08-31, `docs/bench_finetune_dinov2regl.json`):
    #: depth 4 runs 70.2 img/s at 7.7 GiB against 65.7 at 4 and 66.9 at 16, and
    #: 32 OOMs. Past 8 the step is memory-bandwidth bound and a larger chunk
    #: buys nothing but peak memory.
    src_chunk: int = 8
    #: Recompute block activations in the backward instead of storing them.
    #: Buys roughly an order of magnitude of memory for ~30% time, which is
    #: what makes a larger `src_chunk` reachable on one card.
    grad_checkpointing: bool = True
    #: bfloat16, never float16. DINOv3-L overflows float16 at hidden layer 1
    #: and the 2026-08-29 banks came out 100% NaN; a backward pass is strictly
    #: more exposed to that than the forward that found it. bf16 also needs no
    #: GradScaler, which is one fewer piece of state to get wrong per depth.
    amp_dtype: str = "bfloat16"
    #: The tower's MASTER weights, float32 at EVERY depth including 0.
    #:
    #: `load_backbone` hands back float16, which is right for the frozen
    #: inference it was written for and wrong to hold gradients in: fp16
    #: updates at `tower_lr=1e-5` underflow to nothing, so D1..D4 would train a
    #: tower that never moves and the ladder would report that depth does not
    #: help.
    #:
    #: Applied at depth 0 too, and that is the deliberate part. Running D0 in
    #: the cache's fp16 would make it bit-identical to the cached rung, which
    #: is tempting -- but it would also put a SECOND difference (the numerics)
    #: inside a ladder whose whole claim is that consecutive rungs differ by
    #: one thing. One regime for every depth keeps D0 the control it is
    #: presented as. The pixel path's agreement with the cache is verified
    #: separately and exactly, in `tests/train/test_finetune.py`, where a
    #: deterministic stub removes the tower from the question entirely.
    tower_dtype: str = "float32"
    policy_mode: str = "crop"
    crop_side: int = 200
    geometric: bool = False
    train_exclude_generators: tuple[str, ...] = ()
    #: Fraction of the train rows to keep, stratified by (generator, label).
    #: 1.0 is the whole split and the default; anything smaller exists to
    #: measure the DATA-SCALING slope -- how much of a rung's score is bought
    #: by corpus size rather than by depth -- so that a decision about a much
    #: larger corpus rests on a measured slope instead of an argument.
    train_subsample_frac: float = 1.0

    def policy(self) -> CanonPolicy:
        return CanonPolicy(mode=self.policy_mode, crop_side=self.crop_side)


def _forward_tower(model, spec: BackboneSpec, imgs, device: str,
                   dtype: torch.dtype, chunk: int) -> torch.Tensor:
    """Differentiable counterpart of `features.backbones.embed`.

    Reuses `model_inputs` and `_pool` rather than restating them, so the
    preprocessing a D0 tower applies is the SAME preprocessing the cached bank
    was built with -- resize, normalisation, prefix-token stripping and pooling
    included. `embed` itself cannot be reused: it is `@torch.inference_mode()`,
    which is not merely no-grad but poisons its outputs against ever being used
    in a graph.
    """
    outs = []
    for i in range(0, len(imgs), chunk):
        inputs = model_inputs(spec, imgs[i:i + chunk], device, dtype)
        outs.append(_pool(model, spec, inputs))
    return torch.cat(outs, dim=0)


def _step_loss(head, batch: dict, cfg: FinetuneConfig):
    """One micro-batch's loss and its parts, for whichever rung this is.

    Identical in structure to `train_head.train_rung`'s body, deliberately:
    the unfreeze ladder varies the TOWER, so every other term has to be the
    one the cached rungs used or D0 stops being their control.
    """
    if not cfg.use_augmented:
        out_clean = head(batch["f_clean"], None)
        loss = classification_loss(out_clean["logit"], batch["y_clean"])
        v = float(loss.detach())
        return loss, {"cls": v, "deg": 0.0, "con": 0.0, "total": v}

    out_clean = head(batch["f_clean"], None)
    out_deg = head(batch["f_deg"], None)
    if cfg.use_degradation:
        w = cfg.weights if cfg.use_consistency else LossWeights(
            lambda_deg=cfg.weights.lambda_deg, alpha=0.0, beta=0.0)
        return total_loss(out_clean, out_deg, batch, w)

    loss = (classification_loss(out_clean["logit"], batch["y_clean"])
            + classification_loss(out_deg["logit"], batch["y_deg"])) * 0.5
    if cfg.use_consistency:
        loss = loss + consistency_loss(
            out_clean["logit"].detach(), out_deg["logit"],
            out_clean["hidden"].detach(), out_deg["hidden"],
            cfg.weights.alpha, cfg.weights.beta)
    v = float(loss.detach())
    return loss, {"cls": v, "deg": 0.0, "con": 0.0, "total": v}


def _owned_chunks(n_src, src_chunk, rank, world):
    """Which accumulation chunks this rank runs, and where they sit locally.

    Yields `(g0, g1, l0, l1)`: the chunk's source range in the FULL batch and
    its range in this rank's own (shorter) prepared arrays. The two differ
    because a rank decodes only the sources it will forward, so its arrays are
    packed while the targets it indexes -- labels, presence, severity -- are
    still the whole batch's.

    Chunks go round-robin, `c % world == rank`. Round-robin rather than
    contiguous blocks because the last chunk of a batch can be short: handing
    one rank every short chunk would make its share of the work systematically
    smaller, and the point of the split is that four cards finish together.
    """
    out, loc = [], 0
    for c, g0 in enumerate(range(0, n_src, src_chunk)):
        g1 = min(g0 + src_chunk, n_src)
        if c % world == rank:
            out.append((g0, g1, loc, loc + (g1 - g0)))
            loc += g1 - g0
    return out


def _shard_task(task, rank, world, src_chunk):
    """This rank's slice of one batch's CPU work.

    Decoding is ~199 ms/image and every rank would otherwise decode all
    `n_src` sources to forward a quarter of them, which turns a 4x split of
    the GPU into a 4x multiplication of the CPU. Slicing the task first keeps
    both halves of the step parallel.
    """
    paths, row_ids, views, seed, policy, geometric, exclude = task[:7]
    if world == 1:
        return task[:7]
    keep = [i for g0, g1, _, _ in _owned_chunks(len(paths), src_chunk, rank, world)
            for i in range(g0, g1)]
    return ([paths[i] for i in keep], row_ids[keep], views[keep],
            seed, policy, geometric, exclude)


def _dist_init(cfg):
    """Join the process group torchrun started, if there is one.

    Returns `(rank, world)`; `(0, 1)` when run normally, and then NOTHING else
    in this module changes behaviour -- the distributed path has to be exactly
    the single-GPU path when there is one GPU, or every number measured before
    it stops being comparable.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return 0, 1
    import torch.distributed as dist
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local)
    cfg.device = f"cuda:{local}"
    return dist.get_rank(), world


class _GradReducer:
    """One flat buffer, one all-reduce per optimiser step.

    WHY A SUM AND NOT A MEAN. Each micro-chunk already scales its loss by
    `share = len(degs) / n_pairs`, computed against the FULL batch's pair
    count on every rank. The shares of every chunk on every rank therefore sum
    to exactly 1, so summing the ranks' gradients reproduces the gradient the
    single-GPU run would have accumulated -- not approximately, and not up to
    a scale factor. Averaging instead would quietly divide the step by
    `world_size`.

    WHY ONE BUFFER. A 24-block tower is ~400 parameter tensors; reducing them
    one at a time issues ~400 NCCL calls per step, and at tens of microseconds
    of launch latency apiece that is a visible slice of the step this is meant
    to shorten. Two on-device memcpys and one call is cheaper.
    """

    #: Floats per bucket. ONE buffer over every parameter would be simpler and
    #: was the first version, but at d24 that buffer is 302M floats = 1.21 GiB
    #: of card that the run does not have: the tower's own weights, its
    #: gradients and AdamW's two moments already come to ~4.8 GiB before a
    #: single activation, and d24 measured 22.02 GiB of activations at
    #: src_chunk=4 on a 23.5 GiB card. Bucketing caps the extra at 128 MiB and
    #: still issues ~10 NCCL calls per step instead of ~400.
    BUCKET_NUMEL = 32 * 1024 * 1024

    def __init__(self, params, bucket_numel: int | None = None):
        params = [p for p in params]
        if not params:
            raise ValueError("nothing to reduce: no trainable parameters")
        dtypes = {p.dtype for p in params}
        if len(dtypes) != 1:
            raise ValueError(f"mixed grad dtypes {dtypes}; the buffer "
                             "assumes one")
        cap = bucket_numel or self.BUCKET_NUMEL
        self.buckets, cur, n = [], [], 0
        for p in params:
            if cur and n + p.numel() > cap:
                self.buckets.append(cur)
                cur, n = [], 0
            cur.append(p)
            n += p.numel()
        if cur:
            self.buckets.append(cur)
        widest = max(sum(p.numel() for p in b) for b in self.buckets)
        self.buf = torch.zeros(widest, device=params[0].device,
                               dtype=params[0].dtype)

    def reduce(self):
        import torch.distributed as dist
        for bucket in self.buckets:
            off = 0
            for p in bucket:
                n = p.numel()
                # A rank whose chunks never touched a parameter must still put
                # a zero in, or the sum is over a different set of ranks per
                # tensor.
                if p.grad is None:
                    self.buf[off:off + n].zero_()
                else:
                    self.buf[off:off + n].copy_(p.grad.reshape(-1))
                off += n
            dist.all_reduce(self.buf[:off], op=dist.ReduceOp.SUM)
            off = 0
            for p in bucket:
                n = p.numel()
                if p.grad is None:
                    p.grad = torch.empty_like(p)
                p.grad.reshape(-1).copy_(self.buf[off:off + n])
                off += n


def _stratified_subsample(bank, idx, frac, rng):
    """Keep `frac` of `idx`, family by family.

    Stratified on (generator, label) rather than drawn flat, because
    `PairedSampler._draw_stratified` picks a generator family UNIFORMLY and
    then an image inside it. A flat subsample would thin the small families
    and the large ones by the same ratio in absolute terms but would also
    perturb which families survive at all, so the run would differ from its
    full-data twin in composition as well as in size -- two variables, and
    the whole point of the comparison is to move one. Every family keeps at
    least one row for the same reason: a family that vanishes is a mixture
    change wearing a data-size costume.

    `rng` must be a generator of its own, NEVER the sampler's: drawing from
    the sampler's stream here would shift every subsequent batch and the
    subsampled run would stop being comparable to the full one for a reason
    that has nothing to do with data size.
    """
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"train_subsample_frac must be in (0, 1], got {frac}")
    if frac == 1.0:
        return idx
    gens = bank.meta["generator"].to_numpy()[idx]
    labels = bank.meta["label"].to_numpy()[idx]
    keep = []
    for gen, label in sorted(set(zip(gens.tolist(), labels.tolist()))):
        pool = idx[(gens == gen) & (labels == label)]
        n = max(1, int(round(frac * len(pool))))
        keep.append(rng.choice(pool, size=n, replace=False))
    return np.sort(np.concatenate(keep))


def train_finetune(cfg: FinetuneConfig) -> dict:
    """Train one depth rung end to end and write its checkpoint."""
    from concurrent.futures import ThreadPoolExecutor

    rank, world = _dist_init(cfg)
    bank = FeatureBank.open(cfg.bank_dir)
    backbone_name = cfg.backbone or bank.config["backbone"]
    split = bank.meta["split"].to_numpy()
    train_idx = np.where(split == "train")[0]
    if len(train_idx) == 0:
        raise ValueError(f"the bank at {cfg.bank_dir} has no train rows")
    if cfg.train_exclude_generators:
        from aigcdet.train.train_head import _drop_generators
        train_idx = _drop_generators(bank, train_idx, cfg.train_exclude_generators)
    if cfg.train_subsample_frac != 1.0:
        # seed+1, not seed: a separate stream so the sampler's own draws are
        # bit-identical to the full-data run's.
        train_idx = _stratified_subsample(
            bank, train_idx, cfg.train_subsample_frac,
            np.random.default_rng(cfg.seed + 1))

    tower, spec = load_backbone(backbone_name, device=cfg.device)
    tower = tower.to(getattr(torch, cfg.tower_dtype))
    unfrozen = unfreeze_last_n(tower, cfg.depth)
    if cfg.grad_checkpointing and cfg.depth and hasattr(
            tower, "gradient_checkpointing_enable"):
        # Only when something is actually trainable: with a fully frozen tower
        # there is no backward through it to recompute for, so checkpointing
        # would buy nothing and cost the recompute anyway.
        tower.gradient_checkpointing_enable()
    # NEVER `.train()`. See the module docstring: eval mode governs dropout and
    # running statistics, not autograd, and letting those differ between D0 and
    # D1 would put a second difference inside a one-difference ladder.
    tower.eval()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        head = Detector(dim_feat=spec.dim, use_recon=False,
                        use_film=cfg.use_film, hidden=cfg.head_hidden).to(cfg.device)

    tower_params = [p for p in tower.parameters() if p.requires_grad]
    groups = [{"params": list(head.parameters()), "lr": cfg.lr}]
    if tower_params:
        groups.append({"params": tower_params, "lr": cfg.tower_lr})
    opt = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Every rank builds the same head from the same seed and loads the same
    # tower, so the replicas start identical and, given an identical gradient,
    # stay identical without ever broadcasting weights.
    reducer = _GradReducer(list(head.parameters()) + tower_params) if world > 1 else None

    sampler = LiveViewSampler(
        bank, train_idx, root=cfg.root, seed=int(bank.config["seed"]),
        policy=cfg.policy(), geometric=cfg.geometric,
        exclude_families=(), n_src=cfg.n_src, m_deg=cfg.m_deg,
        rng=np.random.default_rng(cfg.seed), device=cfg.device)

    amp = getattr(torch, cfg.amp_dtype)
    param_dtype = next(tower.parameters()).dtype
    history = []
    for epoch in range(cfg.epochs):
        tasks = sampler.batch_tasks()
        # Threads, not processes: the work inside `_prepare_batch` is PIL
        # decode and OpenCV/numpy augmentation, all of which release the GIL,
        # and threads keep the memmapped bank and the already-loaded tower
        # shared instead of paying to re-create them per worker.
        with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as pool:
            pending = [pool.submit(_prepare_batch,
                                   _shard_task(t, rank, world, cfg.src_chunk))
                       for t in tasks]
            for task, fut in zip(tasks, pending):
                prepared = fut.result()
                si, vi = task[7], task[8]
                batch = sampler.targets(si, vi, cfg.device)
                # The FULL batch's pair count on every rank, never this rank's
                # share of it: `share` below has to be each chunk's fraction of
                # the whole step for the summed gradient to equal the
                # single-GPU one.
                n_src_all, n_pairs = len(task[0]), len(si)
                opt.zero_grad(set_to_none=True)
                parts = {"cls": 0.0, "deg": 0.0, "con": 0.0, "total": 0.0}

                # Micro-batches are cut on SOURCE boundaries, never on pair
                # boundaries. A pair's clean and degraded halves must land in
                # the same backward or the consistency term is comparing rows
                # from different graphs, and cutting mid-source would also
                # force the clean image to be forwarded twice.
                for g0, g1, l0, l1 in _owned_chunks(n_src_all, cfg.src_chunk,
                                                    rank, world):
                    srcs = prepared["clean"][l0:l1]
                    # Local indices read this rank's packed pixels; global ones
                    # index the batch-wide targets. Conflating them would pair
                    # an image with another image's label.
                    degs = prepared["deg"][l0 * cfg.m_deg:l1 * cfg.m_deg]
                    p0, p1 = g0 * cfg.m_deg, g1 * cfg.m_deg
                    share = len(degs) / n_pairs
                    with torch.autocast(device_type=cfg.device.split(":")[0],
                                        dtype=amp, enabled=cfg.device != "cpu"):
                        # The clean image is forwarded ONCE per source and its
                        # embedding repeated, exactly as `PairedSampler`
                        # repeats the cached one. Forwarding it `m_deg` times
                        # would give the same number for a third more compute.
                        f_src = _forward_tower(tower, spec, srcs, cfg.device,
                                               param_dtype, cfg.src_chunk)
                        f_clean = f_src.repeat_interleave(cfg.m_deg, dim=0)
                        f_deg = _forward_tower(tower, spec, degs, cfg.device,
                                               param_dtype, cfg.src_chunk)
                    micro = {k: (v[p0:p1] if torch.is_tensor(v) else v)
                             for k, v in batch.items()}
                    micro["f_clean"] = f_clean.float()
                    micro["f_deg"] = f_deg.float()
                    loss, part = _step_loss(head, micro, cfg)
                    # Weighted by this micro-batch's share of the pairs, which
                    # is what makes the accumulated gradient equal the one the
                    # full batch would have produced. Every loss below is a
                    # mean, so the weighting is exact rather than approximate.
                    (loss * share).backward()
                    for k in parts:
                        parts[k] += part[k] * share
                if reducer is not None:
                    reducer.reduce()
                opt.step()
        if world > 1:
            # Each rank only accumulated its own chunks' share, so the loss
            # printed by a rank alone would be a quarter of the batch's.
            import torch.distributed as dist
            t = torch.tensor([parts[k] for k in ("cls", "deg", "con", "total")],
                             device=cfg.device, dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            parts = dict(zip(("cls", "deg", "con", "total"), t.tolist()))
        history.append({"epoch": epoch, **parts})

    out_dir = os.path.join(cfg.out_dir, cfg.name)
    ckpt = os.path.join(out_dir, "checkpoint.pt")
    if world > 1 and rank != 0:
        # The replicas are identical, so a second writer could only race the
        # first to the same bytes.
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
        return {"checkpoint": ckpt, "unfrozen": unfrozen, "history": history,
                "rank": rank}
    os.makedirs(out_dir, exist_ok=True)
    # The TOWER is saved too, and it has to be: at depth > 0 the head alone no
    # longer describes the model, and the eval bank this rung is scored on can
    # only be extracted by the tower that produced its training features.
    torch.save({"state_dict": head.state_dict(),
                "tower_state_dict": tower.state_dict(),
                "config": asdict(cfg), "dim_feat": spec.dim,
                "backbone": backbone_name, "unfrozen": unfrozen}, ckpt)
    result = {"unfrozen": unfrozen, "history": history, "world_size": world}
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
    return {"checkpoint": ckpt, **result}
