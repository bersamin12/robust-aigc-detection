"""The unfreeze depth ladder (D0-D4).

This file is mostly about ONE property, because the ladder's meaning rests on
it: at depth 0 the live trainer must reproduce the cached rung exactly. Two
things have to hold for that, and each fails silently on its own --

  * the PIXELS. `build_view(decoded, v, row_id, seed, ...)` must return
    byte-for-byte what `extract_bank` wrote into column `v`. If it drifts, D0
    trains on slightly different images than the cached rung did, every depth
    inherits the difference, and the ladder measures pixel drift plus
    unfreezing while reporting unfreezing.
  * the BATCH SEQUENCE. `LiveViewSampler` must draw the same rows and views, in
    the same order, from the same seed. One extra `rng` call anywhere and D0
    diverges from its own control for a reason nothing in the output names.

Neither has a shape, a dtype or a row count that differs when it breaks, which
is why they are asserted rather than reviewed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from aigcdet.augment.canonical import MODE_BAND, MODE_CROP, CanonPolicy
from aigcdet.data.manifest import make_dummy_manifest
from aigcdet.features.bank import FeatureBank
from aigcdet.train.finetune import (
    FinetuneConfig, LiveViewSampler, tower_blocks, unfreeze_last_n,
)


# ===========================================================================
# Finding and unfreezing the tower's blocks
# ===========================================================================

class _FakeViT(torch.nn.Module):
    """An HF-shaped encoder: `encoder.layer` plus a trailing `layernorm`."""

    def __init__(self, n=6, d=8):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.layer = torch.nn.ModuleList(
            [torch.nn.Linear(d, d) for _ in range(n)])
        self.layernorm = torch.nn.LayerNorm(d)
        self.embeddings = torch.nn.Linear(d, d)
        for p in self.parameters():
            p.requires_grad_(False)


class _TimmViT(torch.nn.Module):
    def __init__(self, n=4, d=8):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(d, d) for _ in range(n)])
        self.norm = torch.nn.LayerNorm(d)
        for p in self.parameters():
            p.requires_grad_(False)


class _Headless(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.thing = torch.nn.Linear(4, 4)


@pytest.mark.parametrize("model, path, n", [
    (_FakeViT(), "encoder.layer", 6),
    (_TimmViT(), "blocks", 4),
])
def test_the_block_list_is_found_on_both_naming_conventions(model, path, n):
    got_path, blocks = tower_blocks(model)
    assert got_path == path and len(blocks) == n


def test_a_tower_with_no_block_list_raises_rather_than_unfreezing_nothing():
    """The failure this refuses is the one that produces a complete table:
    unfreeze nothing at every depth, D1..D4 all equal D0, and the run reports
    'unfreezing does not help'."""
    with pytest.raises(ValueError, match="cannot find a block list"):
        tower_blocks(_Headless())


def test_depth_zero_leaves_the_tower_exactly_frozen():
    model = _FakeViT()
    rec = unfreeze_last_n(model, 0)
    assert rec["trainable_params"] == 0
    assert not any(p.requires_grad for p in model.parameters())


def test_each_depth_unfreezes_strictly_more_than_the_last():
    """The one thing that distinguishes the rungs of this ladder. If two
    depths had the same trainable set they would be the same experiment run
    twice under different names."""
    counts = [unfreeze_last_n(_FakeViT(), d)["trainable_params"] for d in range(5)]
    assert counts == sorted(counts)
    assert len(set(counts)) == len(counts), counts


def test_the_final_norm_rides_with_the_last_block():
    """It sits after the last block and before the pooling, so freezing it
    while unfreezing the block beneath asks that block to move its output
    distribution underneath a normalisation calibrated for the old one."""
    model = _FakeViT()
    unfreeze_last_n(model, 1)
    assert all(p.requires_grad for p in model.layernorm.parameters())
    assert not any(p.requires_grad for p in model.encoder.layer[0].parameters())


def test_the_frozen_blocks_stay_frozen():
    model = _FakeViT(n=6)
    unfreeze_last_n(model, 2)
    for i in range(4):
        assert not any(p.requires_grad for p in model.encoder.layer[i].parameters())
    for i in (4, 5):
        assert all(p.requires_grad for p in model.encoder.layer[i].parameters())


def test_asking_for_more_blocks_than_exist_is_refused():
    """Clamping would make two rungs of the ladder the same rung."""
    with pytest.raises(ValueError, match="only 6 blocks"):
        unfreeze_last_n(_FakeViT(n=6), 7)


def test_unfreezing_is_idempotent_across_depths_on_one_tower():
    """`unfreeze_last_n` re-freezes everything first, so a tower reused across
    depths does not accumulate trainable blocks from earlier calls -- which
    would make D1-after-D4 secretly D4."""
    model = _FakeViT(n=6)
    deep = unfreeze_last_n(model, 4)["trainable_params"]
    shallow = unfreeze_last_n(model, 1)["trainable_params"]
    assert shallow < deep


# ===========================================================================
# The pixels: build_view against a real extraction
# ===========================================================================

def _eval_conditions():
    """A two-condition grid: enough for an eval bank, cheap enough for a unit
    test. The full 20-condition grid would measure nothing extra here -- what
    is under test is the tower's PROVENANCE, not the condition axis."""
    from aigcdet.augment.recipes import Op, Recipe
    return {"clean": Recipe(()), "jpeg_q50": Recipe((Op("jpeg", {"quality": 50}),))}


def _tree(tmp_path, n=4):
    return make_dummy_manifest(n, str(tmp_path / "img"), np.random.default_rng(0))


def _stub_backbone(monkeypatch, module, dim=8):
    from aigcdet.features.backbones import BackboneSpec
    spec = BackboneSpec("fake", "none", image_size=64, dim=dim,
                        num_prefix_tokens=1, params=0)
    monkeypatch.setattr(module, "load_backbone", lambda n, device: (None, spec))
    # A FINGERPRINT of the exact pixels, not a constant: the mean of each of
    # `dim` horizontal strips. A stub returning zeros would let the
    # reproduction test below pass on any pixels at all, and a plain sum
    # overflows the bank's float16 storage (its own guard refuses the write).
    # Strip means stay in 0..255, so they survive the cast, and eight of them
    # move independently when a crop window does.
    monkeypatch.setattr(module, "embed", lambda model, spec, imgs, device="cpu",
                        batch_size=16: np.stack(
                            [_fingerprint(im, spec.dim) for im in imgs]))
    return spec


def _fingerprint(img, dim):
    a = np.asarray(img, dtype=np.float64)
    return np.array([a[s::dim].mean() for s in range(dim)], dtype=np.float32)


@pytest.mark.parametrize("mode, kw", [(MODE_BAND, {}), (MODE_CROP, {"crop_side": 64})])
def test_build_view_reproduces_every_cached_view(tmp_path, monkeypatch, mode, kw):
    """The foundation of the D0 control, on both canonicalisation policies.

    Crop is the load-bearing case: each view draws its OWN window from
    `canonical_rng(seed, row_id, view)`, so a `build_view` that forgot the key
    would return a different crop of the same picture -- same shape, same
    dtype, different pixels, and no error anywhere.
    """
    from aigcdet.features import extract
    from aigcdet.features.extract import build_view

    df = _tree(tmp_path)
    _stub_backbone(monkeypatch, extract)
    policy = CanonPolicy(mode=mode, **kw)
    out = extract.extract_bank(df, "fake", str(tmp_path / "bank"), seed=7,
                               device="cpu", policy=policy)
    bank = FeatureBank.open(out)

    from PIL import Image
    for pos, (row_id, row) in enumerate(df.iterrows()):
        with Image.open(row["path"]) as im:
            decoded = np.asarray(im.convert("RGB"), dtype=np.uint8)
        for v in range(bank.config["n_views"]):
            pixels, _ = build_view(decoded, v, int(row_id), 7, policy=policy)
            want = np.asarray(bank.feats[pos, v], dtype=np.float32)
            got = _fingerprint(pixels, len(want)).astype(np.float16).astype(np.float32)
            np.testing.assert_array_equal(got, want, err_msg=(
                f"row {row_id} view {v}: build_view produced different pixels "
                f"than extract_bank cached"))


def test_build_view_ignores_a_shared_base_under_crop(tmp_path):
    """A caller passing the band optimisation's hoisted `base` under a crop
    policy would hand all 11 views the same window. Crop ignores it, so the
    mistake cannot be made."""
    from aigcdet.features.extract import build_view

    rng = np.random.default_rng(0)
    decoded = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
    policy = CanonPolicy(mode=MODE_CROP, crop_side=64)
    wrong_base = np.zeros((64, 64, 3), dtype=np.uint8)
    a, _ = build_view(decoded, 3, 11, 7, policy=policy)
    b, _ = build_view(decoded, 3, 11, 7, policy=policy, base=wrong_base)
    np.testing.assert_array_equal(a, b)
    assert a.any(), "the crop window came from the ignored base after all"


# ===========================================================================
# The batch sequence
# ===========================================================================

def _bank_for_sampler(tmp_path, monkeypatch, n=12):
    from aigcdet.features import extract
    df = _tree(tmp_path, n)
    df["split"] = "train"
    _stub_backbone(monkeypatch, extract)
    out = extract.extract_bank(df, "fake", str(tmp_path / "b"), seed=7,
                               device="cpu", policy=CanonPolicy(mode=MODE_BAND))
    return FeatureBank.open(out), df


def test_the_live_sampler_draws_the_same_batches_as_the_cached_one(tmp_path, monkeypatch):
    """If these diverge, D0 differs from the rung it is the control for and
    nothing in the output says why."""
    from aigcdet.models.sampler import PairedSampler

    bank, _ = _bank_for_sampler(tmp_path, monkeypatch)
    idx = np.arange(len(bank.meta))
    kw = dict(n_src=4, m_deg=2, device="cpu")

    cached = PairedSampler(bank, idx, rng=np.random.default_rng(3), **kw)
    live = LiveViewSampler(bank, idx, root=str(tmp_path), seed=7,
                           rng=np.random.default_rng(3), **kw)
    for _ in range(len(cached)):
        a = cached.draw_batch()
        b = live.draw_batch()
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)


def test_the_live_sampler_covers_an_epoch_of_the_same_length(tmp_path, monkeypatch):
    from aigcdet.models.sampler import PairedSampler

    bank, _ = _bank_for_sampler(tmp_path, monkeypatch)
    idx = np.arange(len(bank.meta))
    kw = dict(n_src=4, m_deg=2, device="cpu")
    a = PairedSampler(bank, idx, rng=np.random.default_rng(3), **kw)
    b = LiveViewSampler(bank, idx, root=str(tmp_path), seed=7,
                        rng=np.random.default_rng(3), **kw)
    assert len(a) == len(b) == len(b.batch_tasks())


def test_degradation_targets_come_from_the_bank_not_recomputed(tmp_path, monkeypatch):
    """They are a property of the recipe, which is already on disk. Deriving
    them again here would be a second implementation of something whose only
    job is to agree with the first."""
    bank, _ = _bank_for_sampler(tmp_path, monkeypatch)
    idx = np.arange(len(bank.meta))
    live = LiveViewSampler(bank, idx, root=str(tmp_path), seed=7, n_src=4,
                           m_deg=2, rng=np.random.default_rng(3), device="cpu")
    _, si, vi = live.draw_batch()
    t = live.targets(si, vi, "cpu")
    np.testing.assert_allclose(
        t["presence_deg"].numpy(), np.asarray(bank.presence[si, vi], np.float32))
    np.testing.assert_allclose(
        t["severity_deg"].numpy(), np.asarray(bank.severity[si, vi], np.float32))
    assert (t["y_clean"].numpy()
            == bank.meta["label"].to_numpy()[si].astype(np.float32)).all()


def test_the_live_sampler_never_draws_the_clean_view_as_a_degraded_partner(
        tmp_path, monkeypatch):
    """Inherited from `PairedSampler`, and worth pinning here too: if view 0
    were eligible the consistency loss would compare the clean view against
    itself, both terms would go to zero, and the mechanism would look healthy
    while contributing nothing."""
    bank, _ = _bank_for_sampler(tmp_path, monkeypatch)
    idx = np.arange(len(bank.meta))
    live = LiveViewSampler(bank, idx, root=str(tmp_path), seed=7, n_src=4,
                           m_deg=2, rng=np.random.default_rng(3), device="cpu")
    for task in live.batch_tasks():
        assert (task[2] >= 1).all()


# ===========================================================================
# The config
# ===========================================================================

def test_the_tower_learning_rate_is_far_below_the_heads():
    """The head is random at step 0 and the tower is not. One pretrained-scale
    step at the head's rate would undo more than the ladder is measuring."""
    cfg = FinetuneConfig(name="d1", bank_dir="b", root="r", depth=1)
    assert cfg.tower_lr < cfg.lr / 50


def test_a_finetune_config_is_not_a_rung_config():
    """They share most fields and are different experiments with different
    costs. Sharing the class would let a finetune rung reach `train_rung`,
    which ignores `depth` and would train a frozen head under D4's name."""
    from aigcdet.train.train_head import RungConfig
    assert not issubclass(FinetuneConfig, RungConfig)
    assert "depth" not in RungConfig.__dataclass_fields__


# ===========================================================================
# End to end
# ===========================================================================

class _StubTower(torch.nn.Module):
    """A differentiable stand-in with a real `encoder.layer` block list.

    `unfreeze_last_n` has to find blocks and `train_finetune` has to push
    gradient through them, so the stub cannot be a bare function: it needs
    parameters that live in the right place for the depth ladder to select.
    """

    def __init__(self, dim=8, n=4):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.layer = torch.nn.ModuleList(
            [torch.nn.Linear(dim, dim) for _ in range(n)])
        self.layernorm = torch.nn.LayerNorm(dim)
        self.dim = dim
        for p in self.parameters():
            p.requires_grad_(False)


def _patch_tower(monkeypatch, dim=8, n=4):
    from aigcdet.features.backbones import BackboneSpec
    from aigcdet.train import finetune

    spec = BackboneSpec("fake", "none", image_size=64, dim=dim,
                        num_prefix_tokens=1, params=0)
    # Seeded. A real tower arrives pretrained and is therefore identical run to
    # run; this stub is built from `nn.Linear`'s default init, which reads the
    # global RNG, so without this the determinism test below would compare two
    # different towers and fail for a reason that cannot occur in production.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        tower = _StubTower(dim, n)
    monkeypatch.setattr(finetune, "load_backbone", lambda name, device: (tower, spec))
    monkeypatch.setattr(finetune, "model_inputs",
                        lambda spec, imgs, device, dtype: {"pixel_values": torch.stack(
                            [torch.as_tensor(np.asarray(im, np.float32).reshape(-1)[:spec.dim] / 255.0)
                             for im in imgs])})

    def _pool(model, spec, inputs):
        h = inputs["pixel_values"]
        for blk in model.encoder.layer:
            h = torch.tanh(blk(h))
        return model.layernorm(h)

    monkeypatch.setattr(finetune, "_pool", _pool)
    return tower, spec


def _corpus(tmp_path, monkeypatch, n=16):
    from aigcdet.features import extract
    df = make_dummy_manifest(n, str(tmp_path / "img"), np.random.default_rng(0))
    df["split"] = "train"
    _stub_backbone(monkeypatch, extract, dim=8)
    out = extract.extract_bank(df, "fake", str(tmp_path / "b"), seed=7,
                               device="cpu", policy=CanonPolicy(mode=MODE_BAND))
    return out


def _cfg(tmp_path, bank, depth, **kw):
    return FinetuneConfig(
        name=f"d{depth}", bank_dir=bank, root=str(tmp_path), depth=depth,
        out_dir=str(tmp_path / "out"), device="cpu", epochs=1, n_src=4,
        m_deg=2, workers=2, policy_mode=MODE_BAND, amp_dtype="float32", **kw)


def test_depth_zero_leaves_every_tower_weight_untouched(tmp_path, monkeypatch):
    """D0's whole job is to be the frozen tower. If a single tower weight
    moved, D0 would not be the control it is presented as -- and the ladder's
    baseline would drift by an amount nothing measures."""
    from aigcdet.train.finetune import train_finetune

    tower, _ = _patch_tower(monkeypatch)
    bank = _corpus(tmp_path, monkeypatch)
    before = {k: v.clone() for k, v in tower.state_dict().items()}
    res = train_finetune(_cfg(tmp_path, bank, depth=0))
    for k, v in tower.state_dict().items():
        assert torch.equal(v, before[k]), f"depth 0 moved tower weight {k}"
    assert res["unfrozen"]["trainable_params"] == 0


def test_depth_one_moves_the_last_block_and_nothing_below_it(tmp_path, monkeypatch):
    """The other half: if the tower did NOT move, every depth would be D0 and
    the ladder would report 'unfreezing does not help' from a table that looks
    entirely normal."""
    from aigcdet.train.finetune import train_finetune

    tower, _ = _patch_tower(monkeypatch, n=4)
    bank = _corpus(tmp_path, monkeypatch)
    before = {k: v.clone() for k, v in tower.state_dict().items()}
    train_finetune(_cfg(tmp_path, bank, depth=1))

    moved = {k for k, v in tower.state_dict().items() if not torch.equal(v, before[k])}
    assert any(k.startswith("encoder.layer.3") for k in moved), moved
    assert not any(k.startswith(f"encoder.layer.{i}") for i in (0, 1, 2)
                   for k in moved), moved


def test_the_head_trains_at_every_depth(tmp_path, monkeypatch):
    from aigcdet.train.finetune import train_finetune

    _patch_tower(monkeypatch)
    bank = _corpus(tmp_path, monkeypatch)
    for depth in (0, 2):
        res = train_finetune(_cfg(tmp_path, bank, depth=depth))
        assert res["history"], "no history recorded"
        assert np.isfinite(res["history"][-1]["total"])


def test_the_checkpoint_carries_the_tower_as_well_as_the_head(tmp_path, monkeypatch):
    """At depth > 0 the head alone no longer describes the model, and the eval
    bank this rung is scored against can only be extracted by the tower that
    produced its training features."""
    from aigcdet.train.finetune import train_finetune

    _patch_tower(monkeypatch)
    bank = _corpus(tmp_path, monkeypatch)
    res = train_finetune(_cfg(tmp_path, bank, depth=1))
    ck = torch.load(res["checkpoint"], map_location="cpu", weights_only=False)
    assert "tower_state_dict" in ck and ck["tower_state_dict"]
    assert ck["unfrozen"]["depth"] == 1
    assert ck["config"]["tower_lr"] < ck["config"]["lr"]


def test_two_runs_of_one_depth_agree(tmp_path, monkeypatch):
    """Determinism, which the D0 control needs as much as the pixels do."""
    from aigcdet.train.finetune import train_finetune

    outs = []
    for run in range(2):
        tower, _ = _patch_tower(monkeypatch)
        bank = _corpus(tmp_path / f"r{run}", monkeypatch)
        cfg = _cfg(tmp_path / f"r{run}", bank, depth=1)
        outs.append(train_finetune(cfg)["history"][-1]["total"])
    assert outs[0] == outs[1]


# ===========================================================================
# Gradient accumulation
# ===========================================================================

def test_accumulating_micro_batches_equals_one_full_batch(tmp_path, monkeypatch):
    """The claim that justifies `src_chunk` at all.

    A 64-source batch's activations OOM a 24 GiB card at dinov2regl's 1374
    tokens, so the batch is cut into micro-batches and their gradients summed.
    The easy alternative was to shrink `n_src`, and it would have been wrong:
    batch size changes the gradient, and this ladder has to stay comparable
    with cached rungs trained at n_src=64. Accumulation only changes peak
    memory -- but only if the share-weighting is right, which is what this
    pins. Get the weight wrong and every depth trains on a subtly rescaled
    gradient, which no test of shapes or finiteness would notice.
    """
    from aigcdet.train.finetune import train_finetune

    grads = []
    for chunk in (4, 1):
        tower, _ = _patch_tower(monkeypatch)
        bank = _corpus(tmp_path / f"c{chunk}", monkeypatch)
        cfg = _cfg(tmp_path / f"c{chunk}", bank, depth=1)
        cfg.src_chunk = chunk
        cfg.epochs = 1
        train_finetune(cfg)
        grads.append(torch.cat([p.detach().reshape(-1)
                                for p in tower.encoder.layer[-1].parameters()]))
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-4, atol=1e-6)


def test_a_micro_batch_never_splits_a_clean_degraded_pair(tmp_path, monkeypatch):
    """Cuts land on SOURCE boundaries. A pair split across two backwards would
    put the consistency term's two halves in different graphs -- it would still
    run, and it would silently stop constraining anything."""
    from aigcdet.train import finetune

    seen = []
    real = finetune._step_loss

    def spy(head, batch, cfg):
        seen.append((batch["f_clean"].shape[0], batch["f_deg"].shape[0]))
        return real(head, batch, cfg)

    monkeypatch.setattr(finetune, "_step_loss", spy)
    _patch_tower(monkeypatch)
    bank = _corpus(tmp_path, monkeypatch)
    cfg = _cfg(tmp_path, bank, depth=1)
    cfg.src_chunk = 3
    finetune.train_finetune(cfg)

    assert seen, "no micro-batch ran"
    for n_clean, n_deg in seen:
        assert n_clean == n_deg, (n_clean, n_deg)
        assert n_deg % cfg.m_deg == 0, (
            f"a micro-batch held {n_deg} rows, which is not a whole number of "
            f"{cfg.m_deg}-view sources -- a pair was split")


def test_gradient_checkpointing_is_off_when_nothing_is_trainable(tmp_path, monkeypatch):
    """With a frozen tower there is no backward through it to recompute for, so
    checkpointing would buy nothing and cost the recompute anyway.

    The mechanism asserted here is the module's own `_enable_eval_checkpointing`,
    not HF's `gradient_checkpointing_enable` flag: that flag is consulted as
    `self.gradient_checkpointing and self.training`, and this loop keeps the
    tower in eval mode on purpose, so the flag-based route silently never
    recomputed anything (d24 measured 22 GiB of activations with it "on")."""
    from aigcdet.train import finetune

    _patch_tower(monkeypatch)
    calls = []
    real = finetune._enable_eval_checkpointing
    monkeypatch.setattr(finetune, "_enable_eval_checkpointing",
                        lambda t: calls.append(real(t)))
    bank = _corpus(tmp_path, monkeypatch)
    finetune.train_finetune(_cfg(tmp_path, bank, depth=0))
    assert calls == []
    finetune.train_finetune(_cfg(tmp_path, bank, depth=1))
    assert len(calls) == 1 and calls[0] >= 1  # every block wrapped, once


# ===========================================================================
# The eval bank a finetuned tower produces
# ===========================================================================

def test_a_finetuned_bank_records_the_tower_that_built_it(tmp_path, monkeypatch):
    """The weights become part of the bank's identity, and must be on disk.

    A tower whose weights moved does not produce the features in the frozen
    bank, so every depth has its own -- and two such banks are otherwise
    indistinguishable: same shape, same dtype, same manifest, same conditions.
    `tower_sha256` is what tells them apart, and because `BankWriter` treats
    every unrecognised config key as must-match, recording it is also what
    stops two depths' shards merging into one bank half-computed by each of two
    different models.
    """
    from aigcdet.eval import grid
    from aigcdet.features.bank import FeatureBank
    from aigcdet.train.finetune import train_finetune

    tower, spec = _patch_tower(monkeypatch)
    bank = _corpus(tmp_path, monkeypatch)
    res = train_finetune(_cfg(tmp_path, bank, depth=1))

    _stub_backbone(monkeypatch, grid, dim=8)
    monkeypatch.setattr(grid, "load_backbone", lambda n, device: (tower, spec))
    df = make_dummy_manifest(4, str(tmp_path / "e"), np.random.default_rng(1))
    out = grid.extract_eval_bank(
        df, "fake", str(tmp_path / "eb"), conditions=_eval_conditions(),
        device="cpu", tower_checkpoint=res["checkpoint"])
    cfg = FeatureBank.open(out).config
    assert cfg["unfreeze_depth"] == 1
    assert len(cfg["tower_sha256"]) == 64
    assert cfg["tower_checkpoint"] == "checkpoint.pt"


def test_a_head_only_checkpoint_cannot_re_extract_a_bank(tmp_path, monkeypatch):
    """At depth 0 the tower is unchanged and the frozen bank already holds its
    features; at depth > 0 the head alone does not describe the model. Either
    way a checkpoint with no tower is the wrong thing to hand this flag."""
    import torch as _t
    from aigcdet.eval import grid

    tower, spec = _patch_tower(monkeypatch)
    _stub_backbone(monkeypatch, grid, dim=8)
    monkeypatch.setattr(grid, "load_backbone", lambda n, device: (tower, spec))
    ck = tmp_path / "head_only.pt"
    _t.save({"state_dict": {}, "config": {}}, ck)
    df = make_dummy_manifest(2, str(tmp_path / "e"), np.random.default_rng(1))
    with pytest.raises(ValueError, match="tower_state_dict"):
        grid.extract_eval_bank(df, "fake", str(tmp_path / "eb"),
                               conditions=_eval_conditions(), device="cpu",
                               tower_checkpoint=str(ck))


def test_the_ladder_refuses_two_depths_sharing_one_tower(tmp_path, monkeypatch):
    """Either a bank was reused across depths or a training run did not move
    the tower. Both make the ladder report that depth does not help, from a
    table that looks entirely normal."""
    import importlib.util
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    spec_ = importlib.util.spec_from_file_location(
        "sul", root / "scripts" / "score_unfreeze_ladder.py")
    sul = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(sul)

    class _B:
        def __init__(self, sha):
            self.config = {"manifest_sha256": "m", "conditions": ["clean"],
                           "canon_policy": {"mode": "crop"}, "backbone": "fake",
                           "n_views": 1, "tower_sha256": sha}
            self.meta = pd.DataFrame({"row_id": [0, 1]})

    sul.assert_ladder_comparable({"d0": _B("a"), "d1": _B("b")})
    with pytest.raises(ValueError, match="SAME tower weights"):
        sul.assert_ladder_comparable({"d0": _B("a"), "d1": _B("a")})


def test_the_ladder_refuses_banks_over_different_rows(tmp_path):
    """Scores are joined positionally, so a depth's row would carry another
    image's label -- and both banks would pass every other check."""
    import importlib.util
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    spec_ = importlib.util.spec_from_file_location(
        "sul2", root / "scripts" / "score_unfreeze_ladder.py")
    sul = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(sul)

    def _bank(sha, rows):
        class _B:
            config = {"manifest_sha256": "m", "conditions": ["clean"],
                      "canon_policy": {"mode": "crop"}, "backbone": "fake",
                      "n_views": 1, "tower_sha256": sha}
            meta = pd.DataFrame({"row_id": rows})
        return _B()

    with pytest.raises(ValueError, match="different ORDER"):
        sul.assert_ladder_comparable({"d0": _bank("a", [0, 1]),
                                      "d1": _bank("b", [1, 0])})


# ===========================================================================
# Token pooling (models/pooling.py) through the trainer
# ===========================================================================

class _StubTokenTower(_StubTower):
    """`_StubTower` that also emits a token sequence: CLS, two registers and
    four patches, each a differently scaled copy of the block output, so
    every readout in `TOKEN_POOLS` sees a different vector."""

    def __init__(self, dim=8, n=4, n_tokens=7):
        super().__init__(dim, n)
        self.register_buffer("scales", torch.linspace(0.5, 1.5, n_tokens))

    def forward(self, pixel_values):
        from types import SimpleNamespace
        h = pixel_values
        for blk in self.encoder.layer:
            h = torch.tanh(blk(h))
        h = self.layernorm(h)
        return SimpleNamespace(
            last_hidden_state=h[:, None, :] * self.scales[None, :, None])


def _patch_token_tower(monkeypatch, dim=8, n=4, prefix=3):
    from aigcdet.features.backbones import BackboneSpec
    from aigcdet.train import finetune

    spec = BackboneSpec("fake", "none", image_size=64, dim=dim,
                        num_prefix_tokens=prefix, params=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        tower = _StubTokenTower(dim, n)
    monkeypatch.setattr(finetune, "load_backbone", lambda name, device: (tower, spec))
    monkeypatch.setattr(finetune, "model_inputs",
                        lambda spec, imgs, device, dtype: {"pixel_values": torch.stack(
                            [torch.as_tensor(np.asarray(im, np.float32).reshape(-1)[:spec.dim] / 255.0)
                             for im in imgs])})
    return tower, spec


@pytest.mark.parametrize("mode,width", [("cls", 8), ("cls_mean", 16),
                                        ("cls_reg_mean", 32), ("attn", 8)])
def test_a_token_pool_trains_end_to_end_and_records_its_width(
        tmp_path, monkeypatch, mode, width):
    """`dim_feat` in the checkpoint is the pooled width, the pooler's
    parameters are saved under the head, and `token_pool` is in the config
    the scorer rebuilds from."""
    from aigcdet.train.finetune import train_finetune

    _patch_token_tower(monkeypatch)
    bank = _corpus(tmp_path, monkeypatch)
    res = train_finetune(_cfg(tmp_path, bank, depth=1, token_pool=mode))
    ck = torch.load(res["checkpoint"], map_location="cpu", weights_only=False)
    assert ck["dim_feat"] == width
    assert ck["config"]["token_pool"] == mode
    head_keys = [k for k in ck["state_dict"] if k.startswith("pooler.")]
    if mode == "attn":
        assert "pooler.query" in head_keys
        assert "pooler.query" in ck["swa_state_dict"] if ck["swa_n"] else True
    else:
        assert not head_keys
    assert ck["state_dict"]["classifier.trunk.0.weight"].shape[1] == width


def test_attention_pooling_is_trained_with_the_head(tmp_path, monkeypatch):
    from aigcdet.train.finetune import train_finetune

    _patch_token_tower(monkeypatch)
    bank = _corpus(tmp_path, monkeypatch)
    res = train_finetune(_cfg(tmp_path, bank, depth=0, token_pool="attn"))
    ck = torch.load(res["checkpoint"], map_location="cpu", weights_only=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(ck["config"]["seed"])
        from aigcdet.models.pooling import build_pooler
        fresh = build_pooler("attn", [type("S", (), {"dim": 8, "num_prefix_tokens": 3})()])
    assert not torch.equal(ck["state_dict"]["pooler.query"], fresh.query.detach())


def test_two_attention_pooling_runs_agree(tmp_path, monkeypatch):
    """The pooler is built under the head's seed, so a config is one model."""
    from aigcdet.train.finetune import train_finetune

    outs = []
    for i in range(2):
        _patch_token_tower(monkeypatch)
        bank = _corpus(tmp_path / str(i), monkeypatch)
        res = train_finetune(_cfg(tmp_path / str(i), bank, depth=1, token_pool="attn"))
        outs.append(torch.load(res["checkpoint"], map_location="cpu",
                               weights_only=False)["state_dict"])
    for k in outs[0]:
        assert torch.equal(outs[0][k], outs[1][k]), k


def test_a_checkpoint_from_before_token_pool_existed_resumes_at_the_mean(
        tmp_path, monkeypatch):
    """Every older run was the mean readout. Refusing to resume it over a key
    it could not have carried would strand all of them."""
    from aigcdet.train.finetune import train_finetune

    _patch_tower(monkeypatch)
    bank = _corpus(tmp_path, monkeypatch)
    res = train_finetune(_cfg(tmp_path, bank, depth=1))
    ck = torch.load(res["checkpoint"], map_location="cpu", weights_only=False)
    del ck["config"]["token_pool"]
    torch.save(ck, res["checkpoint"])
    cfg = _cfg(tmp_path, bank, depth=1, resume=True)
    cfg.epochs = 2
    res2 = train_finetune(cfg)
    assert [h["epoch"] for h in res2["history"]] == [0, 1]


def test_resuming_under_a_different_token_pool_is_refused(tmp_path, monkeypatch):
    from aigcdet.train.finetune import train_finetune

    _patch_token_tower(monkeypatch)
    bank = _corpus(tmp_path, monkeypatch)
    train_finetune(_cfg(tmp_path, bank, depth=1, token_pool="cls"))
    cfg = _cfg(tmp_path, bank, depth=1, resume=True, token_pool="cls_mean")
    cfg.epochs = 2
    with pytest.raises(ValueError, match="token_pool"):
        train_finetune(cfg)


def test_the_default_readout_takes_the_historical_pool_path(monkeypatch):
    """`token_pool="mean"` must not change a single bit of an existing run:
    the head has no pooler and `_forward_tower` goes through `_pool`."""
    from aigcdet.train import finetune
    from aigcdet.models.pooling import build_pooler, tower_poolers

    calls = []
    monkeypatch.setattr(finetune, "_pool",
                        lambda m, s, i: calls.append("pool") or torch.zeros(1, 8))
    monkeypatch.setattr(finetune, "tokens",
                        lambda m, s, i: calls.append("tokens") or torch.zeros(1, 4, 8))
    monkeypatch.setattr(finetune, "model_inputs", lambda *a: {})
    spec = type("S", (), {"dim": 8, "num_prefix_tokens": 1})()
    head = finetune.Detector(dim_feat=8, pooler=build_pooler("mean", [spec]))
    finetune._forward_tower(None, spec, [None], "cpu", torch.float32, 1,
                            pooler=tower_poolers(head, 1)[0])
    assert calls == ["pool"]
    head = finetune.Detector(dim_feat=8, pooler=build_pooler("cls", [spec]))
    finetune._forward_tower(None, spec, [None], "cpu", torch.float32, 1,
                            pooler=tower_poolers(head, 1)[0])
    assert calls == ["pool", "tokens"]


def test_the_live_sampler_refuses_a_pre_op_pool_without_crop_clamp(tmp_path, monkeypatch):
    """ntire2 shrinks images below the crop window; a strict crop policy
    would refuse such a view mid-epoch, hours in. Fail at construction."""
    from aigcdet.augment.canonical import CanonPolicy
    from aigcdet.train.finetune import LiveViewSampler

    bank, _ = _bank_for_sampler(tmp_path, monkeypatch)
    bank.config["recipe_pool"] = "ntire2"
    strict = CanonPolicy(mode="crop", crop_side=200, nominal_side=224)
    with pytest.raises(ValueError, match="crop-clamp"):
        LiveViewSampler(bank, np.arange(len(bank.meta)), root=str(tmp_path), seed=7,
                        policy=strict, n_src=4, m_deg=2, rng=np.random.default_rng(0),
                        device="cpu")
    clamp = CanonPolicy(mode="crop", crop_side=200, nominal_side=224, crop_clamp=True)
    s = LiveViewSampler(bank, np.arange(len(bank.meta)), root=str(tmp_path), seed=7,
                        policy=clamp, n_src=4, m_deg=2, rng=np.random.default_rng(0),
                        device="cpu")
    assert s.pool == "ntire2"
    bank.config["recipe_pool"] = "ntire"
    LiveViewSampler(bank, np.arange(len(bank.meta)), root=str(tmp_path), seed=7,
                    policy=strict, n_src=4, m_deg=2, rng=np.random.default_rng(0),
                    device="cpu")


# ===========================================================================
# Degradation-invariant distillation (component 1): the tower's degraded
# feature pulled to its clean one
# ===========================================================================

def _feature_batch(seed=0, n=4, dim=8):
    from aigcdet.augment.recipes import N_FAMILIES
    g = torch.Generator().manual_seed(seed)
    f_clean = torch.randn(n, dim, generator=g)
    f_deg = torch.randn(n, dim, generator=g)
    return {"f_clean": f_clean, "f_deg": f_deg,
            "y_clean": torch.tensor([1.0, 0.0] * (n // 2)),
            "y_deg": torch.tensor([1.0, 0.0] * (n // 2)),
            "presence_deg": (torch.rand(n, N_FAMILIES, generator=g) > 0.5).float(),
            "severity_deg": torch.rand(n, N_FAMILIES, generator=g)}


def _head(dim=8):
    from aigcdet.models.heads import Detector
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        return Detector(dim_feat=dim)


def test_distillation_off_is_the_a3_loss_with_a_zero_term():
    """`distil="off"` must not change a bit of an existing run's objective."""
    from aigcdet.train.finetune import _a3_loss, _step_loss

    head, batch = _head(), _feature_batch()
    cfg = FinetuneConfig(name="x", bank_dir="b", root="r")
    loss, parts = _step_loss(head, batch, cfg)
    ref, ref_parts = _a3_loss(head, batch, cfg)
    assert torch.equal(loss, ref)
    assert parts["dis"] == 0.0 and parts["total"] == ref_parts["total"]
    assert set(parts) == {"cls", "deg", "con", "dis", "total"}


def test_self_distillation_adds_exactly_the_weighted_cosine_term():
    from aigcdet.models.losses import distillation_loss
    from aigcdet.train.finetune import _a3_loss, _step_loss

    head, batch = _head(), _feature_batch()
    batch["f_teacher"] = batch["f_clean"].detach()
    cfg = FinetuneConfig(name="x", bank_dir="b", root="r", distil="self",
                         distil_weight=0.7)
    loss, parts = _step_loss(head, batch, cfg)
    ref, _ = _a3_loss(head, batch, cfg)
    want = distillation_loss(batch["f_deg"], batch["f_clean"])
    assert torch.isclose(loss - ref, 0.7 * want, atol=1e-6)
    assert parts["dis"] == pytest.approx(want.item(), abs=1e-6)
    assert 0.0 < parts["dis"] < 2.0


def test_the_distillation_gradient_never_reaches_the_clean_forward():
    """The clean feature is the target. If the term also pulled the clean
    feature towards the degraded one, both could meet at a constant and the
    detector above them would be reading noise while the term read zero."""
    from aigcdet.train.finetune import _step_loss

    head, batch = _head(), _feature_batch()
    f_clean = batch["f_clean"].clone().requires_grad_(True)
    f_deg = batch["f_deg"].clone().requires_grad_(True)
    cfg_off = FinetuneConfig(name="x", bank_dir="b", root="r", use_degradation=False,
                             use_consistency=False)
    cfg_on = FinetuneConfig(name="x", bank_dir="b", root="r", use_degradation=False,
                            use_consistency=False, distil="self")
    grads = []
    for cfg in (cfg_off, cfg_on):
        b = {**batch, "f_clean": f_clean, "f_deg": f_deg,
             "f_teacher": f_clean.detach()}
        f_clean.grad = f_deg.grad = None
        _step_loss(head, b, cfg)[0].backward()
        grads.append((f_clean.grad.clone(), f_deg.grad.clone()))
    torch.testing.assert_close(grads[0][0], grads[1][0])      # clean: unchanged
    assert not torch.allclose(grads[0][1], grads[1][1])       # degraded: pulled


def test_distillation_needs_degraded_views():
    from aigcdet.train.finetune import _check_distil
    with pytest.raises(ValueError, match="use_augmented"):
        _check_distil(FinetuneConfig(name="x", bank_dir="b", root="r",
                                     distil="self", use_augmented=False))
    with pytest.raises(ValueError, match="distil_weight"):
        _check_distil(FinetuneConfig(name="x", bank_dir="b", root="r",
                                     distil="self", distil_weight=0.0))
    with pytest.raises(ValueError, match="choose from"):
        _check_distil(FinetuneConfig(name="x", bank_dir="b", root="r",
                                     distil="teacher"))
    _check_distil(FinetuneConfig(name="x", bank_dir="b", root="r"))


def test_self_distillation_trains_end_to_end_and_records_the_term(tmp_path, monkeypatch):
    from aigcdet.train.finetune import train_finetune

    _patch_tower(monkeypatch)
    bank = _corpus(tmp_path, monkeypatch)
    res = train_finetune(_cfg(tmp_path, bank, depth=1, distil="self"))
    h = res["history"][-1]
    assert np.isfinite(h["total"]) and 0.0 < h["dis"] < 2.0
    ck = torch.load(res["checkpoint"], map_location="cpu", weights_only=False)
    assert ck["config"]["distil"] == "self"
    assert ck["ema_tower_state_dict"] is None and ck["ema_state_dict"] is None


def test_the_ema_teacher_moves_a_fraction_of_the_way_per_update():
    from aigcdet.train.finetune import _EmaTeacher

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        tower = _StubTower()
    teacher = _EmaTeacher(tower, decay=0.75)
    head = torch.nn.Linear(8, 1)
    teacher.attach_head(head)
    start = {k: v.clone() for k, v in tower.state_dict().items()}
    with torch.no_grad():
        for p in tower.parameters():
            p.add_(1.0)
    teacher.update(tower, head)
    for k, v in teacher.tower.state_dict().items():
        torch.testing.assert_close(v, start[k] + 0.25)
    assert not any(p.requires_grad for p in teacher.tower.parameters())
    assert not any(p.requires_grad for p in teacher.head.parameters())


def test_the_ema_teacher_refuses_a_student_already_wrapped_for_checkpointing():
    """`deepcopy` keeps functions by reference, so a copy taken after the
    wrappers go on has blocks whose forward runs the STUDENT's layers: the
    teacher would be the student under another name and the term would read
    near zero with nothing in the log to say so."""
    from aigcdet.train.finetune import _EmaTeacher, _enable_eval_checkpointing

    tower = _StubTower()
    _EmaTeacher(tower, decay=0.9)               # fine before
    _enable_eval_checkpointing(tower)
    with pytest.raises(ValueError, match="checkpointing"):
        _EmaTeacher(tower, decay=0.9)


def test_ema_distillation_keeps_a_teacher_between_the_start_and_the_student(
        tmp_path, monkeypatch):
    """The teacher is neither the pretrained tower (it moved) nor the
    student (it lags), it only moved where the student did, and it is in the
    checkpoint beside the final weights."""
    from aigcdet.train.finetune import train_finetune

    tower, _ = _patch_tower(monkeypatch, n=4)
    bank = _corpus(tmp_path, monkeypatch)
    start = {k: v.clone() for k, v in tower.state_dict().items()}
    res = train_finetune(_cfg(tmp_path, bank, depth=1, distil="ema",
                              distil_ema_decay=0.5))
    assert 0.0 < res["history"][-1]["dis"] < 2.0
    ck = torch.load(res["checkpoint"], map_location="cpu", weights_only=False)
    ema, final = ck["ema_tower_state_dict"], ck["tower_state_dict"]
    for k in ema:
        if k.startswith("encoder.layer.3") or k.startswith("layernorm"):
            assert not torch.equal(ema[k], start[k]), f"{k}: teacher never moved"
            assert not torch.equal(ema[k], final[k]), f"{k}: teacher is the student"
        else:
            assert torch.equal(ema[k], start[k]), f"{k}: a frozen weight moved"
    assert set(ck["ema_state_dict"]) == set(ck["state_dict"])


def test_an_ema_run_resumed_after_a_kill_equals_the_uninterrupted_one(
        tmp_path, monkeypatch):
    """The teacher is training state. A resume that restarted it from the
    student would hand the remaining epochs a different target from the one
    the finished epochs saw, and nothing downstream could tell."""
    from aigcdet.train.finetune import train_finetune

    def cfg(sub, epochs, **kw):
        c = _cfg(tmp_path / sub, bank, depth=1, distil="ema",
                 distil_ema_decay=0.5, **kw)
        c.epochs = epochs
        return c

    _patch_tower(monkeypatch)
    bank = _corpus(tmp_path / "a", monkeypatch)
    straight = train_finetune(cfg("a", 2))

    _patch_tower(monkeypatch)
    bank = _corpus(tmp_path / "b", monkeypatch)
    train_finetune(cfg("b", 1))
    # A fresh tower, as a new process would load: the stub above hands the
    # same object back, already wrapped for checkpointing by the first run,
    # and the teacher refuses to copy a wrapped tower (see `_EmaTeacher`).
    _patch_tower(monkeypatch)
    resumed = train_finetune(cfg("b", 2, resume=True))

    a = torch.load(straight["checkpoint"], map_location="cpu", weights_only=False)
    b = torch.load(resumed["checkpoint"], map_location="cpu", weights_only=False)
    for key in ("tower_state_dict", "ema_tower_state_dict", "state_dict", "ema_state_dict"):
        for k in a[key]:
            torch.testing.assert_close(a[key][k], b[key][k], rtol=0, atol=0,
                                       msg=lambda m: f"{key}/{k}: {m}")


def test_resuming_an_ema_run_from_a_checkpoint_without_a_teacher_is_refused(
        tmp_path, monkeypatch):
    from aigcdet.train.finetune import train_finetune

    _patch_tower(monkeypatch)
    bank = _corpus(tmp_path, monkeypatch)
    res = train_finetune(_cfg(tmp_path, bank, depth=1, distil="ema"))
    ck = torch.load(res["checkpoint"], map_location="cpu", weights_only=False)
    ck["ema_tower_state_dict"] = None
    torch.save(ck, res["checkpoint"])
    cfg = _cfg(tmp_path, bank, depth=1, distil="ema", resume=True)
    cfg.epochs = 2
    _patch_tower(monkeypatch)                  # a fresh, unwrapped tower
    with pytest.raises(ValueError, match="carries none"):
        train_finetune(cfg)


def test_resuming_under_a_different_distil_mode_is_refused(tmp_path, monkeypatch):
    from aigcdet.train.finetune import train_finetune

    _patch_tower(monkeypatch)
    bank = _corpus(tmp_path, monkeypatch)
    train_finetune(_cfg(tmp_path, bank, depth=1))
    for kw in ({"distil": "self"}, {"distil": "self", "distil_weight": 2.0}):
        cfg = _cfg(tmp_path, bank, depth=1, resume=True, **kw)
        cfg.epochs = 2
        with pytest.raises(ValueError, match="distil"):
            train_finetune(cfg)


def test_a_checkpoint_from_before_distil_existed_resumes_with_it_off(
        tmp_path, monkeypatch):
    """Every older run trained without the term, which is what the default
    says; refusing to resume over keys it could not carry would strand them."""
    from aigcdet.train.finetune import train_finetune

    _patch_tower(monkeypatch)
    bank = _corpus(tmp_path, monkeypatch)
    res = train_finetune(_cfg(tmp_path, bank, depth=1))
    ck = torch.load(res["checkpoint"], map_location="cpu", weights_only=False)
    for k in ("distil", "distil_weight", "distil_ema_decay"):
        del ck["config"][k]
    for k in ("ema_state_dict", "ema_tower_state_dict"):
        del ck[k]
    torch.save(ck, res["checkpoint"])
    cfg = _cfg(tmp_path, bank, depth=1, resume=True)
    cfg.epochs = 2
    assert [h["epoch"] for h in train_finetune(cfg)["history"]] == [0, 1]
