"""Two towers into one head (`finetune_dual`), through the parts it shares
with the single-tower trainer and the parts it restates.

The single-tower file pins the pixels, the batch sequence and the loss; this
one pins what is specific to two towers: both towers are saved in order, and
the distillation teacher exists once PER TOWER and is concatenated in that
same order, so the head's first half is always tower 1.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from aigcdet.augment.canonical import MODE_BAND
from aigcdet.train.finetune import FinetuneConfig
from aigcdet.train.finetune_dual import DualFinetuneConfig, train_dual

from tests.train.test_finetune import _StubTower, _corpus


def _patch_two_towers(monkeypatch, dim=8, n=4):
    """`load_backbone` hands back a FRESH stub per call, as it would per
    tower; the single-tower helper returns one object, which two towers
    would then share."""
    from aigcdet.features.backbones import BackboneSpec
    from aigcdet.train import finetune, finetune_dual

    spec = BackboneSpec("fake", "none", image_size=64, dim=dim,
                        num_prefix_tokens=1, params=0)
    made = []

    def load(name, device):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234 + len(made))
            t = _StubTower(dim, n)
        made.append(t)
        return t, spec

    monkeypatch.setattr(finetune_dual, "load_backbone", load)
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
    return made, spec


def _cfg(tmp_path, bank, **kw):
    base = FinetuneConfig(
        name="dual", bank_dir=bank, root=str(tmp_path), depth=1,
        backbone="fake", out_dir=str(tmp_path / "out"), device="cpu",
        epochs=1, n_src=4, m_deg=2, workers=2, policy_mode=MODE_BAND,
        amp_dtype="float32", **kw)
    return DualFinetuneConfig(base=base)


def test_a_dual_run_saves_both_towers_in_order_and_a_doubled_head(tmp_path, monkeypatch):
    made, _ = _patch_two_towers(monkeypatch)
    bank = _corpus(tmp_path, monkeypatch)
    res = train_dual(_cfg(tmp_path, bank))
    ck = torch.load(res["checkpoint"], map_location="cpu", weights_only=False)
    assert ck["n_towers"] == 2 and ck["dim_feat"] == 16
    assert len(made) == 2
    for t, sd in zip(made, ck["tower_state_dicts"]):
        for k, v in t.state_dict().items():
            assert torch.equal(v, sd[k])
    assert ck["state_dict"]["classifier.trunk.0.weight"].shape[1] == 16
    assert ck["ema_tower_state_dicts"] is None


def test_a_finetune_config_is_refused_by_the_dual_trainer(tmp_path):
    with pytest.raises(TypeError, match="ONE-tower"):
        train_dual(FinetuneConfig(name="x", bank_dir="b", root="r"))


def test_self_distillation_runs_over_the_concatenated_feature(tmp_path, monkeypatch):
    _patch_two_towers(monkeypatch)
    bank = _corpus(tmp_path, monkeypatch)
    res = train_dual(_cfg(tmp_path, bank, distil="self"))
    h = res["history"][-1]
    assert np.isfinite(h["total"]) and 0.0 < h["dis"] < 2.0
    ck = torch.load(res["checkpoint"], map_location="cpu", weights_only=False)
    assert ck["config"]["distil"] == "self"


def test_ema_distillation_keeps_one_teacher_per_tower_in_tower_order(tmp_path, monkeypatch):
    """Two teachers, saved in the towers' order, each having moved only where
    its own tower did; the head's average saved once."""
    made, _ = _patch_two_towers(monkeypatch)
    bank = _corpus(tmp_path, monkeypatch)
    starts = None

    from aigcdet.train import finetune_dual
    real = finetune_dual.unfreeze_last_n

    def spy(t, depth):
        nonlocal starts
        starts = (starts or []) + [{k: v.clone() for k, v in t.state_dict().items()}]
        return real(t, depth)

    monkeypatch.setattr(finetune_dual, "unfreeze_last_n", spy)
    res = train_dual(_cfg(tmp_path, bank, distil="ema", distil_ema_decay=0.5))
    assert 0.0 < res["history"][-1]["dis"] < 2.0
    ck = torch.load(res["checkpoint"], map_location="cpu", weights_only=False)
    emas = ck["ema_tower_state_dicts"]
    assert len(emas) == 2 and len(starts) == 2
    for i, (ema, start, final) in enumerate(zip(emas, starts, ck["tower_state_dicts"])):
        for k in ema:
            if k.startswith("encoder.layer.3") or k.startswith("layernorm"):
                assert not torch.equal(ema[k], start[k]), f"tower {i} {k}: teacher never moved"
                assert not torch.equal(ema[k], final[k]), f"tower {i} {k}: teacher is the student"
            else:
                assert torch.equal(ema[k], start[k]), f"tower {i} {k}: frozen weight moved"
    # The two towers start from different seeds here, so their teachers differ.
    assert not torch.equal(emas[0]["encoder.layer.3.weight"], emas[1]["encoder.layer.3.weight"])
    assert set(ck["ema_state_dict"]) == set(ck["state_dict"])


def test_an_ema_dual_run_resumes_with_its_teachers(tmp_path, monkeypatch):
    _patch_two_towers(monkeypatch)
    bank = _corpus(tmp_path, monkeypatch)
    cfg = _cfg(tmp_path, bank, distil="ema", distil_ema_decay=0.5)
    train_dual(cfg)
    ck = torch.load(str(tmp_path / "out" / "dual" / "checkpoint.pt"),
                    map_location="cpu", weights_only=False)
    ema_before = ck["ema_tower_state_dicts"]
    cfg2 = _cfg(tmp_path, bank, distil="ema", distil_ema_decay=0.5, resume=True)
    cfg2.base.epochs = 2
    _patch_two_towers(monkeypatch)             # fresh towers, as a new process
    res = train_dual(cfg2)
    assert [h["epoch"] for h in res["history"]] == [0, 1]
    ck2 = torch.load(res["checkpoint"], map_location="cpu", weights_only=False)
    for a, b in zip(ema_before, ck2["ema_tower_state_dicts"]):
        assert not torch.equal(a["encoder.layer.3.weight"], b["encoder.layer.3.weight"])
    # And a checkpoint stripped of its teachers is refused.
    ck2["ema_tower_state_dicts"] = None
    torch.save(ck2, res["checkpoint"])
    cfg3 = _cfg(tmp_path, bank, distil="ema", distil_ema_decay=0.5, resume=True)
    cfg3.base.epochs = 3
    _patch_two_towers(monkeypatch)
    with pytest.raises(ValueError, match="carries none"):
        train_dual(cfg3)
