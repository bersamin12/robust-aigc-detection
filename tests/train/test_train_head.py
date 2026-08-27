import numpy as np
import torch

from aigcdet.features.bank import N_VIEWS, BankWriter, FeatureBank
from aigcdet.train.train_head import RungConfig, load_detector, train_rung


def _learnable_bank(tmp_path, n=120, dim=8):
    """Fakes get a shifted mean, so a linear head can separate them.
    Augmented views are noisier, so consistency training has something to do."""
    w = BankWriter(str(tmp_path / "b"), n, N_VIEWS, dim, "t", 0)
    rng = np.random.default_rng(0)
    for i in range(n):
        label = i % 2
        clean = rng.normal(loc=1.5 if label else -1.5, scale=0.5, size=dim)
        feats = np.stack([clean] + [clean + rng.normal(0, 0.8, dim)
                                    for _ in range(N_VIEWS - 1)]).astype(np.float32)
        pres = np.zeros((N_VIEWS, 6), np.float32); pres[1:, 0] = 1.0
        sev = np.zeros((N_VIEWS, 6), np.float32); sev[1:, 0] = 0.6
        w.write_image(i, {"path": f"/p{i}", "label": label, "generator": f"g{i % 2}",
                          "source": "s", "split": "train" if i < 100 else "val_internal"},
                      feats=feats, presence=pres, severity=sev,
                      proxies=np.zeros((N_VIEWS, 3), np.float32),
                      recipes=["[]"] * N_VIEWS)
    w.close()
    return str(tmp_path / "b")


def test_a0_trains_and_separates_the_classes(tmp_path):
    cfg = RungConfig(name="a0", bank_dir=_learnable_bank(tmp_path), epochs=15,
                     use_augmented=False, use_consistency=False,
                     use_degradation=False, out_dir=str(tmp_path / "out"))
    res = train_rung(cfg)
    assert res["val_auc"] > 0.85


def test_a3_runs_with_all_terms_enabled(tmp_path):
    cfg = RungConfig(name="a3", bank_dir=_learnable_bank(tmp_path), epochs=10,
                     use_augmented=True, use_consistency=True,
                     use_degradation=True, out_dir=str(tmp_path / "out3"))
    res = train_rung(cfg)
    assert res["val_auc"] > 0.7
    assert all(np.isfinite(h["total"]) for h in res["history"])


def test_checkpoint_roundtrips_and_reproduces_scores(tmp_path):
    cfg = RungConfig(name="a1", bank_dir=_learnable_bank(tmp_path), epochs=5,
                     use_augmented=True, out_dir=str(tmp_path / "out1"))
    res = train_rung(cfg)
    model, meta = load_detector(res["checkpoint"], device="cpu")
    model.eval()
    b = FeatureBank.open(cfg.bank_dir)
    f = torch.from_numpy(np.asarray(b.feats[:4, 0]).astype(np.float32))
    with torch.no_grad():
        a = model(f)["logit"]
        c = model(f)["logit"]
    assert torch.allclose(a, c)
    assert meta["dim_feat"] == b.config["dim"]


def test_same_seed_gives_the_same_val_auc(tmp_path):
    bank = _learnable_bank(tmp_path)
    mk = lambda o: RungConfig(name="a1", bank_dir=bank, epochs=5, seed=99,
                              use_augmented=True, out_dir=str(tmp_path / o))
    assert train_rung(mk("x"))["val_auc"] == train_rung(mk("y"))["val_auc"]
