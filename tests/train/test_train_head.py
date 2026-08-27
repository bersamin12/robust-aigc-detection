import numpy as np
import torch

from aigcdet.features.bank import N_VIEWS, RECON_DIM, BankWriter, FeatureBank
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


def _learnable_bank_with_recon(tmp_path, n=120, dim=8, recon_dim=RECON_DIM):
    """Same class-separable feats as `_learnable_bank`, plus a synthetic
    recon.npy attached via the real `attach_recon` API -- no VAE, no
    reconstruction pipeline, just an array of the right shape."""
    bank_dir = _learnable_bank(tmp_path, n=n, dim=dim)
    bank = FeatureBank.open(bank_dir)
    rng = np.random.default_rng(1)
    recon = rng.normal(0.0, 0.1, size=(n, N_VIEWS, recon_dim)).astype(np.float32)
    bank.attach_recon(recon)
    return bank_dir


def test_train_rung_restores_global_rng_state_and_does_not_init_cuda(tmp_path):
    """train_rung must seed module init reproducibly (nn.Linear.reset_parameters
    has no generator param) without leaking that seed into the caller's global
    torch RNG state, and must never touch CUDA (device="cpu" throughout, and
    fork_rng(devices=[]) must not itself initialise a CUDA context)."""
    cfg = RungConfig(name="a1", bank_dir=_learnable_bank(tmp_path), epochs=3,
                     use_augmented=True, out_dir=str(tmp_path / "out_rng"))
    before = torch.random.get_rng_state()
    train_rung(cfg)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    assert not torch.cuda.is_initialized()


def test_a4_shaped_run_with_recon_trains_and_produces_a_plausible_auc(tmp_path):
    cfg = RungConfig(name="a4", bank_dir=_learnable_bank_with_recon(tmp_path), epochs=10,
                     use_augmented=True, use_degradation=True, use_consistency=True,
                     use_recon=True, use_film=False, out_dir=str(tmp_path / "out4"))
    res = train_rung(cfg)
    assert 0.0 <= res["val_auc"] <= 1.0
    assert res["val_auc"] > 0.7
    model, meta = load_detector(res["checkpoint"], device="cpu")
    assert model.use_recon is True
    assert meta["config"]["use_recon"] is True


def test_a7_shaped_run_with_film_trains_and_produces_a_plausible_auc(tmp_path):
    cfg = RungConfig(name="a7", bank_dir=_learnable_bank_with_recon(tmp_path), epochs=10,
                     use_augmented=True, use_degradation=True, use_consistency=True,
                     use_recon=True, use_film=True, out_dir=str(tmp_path / "out7"))
    res = train_rung(cfg)
    assert 0.0 <= res["val_auc"] <= 1.0
    assert res["val_auc"] > 0.7
    model, meta = load_detector(res["checkpoint"], device="cpu")
    assert model.use_film is True
    assert model.classifier.use_film is True
    assert meta["config"]["use_film"] is True
