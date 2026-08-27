import numpy as np
import pytest

from aigcdet.features.bank import N_VIEWS, BankWriter, FeatureBank


def _build(tmp_path, n=4, dim=8):
    w = BankWriter(str(tmp_path / "bank"), n_images=n, n_views=N_VIEWS,
                   dim=dim, backbone="test", seed=0)
    rng = np.random.default_rng(0)
    for i in range(n):
        presence = np.zeros((N_VIEWS, 6), np.float32)
        severity = np.zeros((N_VIEWS, 6), np.float32)
        presence[1:, 0] = 1.0                       # view 0 stays clean
        severity[1:, 0] = 0.5
        w.write_image(
            i,
            {"path": f"/x/{i}.png", "label": i % 2, "generator": "g", "source": "s",
             "split": "train"},
            feats=rng.normal(size=(N_VIEWS, dim)).astype(np.float32),
            presence=presence, severity=severity,
            proxies=rng.normal(size=(N_VIEWS, 3)).astype(np.float32),
            recipes=["[]"] + ['[{"name": "jpeg", "params": {"quality": 50}}]'] * (N_VIEWS - 1),
        )
    w.close()
    return FeatureBank.open(str(tmp_path / "bank"))


def test_bank_roundtrips_all_arrays(tmp_path):
    b = _build(tmp_path)
    assert b.feats.shape == (4, N_VIEWS, 8)
    assert b.presence.shape == (4, N_VIEWS, 6)
    assert b.severity.shape == (4, N_VIEWS, 6)
    assert b.proxies.shape == (4, N_VIEWS, 3)
    assert b.recon is None
    assert len(b.meta) == 4 and b.meta["label"].tolist() == [0, 1, 0, 1]
    assert b.config["backbone"] == "test"


def test_view_zero_is_the_clean_view(tmp_path):
    b = _build(tmp_path)
    b.check_invariants()
    assert b.presence[:, 0, :].sum() == 0.0


def test_check_invariants_rejects_a_degraded_view_zero(tmp_path):
    b = _build(tmp_path)
    b.presence[0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="view 0"):
        b.check_invariants()


def test_attach_recon_persists_and_reloads(tmp_path):
    b = _build(tmp_path)
    r = np.arange(4 * N_VIEWS * 12, dtype=np.float32).reshape(4, N_VIEWS, 12)
    b.attach_recon(r)
    b2 = FeatureBank.open(b.path)
    assert b2.recon is not None and b2.recon.shape == (4, N_VIEWS, 12)
    np.testing.assert_allclose(b2.recon[1, 2], r[1, 2])


def test_recipes_are_recoverable_per_view(tmp_path):
    from aigcdet.augment.recipes import Recipe
    b = _build(tmp_path)
    assert Recipe.from_json(b.recipe_json(0, 0)).ops == ()
    assert Recipe.from_json(b.recipe_json(0, 1)).ops[0].name == "jpeg"
