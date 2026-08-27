import numpy as np
import pytest

from aigcdet.data.manifest import make_dummy_manifest
from aigcdet.features.bank import N_VIEWS, FeatureBank


def test_extract_with_a_fake_backbone_produces_a_valid_bank(tmp_path, monkeypatch):
    from aigcdet.features import extract
    from aigcdet.features.backbones import BackboneSpec

    spec = BackboneSpec("fake", "none", 64, 5, 1, 0)
    monkeypatch.setattr(extract, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(
        extract, "embed",
        lambda m, s, imgs, device, batch_size=16:
            np.stack([np.full(s.dim, float(i.mean()), np.float32) for i in imgs]))

    df = make_dummy_manifest(6, str(tmp_path / "img"), np.random.default_rng(0))
    out = extract.extract_bank(df, "fake", str(tmp_path / "bank"), seed=1, device="cpu")

    b = FeatureBank.open(out)
    b.check_invariants()
    assert b.feats.shape == (6, N_VIEWS, 5)
    assert len(b.meta) == 6
    # View 0 must be the clean view: its embedding equals the raw image mean.
    assert b.presence[:, 0, :].sum() == 0.0
    # Augmented views must actually differ from the clean one.
    assert not np.allclose(np.asarray(b.feats[0, 0]), np.asarray(b.feats[0, 1]))


def test_exclude_families_never_samples_the_excluded_family(tmp_path, monkeypatch):
    from aigcdet.augment.recipes import FAMILIES, Recipe
    from aigcdet.features import extract
    from aigcdet.features.backbones import BackboneSpec

    spec = BackboneSpec("fake", "none", 64, 4, 1, 0)
    monkeypatch.setattr(extract, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(extract, "embed",
                        lambda m, s, imgs, device, batch_size=16:
                            np.zeros((len(imgs), s.dim), np.float32))

    df = make_dummy_manifest(4, str(tmp_path / "img2"), np.random.default_rng(0))
    out = extract.extract_bank(df, "fake", str(tmp_path / "bank2"), seed=2,
                               device="cpu", exclude_families=("noise",))
    b = FeatureBank.open(out)
    i_noise = FAMILIES.index("noise")
    assert np.asarray(b.presence)[:, :, i_noise].sum() == 0.0
    for img in range(4):
        for v in range(N_VIEWS):
            assert all(o.name != "noise" for o in Recipe.from_json(b.recipe_json(img, v)).ops)


def test_extraction_is_shard_independent_when_the_manifest_index_is_preserved(tmp_path, monkeypatch):
    """A shard/session handed a slice of the frozen manifest (e.g. Kaggle
    session 2 getting rows [3:6) of a 100k-row manifest) must draw exactly
    the same views for each image as an uninterrupted single run would have
    -- otherwise two shards of the same logical bank disagree with each
    other, and a crash-and-resume changes what a completed bank contains.
    This only holds if the caller does not reset the slice's index (the
    manifest.py convention: `.iloc[]` preserves it, `.reset_index()` erases
    it), which is why the CLI's --split filter avoids reset_index too.
    """
    from aigcdet.features import extract
    from aigcdet.features.backbones import BackboneSpec

    spec = BackboneSpec("fake", "none", 64, 5, 1, 0)
    monkeypatch.setattr(extract, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(
        extract, "embed",
        lambda m, s, imgs, device, batch_size=16:
            np.stack([np.full(s.dim, float(i.mean()), np.float32) for i in imgs]))

    df = make_dummy_manifest(6, str(tmp_path / "img4"), np.random.default_rng(0))
    full = FeatureBank.open(
        extract.extract_bank(df, "fake", str(tmp_path / "full"), seed=9, device="cpu"))
    shard = FeatureBank.open(
        extract.extract_bank(df.iloc[3:6], "fake", str(tmp_path / "shard"), seed=9, device="cpu"))

    for shard_pos, full_pos in enumerate([3, 4, 5]):
        assert shard.recipe_json(shard_pos, 0) == full.recipe_json(full_pos, 0)
        for v in range(N_VIEWS):
            assert shard.recipe_json(shard_pos, v) == full.recipe_json(full_pos, v)
        np.testing.assert_array_equal(
            np.asarray(shard.feats[shard_pos]), np.asarray(full.feats[full_pos]))


def test_extraction_is_reproducible_for_a_fixed_seed(tmp_path, monkeypatch):
    from aigcdet.features import extract
    from aigcdet.features.backbones import BackboneSpec
    spec = BackboneSpec("fake", "none", 64, 3, 1, 0)
    monkeypatch.setattr(extract, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(extract, "embed",
                        lambda m, s, imgs, device, batch_size=16:
                            np.stack([np.full(s.dim, float(i.std()), np.float32) for i in imgs]))
    df = make_dummy_manifest(3, str(tmp_path / "img3"), np.random.default_rng(0))
    a = FeatureBank.open(extract.extract_bank(df, "fake", str(tmp_path / "b1"), seed=5, device="cpu"))
    c = FeatureBank.open(extract.extract_bank(df, "fake", str(tmp_path / "b2"), seed=5, device="cpu"))
    np.testing.assert_allclose(np.asarray(a.feats), np.asarray(c.feats))
