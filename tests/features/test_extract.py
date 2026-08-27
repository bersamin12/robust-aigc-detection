import numpy as np
import pandas as pd
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


def test_extract_bank_rejects_a_manifest_with_a_duplicated_index(tmp_path, monkeypatch):
    """extract_bank keys each image's RNG on its manifest index label
    (test_extraction_is_shard_independent_... above is why). If two rows
    share a label -- e.g. pd.concat of two manifest pieces that each kept
    their own default RangeIndex -- those two different images would
    silently draw identical views instead of failing loudly, so this must
    be a checked precondition, not an assumed one.
    """
    from aigcdet.features import extract
    from aigcdet.features.backbones import BackboneSpec

    spec = BackboneSpec("fake", "none", 64, 4, 1, 0)
    # Defensive: the guard must fire before load_backbone/embed are ever
    # called, but monkeypatch them anyway so a regression that removes the
    # guard fails on an assertion, never by starting a real GPU load.
    monkeypatch.setattr(extract, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(extract, "embed",
                        lambda m, s, imgs, device, batch_size=16:
                            np.zeros((len(imgs), s.dim), np.float32))

    a = make_dummy_manifest(3, str(tmp_path / "part_a"), np.random.default_rng(0))
    b = make_dummy_manifest(3, str(tmp_path / "part_b"), np.random.default_rng(1))
    dupes = pd.concat([a, b])  # both parts carry a fresh 0,1,2 RangeIndex

    with pytest.raises(ValueError, match="duplicated"):
        extract.extract_bank(dupes, "fake", str(tmp_path / "bank_dupe"), seed=3, device="cpu")


# --- H3: the LOTO bank must differ by one family and NOTHING else ----------

def _op_count_histogram(exclude, n=30_000, seed=20260827):
    from aigcdet.features.extract import _sample_recipe_excluding

    rng = np.random.default_rng(seed)
    ks = [len(_sample_recipe_excluding(rng, exclude).ops) for _ in range(n)]
    return np.bincount(ks, minlength=4)[1:], float(np.mean(ks))


def test_excluding_a_family_does_not_change_the_op_count_distribution():
    """The A3-vs-A3-LOTO comparison exists to isolate ONE transform family.
    The previous rejection sampler discarded whole recipes containing the
    excluded family, which throws away long chains disproportionately (an
    excluded family is likelier to appear in a 3-op chain than a 1-op one):
    measured mean 1.993 ops with no exclusion vs 1.827 with
    exclude=("noise",), so the LOTO bank trained on ~8% lighter augmentation
    overall. That violates the project's "identical view coverage across
    compared rungs" hard constraint in the one comparison it was written to
    bind. Fixed seeds, so this is deterministic rather than flaky.
    """
    from scipy import stats

    base_counts, base_mean = _op_count_histogram(())
    loto_counts, loto_mean = _op_count_histogram(("noise",))

    assert abs(base_mean - loto_mean) < 0.03, (base_mean, loto_mean)
    _chi2, p, _dof, _exp = stats.chi2_contingency(
        np.vstack([base_counts, loto_counts]))
    assert p > 0.01, f"op-count distributions differ (chi2 p={p:.4g})"


def test_op_count_stays_uniform_even_when_two_families_are_excluded():
    """Two exclusions made the old bias worse (mean 1.630). With four kept
    families a 1-3 op chain is still fully available, so the distribution must
    be unchanged."""
    _base_counts, base_mean = _op_count_histogram(())
    _counts, mean = _op_count_histogram(("jpeg", "blur"))
    assert abs(base_mean - mean) < 0.03, (base_mean, mean)


def test_exclusion_removes_only_the_named_family():
    """Every kept family must still be reachable -- restricting the pool must
    not accidentally narrow it further."""
    from aigcdet.augment.recipes import FAMILIES
    from aigcdet.features.extract import _sample_recipe_excluding

    rng = np.random.default_rng(7)
    seen = set()
    for _ in range(5_000):
        for op in _sample_recipe_excluding(rng, ("noise",)).ops:
            seen.add(op.name)
    assert seen == set(FAMILIES) - {"noise"}


def test_excluding_every_family_fails_loudly():
    from aigcdet.augment.recipes import FAMILIES
    from aigcdet.features.extract import _sample_recipe_excluding

    with pytest.raises(ValueError, match="no transform families"):
        _sample_recipe_excluding(np.random.default_rng(0), tuple(FAMILIES))


# --- H1/M3/L5: what extract_bank records about its own inputs and output ---

def test_extract_bank_stores_the_manifest_row_id_not_its_write_position(tmp_path, monkeypatch):
    """The replay key (seed, row_id, view_idx) must live in the bank. Before
    it did, `attach_recon_to_bank` had to recover it from a manifest the
    caller passed in, which nothing verified."""
    from aigcdet.features import extract
    from aigcdet.features.backbones import BackboneSpec

    spec = BackboneSpec("fake", "none", 64, 4, 1, 0)
    monkeypatch.setattr(extract, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(extract, "embed",
                        lambda m, s, imgs, device, batch_size=16:
                            np.zeros((len(imgs), s.dim), np.float32))

    df = make_dummy_manifest(6, str(tmp_path / "img_rid"), np.random.default_rng(0))
    shard = df.iloc[3:6]                       # index labels 3, 4, 5
    out = extract.extract_bank(shard, "fake", str(tmp_path / "bank_rid"),
                                seed=11, device="cpu")
    b = FeatureBank.open(out)
    np.testing.assert_array_equal(b.row_ids, [3, 4, 5])
    assert b.meta["image_idx"].tolist() == [0, 1, 2]


def test_extract_bank_records_the_manifest_fingerprint(tmp_path, monkeypatch):
    from aigcdet.features import extract
    from aigcdet.features.backbones import BackboneSpec
    from aigcdet.features.bank import manifest_fingerprint

    spec = BackboneSpec("fake", "none", 64, 4, 1, 0)
    monkeypatch.setattr(extract, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(extract, "embed",
                        lambda m, s, imgs, device, batch_size=16:
                            np.zeros((len(imgs), s.dim), np.float32))

    df = make_dummy_manifest(5, str(tmp_path / "img_fp"), np.random.default_rng(0))
    out = extract.extract_bank(df, "fake", str(tmp_path / "bank_fp"), seed=1,
                                device="cpu")
    b = FeatureBank.open(out)
    assert b.config["manifest_sha256"] == manifest_fingerprint(df)
    b.verify_against_manifest(df)
    with pytest.raises(ValueError, match="not the manifest the bank was built from"):
        b.verify_against_manifest(df.assign(path=df["path"] + ".bak"))


def test_extract_bank_checks_the_invariants_of_what_it_just_wrote(tmp_path, monkeypatch):
    """The cheapest possible post-condition on a job that runs for hours."""
    from aigcdet.features import extract
    from aigcdet.features.backbones import BackboneSpec

    spec = BackboneSpec("fake", "none", 64, 4, 1, 0)
    monkeypatch.setattr(extract, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(extract, "embed",
                        lambda m, s, imgs, device, batch_size=16:
                            np.zeros((len(imgs), s.dim), np.float32))

    calls: list[str] = []
    real_check = FeatureBank.check_invariants
    monkeypatch.setattr(FeatureBank, "check_invariants",
                        lambda self: (calls.append(self.path), real_check(self)))

    df = make_dummy_manifest(3, str(tmp_path / "img_inv"), np.random.default_rng(0))
    out = extract.extract_bank(df, "fake", str(tmp_path / "bank_inv"), seed=1,
                                device="cpu")
    assert calls == [out]
