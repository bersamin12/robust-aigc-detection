import numpy as np
import pandas as pd
import pytest

from aigcdet.augment.recipes import Op, Recipe
from aigcdet.augment.scenarios import EVAL_GRID
from aigcdet.data.manifest import make_dummy_manifest
from aigcdet.features.bank import FeatureBank


def _spec():
    from aigcdet.features.backbones import BackboneSpec
    return BackboneSpec("fake", "none", 64, 4, 1, 0)


def _patch_backbone(monkeypatch, grid, embed_fn=None):
    """Every test in this file runs against a stubbed backbone.

    The project's standing limit forbids loading real weights or touching the
    GPU here, so `load_backbone`/`embed` are always monkeypatched.
    """
    spec = _spec()
    monkeypatch.setattr(grid, "load_backbone", lambda n, device: (None, spec))
    if embed_fn is None:
        def embed_fn(m, s, imgs, device, batch_size=16):
            return np.stack([np.full(s.dim, float(i.mean()), np.float32) for i in imgs])
    monkeypatch.setattr(grid, "embed", embed_fn)
    return spec


def test_eval_bank_view_axis_is_the_condition_axis(tmp_path, monkeypatch):
    from aigcdet.eval import grid
    _patch_backbone(monkeypatch, grid)

    df = make_dummy_manifest(5, str(tmp_path / "img"), np.random.default_rng(0))
    out = grid.extract_eval_bank(df, "fake", str(tmp_path / "eb"), device="cpu")
    b = FeatureBank.open(out)
    assert b.config["conditions"][0] == "clean"
    assert list(b.config["conditions"]) == list(EVAL_GRID)
    assert b.feats.shape == (5, len(EVAL_GRID), 4)
    b.check_invariants()   # view 0 clean


def test_eval_bank_view_j_carries_condition_j_labels(tmp_path, monkeypatch):
    """The view axis is the condition axis for every image identically.

    Shape alone would pass if every view held the clean recipe, so pin the
    per-view supervision against `EVAL_GRID`'s own recipes.
    """
    from aigcdet.eval import grid
    _patch_backbone(monkeypatch, grid)

    df = make_dummy_manifest(3, str(tmp_path / "imgl"), np.random.default_rng(1))
    b = FeatureBank.open(
        grid.extract_eval_bank(df, "fake", str(tmp_path / "ebl"), device="cpu"))
    names = list(EVAL_GRID)
    for i in range(len(b.meta)):
        for j, name in enumerate(names):
            recipe = EVAL_GRID[name]
            assert Recipe.from_json(b.recipe_json(i, j)) == recipe
            np.testing.assert_allclose(np.asarray(b.presence[i, j]),
                                       recipe.labels()["presence"])
            np.testing.assert_allclose(np.asarray(b.severity[i, j]),
                                       recipe.labels()["severity"], atol=1e-6)
    # not vacuous: the degraded conditions really do carry presence
    assert float(np.asarray(b.presence)[:, 1:, :].sum()) > 0.0


def test_eval_bank_records_the_manifest_fingerprint(tmp_path, monkeypatch):
    from aigcdet.eval import grid
    _patch_backbone(monkeypatch, grid)

    df = make_dummy_manifest(4, str(tmp_path / "imgf"), np.random.default_rng(2))
    b = FeatureBank.open(
        grid.extract_eval_bank(df, "fake", str(tmp_path / "ebf"), device="cpu"))
    b.verify_against_manifest(df)
    with pytest.raises(ValueError, match="not the manifest"):
        b.verify_against_manifest(df.iloc[::-1])


def test_eval_views_are_keyed_on_row_id_not_loop_position(tmp_path, monkeypatch):
    """A shard of the manifest must reproduce the full run's exact pixels.

    Eval extraction is deterministic by design, but `noise` and `jitter`
    conditions still consume randomness. Keying each view on
    (seed, row_id, view_idx) -- never on this call's loop position -- is what
    makes the eval bank shardable and restartable like the training bank.
    """
    from aigcdet.eval import grid

    captured: dict[str, list] = {}

    def recorder(key):
        def _embed(m, s, imgs, device, batch_size=16):
            captured.setdefault(key, []).append([np.asarray(v).copy() for v in imgs])
            return np.zeros((len(imgs), s.dim), np.float32)
        return _embed

    df = make_dummy_manifest(5, str(tmp_path / "imgs"), np.random.default_rng(3))

    _patch_backbone(monkeypatch, grid, recorder("full"))
    grid.extract_eval_bank(df, "fake", str(tmp_path / "full"), device="cpu")
    _patch_backbone(monkeypatch, grid, recorder("shard"))
    grid.extract_eval_bank(df.iloc[2:], "fake", str(tmp_path / "shard"), device="cpu")

    noise_view = list(EVAL_GRID).index("noise_s0.05")
    for offset in range(3):
        for j in range(len(EVAL_GRID)):
            np.testing.assert_array_equal(captured["full"][2 + offset][j],
                                          captured["shard"][offset][j])
    # not vacuous: the noise view differs between images and from the clean view
    assert not np.array_equal(captured["full"][0][noise_view],
                              captured["full"][1][noise_view])
    assert not np.array_equal(captured["full"][0][noise_view],
                              captured["full"][0][0])


def test_extract_eval_bank_rejects_conditions_whose_view_zero_is_not_clean(tmp_path, monkeypatch):
    from aigcdet.eval import grid
    _patch_backbone(monkeypatch, grid)
    df = make_dummy_manifest(3, str(tmp_path / "imgr"), np.random.default_rng(4))

    reordered = {k: EVAL_GRID[k] for k in list(EVAL_GRID)[1:]}
    with pytest.raises(ValueError, match="condition 0"):
        grid.extract_eval_bank(df, "fake", str(tmp_path / "bad1"),
                               conditions=reordered, device="cpu")

    lying = {**EVAL_GRID, "clean": Recipe((Op("jpeg", {"quality": 50}),))}
    with pytest.raises(ValueError, match="condition 0"):
        grid.extract_eval_bank(df, "fake", str(tmp_path / "bad2"),
                               conditions=lying, device="cpu")


def test_extract_eval_bank_rejects_a_duplicated_manifest_index(tmp_path, monkeypatch):
    from aigcdet.eval import grid
    _patch_backbone(monkeypatch, grid)
    df = make_dummy_manifest(3, str(tmp_path / "imgd"), np.random.default_rng(5))
    with pytest.raises(ValueError, match="duplicated label"):
        grid.extract_eval_bank(pd.concat([df, df]), "fake", str(tmp_path / "bad3"),
                               device="cpu")


def test_score_grid_returns_one_row_per_image_and_condition(tmp_path, monkeypatch):
    from aigcdet.eval import grid
    from aigcdet.models.heads import Detector
    _patch_backbone(monkeypatch, grid,
                    lambda m, s, imgs, device, batch_size=16:
                        np.zeros((len(imgs), s.dim), np.float32))
    df = make_dummy_manifest(4, str(tmp_path / "img2"), np.random.default_rng(0))
    b = FeatureBank.open(grid.extract_eval_bank(df, "fake", str(tmp_path / "eb2"),
                                                device="cpu"))
    model = Detector(dim_feat=4, use_recon=False)
    out = grid.score_grid(model, b, use_recon=False, device="cpu")
    assert len(out) == 4 * len(EVAL_GRID)
    assert set(out.columns) >= {"condition", "image_idx", "label", "generator",
                                "source", "score"}
    assert out["condition"].nunique() == len(EVAL_GRID)


def test_score_grid_carries_each_rows_own_metadata_and_score(tmp_path, monkeypatch):
    """Scores must track the per-view features, not be a broadcast constant."""
    from aigcdet.eval import grid
    from aigcdet.models.heads import Detector
    _patch_backbone(monkeypatch, grid)     # feats = mean pixel value per view
    df = make_dummy_manifest(4, str(tmp_path / "img3"), np.random.default_rng(6))
    b = FeatureBank.open(grid.extract_eval_bank(df, "fake", str(tmp_path / "eb3"),
                                                device="cpu"))
    model = Detector(dim_feat=4, use_recon=False)
    out = grid.score_grid(model, b, use_recon=False, device="cpu")

    clean = out[out["condition"] == "clean"].sort_values("image_idx")
    assert clean["label"].tolist() == b.meta["label"].tolist()
    assert clean["generator"].tolist() == b.meta["generator"].tolist()
    assert clean["source"].tolist() == b.meta["source"].tolist()
    assert out["score"].nunique() > 1
    # every (condition, image) pair appears exactly once
    assert not out.duplicated(["condition", "image_idx"]).any()


def test_score_grid_uses_recon_features_when_asked(tmp_path, monkeypatch):
    from aigcdet.eval import grid
    from aigcdet.models.heads import Detector
    _patch_backbone(monkeypatch, grid,
                    lambda m, s, imgs, device, batch_size=16:
                        np.zeros((len(imgs), s.dim), np.float32))
    df = make_dummy_manifest(3, str(tmp_path / "img4"), np.random.default_rng(7))
    b = FeatureBank.open(grid.extract_eval_bank(df, "fake", str(tmp_path / "eb4"),
                                                device="cpu"))
    rng = np.random.default_rng(0)
    b.attach_recon(rng.normal(
        size=(len(b.meta), len(EVAL_GRID), 12)).astype(np.float32))
    model = Detector(dim_feat=4, use_recon=True)
    out = grid.score_grid(model, b, use_recon=True, device="cpu")
    assert len(out) == 3 * len(EVAL_GRID)
    # feats are all-zero, so any variation in the score comes from recon
    assert out["score"].nunique() > 1

    plain = FeatureBank.open(str(tmp_path / "eb4"))
    plain.recon = None
    with pytest.raises(ValueError, match="no recon"):
        grid.score_grid(model, plain, use_recon=True, device="cpu")


def test_score_grid_rejects_a_bank_that_is_not_an_eval_bank(tmp_path, monkeypatch):
    from aigcdet.eval import grid
    from aigcdet.features.bank import BankWriter
    from aigcdet.models.heads import Detector

    w = BankWriter(str(tmp_path / "train_bank"), n_images=1, n_views=2, dim=4,
                   backbone="fake", seed=0)
    w.write_image(0, {"path": "/a.png", "label": 0, "generator": "",
                      "source": "s", "split": "train"},
                  feats=np.zeros((2, 4), np.float32),
                  presence=np.zeros((2, 6), np.float32),
                  severity=np.zeros((2, 6), np.float32),
                  proxies=np.zeros((2, 3), np.float32),
                  recipes=["[]", "[]"])
    w.close()
    model = Detector(dim_feat=4, use_recon=False)
    with pytest.raises(ValueError, match="conditions"):
        grid.score_grid(model, FeatureBank.open(str(tmp_path / "train_bank")),
                        use_recon=False, device="cpu")


def test_score_grid_verifies_the_manifest_when_one_is_supplied(tmp_path, monkeypatch):
    from aigcdet.eval import grid
    from aigcdet.models.heads import Detector
    _patch_backbone(monkeypatch, grid)
    df = make_dummy_manifest(4, str(tmp_path / "img5"), np.random.default_rng(8))
    b = FeatureBank.open(grid.extract_eval_bank(df, "fake", str(tmp_path / "eb5"),
                                                device="cpu"))
    model = Detector(dim_feat=4, use_recon=False)
    assert len(grid.score_grid(model, b, use_recon=False, device="cpu",
                               manifest_df=df)) == 4 * len(EVAL_GRID)
    with pytest.raises(ValueError, match="not the manifest"):
        grid.score_grid(model, b, use_recon=False, device="cpu",
                        manifest_df=df.iloc[::-1])


# --- R24: rung comparisons must be over identical view coverage ------------

def _eval_bank(tmp_path, monkeypatch, name, df, backbone="fake", conditions=None):
    from aigcdet.eval import grid
    _patch_backbone(monkeypatch, grid)
    return FeatureBank.open(grid.extract_eval_bank(
        df, backbone, str(tmp_path / name), conditions=conditions, device="cpu"))


def test_assert_banks_comparable_accepts_two_banks_from_the_same_extraction(
        tmp_path, monkeypatch):
    from aigcdet.eval.grid import assert_banks_comparable
    df = make_dummy_manifest(3, str(tmp_path / "srcc"), np.random.default_rng(9))
    a = _eval_bank(tmp_path, monkeypatch, "ca", df)
    b = _eval_bank(tmp_path, monkeypatch, "cb", df)
    assert_banks_comparable([a, b])
    assert_banks_comparable([a])


@pytest.mark.parametrize("differ", ["n_views", "backbone", "manifest_sha256"])
def test_assert_banks_comparable_rejects_differing_coverage(tmp_path, monkeypatch,
                                                            differ):
    """Two rungs compared over different view coverage compare augmentation
    budgets, not models -- the guard R24 exists for."""
    from aigcdet.eval.grid import assert_banks_comparable
    df = make_dummy_manifest(3, str(tmp_path / f"src{differ}"),
                             np.random.default_rng(9))
    a = _eval_bank(tmp_path, monkeypatch, f"x{differ}a", df)
    if differ == "n_views":
        subset = {k: EVAL_GRID[k] for k in list(EVAL_GRID)[:5]}
        b = _eval_bank(tmp_path, monkeypatch, f"x{differ}b", df, conditions=subset)
    elif differ == "backbone":
        b = _eval_bank(tmp_path, monkeypatch, f"x{differ}b", df, backbone="other")
    else:
        other_df = make_dummy_manifest(3, str(tmp_path / f"src{differ}2"),
                                       np.random.default_rng(11))
        b = _eval_bank(tmp_path, monkeypatch, f"x{differ}b", other_df)
    with pytest.raises(ValueError, match=differ):
        assert_banks_comparable([a, b])


# --- stratified subsample --------------------------------------------------

def test_stratified_subsample_preserves_class_balance_and_is_reproducible():
    from aigcdet.eval.grid import stratified_subsample
    meta = pd.DataFrame({
        "label": [0] * 60 + [1] * 60,
        "generator": [""] * 60 + ["g1"] * 30 + ["g2"] * 30,
        "source": ["coco"] * 60 + ["wf"] * 60,
    })
    a = stratified_subsample(meta, 40, seed=1)
    b = stratified_subsample(meta, 40, seed=1)
    assert np.array_equal(a, b)
    assert len(a) == 40
    picked = meta.iloc[a]
    assert abs((picked["label"] == 1).mean() - 0.5) < 0.15
    assert picked["generator"].nunique() >= 2


def test_stratified_subsample_balances_generators_within_a_class():
    from aigcdet.eval.grid import stratified_subsample
    meta = pd.DataFrame({
        "label": [0] * 60 + [1] * 60,
        "generator": [""] * 60 + ["g1"] * 50 + ["g2"] * 10,
        "source": ["coco"] * 60 + ["wf"] * 60,
    })
    picked = meta.iloc[stratified_subsample(meta, 40, seed=2)]
    assert (picked["label"] == 1).sum() == 20
    # the small family is not swamped by the large one
    assert (picked["generator"] == "g2").sum() == 10
    assert (picked["generator"] == "g1").sum() == 10


def test_stratified_subsample_is_seed_sensitive_and_not_just_the_first_rows():
    from aigcdet.eval.grid import stratified_subsample
    meta = pd.DataFrame({"label": [0, 1] * 50, "generator": ["", "g"] * 50,
                         "source": ["a"] * 100})
    a = stratified_subsample(meta, 20, seed=1)
    b = stratified_subsample(meta, 20, seed=2)
    assert not np.array_equal(a, b)
    assert not np.array_equal(a, np.arange(20))
    assert np.array_equal(a, np.unique(a))          # sorted, no duplicates


def test_stratified_subsample_returns_exactly_n_when_strata_outnumber_it():
    from aigcdet.eval.grid import stratified_subsample
    meta = pd.DataFrame({"label": list(range(2)) * 10,
                         "generator": [f"g{i}" for i in range(20)],
                         "source": ["a"] * 20})
    idx = stratified_subsample(meta, 5, seed=3)
    assert len(idx) == 5 and len(np.unique(idx)) == 5


def test_stratified_subsample_returns_everything_when_n_exceeds_pool():
    from aigcdet.eval.grid import stratified_subsample
    meta = pd.DataFrame({"label": [0, 1, 0, 1], "generator": ["", "g", "", "g"],
                         "source": ["a"] * 4})
    assert len(stratified_subsample(meta, 100, seed=0)) == 4


def test_stratified_subsample_ignores_the_frames_index_labels():
    """Returned indices are POSITIONAL, so a filtered frame still works."""
    from aigcdet.eval.grid import stratified_subsample
    meta = pd.DataFrame({"label": [0, 1] * 20, "generator": ["", "g"] * 20,
                         "source": ["a"] * 40}, index=range(100, 140))
    idx = stratified_subsample(meta, 10, seed=4)
    assert idx.max() < len(meta)
    assert (meta.iloc[idx]["label"] == 1).sum() == 5
