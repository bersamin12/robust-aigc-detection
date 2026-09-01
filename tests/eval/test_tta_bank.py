"""Rung A6: the bank whose view axis is condition x tta_view.

A6 is the rung most likely to be wired up last, by whoever has the least
context, and every way of getting it wrong is silent. A TTA bank and a plain
eval bank differ only in the WIDTH of an axis both of them call `n_views`, and
a width is not something any downstream consumer checks; a frame of scores
produced by reading the wrong columns has the right shape, the right dtypes,
the right row count and the right condition labels attached to the wrong
pixels. So the properties are pinned here rather than left to review:

  * the identity view reproduces the plain bank BIT FOR BIT, which is the one
    check that proves the composition order and the RNG keying at once. It is
    exact HERE because the embedder is a deterministic stub; on a GPU the same
    pixels embedded in a 20-view and a 160-view batch take different GEMM
    reduction orders and their float16 casts land within a ULP of each other,
    which is why `scripts/verify_tta_bank.py` measures the distance in ULPs
    instead. The pixel-level claim is the one that matters and this is where
    it is actually pinned;
  * `score_grid` refuses the bank it would misread;
  * `score_grid_tta` refuses the bank it would average over the wrong axis;
  * the reduced `n_views` that `TtaEvalBank` declares is stated on purpose and
    the eight-fold cost is recorded next to it.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from aigcdet.data.manifest import make_dummy_manifest
from aigcdet.features.bank import FeatureBank


VIEWS = ["identity", "hflip", "jpeg_95"]


def _patch_backbone(monkeypatch, grid, embed_fn):
    """No real weights and no GPU in the test suite (standing project limit)."""
    from aigcdet.features.backbones import BackboneSpec
    spec = BackboneSpec("fake", "none", 64, 4, 1, 0)
    monkeypatch.setattr(grid, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(grid, "embed", embed_fn)
    return spec


def _conditions():
    """Three conditions, one of them RANDOM.

    `noise` draws from the per-image RNG, which is what makes the identity
    check below load-bearing: if the TTA loop re-keyed the condition RNG per
    view, the eight views of `noise_s0.05` would be eight different noise
    realisations and the identity column would stop matching the plain bank.
    A grid of deterministic conditions could not tell the two apart.
    """
    from aigcdet.augment.recipes import Op, Recipe
    return {
        "clean": Recipe(()),
        "jpeg_q50": Recipe((Op("jpeg", {"quality": 50}),)),
        "noise_s0.05": Recipe((Op("noise", {"sigma": 0.05}),)),
    }


def _mean_embed(m, s, imgs, device, batch_size=16):
    return np.stack([np.full(s.dim, float(i.mean()), np.float32) for i in imgs])


def _build(tmp_path, monkeypatch, tta_views, n=4, seed=7, out="bank"):
    from aigcdet.eval import grid
    _patch_backbone(monkeypatch, grid, _mean_embed)
    df = make_dummy_manifest(n, str(tmp_path / "img"), np.random.default_rng(0))
    path = grid.extract_eval_bank(
        df, "fake", str(tmp_path / out), conditions=_conditions(),
        device="cpu", seed=seed, tta_views=tta_views)
    return FeatureBank.open(path), df


# ===========================================================================
# The axis
# ===========================================================================

def test_the_view_axis_is_the_cross_product_and_says_so(tmp_path, monkeypatch):
    bank, _ = _build(tmp_path, monkeypatch, VIEWS)
    assert bank.config["conditions"] == ["clean", "jpeg_q50", "noise_s0.05"]
    assert bank.config["tta_views"] == VIEWS
    assert bank.config["n_views"] == 3 * len(VIEWS)
    assert bank.feats.shape[1] == 3 * len(VIEWS)


def test_a_plain_eval_bank_declares_no_tta_axis(tmp_path, monkeypatch):
    """The absence has to be an absence, not a `tta_views: []` that reads as
    'a TTA bank with no views' to anything doing a truthiness test."""
    bank, _ = _build(tmp_path, monkeypatch, None)
    assert "tta_views" not in bank.config
    from aigcdet.eval.grid import tta_axis
    assert tta_axis(bank) == []


def test_an_empty_view_list_is_refused_rather_than_silently_ignored(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="zero-width"):
        _build(tmp_path, monkeypatch, [])


def test_an_unknown_view_name_fails_before_anything_is_decoded(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="unknown TTA view"):
        _build(tmp_path, monkeypatch, ["identity", "hfilp"])


# ===========================================================================
# The identity column: composition order and RNG keying, in one check
# ===========================================================================

def test_the_identity_column_reproduces_the_plain_bank_bit_for_bit(tmp_path, monkeypatch):
    """The single strongest statement available about this bank.

    Column `j * len(views)` of the TTA bank is condition `j` with the identity
    view applied, so it must equal column `j` of the plain bank built from the
    same manifest and seed -- exactly, not approximately. It fails if the
    composition order flips (TTA before the condition), if the condition RNG is
    re-keyed on the flattened index (each view would get its own noise draw),
    if the flattening is transposed (`k * n_cond + j`), or if canonicalisation
    moved. Four distinct bugs, none of which changes a single shape.
    """
    plain, _ = _build(tmp_path, monkeypatch, None, out="plain")
    tta, _ = _build(tmp_path, monkeypatch, VIEWS, out="tta")

    n = len(VIEWS)
    for j in range(len(plain.config["conditions"])):
        np.testing.assert_array_equal(
            np.asarray(tta.feats[:, j * n, :]), np.asarray(plain.feats[:, j, :]),
            err_msg=f"condition {j}: identity view != plain bank")


def test_every_view_of_a_condition_sees_the_same_noise_draw(tmp_path, monkeypatch):
    """`hflip` of a noised image is a flip of ONE noise realisation.

    With the stub embedder returning the image mean, an hflip cannot change the
    embedding at all -- so under the correct keying `hflip` and `identity`
    agree on every condition, and under per-view re-keying they would differ on
    the random one. That is exactly the confound this pins: an average over
    eight independent noise draws is a strictly easier problem than an average
    over eight transforms, and it would flatter A6 for a reason that has
    nothing to do with test-time augmentation.
    """
    tta, _ = _build(tmp_path, monkeypatch, VIEWS)
    n = len(VIEWS)
    noise_j = tta.config["conditions"].index("noise_s0.05")
    np.testing.assert_array_equal(
        np.asarray(tta.feats[:, noise_j * n + VIEWS.index("hflip"), :]),
        np.asarray(tta.feats[:, noise_j * n, :]))


def test_the_degradation_labels_describe_the_condition_not_the_tta_view(tmp_path, monkeypatch):
    """`jpeg_95` is something the DETECTOR did, not something that happened to
    the image. Labelling that view as JPEG-degraded would teach the degradation
    readout to report the detector's own preprocessing as evidence about the
    image's history."""
    tta, _ = _build(tmp_path, monkeypatch, VIEWS)
    n = len(VIEWS)
    for j in range(len(tta.config["conditions"])):
        for k in range(n):
            np.testing.assert_array_equal(tta.presence[:, j * n + k],
                                          tta.presence[:, j * n])
            np.testing.assert_array_equal(tta.severity[:, j * n + k],
                                          tta.severity[:, j * n])


def test_every_view_of_a_condition_stores_that_conditions_recipe(tmp_path, monkeypatch):
    """`recipe_json` is what the WORLD did to the image, so all eight views of
    a condition carry that condition's recipe and nothing about the TTA view.

    It also has to stay a parseable `Recipe`, because `check_invariants`
    replays it at the end of every extraction -- a richer object naming the
    view would fail there. Which view a column is stays recoverable from
    `config["tta_views"]` and the flattening, and `tta_axis` refuses a bank
    where those two disagree.
    """
    from aigcdet.augment.recipes import Recipe
    tta, _ = _build(tmp_path, monkeypatch, VIEWS)
    n = len(VIEWS)
    for j, cond in enumerate(tta.config["conditions"]):
        first = tta.recipe_json(0, j * n)
        assert Recipe.from_json(first) == _conditions()[cond]
        for k in range(n):
            assert tta.recipe_json(0, j * n + k) == first


# ===========================================================================
# Scoring
# ===========================================================================

class _Sum(torch.nn.Module):
    """logit = sum(features), so a view's logit is a readable function of its
    pixels and the average over views is checkable by hand."""

    use_recon = False

    def forward(self, f, r=None):
        return {"logit": f.sum(dim=-1)}


def test_score_grid_tta_averages_the_views_in_logit_space(tmp_path, monkeypatch):
    from aigcdet.eval.grid import score_grid_tta
    tta, _ = _build(tmp_path, monkeypatch, VIEWS)
    out = score_grid_tta(_Sum(), tta, device="cpu")

    n = len(VIEWS)
    names = tta.config["conditions"]
    assert list(out["condition"].unique()) == names
    assert len(out) == len(tta.meta) * len(names)
    for j, cond in enumerate(names):
        want = np.mean([np.asarray(tta.feats[:, j * n + k, :], dtype=np.float32).sum(-1)
                        for k in range(n)], axis=0)
        got = out[out["condition"] == cond]["score"].to_numpy()
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_score_grid_tta_returns_the_shape_score_grid_returns(tmp_path, monkeypatch):
    """A6 is a row, not a footnote, and that rests on the two frames being
    interchangeable to everything downstream of them."""
    from aigcdet.eval.grid import score_grid, score_grid_tta
    plain, _ = _build(tmp_path, monkeypatch, None, out="plain")
    tta, _ = _build(tmp_path, monkeypatch, VIEWS, out="tta")

    a = score_grid(_Sum(), plain, device="cpu")
    b = score_grid_tta(_Sum(), tta, device="cpu")
    assert list(a.columns) == list(b.columns)
    assert len(a) == len(b)
    for col in ("condition", "image_idx", "label", "generator", "source"):
        pd.testing.assert_series_equal(a[col], b[col])


def test_a_single_identity_view_reproduces_score_grid_exactly(tmp_path, monkeypatch):
    """The degenerate case has to be an identity, or the averaging is doing
    something beyond averaging."""
    from aigcdet.eval.grid import score_grid, score_grid_tta
    plain, _ = _build(tmp_path, monkeypatch, None, out="plain")
    one, _ = _build(tmp_path, monkeypatch, ["identity"], out="one")
    np.testing.assert_array_equal(
        score_grid(_Sum(), plain, device="cpu")["score"].to_numpy(),
        score_grid_tta(_Sum(), one, device="cpu")["score"].to_numpy())


def test_score_grid_refuses_a_tta_bank(tmp_path, monkeypatch):
    """It would read the first `len(conditions)` of the 9 columns, label them
    with the condition names, and return a frame in which `jpeg_q50` is really
    an hflip of `clean`."""
    from aigcdet.eval.grid import score_grid
    tta, _ = _build(tmp_path, monkeypatch, VIEWS)
    with pytest.raises(ValueError, match="score_grid_tta"):
        score_grid(_Sum(), tta, device="cpu")


def test_score_grid_tta_refuses_a_plain_bank(tmp_path, monkeypatch):
    """The mirror error, and the worse one: averaging a plain bank's axis
    averages over CONDITIONS, which produces a single robustness-looking number
    that is nothing of the kind."""
    from aigcdet.eval.grid import score_grid_tta
    plain, _ = _build(tmp_path, monkeypatch, None)
    with pytest.raises(ValueError, match="no TTA axis|tta_views"):
        score_grid_tta(_Sum(), plain, device="cpu")


def test_a_bank_whose_width_contradicts_its_axis_is_refused(tmp_path, monkeypatch):
    """Written by a version that disagreed with this one about the flattening.
    Every column index computed from it would be off by a silent amount."""
    from aigcdet.eval.grid import tta_axis
    tta, _ = _build(tmp_path, monkeypatch, VIEWS)
    tta.config["tta_views"] = VIEWS + ["blur_0.3"]
    with pytest.raises(ValueError, match="flattening"):
        tta_axis(tta)


# ===========================================================================
# The evaluation identity
# ===========================================================================

def test_tta_eval_bank_declares_the_reduced_axis_and_the_real_cost(tmp_path, monkeypatch):
    from aigcdet.eval.grid import TtaEvalBank
    tta, _ = _build(tmp_path, monkeypatch, VIEWS)
    ident = TtaEvalBank(tta)
    assert ident.config["n_views"] == 3           # conditions, after averaging
    assert ident.config["physical_n_views"] == 9  # what it actually holds
    assert ident.config["tta_cost_multiplier"] == 3
    assert ident.config["tta_views"] == VIEWS
    assert ident.config["n_images"] == len(tta.meta)


def test_tta_eval_bank_is_comparable_with_a_plain_bank(tmp_path, monkeypatch):
    """The whole point of the reduced axis: A6 must be tabulable beside its own
    base rung. If it is not, A6 is unreportable -- the A5 problem (R43) in a
    different costume."""
    from aigcdet.eval.grid import TtaEvalBank, assert_banks_comparable
    plain, _ = _build(tmp_path, monkeypatch, None, out="plain")
    tta, _ = _build(tmp_path, monkeypatch, VIEWS, out="tta")
    assert_banks_comparable([plain, TtaEvalBank(tta)])


def test_tta_eval_bank_refuses_a_bank_with_no_tta_axis(tmp_path, monkeypatch):
    """Registering a plain bank here would state a cost multiplier of one for a
    rung that paid eight."""
    from aigcdet.eval.grid import TtaEvalBank
    plain, _ = _build(tmp_path, monkeypatch, None)
    with pytest.raises(ValueError, match="cost multiplier"):
        TtaEvalBank(plain)


# ===========================================================================
# Matching the run it is tabulated in
# ===========================================================================

def test_matching_bank_passes(tmp_path, monkeypatch):
    from aigcdet.eval.grid import assert_tta_bank_matches
    plain, _ = _build(tmp_path, monkeypatch, None, out="plain")
    tta, _ = _build(tmp_path, monkeypatch, VIEWS, out="tta")
    assert_tta_bank_matches(plain, tta)


@pytest.mark.parametrize("key, value", [
    ("manifest_sha256", "deadbeef"),
    ("conditions", ["clean", "noise_s0.05", "jpeg_q50"]),   # reordered
    ("backbone", "other"),
    ("canon_policy", {"mode": "band", "nominal_side": 512}),
])
def test_a_tta_bank_from_another_evaluation_is_refused(tmp_path, monkeypatch, key, value):
    """Each of these has the right shape and the wrong meaning, and the score
    it produces lands in the table beside A3 as though the two were
    comparable. The reordered condition list is the subtle one: same names,
    same count, and view j means a different thing in each bank."""
    from aigcdet.eval.grid import assert_tta_bank_matches
    plain, _ = _build(tmp_path, monkeypatch, None, out="plain")
    tta, _ = _build(tmp_path, monkeypatch, VIEWS, out="tta")
    tta.config[key] = value
    with pytest.raises(ValueError, match=key):
        assert_tta_bank_matches(plain, tta)


def test_a_tta_bank_over_different_rows_is_refused(tmp_path, monkeypatch):
    from aigcdet.eval.grid import assert_tta_bank_matches
    plain, _ = _build(tmp_path, monkeypatch, None, n=4, out="plain")
    tta, _ = _build(tmp_path, monkeypatch, VIEWS, n=3, out="tta")
    # A different row count usually changes the manifest fingerprint too, and
    # that check fires first. Matched here so the ROW COUNT branch is the one
    # under test -- a bank can legitimately share a fingerprint and hold fewer
    # rows (a killed extraction, a shard), and that is the case this catches.
    tta.config["manifest_sha256"] = plain.config["manifest_sha256"]
    with pytest.raises(ValueError, match="rows"):
        assert_tta_bank_matches(plain, tta)


def test_a_tta_bank_in_a_different_row_order_is_refused(tmp_path, monkeypatch):
    """Scores are joined positionally, so A6's row would carry another image's
    label -- and both banks would pass every other check in this file."""
    from aigcdet.eval.grid import assert_tta_bank_matches
    plain, _ = _build(tmp_path, monkeypatch, None, out="plain")
    tta, _ = _build(tmp_path, monkeypatch, VIEWS, out="tta")
    tta.meta["row_id"] = tta.meta["row_id"].to_numpy()[::-1]
    with pytest.raises(ValueError, match="different ORDER"):
        assert_tta_bank_matches(plain, tta)
