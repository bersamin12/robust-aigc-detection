"""Rung A5: fusing two independently-trained banks at scoring time.

No GPU, no weights, no downloads: fusion operates on score frames, so every
test here is pure pandas over synthetic frames, plus two tiny banks written
directly with `BankWriter` for the parentage checks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aigcdet.eval.fusion import (
    ALL_ROWS, FIT_SPLITS_FOR_SELECTION, POPULATION_COLUMN, FusedEvalBank,
    assert_fusion_parents, fuse_scores, fused_splits, zscore_by_condition,
)
from aigcdet.eval.metrics import roc_auc
from aigcdet.features.bank import N_FAMILIES, BankWriter, FeatureBank


def _df(seed, scale, n=200, conditions=("clean", "jpeg_q30")):
    rng = np.random.default_rng(seed)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    rows = []
    for cond in conditions:
        rows.append(pd.DataFrame({
            "condition": cond, "image_idx": np.arange(n), "label": y,
            "generator": "g", "source": "s",
            "score": (rng.normal(y * 1.0, 1.0)) * scale,
        }))
    return pd.concat(rows, ignore_index=True)


# --- z-scoring -------------------------------------------------------------

def test_zscore_makes_each_condition_zero_mean_unit_variance():
    z = zscore_by_condition(_df(0, scale=50.0))
    for _, g in z.groupby("condition"):
        assert abs(g["score"].mean()) < 1e-6
        assert abs(g["score"].std(ddof=0) - 1.0) < 1e-6


def _offset_df(n=200):
    """Two conditions on deliberately different score scales AND offsets.

    The offsets are what make a whole-frame z-score distinguishable from a
    per-condition one: standardising the frame as a whole leaves `jpeg_q30`
    sitting 20 units below `clean`, so neither condition ends up at mean zero.
    """
    rng = np.random.default_rng(3)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    rows = []
    for cond, offset, scale in (("clean", 0.0, 1.0), ("jpeg_q30", -20.0, 7.0)):
        rows.append(pd.DataFrame({
            "condition": cond, "image_idx": np.arange(n), "label": y,
            "generator": "g", "source": "s",
            "score": rng.normal(y * 1.0, 1.0) * scale + offset,
        }))
    return pd.concat(rows, ignore_index=True)


def test_the_grouping_is_per_condition_not_over_the_whole_frame():
    """Kills `(score - score.mean()) / score.std()` computed frame-wide.

    A whole-frame z-score is still "standardised" -- it just standardises the
    wrong thing, and the mixture of two offset conditions then leaves each
    condition off-centre, which is exactly the imbalance fusion exists to
    remove. The fixture is built with a 20-unit offset between the conditions
    so the two groupings cannot coincide.
    """
    df = _offset_df()
    whole = df["score"].to_numpy()
    whole = (whole - whole.mean()) / whole.std(ddof=0)
    z = zscore_by_condition(df)

    for _, g in z.groupby("condition"):
        assert abs(g["score"].mean()) < 1e-9
    assert not np.allclose(z["score"].to_numpy(), whole), \
        "zscore_by_condition standardised the whole frame, not each condition"


def test_the_grouping_does_not_include_the_label():
    """Kills `groupby(["condition", "label"])`, which leaks the answer.

    Standardising within class centres BOTH classes on zero, so the fused score
    would carry no class signal at all while still passing every "is it
    standardised?" assertion: a per-condition mean of 0 and a per-condition
    variance of 1 both survive it exactly, because two equal-sized unit-variance
    groups with equal means pool to mean 0 and variance 1.
    """
    df = _offset_df()
    z = zscore_by_condition(df)
    for cond, g in z.groupby("condition"):
        by_class = g.groupby("label")["score"].mean()
        assert by_class[1] - by_class[0] > 0.5, (
            f"condition {cond}: the class separation was standardised away, so "
            "the z-score was computed within (condition, label)")
        # And the ranking, hence every threshold-free metric, is untouched.
        raw = df[df["condition"] == cond]
        assert roc_auc(g["label"], g["score"]) == \
            pytest.approx(roc_auc(raw["label"], raw["score"]))


def test_zscoring_a_scaled_frame_gives_the_same_answer():
    a, b = _df(0, scale=1.0), _df(0, scale=50.0)
    np.testing.assert_allclose(zscore_by_condition(a)["score"].to_numpy(),
                               zscore_by_condition(b)["score"].to_numpy(),
                               atol=1e-9)


def test_a_constant_condition_is_zeroed_rather_than_divided_by_zero():
    df = _df(0, scale=1.0)
    df.loc[df["condition"] == "clean", "score"] = 4.0
    z = zscore_by_condition(df)
    clean = z[z["condition"] == "clean"]["score"].to_numpy()
    assert np.all(np.isfinite(z["score"].to_numpy()))
    np.testing.assert_allclose(clean, 0.0)


def test_zscore_leaves_the_other_columns_and_the_input_alone():
    df = _df(0, scale=50.0)
    before = df["score"].to_numpy().copy()
    z = zscore_by_condition(df)
    np.testing.assert_array_equal(df["score"].to_numpy(), before)
    for col in ("condition", "image_idx", "label", "generator", "source"):
        np.testing.assert_array_equal(z[col].to_numpy(), df[col].to_numpy())


# --- fusion ----------------------------------------------------------------

def test_fusion_is_invariant_to_one_backbones_logit_scale():
    """Without z-scoring, a backbone with 50x larger logits would dominate."""
    a, b = _df(0, scale=1.0), _df(1, scale=50.0)
    fused = fuse_scores([a, b])
    a2, b2 = _df(0, scale=1.0), _df(1, scale=1.0)
    fused2 = fuse_scores([a2, b2])
    clean = fused[fused["condition"] == "clean"]
    clean2 = fused2[fused2["condition"] == "clean"]
    assert abs(roc_auc(clean["label"], clean["score"])
               - roc_auc(clean2["label"], clean2["score"])) < 1e-9


def test_without_zscoring_the_larger_scale_would_have_dominated():
    """Proves the previous test's fixture can express the failure it rules out.

    A raw average of the same two frames is NOT scale-invariant: the 50x
    backbone decides the ranking on its own, and the fused AUC moves. If this
    assertion ever fails, the invariance test above is asserting nothing.
    """
    a, b = _df(0, scale=1.0), _df(1, scale=50.0)
    a2, b2 = _df(0, scale=1.0), _df(1, scale=1.0)

    def raw_mean(x, y):
        out = x.copy()
        out["score"] = (x["score"].to_numpy() + y["score"].to_numpy()) / 2.0
        return out[out["condition"] == "clean"]

    scaled, unscaled = raw_mean(a, b), raw_mean(a2, b2)
    assert abs(roc_auc(scaled["label"], scaled["score"])
               - roc_auc(unscaled["label"], unscaled["score"])) > 1e-3


def test_fusion_of_two_noisy_views_beats_the_BETTER_of_the_two():
    """Against `max`, not `min`.

    `auc_f >= min(auc_a, auc_b)` is satisfied by any output at least as good as
    the WORSE parent -- including one that simply returns the better parent, and
    including several that are worse than it. The two frames here are
    independent noisy views of the same signal, which is the case where
    averaging is supposed to help, so the fused rung must beat BOTH parents by a
    margin the noise cannot account for.
    """
    a, b = _df(0, 1.0), _df(7, 1.0)
    fused = fuse_scores([a, b])
    sel = lambda d: d[d["condition"] == "clean"]           # noqa: E731
    auc_a = roc_auc(sel(a)["label"], sel(a)["score"])
    auc_b = roc_auc(sel(b)["label"], sel(b)["score"])
    auc_f = roc_auc(sel(fused)["label"], sel(fused)["score"])
    assert auc_a != auc_b, "fixture cannot tell `max` from `min`"
    assert auc_f > max(auc_a, auc_b) + 0.02, (auc_a, auc_b, auc_f)


def test_fusion_preserves_row_count_and_keys():
    a, b = _df(0, 1.0), _df(1, 1.0)
    fused = fuse_scores([a, b])
    assert len(fused) == len(a)
    assert set(fused.columns) >= {"condition", "image_idx", "label", "score"}


def test_fusion_preserves_the_condition_order_of_the_first_frame():
    """The robustness table compares rungs on condition ORDER, not just set.

    `report._check_rungs_comparable` rejects two rungs whose frames list the
    same conditions in a different order ("same set, different order"), so a
    fusion that sorted its output would make the A5 row incomparable with every
    other rung in the table.

    The conditions here are deliberately NOT in alphabetical order -- the eval
    grid's are not either. On `("clean", "jpeg_q30")` a sort is a no-op and this
    test asserts nothing at all, which is how the mutation run first found it.
    """
    conditions = ("clean", "blur_s2.0")
    assert sorted(conditions) != list(conditions), \
        "fixture cannot tell a preserved order from a sorted one"
    a, b = _df(0, 1.0, conditions=conditions), _df(1, 1.0, conditions=conditions)
    fused = fuse_scores([a, b])
    order = lambda d: list(dict.fromkeys(d["condition"].tolist()))   # noqa: E731
    assert order(fused) == order(a) == list(conditions)
    np.testing.assert_array_equal(fused["image_idx"].to_numpy(),
                                  a["image_idx"].to_numpy())
    np.testing.assert_array_equal(fused["condition"].to_numpy(),
                                  a["condition"].to_numpy())


def test_fusion_is_not_just_the_first_frame():
    """Would `fuse_scores` still pass if it returned its first input unchanged?

    Every other assertion here is satisfied by `return dfs[0]`, so the fused
    score is compared against both parents directly.
    """
    a, b = _df(0, 1.0), _df(7, 1.0)
    fused = fuse_scores([a, b])
    for parent in (a, b):
        assert not np.allclose(fused["score"].to_numpy(),
                               parent["score"].to_numpy())
    expected = (zscore_by_condition(a)["score"].to_numpy()
                + zscore_by_condition(b)["score"].to_numpy()) / 2.0
    np.testing.assert_allclose(fused["score"].to_numpy(), expected, atol=1e-12)


def test_weights_are_honoured():
    a, b = _df(0, 1.0), _df(1, 1.0)
    only_a = fuse_scores([a, b], weights=[1.0, 0.0])
    za = zscore_by_condition(a).sort_values(["condition", "image_idx"])
    of = only_a.sort_values(["condition", "image_idx"])
    np.testing.assert_allclose(of["score"].to_numpy(), za["score"].to_numpy(),
                               atol=1e-9)


def test_weights_are_normalised_not_taken_literally():
    a, b = _df(0, 1.0), _df(7, 1.0)
    np.testing.assert_allclose(fuse_scores([a, b], weights=[2.0, 2.0])["score"],
                               fuse_scores([a, b])["score"], atol=1e-12)


def test_a_wrong_length_weight_vector_is_rejected():
    a, b = _df(0, 1.0), _df(1, 1.0)
    with pytest.raises(ValueError, match="weights must match"):
        fuse_scores([a, b], weights=[1.0])


def test_fusing_nothing_is_rejected():
    with pytest.raises(ValueError, match="nothing to fuse"):
        fuse_scores([])


def test_mismatched_frames_are_rejected():
    a = _df(0, 1.0)
    b = _df(1, 1.0).iloc[:10]
    with pytest.raises(ValueError, match="same rows"):
        fuse_scores([a, b])


def test_frames_that_disagree_on_a_rows_label_are_rejected():
    """Same keys, different truth. `fuse_scores` builds its output from the
    first frame, so a label disagreement would be resolved silently in favour
    of whichever bank happened to be passed first -- and every metric computed
    downstream would then be scored against one parent's labels."""
    a, b = _df(0, 1.0), _df(1, 1.0)
    b.loc[0, "label"] = 1 - int(b.loc[0, "label"])
    with pytest.raises(ValueError, match="disagree on the label"):
        fuse_scores([a, b])


# --- C2: which rows set the fusion weights ---------------------------------
#
# The ablation tier's bank is 5k internal validation plus a 5k subsample of the
# organisers' benchmark (spec §4.4a). Standardising over the whole frame lets
# the benchmark rows set half of each parent's sigma, and sigma_1/sigma_2 is
# exactly how much each backbone contributes -- so A5's §6.4 selection number
# becomes a function of how the demo set happens to spread. The fixtures below
# make that reachable (the two heads separate the benchmark differently, which
# §6.5 expects) and then assert the fix removes it.

_N_SELECTION, _N_BENCHMARK = 200, 400
_FIT = FIT_SPLITS_FOR_SELECTION


def _selection_splits():
    """A bank's split column: a §6.4 selection population plus benchmark rows."""
    half = _N_SELECTION // 2
    split = np.array(["val_internal"] * half + ["heldout_generator"] * half
                     + ["benchmark"] * _N_BENCHMARK)
    label = np.array([0] * half + [1] * half + [0, 1] * (_N_BENCHMARK // 2))
    return split, label


def _tiered_df(seed, benchmark_separation, conditions=("clean", "jpeg_q30")):
    """One head's scores over a bank that holds both tiers' rows.

    `benchmark_separation` is how well this head separates the benchmark half.
    The selection rows are drawn identically whatever it is, so any change in a
    selection-population metric came from the benchmark rows and nothing else.
    """
    split, label = _selection_splits()
    rng = np.random.default_rng(seed)
    rows = []
    for cond in conditions:
        separation = np.where(split == "benchmark", benchmark_separation, 1.0)
        rows.append(pd.DataFrame({
            "condition": cond, "image_idx": np.arange(len(label)),
            "label": label, "generator": "g", "source": "s",
            "score": rng.normal(label * separation, 1.0),
        }))
    return pd.concat(rows, ignore_index=True)


def _on_selection_rows(fused, split):
    return fused["score"].to_numpy()[np.isin(split[fused["image_idx"].to_numpy()],
                                             list(_FIT))]


def test_the_benchmark_rows_do_not_set_a_declared_fusions_weights():
    """The C2 guarantee, asserted where it bites: A5's selection metric.

    Same two heads, same selection rows; only how well each separates the
    organisers' benchmark varies. With the population declared, the fused score
    on every selection row is bit-identical and so is the §6.4 selection metric.
    """
    from aigcdet.eval.errors import heldout_robust_tpr
    split, _ = _selection_splits()
    baseline = fuse_scores([_tiered_df(0, 1.0), _tiered_df(7, 1.0)],
                           splits=split, fit_splits=_FIT)
    for separation in (3.0, 6.0, 10.0, 25.0):
        fused = fuse_scores([_tiered_df(0, separation), _tiered_df(7, 1.0)],
                            splits=split, fit_splits=_FIT)
        np.testing.assert_allclose(_on_selection_rows(fused, split),
                                   _on_selection_rows(baseline, split), atol=1e-12)
        assert heldout_robust_tpr(fused, split) == \
            heldout_robust_tpr(baseline, split)


def test_without_a_declared_population_the_benchmark_rows_move_the_selection_metric():
    """Proves the fixture above can express the failure it rules out.

    This is the defect, run: the whole-frame fit is the shipped default, and
    under it the §6.4 selection metric of an unchanged pair of heads on an
    unchanged selection population moves because the demo set's spread moved.
    If this ever stops failing to be invariant, the test above asserts nothing.
    """
    from aigcdet.eval.errors import heldout_robust_tpr
    split, _ = _selection_splits()
    baseline = fuse_scores([_tiered_df(0, 1.0), _tiered_df(7, 1.0)])
    moved = [heldout_robust_tpr(
        fuse_scores([_tiered_df(0, s), _tiered_df(7, 1.0)]), split)
        for s in (3.0, 6.0, 10.0, 25.0)]
    assert any(m != heldout_robust_tpr(baseline, split) for m in moved), moved
    scaled = fuse_scores([_tiered_df(0, 25.0), _tiered_df(7, 1.0)])
    assert np.abs(_on_selection_rows(scaled, split)
                  - _on_selection_rows(baseline, split)).max() > 0.1


def test_a_declared_fit_still_standardises_and_returns_every_row():
    """Restricting the FIT is not dropping rows: the fused frame covers the
    whole bank, because the robustness table is built over all of it."""
    split, _ = _selection_splits()
    a, b = _tiered_df(0, 8.0), _tiered_df(7, 1.0)
    fused = fuse_scores([a, b], splits=split, fit_splits=_FIT)
    assert len(fused) == len(a)
    np.testing.assert_array_equal(fused["image_idx"].to_numpy(),
                                  a["image_idx"].to_numpy())
    assert fused["score"].notna().all()
    # The fit rows are what came out standardised; the benchmark rows did not,
    # which is the whole point -- they were measured against the selection
    # population's scale rather than setting it.
    fit_rows = np.isin(split[fused["image_idx"].to_numpy()], list(_FIT))
    for cond in ("clean", "jpeg_q30"):
        here = (fused["condition"] == cond).to_numpy() & fit_rows
        assert abs(float(fused["score"].to_numpy()[here].mean())) < 0.35


def test_the_zscore_population_is_recorded_on_the_output():
    """The fused score is not a fixed function of the two heads without it."""
    split, _ = _selection_splits()
    a, b = _tiered_df(0, 8.0), _tiered_df(7, 1.0)
    declared = fuse_scores([a, b], splits=split, fit_splits=_FIT)
    assert set(declared[POPULATION_COLUMN]) == \
        {"split=heldout_generator+val_internal"}
    assert declared.attrs[POPULATION_COLUMN] == \
        "split=heldout_generator+val_internal"
    undeclared = fuse_scores([a, b])
    assert set(undeclared[POPULATION_COLUMN]) == {ALL_ROWS}
    assert undeclared.attrs[POPULATION_COLUMN] == ALL_ROWS
    assert ALL_ROWS != "split=heldout_generator+val_internal"


def test_the_fit_population_for_selection_is_the_selection_population():
    """`FIT_SPLITS_FOR_SELECTION` must not drift from `errors.SELECTION_SPLITS`.

    They are the same rows on purpose: the rows that decide how much each
    backbone contributes are the rows the selection metric is read on, and the
    organisers' benchmark subsample is in neither.
    """
    from aigcdet.eval.errors import SELECTION_SPLITS
    assert FIT_SPLITS_FOR_SELECTION == tuple(SELECTION_SPLITS)
    assert "benchmark" not in FIT_SPLITS_FOR_SELECTION


def test_splits_and_fit_splits_must_be_given_together():
    split, _ = _selection_splits()
    a, b = _tiered_df(0, 1.0), _tiered_df(7, 1.0)
    with pytest.raises(ValueError, match="go together"):
        fuse_scores([a, b], splits=split)
    with pytest.raises(ValueError, match="go together"):
        fuse_scores([a, b], fit_splits=_FIT)
    with pytest.raises(ValueError, match="fit_splits is empty"):
        fuse_scores([a, b], splits=split, fit_splits=[])


def test_a_declared_population_the_bank_does_not_contain_is_refused():
    """Otherwise the recorded population names rows that were never there."""
    split, _ = _selection_splits()
    a, b = _tiered_df(0, 1.0), _tiered_df(7, 1.0)
    with pytest.raises(ValueError, match="does not contain"):
        fuse_scores([a, b], splits=split, fit_splits=("val_internal", "test_ood"))


def test_a_condition_with_no_row_in_the_fit_population_is_refused():
    """Standardising a condition on another condition's rows is not
    standardising it; a NaN column would propagate through every metric."""
    split, _ = _selection_splits()
    a, b = _tiered_df(0, 1.0), _tiered_df(7, 1.0)
    fit = np.isin(split, list(_FIT)) & (np.arange(len(split)) < 0)
    with pytest.raises(ValueError, match="nothing to fit"):
        zscore_by_condition(a, np.zeros(len(a), dtype=bool))
    lopsided = np.isin(split[a["image_idx"].to_numpy()], list(_FIT)) & \
        (a["condition"].to_numpy() == "clean")
    with pytest.raises(ValueError, match="no row in the declared"):
        zscore_by_condition(a, lopsided)
    assert not fit.any()


def test_a_fit_mask_of_the_wrong_length_is_refused():
    a = _df(0, 1.0)
    with pytest.raises(ValueError, match="one boolean per row"):
        zscore_by_condition(a, np.ones(len(a) - 1, dtype=bool))


def test_split_column_shorter_than_the_frames_image_idx_is_refused():
    """`splits` is positionally indexed by image_idx, as in `errors`."""
    split, _ = _selection_splits()
    a, b = _tiered_df(0, 1.0), _tiered_df(7, 1.0)
    truncated = split[:_N_SELECTION - 50]          # still holds both fit splits
    assert set(truncated) == set(_FIT)
    with pytest.raises(ValueError, match="positionally indexed by image_idx"):
        fuse_scores([a, b], splits=truncated, fit_splits=_FIT)


def test_the_fit_is_matched_on_image_idx_and_not_on_row_order():
    """A frame listed in another order must get the same fused score.

    `_aligned` already matches rows on keys rather than position; the fit mask
    has to be derived the same way or a shuffled parent would be standardised
    against another row's split.
    """
    split, _ = _selection_splits()
    a, b = _tiered_df(0, 8.0), _tiered_df(7, 1.0)
    shuffled = b.sample(frac=1.0, random_state=5).reset_index(drop=True)
    straight = fuse_scores([a, b], splits=split, fit_splits=_FIT)
    mixed = fuse_scores([a, shuffled], splits=split, fit_splits=_FIT)
    np.testing.assert_allclose(mixed["score"].to_numpy(),
                               straight["score"].to_numpy(), atol=1e-12)


# --- which bank's splits apply to a fused frame ----------------------------

DIM = 4


def _bank(tmp_path, name, fingerprint="eb", backbone="fake", n=8,
          splits=None, labels=None, conditions=("clean", "jpeg_q30")):
    out = str(tmp_path / name)
    w = BankWriter(out, n, len(conditions), DIM, backbone, 0,
                   manifest_sha256=fingerprint,
                   extra_config={"conditions": list(conditions)})
    for i in range(n):
        label = (i % 2) if labels is None else int(labels[i])
        split = "val_internal" if splits is None else splits[i]
        w.write_image(i, {"path": f"/b{i}.png", "label": label,
                          "generator": "g", "source": "s", "split": split},
                      feats=np.zeros((len(conditions), DIM), np.float32),
                      presence=np.zeros((len(conditions), N_FAMILIES), np.float32),
                      severity=np.zeros((len(conditions), N_FAMILIES), np.float32),
                      proxies=np.zeros((len(conditions), 3), np.float32),
                      recipes=["[]"] * len(conditions))
    w.close()
    return FeatureBank.open(out)


def test_fused_splits_returns_the_shared_split_column(tmp_path):
    splits = ["val_internal", "heldout_generator"] * 4
    a = _bank(tmp_path, "a", splits=splits)
    b = _bank(tmp_path, "b", splits=splits, backbone="other")
    np.testing.assert_array_equal(fused_splits([a, b]), np.asarray(splits))


def test_fused_splits_refuses_parents_from_different_manifests(tmp_path):
    a = _bank(tmp_path, "a", fingerprint="one")
    b = _bank(tmp_path, "b", fingerprint="two")
    with pytest.raises(ValueError, match="different frozen manifests"):
        fused_splits([a, b])


def test_fused_splits_refuses_a_parent_with_no_manifest_fingerprint(tmp_path):
    a = _bank(tmp_path, "a", fingerprint=None)
    b = _bank(tmp_path, "b", fingerprint=None)
    with pytest.raises(ValueError, match="records no manifest_sha256"):
        fused_splits([a, b])


def test_fused_splits_refuses_parents_whose_split_columns_disagree(tmp_path):
    """Two banks over the same manifest fingerprint whose split columns still
    differ: the fingerprint covers the path column, so a re-split that kept the
    paths would pass it. Whose splits apply is then genuinely undefined, and
    scoring A5 against either parent's would label half its rows wrongly."""
    a = _bank(tmp_path, "a", splits=["val_internal"] * 8)
    b = _bank(tmp_path, "b", splits=["val_internal"] * 7 + ["benchmark"])
    with pytest.raises(ValueError, match="disagree on the split"):
        fused_splits([a, b])


def test_fusion_parents_must_share_the_condition_axis(tmp_path):
    a = _bank(tmp_path, "a")
    b = _bank(tmp_path, "b", conditions=("clean", "blur_s2.0"))
    with pytest.raises(ValueError, match="condition"):
        assert_fusion_parents([a, b])


def test_fusion_parents_may_differ_on_the_backbone(tmp_path):
    """A5 IS the two-backbone rung (spec §6.4: "+ second backbone"), so a
    differing backbone is the treatment, not a confound, and the fusion itself
    must not refuse it. Whether the resulting row may sit in the same
    robustness table as a single-backbone rung is a separate question, decided
    by `eval.grid.assert_banks_comparable` over `FusedEvalBank.config`."""
    a = _bank(tmp_path, "a", backbone="dinov3l")
    b = _bank(tmp_path, "b", backbone="siglip2l")
    assert_fusion_parents([a, b])


def test_a_single_parent_is_not_a_fusion(tmp_path):
    with pytest.raises(ValueError, match="at least two"):
        assert_fusion_parents([_bank(tmp_path, "a")])


# --- the fused rung's evaluation identity ----------------------------------

def test_the_fused_bank_reports_both_parents_and_the_shared_axis(tmp_path):
    a = _bank(tmp_path, "a", backbone="dinov3l")
    b = _bank(tmp_path, "b", backbone="siglip2l")
    fused = FusedEvalBank([a, b])
    assert fused.config["conditions"] == ["clean", "jpeg_q30"]
    assert fused.config["n_views"] == 2
    assert fused.config["manifest_sha256"] == "eb"
    assert fused.config["fused_from"] == [a.path, b.path]
    assert a.path in fused.path and b.path in fused.path


def test_the_fused_banks_backbone_names_every_backbone_that_produced_it(tmp_path):
    """Kills the fused bank that borrows one parent's backbone name.

    `assert_banks_comparable` compares rungs on `backbone`, so reporting
    `dinov3l` for a row that is half SigLIP2 would let a two-backbone row into
    a single-backbone table unremarked -- the R24 confound, laundered through a
    label. When the parents agree the composite collapses to the one true name.
    """
    a = _bank(tmp_path, "a", backbone="dinov3l")
    b = _bank(tmp_path, "b", backbone="siglip2l")
    assert FusedEvalBank([a, b]).config["backbone"] == "dinov3l+siglip2l"

    same = _bank(tmp_path, "c", backbone="dinov3l")
    assert FusedEvalBank([a, same]).config["backbone"] == "dinov3l"


def test_a_declared_cross_backbone_fused_row_may_share_the_table(tmp_path):
    """R43: the spec's A5 is DINOv3 + SigLIP2 fused (§6.4).

    With `backbone` compared for equality across every bank, that row is
    refused a place beside any single-backbone rung -- so the A5 the spec
    describes could not appear in the results at all. A DECLARED composite is
    exempt from that key; its parents are already forced to agree on every
    other one by `assert_fusion_parents`.
    """
    from aigcdet.eval.grid import assert_banks_comparable
    a = _bank(tmp_path, "a", backbone="dinov3l")
    b = _bank(tmp_path, "b", backbone="siglip2l")
    fused = FusedEvalBank([a, b])
    assert fused.config["backbone"] == "dinov3l+siglip2l"
    assert_banks_comparable([a, fused])
    assert_banks_comparable([fused, a])
    same = _bank(tmp_path, "c", backbone="dinov3l")
    assert_banks_comparable([a, FusedEvalBank([a, same])])


def test_the_exemption_is_only_for_the_backbone_key(tmp_path):
    """A composite still has to have been the same evaluation.

    The exemption is for the treatment under test (a second backbone), not for
    the condition axis, the manifest or the row set -- those are what make two
    rungs a model comparison rather than a comparison of evaluations.
    """
    from aigcdet.eval.grid import assert_banks_comparable
    a = _bank(tmp_path, "a", backbone="dinov3l")
    b = _bank(tmp_path, "b", backbone="siglip2l")
    other_axis = _bank(tmp_path, "d", conditions=("clean", "blur_s2.0"))
    with pytest.raises(ValueError, match="not comparable"):
        assert_banks_comparable([other_axis, FusedEvalBank([a, b])])
    other_manifest = _bank(tmp_path, "e", fingerprint="different")
    with pytest.raises(ValueError, match="not comparable"):
        assert_banks_comparable([other_manifest, FusedEvalBank([a, b])])


def test_a_composite_that_borrows_one_parents_name_is_refused(tmp_path):
    """The exemption is granted to the declaration, not to the row.

    A row claiming to be fused from DINOv3 and SigLIP2 while calling itself
    `dinov3l` would take the exemption AND hide half of what produced it --
    the R24 confound wearing a label. Kills the mutant that grants the
    exemption on the presence of `fused_from` alone.
    """
    from aigcdet.eval.grid import assert_banks_comparable
    a = _bank(tmp_path, "a", backbone="dinov3l")
    b = _bank(tmp_path, "b", backbone="siglip2l")
    liar = FusedEvalBank([a, b])
    liar.config["backbone"] = "dinov3l"
    with pytest.raises(ValueError, match="borrows a parent's name"):
        assert_banks_comparable([a, liar])
    half = FusedEvalBank([a, b])
    half.config["fused_backbones"] = ["dinov3l"]
    with pytest.raises(ValueError, match="at least two parents"):
        assert_banks_comparable([a, half])
    unsaid = FusedEvalBank([a, b])
    unsaid.config["fused_from"] = None
    with pytest.raises(ValueError, match="does not say what from"):
        assert_banks_comparable([a, unsaid])


def test_a_composite_cannot_bridge_two_single_backbone_rungs(tmp_path):
    """The hole the exemption would open if it were checked against bank 0.

    `assert_banks_comparable` compares every bank against the first. If the
    first is a composite and the backbone key is simply skipped for the pair, a
    DINOv3 rung and a SigLIP2 rung both pass -- and a table comparing two
    different embeddings is the R24 confound the guard exists to stop.
    """
    from aigcdet.eval.grid import assert_banks_comparable
    a = _bank(tmp_path, "a", backbone="dinov3l")
    b = _bank(tmp_path, "b", backbone="siglip2l")
    fused = FusedEvalBank([a, b])
    with pytest.raises(ValueError, match="not comparable"):
        assert_banks_comparable([fused, a, b])
    with pytest.raises(ValueError, match="not comparable"):
        assert_banks_comparable([a, b, fused])


# --- C3: the fused row in the robustness table -----------------------------

def _scores_for(bank, seed=0):
    """A score frame covering exactly the rows and conditions of `bank`."""
    rng = np.random.default_rng(seed)
    meta = bank.meta
    rows = []
    for cond in bank.config["conditions"]:
        rows.append(pd.DataFrame({
            "condition": cond, "image_idx": meta["image_idx"].to_numpy(),
            "label": meta["label"].to_numpy(), "generator": "g", "source": "s",
            "score": rng.normal(meta["label"].to_numpy() * 1.0, 1.0),
        }))
    return pd.concat(rows, ignore_index=True)


def test_the_fused_bank_states_how_many_images_it_holds(tmp_path):
    """Without `n_images`, `_check_banks` SKIPS its bank-size check.

    That check is what stops a 5k selection-tier bank sitting beside a 13.8k
    final-report score frame. A config that omits the key made the A5 row the
    one row in the table exempt from it -- and the table still reported
    `banks_verified = True`.
    """
    a = _bank(tmp_path, "a", backbone="dinov3l", n=8)
    b = _bank(tmp_path, "b", backbone="siglip2l", n=8)
    assert FusedEvalBank([a, b]).config["n_images"] == 8


def test_a_bank_that_will_not_say_how_many_images_it_holds_is_refused(tmp_path):
    """C3: absent `n_images` must RAISE, not skip -- the same reasoning
    `_check_banks` already applies to an absent `manifest_sha256`."""
    from aigcdet.eval.report import robustness_table
    a = _bank(tmp_path, "a", backbone="dinov3l", n=8)
    b = _bank(tmp_path, "b", backbone="siglip2l", n=8)
    fused = FusedEvalBank([a, b])
    scores = {"a3": _scores_for(a, seed=1), "a5": _scores_for(a, seed=2)}
    banks = {"a3": a, "a5": fused}
    robustness_table(scores, tier="smoke", n_boot=8, banks=banks)  # the control

    del fused.config["n_images"]
    with pytest.raises(ValueError, match="records no n_images"):
        robustness_table(scores, tier="smoke", n_boot=8, banks=banks)


def test_a_fused_row_from_the_wrong_tiers_bank_is_refused(tmp_path):
    """The check the missing `n_images` was skipping, run on a fused row.

    A four-image fused bank against a score frame covering twenty images is two
    different evaluation tiers in one table; before `n_images` reached
    `FusedEvalBank`, this was accepted and rendered with no banner.
    """
    from aigcdet.eval.report import robustness_table
    a = _bank(tmp_path, "a", backbone="dinov3l", n=4)
    b = _bank(tmp_path, "b", backbone="siglip2l", n=4)
    big = _bank(tmp_path, "big", backbone="dinov3l", n=20)
    scores = {"a3": _scores_for(big, seed=1), "a5": _scores_for(big, seed=2)}
    with pytest.raises(ValueError, match="do not belong to these scores"):
        robustness_table(scores, tier="smoke", n_boot=8,
                         banks={"a3": big, "a5": FusedEvalBank([a, b])})


def test_the_fused_bank_refuses_a_parent_that_misstates_its_own_row_count(tmp_path):
    """The fused row's size is what `_check_banks` compares against the scores,
    so it cannot be stated from a bank that disagrees with itself."""
    a = _bank(tmp_path, "a", backbone="dinov3l", n=8)
    b = _bank(tmp_path, "b", backbone="siglip2l", n=8)
    b.config["n_images"] = 9
    with pytest.raises(ValueError, match="but holds 8 metadata row"):
        FusedEvalBank([a, b])


def test_the_fused_bank_says_what_it_is():
    """`repr` shows up in `_check_banks`'s own error messages."""
    class _Stub:
        path = "banks/one"
        config = {"n_views": 2, "conditions": ["clean", "x"],
                  "manifest_sha256": "m", "backbone": "b"}
        meta = pd.DataFrame({"image_idx": [0], "split": ["val_internal"],
                             "label": [0]})

    a, b = _Stub(), _Stub()
    b.path = "banks/two"
    assert repr(FusedEvalBank([a, b])) == \
        "FusedEvalBank('fused(banks/one, banks/two)')"
