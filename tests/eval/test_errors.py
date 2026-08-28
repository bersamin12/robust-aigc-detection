"""Tests for the §6.4 selection rule and the §6.6 error-analysis helpers.

Everything here is hermetic: no GPU, no weights, no downloads, and every file
written lands under `tmp_path`.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from aigcdet.eval.errors import (
    ELIGIBLE_RUNGS, SELECTION_METRIC, SELECTION_POPULATION, SELECTION_SPLITS,
    IneligibleRungWarning, contact_sheet, fp_rate_by_source,
    heldout_robust_tpr, select_headline, selection_report, top_errors,
)


def _scores():
    rng = np.random.default_rng(0)
    n = 200
    y = np.array([0] * 100 + [1] * 100)
    s = np.concatenate([rng.normal(-1, 1, 100), rng.normal(1, 1, 100)])
    return pd.DataFrame({"condition": "clean", "image_idx": np.arange(n),
                         "label": y, "score": s, "generator": "g",
                         "source": ["a"] * 100 + ["b"] * 100,
                         "path": [f"/img/{i}.png" for i in range(n)]})


# --- top_errors ------------------------------------------------------------

def test_top_false_positives_are_authentic_images_with_high_scores():
    fp = top_errors(_scores(), k=10, kind="fp")
    assert len(fp) == 10
    assert (fp["label"] == 0).all()
    assert fp["score"].is_monotonic_decreasing


def test_top_false_negatives_are_generated_images_with_low_scores():
    fn = top_errors(_scores(), k=10, kind="fn")
    assert (fn["label"] == 1).all()
    assert fn["score"].is_monotonic_increasing


def test_top_errors_are_the_globally_most_confident_mistakes():
    """Not merely sorted -- the right rows.

    Kills a mutant that takes the head of the pool and sorts THAT (a plausible
    `pool.head(k).sort_values(...)`), which yields a monotonic frame of the
    wrong images and passes both tests above.
    """
    df = _scores()
    fp = top_errors(df, k=10, kind="fp")
    authentic = df[df["label"] == 0]["score"]
    assert set(fp["score"]) == set(authentic.nlargest(10))
    fn = top_errors(df, k=10, kind="fn")
    assert set(fn["score"]) == set(df[df["label"] == 1]["score"].nsmallest(10))


def test_top_errors_break_score_ties_deterministically():
    """Identical data in a different row order must give an identical sheet.

    Kills the mutant that drops `TIE_BREAK_COLUMNS` from the sort (equivalently
    `nlargest`/`nsmallest`, whose tie-breaking follows input row order): with
    every score tied, the un-tie-broken version returns whichever rows happened
    to come first in the frame it was handed.
    """
    df = pd.DataFrame({"condition": "clean", "image_idx": np.arange(8),
                       "label": 0, "score": 1.0, "source": "a",
                       "path": [f"/img/{i}.png" for i in range(8)]})
    shuffled = df.iloc[[5, 2, 7, 0, 3, 6, 1, 4]].reset_index(drop=True)
    first = top_errors(df, k=4, kind="fp")
    second = top_errors(shuffled, k=4, kind="fp")
    assert first["image_idx"].tolist() == [0, 1, 2, 3]
    assert second["image_idx"].tolist() == first["image_idx"].tolist()


def test_top_errors_returns_what_exists_when_k_exceeds_the_pool():
    assert len(top_errors(_scores(), k=500, kind="fp")) == 100


def test_top_errors_rejects_an_unknown_kind_and_a_useless_k():
    with pytest.raises(ValueError, match="kind must be"):
        top_errors(_scores(), kind="fpr")
    with pytest.raises(ValueError, match="k must be at least 1"):
        top_errors(_scores(), k=0)


# --- fp_rate_by_source -----------------------------------------------------

def test_fp_rate_by_source_reports_every_source():
    out = fp_rate_by_source(_scores(), threshold=0.0)
    assert set(out["source"]) == {"a", "b"}
    defined = out["fp_rate"].dropna()
    assert ((defined >= 0) & (defined <= 1)).all()


def test_a_source_with_no_authentic_images_gets_nan_not_zero():
    """`_scores()` puts every authentic image in source "a" and every generated
    one in source "b", so "b" has an empty denominator.

    Kills two mutants at once: the brief's `scores_df[label == 0].groupby(...)`,
    which drops "b" from the table entirely, and a `fillna(0.0)` that would
    report source "b" as never producing a false positive.
    """
    out = fp_rate_by_source(_scores(), threshold=0.0).set_index("source")
    assert out.loc["b", "n_authentic"] == 0
    assert np.isnan(out.loc["b", "fp_rate"])
    assert out.loc["a", "n_authentic"] == 100
    assert out.loc["a", "fp_rate"] == out.loc["a", "n_fp"] / 100


def test_fp_rate_counts_a_score_exactly_at_the_threshold_as_a_false_positive():
    """`>=`, matching `calibrate.policy.decide`. Kills the `>` mutant."""
    df = pd.DataFrame({"condition": "clean", "image_idx": [0, 1],
                       "label": [0, 0], "score": [0.5, 0.4], "source": ["a", "a"]})
    out = fp_rate_by_source(df, threshold=0.5).set_index("source")
    assert out.loc["a", "n_fp"] == 1


def test_fp_rate_by_source_ignores_generated_rows_in_the_numerator():
    """Generated images scored above the threshold are TRUE positives.

    Kills the mutant that drops the `label == 0` term from the `_fp` mask,
    which would report a source of confidently-detected fakes as a source of
    false positives.
    """
    df = pd.DataFrame({"condition": "clean", "image_idx": [0, 1, 2, 3],
                       "label": [0, 1, 1, 1], "score": [-1.0, 9.0, 9.0, 9.0],
                       "source": "a"})
    out = fp_rate_by_source(df, threshold=0.0).set_index("source")
    assert out.loc["a", "n_fp"] == 0
    assert out.loc["a", "fp_rate"] == 0.0
    assert out.loc["a", "n_images"] == 4


# --- select_headline -------------------------------------------------------

def test_select_headline_uses_robust_tpr_on_heldout_generators():
    """A4 has the best clean AUC and must still lose.

    Kills the mutant that reads `clean_auc` (or `val_auc`) instead of
    SELECTION_METRIC -- the exact substitution C-C warns about.
    """
    results = {
        "a3": {"heldout_robust_tpr_at_1pct": 0.71, "clean_auc": 0.99},
        "a4": {"heldout_robust_tpr_at_1pct": 0.66, "clean_auc": 0.999},  # better clean
        "a5": {"heldout_robust_tpr_at_1pct": 0.74, "clean_auc": 0.95},
    }
    assert select_headline(results) == "a5"


def test_select_headline_ignores_rungs_outside_a3_to_a6():
    """A0 wins the metric outright and is still not the headline.

    The exclusion is asserted as a warning as well as an outcome: a silent
    filter and a bug that dropped a rung look identical from the return value.
    """
    results = {
        "a0": {"heldout_robust_tpr_at_1pct": 0.99, "clean_auc": 0.99},
        "a3": {"heldout_robust_tpr_at_1pct": 0.60, "clean_auc": 0.90},
    }
    with pytest.warns(IneligibleRungWarning, match="a0"):
        assert select_headline(results) == "a3"


def test_select_headline_is_quiet_when_no_control_outscores_the_winner():
    """The warning must fire on the finding, not on every call with a control
    present -- otherwise it is noise and gets filtered out wholesale."""
    results = {
        "a0": {"heldout_robust_tpr_at_1pct": 0.40},
        "a3": {"heldout_robust_tpr_at_1pct": 0.60},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert select_headline(results) == "a3"


def test_select_headline_raises_when_no_eligible_rung_present():
    with pytest.raises(ValueError, match="no eligible"):
        select_headline({"a0": {"heldout_robust_tpr_at_1pct": 0.9, "clean_auc": 0.9}})


def test_select_headline_refuses_a_result_that_carries_only_val_auc():
    """A rung whose dict has AUCs but not the selection metric is an error,
    never a fallback.

    Kills the mutant `results[k].get(SELECTION_METRIC, results[k]["val_auc"])`,
    which would silently select on the clean-view AUC.
    """
    results = {"a3": {"val_auc": 0.99, "val_auc_mean_views": 0.90},
               "a4": {"val_auc": 0.80, "val_auc_mean_views": 0.95}}
    with pytest.raises(ValueError, match="val_auc.*NOT substitutes|no "
                                         f"{SELECTION_METRIC}"):
        select_headline(results)


def test_select_headline_refuses_a_declared_benchmark_population():
    results = {"a3": {SELECTION_METRIC: 0.7, "population": "benchmark"},
               "a4": {SELECTION_METRIC: 0.6, "population": "benchmark"}}
    with pytest.raises(ValueError, match="declares population"):
        select_headline(results)


def test_select_headline_accepts_the_population_run_ablation_declares():
    results = {"a3": {SELECTION_METRIC: 0.7, "population": SELECTION_POPULATION,
                      "splits": list(SELECTION_SPLITS)}}
    assert select_headline(results) == "a3"


def test_select_headline_refuses_a_nan_metric():
    """NaN loses every `>` comparison, so a rung whose metric failed to compute
    would be silently ranked last rather than reported as uncomputed."""
    results = {"a3": {SELECTION_METRIC: float("nan")},
               "a4": {SELECTION_METRIC: 0.5}}
    with pytest.raises(ValueError, match="non-finite"):
        select_headline(results)


def test_select_headline_breaks_ties_by_rung_name():
    """Same numbers, different insertion order, same answer."""
    tied = {SELECTION_METRIC: 0.7}
    assert select_headline({"a5": dict(tied), "a3": dict(tied)}) == "a3"
    assert select_headline({"a3": dict(tied), "a5": dict(tied)}) == "a3"


def test_rung_names_are_matched_case_insensitively():
    """`A5` is a candidate. Kills a mutant that filters on the literal keys and
    would drop an upper-cased rung name into the ineligible pile."""
    results = {"A5": {SELECTION_METRIC: 0.8}, "a3": {SELECTION_METRIC: 0.7}}
    assert select_headline(results) == "A5"


def test_eligible_rungs_are_exactly_a3_to_a6():
    assert ELIGIBLE_RUNGS == ("a3", "a4", "a5", "a6")


# --- selection_report ------------------------------------------------------

def test_selection_report_records_the_rule_the_population_and_the_exclusions():
    results = {"a0": {SELECTION_METRIC: 0.9}, "a3": {SELECTION_METRIC: 0.6},
               "a4": {SELECTION_METRIC: 0.7}}
    with pytest.warns(IneligibleRungWarning):
        report = selection_report(results)
    assert report["headline"] == "a4"
    assert report["metric"] == SELECTION_METRIC
    assert report["population"] == SELECTION_POPULATION
    assert report["excluded_as_ineligible"] == {"a0": 0.9}
    assert report["candidates"] == {"a3": 0.6, "a4": 0.7}
    assert "6.4" in report["rule"] and "1% FPR" in report["rule"]


def test_selection_report_records_a_failed_selection_instead_of_raising():
    report = selection_report({"a0": {SELECTION_METRIC: 0.9}})
    assert report["headline"] is None
    assert "no eligible" in report["headline_error"]
    assert report["summary"] == {"a0": {SELECTION_METRIC: 0.9}}


# --- heldout_robust_tpr ----------------------------------------------------

def _grid_scores(n=40, conditions=("clean", "jpeg_q30", "blur_s2.0"), seed=0,
                 separation=(2.0, 1.0, 0.2)):
    """A score frame over `n` bank rows and a few conditions.

    Rows 0..n/4 are val_internal authentic, n/4..n/2 val_internal generated
    (SEEN generators), n/2..3n/4 heldout_generator generated, 3n/4..n benchmark
    authentic. `splits` is returned alongside, as the eval bank would hold it.
    """
    q = n // 4
    splits = np.array(["val_internal"] * q + ["val_internal"] * q
                      + ["heldout_generator"] * q + ["benchmark"] * q)
    label = np.array([0] * q + [1] * q + [1] * q + [0] * q)
    rng = np.random.default_rng(seed)
    frames = []
    for cond, sep in zip(conditions, separation):
        frames.append(pd.DataFrame({
            "condition": cond, "image_idx": np.arange(n), "label": label,
            "generator": "g", "source": "s",
            "score": rng.normal(label * sep, 0.5, n)}))
    return pd.concat(frames, ignore_index=True), splits


def test_heldout_robust_tpr_ignores_benchmark_rows_entirely():
    """The external demo set must not reach the selection metric.

    Kills the brief's `sub = g[heldout | (label == 0)]`, which keeps EVERY
    authentic row -- benchmark included. Here the benchmark authentic rows are
    pushed far above every generated score, so including them would collapse
    the TPR; the assertion is that the value does not move at all.
    """
    scores, splits = _grid_scores()
    contaminated = scores.copy()
    is_benchmark = splits[contaminated["image_idx"].to_numpy()] == "benchmark"
    contaminated.loc[is_benchmark, "score"] += 50.0
    assert heldout_robust_tpr(contaminated, splits) == \
        heldout_robust_tpr(scores, splits)


def test_heldout_robust_tpr_ignores_val_internal_fakes():
    """Generated images from generators the head TRAINED on are not the §6.4
    population; only held-out generators are."""
    scores, splits = _grid_scores()
    seen_fake = (splits[scores["image_idx"].to_numpy()] == "val_internal") \
        & (scores["label"].to_numpy() == 1)
    moved = scores.copy()
    moved.loc[seen_fake, "score"] -= 50.0
    assert heldout_robust_tpr(moved, splits) == heldout_robust_tpr(scores, splits)


def test_heldout_robust_tpr_excludes_the_clean_condition():
    """"Robust" is a mean over the DEGRADED conditions (spec §6.1).

    Kills the mutant that averages every condition: clean is by far the
    easiest, so including it inflates the number the headline is chosen on.
    """
    scores, splits = _grid_scores()
    degraded_only = scores[scores["condition"] != "clean"]
    assert heldout_robust_tpr(scores, splits) == \
        heldout_robust_tpr(degraded_only, splits)

    boosted = scores.copy()
    boosted.loc[boosted["condition"] == "clean", "score"] *= 10.0
    assert heldout_robust_tpr(boosted, splits) == heldout_robust_tpr(scores, splits)


def test_heldout_robust_tpr_is_the_mean_over_degraded_conditions():
    from aigcdet.eval.metrics import tpr_at_fpr
    scores, splits = _grid_scores()
    row_split = splits[scores["image_idx"].to_numpy()]
    label = scores["label"].to_numpy()
    keep = ((row_split == "val_internal") & (label == 0)) | \
           ((row_split == "heldout_generator") & (label == 1))
    sub = scores[keep]
    expected = np.mean([
        tpr_at_fpr(g["label"].to_numpy(), g["score"].to_numpy(), 0.01)
        for cond, g in sub.groupby("condition", sort=False) if cond != "clean"])
    assert heldout_robust_tpr(scores, splits) == pytest.approx(float(expected))


def test_the_selection_metric_is_taken_at_one_percent_fpr():
    """The 1% in "robust TPR @ 1% FPR" is the operating point the whole project
    is specified at (spec §6.1), so the default must be 1% and not merely
    "some low FPR".

    Kills the `target_fpr=0.05` mutant. It needs a population big enough for
    the two rates to be distinguishable: with only a handful of authentic rows
    the reachable FPRs are so coarse that 1% and 5% land on the same threshold,
    which is why the smaller fixture above cannot pin this.
    """
    from aigcdet.eval.metrics import tpr_at_fpr
    scores, splits = _grid_scores(n=800, seed=3, separation=(2.0, 0.9, 0.6))
    row_split = splits[scores["image_idx"].to_numpy()]
    label = scores["label"].to_numpy()
    sub = scores[((row_split == "val_internal") & (label == 0))
                 | ((row_split == "heldout_generator") & (label == 1))]

    def mean_at(target):
        return float(np.mean([
            tpr_at_fpr(g["label"].to_numpy(), g["score"].to_numpy(), target)
            for cond, g in sub.groupby("condition", sort=False) if cond != "clean"]))

    assert mean_at(0.05) > mean_at(0.01), "fixture cannot tell the two apart"
    assert heldout_robust_tpr(scores, splits) == pytest.approx(mean_at(0.01))


def test_heldout_robust_tpr_refuses_a_bank_with_no_heldout_generators():
    """Kills the brief's `float(np.mean(vals)) if vals else 0.0`, under which
    every rung scores 0.0 and the headline goes to whichever sorts first."""
    scores, splits = _grid_scores()
    splits = np.where(splits == "heldout_generator", "benchmark", splits)
    with pytest.raises(ValueError, match="selection population is empty"):
        heldout_robust_tpr(scores, splits)


def test_heldout_robust_tpr_refuses_a_single_class_condition():
    """Kills the brief's `if sub["label"].nunique() == 2` skip, which averages
    an unstated subset of the grid instead of saying a condition is missing."""
    scores, splits = _grid_scores()
    row_split = splits[scores["image_idx"].to_numpy()]
    drop = (scores["condition"] == "blur_s2.0") & (row_split == "heldout_generator")
    with pytest.raises(ValueError, match="only class"):
        heldout_robust_tpr(scores[~drop], splits)


def test_heldout_robust_tpr_refuses_an_authentic_row_in_the_heldout_split():
    scores, splits = _grid_scores()
    bad = scores.copy()
    row_split = splits[bad["image_idx"].to_numpy()]
    bad.loc[row_split == "heldout_generator", "label"] = 0
    with pytest.raises(ValueError, match="heldout_generator split carry label 0"):
        heldout_robust_tpr(bad, splits)


def test_heldout_robust_tpr_refuses_splits_that_do_not_cover_the_frame():
    scores, splits = _grid_scores()
    with pytest.raises(ValueError, match="must be the eval bank's own"):
        heldout_robust_tpr(scores, splits[:5])


# --- contact_sheet ---------------------------------------------------------

def _image_rows(tmp_path, n=3):
    rows = []
    for i in range(n):
        p = tmp_path / f"im{i}.png"
        Image.new("RGB", (24, 24), (10 * i, 20, 30)).save(p)
        rows.append({"path": str(p), "score": float(i), "image_idx": i, "label": 0})
    return pd.DataFrame(rows)


def test_contact_sheet_writes_a_readable_png(tmp_path):
    out = tmp_path / "sheet.png"
    contact_sheet(_image_rows(tmp_path), str(out))
    assert out.exists() and out.stat().st_size > 0
    with Image.open(out) as im:
        assert im.size[0] > 0 and im.size[1] > 0


def test_contact_sheet_accepts_annotations_one_per_row(tmp_path):
    out = tmp_path / "annotated.png"
    contact_sheet(_image_rows(tmp_path), str(out), ["a", "b", "c"])
    assert out.exists()


def test_contact_sheet_refuses_a_mismatched_annotation_list(tmp_path):
    with pytest.raises(ValueError, match="exactly one per row"):
        contact_sheet(_image_rows(tmp_path), str(tmp_path / "x.png"), ["only one"])


def test_contact_sheet_refuses_an_empty_frame(tmp_path):
    with pytest.raises(ValueError, match="nothing to render"):
        contact_sheet(pd.DataFrame(columns=["path", "score"]),
                      str(tmp_path / "x.png"))
