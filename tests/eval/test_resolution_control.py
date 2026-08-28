"""Tests for `aigcdet.eval.resolution_control`.

Fixture design, because the recurring failure on this project is a fixture
that makes the property under test unreachable:

* `uneven_frame` has FIVE strata with deliberately ASYMMETRIC counts --
  (7 authentic, 3 generated), (2, 9), (5, 5), (11, 0), (0, 6). A frame that
  was already balanced within every stratum could not detect a matcher that
  does nothing, and a frame with one stratum could not detect a matcher that
  balances across the whole frame instead of within strata. The two
  single-class strata are on OPPOSITE sides so a drop rule that only handles
  "no generated rows" is detectable.
* The global class totals are 25 authentic / 23 generated -- close but NOT
  equal, so a whole-frame matcher would return 46 rows and be mistaken for a
  correct one if the totals happened to agree.
* Whole-frame matching would keep 46 rows; per-stratum matching keeps 20.
  The two numbers differ, which is what makes the cross-stratum mutation
  visible.
* Every row carries distinct `width x height` AND a distinct `path`, so a row
  selected from the wrong stratum is identifiable from the row itself.
* `_short_side_frame` gives two rows the same short side but different
  dimensions, so `exact_short_side` and `exact_dimensions` cannot be
  confused for one another.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import pytest

from aigcdet.eval.resolution_control import (
    DEFAULT_STRATEGY,
    DROP_NO_COUNTERPART,
    DROP_SURPLUS,
    MIN_GENERATED,
    ResolutionMatchReport,
    ResolutionMatchTooSmall,
    binned_short_side,
    exact_dimensions,
    exact_short_side,
    minimum_authentic_rows,
    resolution_leakage,
    resolution_matched_subset,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_REAL_MANIFEST = os.path.join(_REPO_ROOT, "data", "manifest.parquet")

#: (short side, n_authentic, n_generated). Asymmetric on purpose; see module
#: docstring. Totals: 25 authentic, 23 generated, 48 rows.
STRATA_SPEC = [(100, 7, 3), (200, 2, 9), (300, 5, 5), (400, 11, 0), (500, 0, 6)]

#: What per-stratum matching must keep: 2*min(a, g) summed over strata.
EXPECTED_MATCHED = sum(2 * min(a, g) for _, a, g in STRATA_SPEC)          # 20
#: What a WHOLE-FRAME matcher would keep. Must differ from the above.
EXPECTED_IF_MATCHED_GLOBALLY = 2 * min(sum(a for _, a, _ in STRATA_SPEC),
                                       sum(g for _, _, g in STRATA_SPEC))  # 46


def _frame(spec=STRATA_SPEC) -> pd.DataFrame:
    rows = []
    for ss, n_auth, n_gen in spec:
        for label, n in ((0, n_auth), (1, n_gen)):
            for i in range(n):
                rows.append({
                    "path": f"/img/ss{ss}_l{label}_{i:03d}.png",
                    "label": label,
                    # long side varies within the stratum so `exact_dimensions`
                    # is strictly finer than `exact_short_side` here.
                    "width": ss,
                    "height": ss + i,
                })
    return pd.DataFrame(rows)


@pytest.fixture()
def uneven_frame() -> pd.DataFrame:
    return _frame()


def _big_frame(n_auth: int, n_gen: int, n_strata: int = 4) -> pd.DataFrame:
    """`n_auth` authentic and `n_gen` generated rows IN TOTAL, spread evenly
    over `n_strata` strata. Pass multiples of `n_strata`."""
    assert n_auth % n_strata == 0 and n_gen % n_strata == 0
    rows = []
    for s in range(n_strata):
        for label, total in ((0, n_auth), (1, n_gen)):
            for i in range(total // n_strata):
                rows.append({"path": f"/img/s{s}_l{label}_{i:05d}.png",
                             "label": label, "width": 100 + s, "height": 100 + s})
    return pd.DataFrame(rows)


def _match(df, **kw):
    """Match with the size guard off -- the small fixtures are about the
    matching, not about whether the result could carry a TPR."""
    kw.setdefault("enforce_minimum", False)
    return resolution_matched_subset(df, **kw)


# --------------------------------------------------------------------------
# the fixture itself must be able to fail the tests
# --------------------------------------------------------------------------

def test_the_fixture_is_unbalanced_within_strata_and_multi_stratum(uneven_frame):
    """Guards the guard. If this frame were already matched, or had one
    stratum, every matching test below would pass against a no-op."""
    ss = np.minimum(uneven_frame["width"], uneven_frame["height"])
    counts = uneven_frame.groupby([ss, "label"]).size().unstack(fill_value=0)
    assert len(counts) == 5, "need several strata to detect cross-stratum leakage"
    assert not (counts[0] == counts[1]).all(), "a pre-balanced fixture proves nothing"
    assert (counts[0] == 0).any() and (counts[1] == 0).any(), \
        "need single-class strata on BOTH sides"
    assert EXPECTED_MATCHED != EXPECTED_IF_MATCHED_GLOBALLY
    assert EXPECTED_MATCHED == 20 and EXPECTED_IF_MATCHED_GLOBALLY == 46
    # global totals unequal, so a whole-frame matcher is not accidentally right
    assert (uneven_frame["label"] == 0).sum() == 25
    assert (uneven_frame["label"] == 1).sum() == 23


# --------------------------------------------------------------------------
# matching happens WITHIN strata
# --------------------------------------------------------------------------

def test_matching_is_within_strata_not_across_the_whole_frame(uneven_frame):
    """MUTATION: match on the whole frame instead of per stratum.

    Whole-frame matching returns 46 of 48 rows and a 50/50 split, which looks
    correct from the class balance alone -- so this pins the row COUNT and the
    per-stratum balance, not just the global one.
    """
    subset, report = _match(uneven_frame, strategy="exact_short_side")
    assert len(subset) == EXPECTED_MATCHED == 20
    assert len(subset) != EXPECTED_IF_MATCHED_GLOBALLY
    assert report.n_out == 20


def test_every_retained_stratum_is_exactly_balanced(uneven_frame):
    """MUTATION: the minority count is not the cap (or an off-by-one).

    Checked per stratum with the EXPECTED per-stratum numbers spelled out, so
    a matcher that keeps min+1 or the majority count is caught even though the
    global balance would still come out 50/50 for some of those bugs.
    """
    subset, _ = _match(uneven_frame, strategy="exact_short_side")
    ss = np.minimum(subset["width"], subset["height"])
    got = subset.groupby([ss, "label"]).size().unstack(fill_value=0)
    expected = {ss_: min(a, g) for ss_, a, g in STRATA_SPEC if min(a, g) > 0}
    assert set(got.index) == set(expected), "wrong strata survived"
    for ss_, k in expected.items():
        assert got.loc[ss_, 0] == k, f"stratum {ss_} authentic count"
        assert got.loc[ss_, 1] == k, f"stratum {ss_} generated count"
    assert expected == {100: 3, 200: 2, 300: 5}


def test_single_class_strata_are_dropped_entirely(uneven_frame):
    """Both directions: 400 is authentic-only, 500 is generated-only."""
    subset, report = _match(uneven_frame, strategy="exact_short_side")
    ss = set(np.minimum(subset["width"], subset["height"]).tolist())
    assert 400 not in ss and 500 not in ss
    assert ss == {100, 200, 300}
    assert report.n_strata_in == 5
    assert report.n_strata_kept == 3


def test_matched_subset_leaks_nothing_and_the_input_leaked_a_lot(uneven_frame):
    """The property the module exists for, measured rather than asserted."""
    before = resolution_leakage(uneven_frame, "exact_short_side")
    assert before.advantage_over_majority > 0.1
    subset, _ = _match(uneven_frame, strategy="exact_short_side")
    after = resolution_leakage(subset, "exact_short_side")
    assert after.advantage_over_majority == pytest.approx(0.0)
    assert after.stratum_majority_accuracy == pytest.approx(0.5)
    assert after.n_single_class_strata == 0


# --------------------------------------------------------------------------
# discard accounting
# --------------------------------------------------------------------------

def test_the_report_accounts_for_every_dropped_row_by_reason(uneven_frame):
    """MUTATION: under-report the discards.

    The two reasons are pinned SEPARATELY with hand-computed numbers. A report
    that counts every drop as `surplus`, or that only counts strata it
    iterated past, disagrees with one of these even though the total is right.
    """
    _, report = _match(uneven_frame, strategy="exact_short_side")
    assert report.n_in == 48
    assert report.n_out == 20
    assert report.n_dropped == 28
    # 400 -> 11 authentic, 500 -> 6 generated
    assert report.n_dropped_no_counterpart == 17
    # 100 -> 7-3=4, 200 -> 9-2=7, 300 -> 0
    assert report.n_dropped_surplus == 11
    assert report.n_dropped_no_counterpart + report.n_dropped_surplus == report.n_dropped
    assert report.retention == pytest.approx(20 / 48)


def test_the_strata_table_covers_the_whole_input_frame(uneven_frame):
    """Including the strata that contributed nothing -- a table that lists
    only the survivors cannot show what the subset cost."""
    _, report = _match(uneven_frame, strategy="exact_short_side")
    t = report.strata
    assert len(t) == 5, "every input stratum must appear, survivors or not"
    assert int(t["n_kept"].sum()) == report.n_out
    assert int(t["n_dropped"].sum()) == report.n_dropped
    assert int((t["n_authentic_in"] + t["n_generated_in"]).sum()) == report.n_in
    by = t.set_index("stratum")
    assert by.loc["400", "drop_reason"] == DROP_NO_COUNTERPART
    assert by.loc["500", "drop_reason"] == DROP_NO_COUNTERPART
    assert by.loc["100", "drop_reason"] == DROP_SURPLUS
    assert by.loc["300", "drop_reason"] == ""
    assert by.loc["400", "n_dropped"] == 11
    assert by.loc["200", "n_dropped"] == 7


def test_class_balance_before_and_after_is_reported(uneven_frame):
    _, report = _match(uneven_frame, strategy="exact_short_side")
    assert report.n_in_authentic == 25 and report.n_in_generated == 23
    assert report.generated_share_in == pytest.approx(23 / 48)
    assert report.n_out_authentic == report.n_out_generated == 10
    assert report.generated_share_out == pytest.approx(0.5)


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------

def test_the_same_seed_gives_the_same_rows(uneven_frame):
    a, _ = _match(uneven_frame, strategy="exact_short_side", seed=7)
    b, _ = _match(uneven_frame, strategy="exact_short_side", seed=7)
    assert list(a["path"]) == list(b["path"])


def test_different_seeds_give_different_rows(uneven_frame):
    """MUTATION: the seed is ignored.

    Needs a stratum with a real choice to make: stratum 200 keeps 2 of its 9
    generated rows, so there are 36 possible draws and two seeds agreeing by
    chance is unlikely. A fixture with nothing to choose (every stratum
    already balanced) would make a seed-ignoring matcher undetectable, which
    is why the search runs over seeds and asserts SOME pair differs.
    """
    draws = {tuple(_match(uneven_frame, strategy="exact_short_side", seed=s)[0]["path"])
             for s in range(12)}
    assert len(draws) > 1, "the seed changed nothing -- is it being used?"


def test_the_selection_does_not_depend_on_the_frames_row_order(uneven_frame):
    """A frame sorted by score must match to the same rows as the same frame
    sorted by path, or two reports of 'the same' matched number disagree."""
    shuffled = uneven_frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    a, _ = _match(uneven_frame, strategy="exact_short_side", seed=5)
    b, _ = _match(shuffled, strategy="exact_short_side", seed=5)
    assert sorted(a["path"]) == sorted(b["path"])


def test_the_subset_keeps_the_input_frames_columns_and_index_labels(uneven_frame):
    df = uneven_frame.set_index("path", drop=False)
    subset, _ = _match(df, strategy="exact_short_side")
    assert list(subset.columns) == list(df.columns)
    assert set(subset.index) <= set(df.index)
    assert list(subset.index) == list(subset["path"])


# --------------------------------------------------------------------------
# the too-small guard
# --------------------------------------------------------------------------

def test_minimum_authentic_rows_is_ten_over_the_target_fpr():
    assert minimum_authentic_rows(0.01) == 1000
    assert minimum_authentic_rows(0.05) == 200
    assert minimum_authentic_rows(0.01, min_exceedances=1) == 100
    assert minimum_authentic_rows(0.03) == math.ceil(10 / 0.03) == 334


def test_a_subset_too_small_for_the_metric_is_refused(uneven_frame):
    """MUTATION: remove the guard.

    The fixture matches to 10 authentic rows, far under the 1,000 floor, so a
    removed guard returns a subset instead of raising.
    """
    with pytest.raises(ResolutionMatchTooSmall) as exc:
        resolution_matched_subset(uneven_frame, strategy="exact_short_side")
    msg = str(exc.value)
    assert "10 authentic" in msg and "1000" in msg, msg
    assert "no counterpart" in msg


def test_the_refusal_names_the_actual_counts_and_carries_the_report(uneven_frame):
    with pytest.raises(ResolutionMatchTooSmall) as exc:
        resolution_matched_subset(uneven_frame, strategy="exact_short_side")
    report = exc.value.report
    assert isinstance(report, ResolutionMatchReport)
    assert report.n_out == 20 and report.n_out_authentic == 10
    assert report.n_dropped_no_counterpart == 17
    assert "dropped 28 rows" in report.describe()


def test_the_generated_floor_binds_when_the_authentic_floor_is_lowered(uneven_frame):
    """Matching equalises the two classes, so the EFFECTIVE floor is the larger
    of the two parameters and `min_generated` is unreachable at its default.
    It exists for the caller who lowers `min_authentic`; lowering that alone
    must not buy a ten-row subset.
    """
    with pytest.raises(ResolutionMatchTooSmall) as exc:
        resolution_matched_subset(uneven_frame, strategy="exact_short_side",
                                  min_authentic=1)
    msg = str(exc.value)
    assert "10 authentic and 10 generated" in msg, msg
    assert f"1 authentic and {MIN_GENERATED} generated" in msg, msg


def test_a_frame_that_clears_the_floor_is_returned():
    n_auth = minimum_authentic_rows(0.01)
    assert n_auth == 1000
    big = _big_frame(n_auth=1040, n_gen=1008)
    subset, report = resolution_matched_subset(big)
    assert report.n_out_authentic == report.n_out_generated == 1008
    assert report.n_out_authentic >= n_auth
    assert len(subset) == report.n_out == 2016


def test_enforce_minimum_false_returns_the_small_subset(uneven_frame):
    subset, report = resolution_matched_subset(
        uneven_frame, strategy="exact_short_side", enforce_minimum=False)
    assert len(subset) == 20 == report.n_out


def test_an_explicit_lower_floor_is_honoured_but_must_be_asked_for(uneven_frame):
    subset, _ = resolution_matched_subset(
        uneven_frame, strategy="exact_short_side",
        min_authentic=10, min_generated=10)
    assert len(subset) == 20
    with pytest.raises(ResolutionMatchTooSmall):
        resolution_matched_subset(uneven_frame, strategy="exact_short_side",
                                  min_authentic=11, min_generated=10)


def test_a_looser_target_fpr_lowers_the_floor():
    """The floor is derived from the operating point, not hard-coded."""
    frame = _big_frame(n_auth=400, n_gen=400)
    assert minimum_authentic_rows(0.01) == 1000 > 400 > minimum_authentic_rows(0.05)
    with pytest.raises(ResolutionMatchTooSmall):
        resolution_matched_subset(frame, target_fpr=0.01)
    subset, _ = resolution_matched_subset(frame, target_fpr=0.05)
    assert len(subset) == 800


# --------------------------------------------------------------------------
# stratum strategy is a parameter
# --------------------------------------------------------------------------

def _short_side_frame() -> pd.DataFrame:
    """Two shapes sharing a short side of 100: 100x100 and 100x200.

    Authentic rows are all 100x100, generated rows all 100x200. On the short
    side the two classes overlap completely; on exact dimensions they do not
    overlap at all. That is exactly the demo benchmark's situation in
    miniature, and it makes the two strategies impossible to confuse.
    """
    rows = [{"path": f"/a{i}.png", "label": 0, "width": 100, "height": 100}
            for i in range(6)]
    rows += [{"path": f"/g{i}.png", "label": 1, "width": 100, "height": 200}
             for i in range(6)]
    return pd.DataFrame(rows)


def test_exact_dimensions_is_stricter_than_exact_short_side():
    frame = _short_side_frame()
    loose, loose_rep = _match(frame, strategy="exact_short_side")
    assert loose_rep.n_out == 12, "the short side matches everything here"
    assert loose_rep.residual_exact_advantage == pytest.approx(0.5), \
        "and leaves the exact size perfectly separating"

    strict, strict_rep = _match(frame, strategy="exact_dimensions")
    assert strict_rep.n_out == 0
    assert strict_rep.n_dropped_no_counterpart == 12
    assert "NOTHING SURVIVED" in strict_rep.describe()


def test_the_default_strategy_is_the_strict_one():
    assert DEFAULT_STRATEGY == "exact_dimensions"
    frame = _short_side_frame()
    _, report = _match(frame)
    assert report.n_out == 0, "the default must not be the permissive strategy"
    assert report.strategy == "exact_dimensions"


def test_describe_downgrades_its_claim_when_control_is_partial():
    """A coarse stratum balances the BINS while leaving the exact sizes as
    separating as ever. `describe()` must not call that a clean control."""
    frame = _short_side_frame()
    _, partial = _match(frame, strategy="exact_short_side")
    text = partial.describe()
    assert "PARTIAL CONTROL ONLY" in text
    assert "+0.5000" in text
    assert "MEANS: on images whose resolution gives away nothing" not in text

    _, clean = _match(_frame(), strategy="exact_dimensions")
    clean_text = clean.describe()
    assert clean.residual_exact_advantage == pytest.approx(0.0)
    assert "PARTIAL CONTROL ONLY" not in clean_text
    assert "MEANS: on images whose resolution gives away nothing" in clean_text
    assert "DOES NOT MEAN" in clean_text


def test_a_callable_strategy_is_accepted(uneven_frame):
    """Stratum definition is a parameter, not a constant -- a caller can
    control on something this module never anticipated."""
    def by_hundreds(df):
        return (df["width"].to_numpy() // 100).astype(str)

    subset, report = _match(uneven_frame, strategy=by_hundreds)
    assert report.strategy == "by_hundreds"
    assert report.n_strata_in == 5
    assert len(subset) == 20


def test_binned_short_side_merges_neighbouring_resolutions():
    frame = pd.DataFrame(
        [{"path": f"/a{i}.png", "label": 0, "width": 130, "height": 130} for i in range(4)]
        + [{"path": f"/g{i}.png", "label": 1, "width": 250, "height": 250} for i in range(4)])
    _, exact = _match(frame, strategy="exact_short_side")
    assert exact.n_out == 0, "130 and 250 are different exact strata"
    _, binned = _match(frame, strategy=binned_short_side())
    assert binned.n_out == 8, "128..256 is one octave bin"
    assert binned.strategy == "binned_short_side(edges=[0.0, 64.0, 128.0, 256.0, " \
                              "512.0, 1024.0, 2048.0, inf])"
    assert binned.residual_exact_advantage == pytest.approx(0.5)


def test_binned_short_side_rejects_bad_edges_and_out_of_range_rows():
    with pytest.raises(ValueError, match="strictly increasing"):
        binned_short_side([0, 256, 128])
    with pytest.raises(ValueError, match="at least two bin edges"):
        binned_short_side([256])
    narrow = binned_short_side([100, 200])
    frame = pd.DataFrame([{"path": "/a.png", "label": 0, "width": 50, "height": 50},
                          {"path": "/g.png", "label": 1, "width": 150, "height": 150}])
    with pytest.raises(ValueError, match="outside the bin edges"):
        _match(frame, strategy=narrow)


def test_an_unknown_strategy_name_is_rejected(uneven_frame):
    with pytest.raises(KeyError, match="unknown stratum strategy"):
        _match(uneven_frame, strategy="short_side_ish")


def test_exact_short_side_and_exact_dimensions_compute_what_they_say():
    frame = pd.DataFrame([{"label": 0, "width": 300, "height": 200},
                          {"label": 1, "width": 200, "height": 300}])
    assert list(exact_short_side(frame)) == [200, 200]
    assert list(exact_dimensions(frame)) == ["300x200", "200x300"]


# --------------------------------------------------------------------------
# leakage diagnostic
# --------------------------------------------------------------------------

def test_resolution_leakage_reports_the_single_class_strata(uneven_frame):
    leak = resolution_leakage(uneven_frame, "exact_short_side")
    assert leak.n_rows == 48
    assert leak.n_strata == 5
    assert leak.n_single_class_strata == 2          # 400 and 500
    assert leak.n_rows_in_single_class_strata == 17  # 11 + 6
    assert leak.share_in_single_class_strata == pytest.approx(17 / 48)
    # per-stratum majorities: 7 + 9 + 5 + 11 + 6 = 38
    assert leak.stratum_majority_accuracy == pytest.approx(38 / 48)
    assert leak.majority_baseline_accuracy == pytest.approx(25 / 48)
    assert leak.advantage_over_majority == pytest.approx(13 / 48)
    assert "NOT evidence that the model read pixels" in leak.describe()


def test_resolution_leakage_says_so_when_there_is_no_leak():
    frame = pd.DataFrame(
        [{"label": lab, "width": ss, "height": ss}
         for ss in (100, 200) for lab in (0, 1) for _ in range(4)])
    leak = resolution_leakage(frame, "exact_short_side")
    assert leak.advantage_over_majority == pytest.approx(0.0)
    assert "Resolution is uninformative" in leak.describe()


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("drop", ["label", "width", "height"])
def test_a_frame_missing_a_required_column_is_rejected(uneven_frame, drop):
    with pytest.raises(KeyError, match=drop):
        _match(uneven_frame.drop(columns=[drop]))


def test_an_empty_frame_is_rejected(uneven_frame):
    with pytest.raises(ValueError, match="empty"):
        _match(uneven_frame.iloc[:0])


def test_non_positive_or_missing_dimensions_are_rejected(uneven_frame):
    bad = uneven_frame.copy()
    bad.loc[0, "width"] = 0
    with pytest.raises(ValueError, match="non-positive"):
        _match(bad)
    missing = uneven_frame.copy()
    missing.loc[0, "height"] = np.nan
    with pytest.raises(ValueError, match="non-numeric or missing"):
        _match(missing)


def test_a_label_that_is_not_zero_or_one_is_rejected(uneven_frame):
    bad = uneven_frame.copy()
    bad.loc[0, "label"] = 2
    with pytest.raises(ValueError, match="label must be 0"):
        _match(bad)


# --------------------------------------------------------------------------
# the real frozen manifest, read-only
# --------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.isfile(_REAL_MANIFEST),
                    reason="the frozen manifest is not materialised here")
def test_the_real_manifest_leaks_resolution_and_survives_matching():
    """Read-only pin on the data ACTUALLY on disk.

    The synthetic fixtures above can only show that the code does what it
    says. Whether the *finding* is real -- and whether the matched subset is
    big enough to publish from -- is a question about this file, and it is the
    reason the module exists. Numbers here were measured on the 138,116-row
    frozen manifest; if a rebuild moves them the assertions are ranges, not
    exact pins, but the qualitative claims must survive.
    """
    df = pd.read_parquet(_REAL_MANIFEST)
    assert {"width", "height", "label"} <= set(df.columns)

    leak = resolution_leakage(df, "exact_short_side")
    # The finding: resolution alone beats the majority baseline by a mile.
    assert leak.advantage_over_majority > 0.15, leak.describe()
    assert leak.n_single_class_strata > 0.5 * leak.n_strata
    assert leak.share_in_single_class_strata > 0.2

    subset, report = resolution_matched_subset(df)
    assert report.strategy == "exact_dimensions"
    # Fully controlled: no residual leak is left on what came back.
    assert report.residual_exact_advantage == pytest.approx(0.0)
    assert resolution_leakage(subset, "exact_dimensions").advantage_over_majority \
        == pytest.approx(0.0)
    # 50/50 by construction, from an input that was not.
    assert report.n_out_authentic == report.n_out_generated
    assert report.generated_share_out == pytest.approx(0.5)
    assert report.generated_share_in != pytest.approx(0.5)
    # It cost a lot, and it is still enough to carry TPR@1%FPR.
    assert 0.1 < report.retention < 0.6, report.describe()
    assert report.n_out_authentic >= minimum_authentic_rows(0.01)
    assert report.n_dropped_no_counterpart > 0


@pytest.mark.skipif(not os.path.isfile(_REAL_MANIFEST),
                    reason="the frozen manifest is not materialised here")
def test_matching_the_real_manifest_on_the_short_side_leaves_a_residual():
    """Why the default is `exact_dimensions` and not `exact_short_side`.

    On this data, matching the short side leaves the long side free and an
    exact-dimensions rule still beats chance on the "matched" subset. If this
    ever stops being true the default could be relaxed -- until then, relaxing
    it would publish a partially-controlled number as a controlled one.
    """
    df = pd.read_parquet(_REAL_MANIFEST)
    _, report = resolution_matched_subset(df, strategy="exact_short_side")
    assert report.residual_exact_advantage > 0.1, report.describe()
    assert "PARTIAL CONTROL ONLY" in report.describe()
