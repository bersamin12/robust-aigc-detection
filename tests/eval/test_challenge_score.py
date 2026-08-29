"""The organisers' announced score: 0.50 x AUC_clean + 0.50 x AUC_robust.

The arithmetic is trivial and is not what these tests are about. What they
pin is WHICH conditions the robust half averages over. Our ablation tier
carries five composed conditions we invented (`social_repost`,
`messaging_app`, ...) alongside the brief's fifteen. Averaging over all of
them produces a number that is not the announced score, differs from it by
however our own scenarios happen to land, and looks exactly like it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aigcdet.augment.scenarios import CORE_CONDITIONS, EVAL_GRID
from aigcdet.eval.report import (
    BANKS_NOT_VERIFIED, CHALLENGE_ROBUST_CONDITIONS, CHALLENGE_WEIGHTS,
    challenge_score, robustness_table,
)

_NOBANK = BANKS_NOT_VERIFIED


def _table(rungs: dict[str, dict[str, float]], tier: str = "ablation",
           metric: str = "auc") -> pd.DataFrame:
    """A robustness-table-shaped frame with the values a test wants.

    Hand-built rather than run through `robustness_table`, because these tests
    assert on exact arithmetic and a bootstrapped table cannot supply exact
    inputs. One test below does go through the real builder, so the two shapes
    cannot drift apart unnoticed.
    """
    conds = list(EVAL_GRID) if tier == "ablation" else list(CORE_CONDITIONS)
    rows = {}
    for rung, vals in rungs.items():
        row = {c: float(vals.get(c, 0.5)) for c in conds}
        degraded = [c for c in conds if c != "clean"]
        row[f"robust_{metric}"] = float(np.mean([row[c] for c in degraded]))
        row["tier"] = tier
        rows[rung] = row
    return pd.DataFrame.from_dict(rows, orient="index")


# --------------------------------------------------------------- arithmetic

def test_the_score_is_half_clean_and_half_robust():
    t = _table({"a3": {c: 0.8 for c in EVAL_GRID} | {"clean": 1.0}})
    out = challenge_score(t)
    assert out.loc["a3", "auc_clean"] == pytest.approx(1.0)
    assert out.loc["a3", "auc_robust"] == pytest.approx(0.8)
    assert out.loc["a3", "challenge_score"] == pytest.approx(0.9)


def test_the_weights_are_the_announced_half_and_half():
    """Pinned by a case where the two halves are 1 and 0, so any other
    weighting gives a different number. Asserting the constant equals (0.5,
    0.5) would only restate the constant."""
    t = _table({"a0": {c: 0.0 for c in EVAL_GRID} | {"clean": 1.0}})
    assert challenge_score(t).loc["a0", "challenge_score"] == pytest.approx(0.5)
    assert CHALLENGE_WEIGHTS == (0.5, 0.5)


def test_each_weight_is_applied_to_its_own_half():
    """With the announced 0.50/0.50 the two weights are interchangeable, so
    a swap is invisible today and wrong the moment the organisers publish any
    other split. Forcing (1, 0) makes the mapping observable."""
    t = _table({"a0": {c: 0.2 for c in EVAL_GRID} | {"clean": 0.9}})
    monkey = pytest.MonkeyPatch()
    with monkey.context() as m:
        m.setattr("aigcdet.eval.report.CHALLENGE_WEIGHTS", (1.0, 0.0))
        assert challenge_score(t).loc["a0", "challenge_score"] == pytest.approx(0.9)
        m.setattr("aigcdet.eval.report.CHALLENGE_WEIGHTS", (0.0, 1.0))
        assert challenge_score(t).loc["a0", "challenge_score"] == pytest.approx(0.2)


def test_clean_is_not_also_counted_in_the_robust_half():
    """A perfect clean AUC beside a mediocre degraded one must not pull the
    robust half up. Counting clean twice inflates every score, and inflates
    the least robust rungs the most -- reversing the comparison the table
    exists to make."""
    t = _table({"a0": {c: 0.5 for c in EVAL_GRID} | {"clean": 1.0}})
    assert challenge_score(t).loc["a0", "auc_robust"] == pytest.approx(0.5)


# ------------------------------------------------ which conditions it averages

def test_robust_averages_the_briefs_transforms_and_not_our_own_scenarios():
    """The heart of it. `social_repost` and friends are OUR conditions; the
    judges do not score them. A table where they are perfect and the brief's
    are not must still report the brief's number."""
    ours = [c for c in EVAL_GRID if c not in CORE_CONDITIONS]
    assert ours, "the ablation tier no longer adds conditions; update this test"
    vals = {c: 0.6 for c in EVAL_GRID}
    vals["clean"] = 1.0
    for c in ours:
        vals[c] = 1.0

    out = challenge_score(_table({"a3": vals}))

    assert out.loc["a3", "auc_robust"] == pytest.approx(0.6)
    assert set(CHALLENGE_ROBUST_CONDITIONS) == set(CORE_CONDITIONS) - {"clean"}


def test_the_score_does_not_reuse_the_tables_own_robust_column():
    """`robust_auc` is a §6.1 reporting mean over EVERY degraded condition in
    whatever tier the table holds. It is the obvious column to grab and it is
    the wrong one at the ablation tier. This is the mutation that matters."""
    ours = [c for c in EVAL_GRID if c not in CORE_CONDITIONS]
    vals = {c: 0.6 for c in EVAL_GRID} | {"clean": 1.0}
    for c in ours:
        vals[c] = 1.0
    t = _table({"a3": vals})

    assert t.loc["a3", "robust_auc"] != pytest.approx(0.6)   # the trap
    assert challenge_score(t).loc["a3", "auc_robust"] == pytest.approx(0.6)


def test_a_final_report_table_scores_the_same_conditions():
    """The final-report tier IS the brief's fifteen, so the two tiers must
    agree on the robust half given equal per-condition numbers. If they do
    not, the tier is silently changing the headline."""
    vals = {c: 0.7 for c in CORE_CONDITIONS} | {"clean": 0.95}
    abl = challenge_score(_table({"a3": dict(vals)}, tier="ablation"))
    fin = challenge_score(_table({"a3": dict(vals)}, tier="final_report"))
    assert abl.loc["a3", "challenge_score"] == pytest.approx(
        fin.loc["a3", "challenge_score"])


# ------------------------------------------------------------------- refusals

def test_a_table_of_a_different_metric_is_refused():
    """The formula is defined on ROC AUC. A TPR table has the same shape and
    the same column names, so the number would render, be in [0, 1], move
    sensibly with the model, and not be the announced score."""
    t = _table({"a0": {c: 0.7 for c in EVAL_GRID}}, metric="tpr_at_1pct")
    with pytest.raises(ValueError, match="auc"):
        challenge_score(t)


def test_an_unpublishable_tier_is_refused():
    t = _table({"a0": {c: 0.7 for c in EVAL_GRID}})
    t["tier"] = "smoke"
    with pytest.raises(ValueError, match="smoke"):
        challenge_score(t)


def test_a_missing_required_condition_is_named():
    t = _table({"a0": {c: 0.7 for c in EVAL_GRID}}).drop(columns=["jpeg_q30"])
    with pytest.raises(ValueError, match="missing required condition"):
        challenge_score(t)


def test_a_nan_is_refused_rather_than_averaged_away():
    """`np.mean` over a NaN gives NaN, which renders as an empty cell in the
    markdown table -- a missing headline that looks like a formatting
    problem."""
    t = _table({"a0": {c: 0.7 for c in EVAL_GRID}})
    t.loc["a0", "blur_s2.0"] = np.nan
    with pytest.raises(ValueError, match="blur_s2.0"):
        challenge_score(t)


# ---------------------------------------------------------------- integration

def test_it_reads_a_table_the_real_builder_produced():
    """The hand-built frames above are a convenience. This one is the shape
    that actually ships."""
    rng = np.random.default_rng(0)
    rows = []
    for cond in EVAL_GRID:
        y = np.array([0] * 40 + [1] * 40)
        rows.append(pd.DataFrame({
            "condition": cond, "image_idx": np.arange(80), "label": y,
            "generator": "g", "source": "s",
            "score": rng.normal(y * 2.0, 1.0)}))
    scores = pd.concat(rows, ignore_index=True)
    t = robustness_table({"a0": scores}, tier="ablation", n_boot=20,
                         banks=_NOBANK)

    out = challenge_score(t)

    assert list(out.index) == ["a0"]
    assert set(out.columns) == {"auc_clean", "auc_robust", "challenge_score"}
    assert 0.0 <= out.loc["a0", "challenge_score"] <= 1.0


def test_rungs_keep_the_tables_order():
    """The caller sorts. Returning a silently re-sorted frame would put a
    different rung first in every rendering that trusts row order."""
    t = _table({"a3": {c: 0.9 for c in EVAL_GRID},
                "a0": {c: 0.6 for c in EVAL_GRID}})
    assert list(challenge_score(t).index) == ["a3", "a0"]
