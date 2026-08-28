"""The project's single false-positive operating point.

The point of the module is that FOUR places -- §6.4 selection, §6.1's reported
TPR column, the deployed decision policy and the error sheet's diagnostic
threshold -- are the same operating point. These tests pin that they all read
it from one constant, and that the reported column's NAME is derived from the
number rather than written out beside it.
"""
import inspect

import numpy as np
import pandas as pd
import pytest

from aigcdet.operating_point import TARGET_FPR, fpr_label, tpr_column_name


def test_the_default_column_name_is_the_one_every_artefact_already_uses():
    """Deriving the name must not rename today's column: `docs/robustness_table.md`
    and every test that reads it say `tpr_at_1pct`."""
    assert TARGET_FPR == 0.01
    assert tpr_column_name() == "tpr_at_1pct"
    assert fpr_label() == "1%"


def test_the_name_moves_with_the_operating_point():
    """Kills the mutant that returns the literal `"tpr_at_1pct"` regardless.

    A column named for 1% holding TPR at 5% is unfalsifiable once written: the
    value is in [0, 1] either way and nothing downstream can tell.
    """
    assert tpr_column_name(0.05) == "tpr_at_5pct"
    assert tpr_column_name(0.10) == "tpr_at_10pct"
    assert tpr_column_name(0.005) == "tpr_at_0.5pct"


def test_a_sub_one_percent_point_is_not_rendered_as_zero():
    """Kills the `:.0%` formatter, which renders 0.005 as `0%`.

    `tpr_at_0pct` reads as "no false positives at all" and collides with any
    other point under 0.5%, so two different operating points would share one
    column name.
    """
    assert fpr_label(0.005) == "0.5%"
    assert tpr_column_name(0.005) != tpr_column_name(0.004)


@pytest.mark.parametrize("bad", [0.0, -0.01, 1.5, 100])
def test_a_rate_outside_zero_to_one_is_refused(bad):
    with pytest.raises(ValueError, match="target_fpr"):
        tpr_column_name(bad)


def test_every_operating_point_in_the_project_reads_the_same_constant():
    """The four call sites the module docstring names.

    Kills a mutant that re-literalises any one of them: the failure mode is
    silent, because moving one leaves the other three at 1% while every name,
    key and rule string still says 1%.
    """
    from aigcdet.calibrate.policy import fit_policy
    from aigcdet.eval.errors import SELECTION_METRIC, SELECTION_TARGET_FPR
    from aigcdet.eval.report import METRIC_COLUMNS, condition_metrics

    assert SELECTION_TARGET_FPR is TARGET_FPR
    assert SELECTION_METRIC.endswith(tpr_column_name(TARGET_FPR))
    assert tpr_column_name(TARGET_FPR) in METRIC_COLUMNS
    assert (inspect.signature(condition_metrics).parameters["target_fpr"].default
            == TARGET_FPR)
    # `calibrate.policy` is not this change's to edit, and its default is still
    # the literal 0.01. Compared by VALUE on purpose: the assertion holds today
    # and fails the moment TARGET_FPR moves without `fit_policy` following,
    # which is the coupling §1.3's deployed operating point has to §6.4's.
    # `policy.py` should take `target_fpr: float = TARGET_FPR` from this module.
    assert inspect.signature(fit_policy).parameters["target_fpr"].default == TARGET_FPR


def test_the_error_sheets_default_is_the_project_operating_point():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "make_error_sheet.py"
    spec = importlib.util.spec_from_file_location("mes_for_op", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.build_parser().parse_args(
        ["--scores", "s", "--eval-bank", "b"]).target_fpr == TARGET_FPR


def test_condition_metrics_measures_and_names_at_the_same_point():
    """Kills two mutants at once: `tpr_at_fpr(y, s, 0.01)` with a `target_fpr`
    parameter that is ignored, and a hardcoded column NAME beside a moved value.

    200 rows (100 authentic) so 1% and 50% are reachable and different; with a
    handful of authentic rows the coarsest reachable FPR exceeds 1% and the two
    columns collapse to the same number, which would make this test pass under
    both mutants.
    """
    from aigcdet.eval.report import condition_metrics

    rng = np.random.default_rng(0)
    n = 200
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    df = pd.DataFrame({"condition": "clean", "image_idx": np.arange(n), "label": y,
                       "score": rng.normal(y * 1.0, 1.0)})
    tight = condition_metrics(df, seed=0, n_boot=20)
    loose = condition_metrics(df, seed=0, n_boot=20, target_fpr=0.5)
    assert "tpr_at_1pct" in tight.columns and "tpr_at_50pct" not in tight.columns
    assert "tpr_at_50pct" in loose.columns and "tpr_at_1pct" not in loose.columns
    assert float(loose.loc[0, "tpr_at_50pct"]) > float(tight.loc[0, "tpr_at_1pct"])
    assert float(tight.loc[0, "target_fpr"]) == 0.01
    assert float(loose.loc[0, "target_fpr"]) == 0.5
