"""The project's ONE false-positive operating point (spec §1.3, §6.1, §6.4).

Four things in this repository are specified at the same 1% false-positive
rate, and until this module existed each one spelled it out for itself:

1. `eval.errors.SELECTION_TARGET_FPR` -- the §6.4 model-selection rule.
2. `eval.report.condition_metrics`, whose per-condition TPR column was
   `tpr_at_fpr(y, s, 0.01)` written as a literal, in a column whose NAME was
   also the literal `"tpr_at_1pct"`.
3. `calibrate.policy.fit_policy(target_fpr=0.01)` -- §1.3's deployed operating
   point, the one behind the reviewer-load impact figure.
4. `scripts/make_error_sheet.py --target-fpr`, whose default was `0.01`.

Four copies of one number is four chances for three of them to stay behind.
The failure is not hypothetical and it is silent: moving the selection rule to
5% leaves a column still *named* `tpr_at_1pct` holding 1% values beside a
headline chosen at 5%, and a decision policy still deployed at 1%.

So the number lives here, and the column NAME is DERIVED from it
(`tpr_column_name`) rather than written out. A changed operating point renames
the column, which is visible in every artefact, instead of leaving a
mislabelled one behind.

Changing `TARGET_FPR` changes the selection rule, the reported TPR column, the
deployed decision policy and the error sheet's diagnostic threshold together.
That is the point: they are the same operating point, and any of them moving
alone is a bug.
"""
from __future__ import annotations

#: The false-positive rate every operating point in this project is set at.
TARGET_FPR: float = 0.01


def _check(target_fpr: float) -> float:
    fpr = float(target_fpr)
    if not 0.0 < fpr <= 1.0:
        raise ValueError(
            f"target_fpr must be in (0, 1], got {target_fpr!r}; it is a "
            "false-positive RATE, not a percentage")
    return fpr


def fpr_label(target_fpr: float = TARGET_FPR) -> str:
    """The operating point as a human-readable percentage, e.g. `1%`.

    `format(..., "g")` rather than `:.0%`: at 0.005 the latter renders `0%`,
    which is not the operating point and reads as "no false positives at all".
    """
    return f"{format(_check(target_fpr) * 100, 'g')}%"


def tpr_column_name(target_fpr: float = TARGET_FPR) -> str:
    """The column name for TPR at `target_fpr`, e.g. `tpr_at_1pct`.

    Derived, never written out, so a moved operating point cannot leave a
    column labelled with the old one. At the project default this returns
    exactly `"tpr_at_1pct"`, which is the name every existing artefact uses.
    """
    return "tpr_at_" + fpr_label(target_fpr).replace("%", "pct")
