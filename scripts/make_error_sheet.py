"""Contact sheets of representative false positives and negatives (spec §6.6).

    python scripts/make_error_sheet.py --scores outputs/scores.parquet \
        --eval-bank banks/eval_dinov3l --condition clean --out docs/errors

Writes, per condition:

- `<condition>_fp.png` / `<condition>_fn.png` -- the `k` most confidently
  mis-scored authentic / generated images, so a human can see whether the
  errors share a visual cause.
- `fp_by_source.md` -- the false-positive rate per source dataset. False
  positives concentrated in one source indicate a confound in that source
  rather than a weakness of the detector, which is why the split is reported
  alongside the sheets rather than as an afterthought.

The threshold behind `fp_by_source.md` is a DIAGNOSTIC threshold, stated in
the file, and it is fitted ON THE VERY ROWS the file then reports. Two
consequences the file spells out rather than leaving to the reader:

- the AGGREGATE false-positive rate across all sources is `--target-fpr` (1% by
  default) BY CONSTRUCTION. It measures the threshold, not the detector. Only
  the RELATIVE concentration across sources carries information -- which is
  what §6.6 asks this sheet for.
- it is not the deployment operating point. That one comes from
  `calibrate.policy`, fitted on internal validation and shipped with the
  model; quoting this number as a deployed false-positive rate would be wrong.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from aigcdet.eval.errors import contact_sheet, fp_rate_by_source, top_errors
from aigcdet.eval.metrics import threshold_at_fpr
from aigcdet.features.bank import FeatureBank

#: Columns pulled off the eval bank onto the scores. `path` is what the sheet
#: renders; `split` is provenance -- an error sheet spanning benchmark rows and
#: internal-validation rows is fine, but the file must say which it covers.
META_COLUMNS = ("image_idx", "path", "split")


def markdown_table(df: pd.DataFrame) -> str:
    """A GitHub-flavoured markdown table.

    Hand-rolled for the same reason `eval.report` hand-rolls its own:
    `DataFrame.to_markdown` needs the optional `tabulate` package, which is not
    a project dependency and is not installed here, so the pandas route raises
    ImportError at write time.
    """
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "" if np.isnan(value) else f"{float(value):.4f}"
        return str(value)

    header = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines)


def attach_paths(scores: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Join the bank's per-image metadata onto the scores, by image_idx.

    On the COLUMN, not on the index: `meta.parquet` is written with
    `index=False`, so its RangeIndex is an artefact of the read rather than the
    bank's own row identity. They agree today; joining on the identity that is
    actually stored keeps them agreeing if a bank is ever written in another
    order.
    """
    columns = [c for c in META_COLUMNS if c in meta.columns]
    if "path" not in columns:
        raise ValueError("the eval bank's meta has no `path` column, so the "
                         "error sheets cannot be rendered from it")
    merged = scores.merge(meta[columns], on="image_idx", how="left",
                          validate="many_to_one")
    unmatched = int(merged["path"].isna().sum())
    if unmatched:
        raise ValueError(
            f"{unmatched} scored row(s) have an image_idx that is not in the "
            "eval bank's meta; the scores and the bank do not belong together")
    return merged


def write_fp_by_source(path: str, by_source: pd.DataFrame, condition: str,
                       threshold: float, provenance: str, splits: dict,
                       target_fpr: float | None) -> None:
    """Write `fp_by_source.md`, as UTF-8.

    `encoding="utf-8"` is not decoration: a source name, a generator name or a
    path in this project may be non-ASCII, and a bare `open(path, "w")` encodes
    through the locale codec -- under LC_ALL=C that is ANSI_X3.4-1968 and the
    write dies with UnicodeEncodeError after both contact sheets have already
    been rendered.
    """
    fitted = ("" if target_fpr is None else
              f"It was fitted on the very rows tabulated below, so the "
              f"AGGREGATE `fp_rate` across all sources is {target_fpr:.1%} by "
              "construction and measures the threshold rather than the "
              "detector. Only the RELATIVE concentration across sources carries "
              "information. ")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# False-positive rate by source\n\n")
        f.write(f"**Condition:** `{condition}`  \n")
        f.write(f"**Rows by split:** {splits or 'not recorded in this bank'}\n\n")
        f.write("Concentration in one source indicates a confound in that "
                "dataset, not a detector weakness (spec §6.6).\n\n")
        f.write(f"**Diagnostic threshold:** {threshold:.6f} -- {provenance}. "
                + fitted +
                "This is NOT the deployment operating point; that one is fitted "
                "on internal validation by `calibrate.policy` and reported "
                "there.\n\n")
        f.write("`fp_rate` is blank for a source that contributed no authentic "
                "image: an empty denominator is not a rate of zero.\n\n")
        f.write(markdown_table(by_source) + "\n")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True,
                    help="parquet of eval.grid.score_grid output")
    ap.add_argument("--eval-bank", required=True)
    ap.add_argument("--condition", default="clean")
    ap.add_argument("--out", default="docs/errors")
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--target-fpr", type=float, default=0.01,
                    help="FPR the diagnostic threshold is placed at")
    ap.add_argument("--threshold", type=float, default=None,
                    help="explicit diagnostic threshold; overrides --target-fpr")
    return ap


def _make_stdout_encoding_safe() -> None:
    """Never let a log line kill a run that has already done the work.

    Several messages below quote spec sections (`§`), and error strings from
    `eval.errors` do too. Python encodes stdout with the LOCALE codec and
    `errors="strict"`; under LC_ALL=C -- the default in many container and CI
    images -- that is ASCII, and a single `print` raises UnicodeEncodeError
    after the artefacts are already on disk. stderr already defaults to
    `backslashreplace` for exactly this reason; this gives stdout the same
    treatment rather than making the messages illegible to avoid the codec.
    """
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, OSError):      # not a reconfigurable stream
        pass


def main(argv=None) -> str:
    a = build_parser().parse_args(argv)
    _make_stdout_encoding_safe()
    os.makedirs(a.out, exist_ok=True)

    scores = pd.read_parquet(a.scores)
    available = sorted(set(scores["condition"]))
    if a.condition not in available:
        raise ValueError(f"condition {a.condition!r} is not in the scores; "
                         f"they cover {available}")
    scores = scores[scores["condition"] == a.condition]
    scores = attach_paths(scores, FeatureBank.open(a.eval_bank).meta)

    for kind in ("fp", "fn"):
        rows = top_errors(scores, k=a.k, kind=kind)
        if rows.empty:
            print(f"no {kind} candidates for condition {a.condition!r}; "
                  "no sheet written")
            continue
        annotations = [f"{r.score:+.2f} {r.generator or 'real'}"
                       for r in rows.itertuples()]
        contact_sheet(rows, os.path.join(a.out, f"{a.condition}_{kind}.png"),
                      annotations)

    authentic = scores[scores["label"] == 0]
    if a.threshold is not None:
        threshold, provenance = float(a.threshold), "supplied on the command line"
    elif authentic.empty or scores["label"].nunique() < 2:
        raise ValueError(
            "cannot place a diagnostic threshold: condition "
            f"{a.condition!r} does not contain both classes. Pass --threshold.")
    else:
        threshold = threshold_at_fpr(scores["label"].to_numpy(),
                                     scores["score"].to_numpy(), a.target_fpr)
        provenance = (f"the lowest threshold whose FPR over the {len(authentic)} "
                      f"authentic rows of this condition does not exceed "
                      f"{a.target_fpr:.1%}")

    by_source = fp_rate_by_source(scores, threshold=threshold)
    splits = (scores.groupby("split").size().to_dict()
              if "split" in scores.columns else {})
    out_md = os.path.join(a.out, "fp_by_source.md")
    write_fp_by_source(out_md, by_source, a.condition, threshold, provenance,
                       splits, None if a.threshold is not None else a.target_fpr)
    print(by_source.to_string(index=False))
    print(f"wrote {out_md}")
    return out_md


if __name__ == "__main__":
    main()
