#!/usr/bin/env python3
"""One markdown table from a directory of `score_plan_splits.py` metrics
files: a row per checkpoint, the challenge's robust / clean ROC AUC per split.

    python scripts/metrics_table.py outputs/scores/epochs --out outputs/scores/epochs/summary.md
    python scripts/metrics_table.py outputs/scores/ablation --pattern 'abl_pool_*_metrics.json'

Rows are labelled by the file stem (`<prefix>_metrics.json` -> `<prefix>`),
sorted by name, so `ntire_dual224_ep1`, `ntire_dual224_ep2`, ... read as a
curve. The leaderboard reference (arXiv 2604.11487, winner) is printed under
the table for scale, never as a row: it is a private-test number and these
are not.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

LEADERBOARD = {"MICV (1st)": (0.9723, 0.9974), "Ant International (2nd)": (0.9721, 0.9972),
               "TeleAI-TeleGuard (3rd)": (0.9251, 0.9786)}


def _f(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def _epoch_key(name: str):
    m = re.search(r"_ep(\d+)$", name)
    return (name[:m.start()], int(m.group(1))) if m else (name, -1)


def load(scores_dir: str, pattern="*_metrics.json") -> list[dict]:
    """`pattern` may be one glob or a list of them (a file matched by two is
    one row), so a table can hold an ablation's arms beside the earlier run
    that is their baseline without re-running it."""
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    paths = sorted({p for pat in patterns
                    for p in glob.glob(os.path.join(scores_dir, pat))})
    rows = []
    for p in paths:
        with open(p) as f:
            m = json.load(f)
        name = os.path.basename(p)[:-len("_metrics.json")]
        rows.append({"name": name, **m})
    rows.sort(key=lambda r: _epoch_key(r["name"]))
    return rows


def render(rows: list[dict], splits: list[str] | None = None) -> str:
    if not rows:
        return "_no metrics files yet_\n"
    if splits is None:
        seen: list[str] = []
        for r in rows:
            for sp in r.get("splits", {}):
                if sp not in seen:
                    seen.append(sp)
        splits = seen
    head = ["run", "epoch", "weights", "pool", "dim"]
    for sp in splits:
        head += [f"{sp} robust", f"{sp} clean", f"{sp} AUC"]
    head.append("objective")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["name"], str(r.get("epoch", "")), str(r.get("weights", "")),
                 str(r.get("token_pool", "mean")), str(r.get("dim_feat", ""))]
        obj = []
        for sp in splits:
            s = r.get("splits", {}).get(sp)
            if not s:
                cells += ["", "", ""]
                continue
            cells += [_f(s.get("robust_auc")), _f(s.get("clean_auc")), _f(s.get("auc"))]
            if sp.startswith("ntire_val") and s.get("robust_auc") is not None \
                    and s.get("clean_auc") is not None:
                obj.append((s["robust_auc"] + s["clean_auc"]) / 2)
        # The fusion objective: mean(robust, clean) over the selection splits.
        cells.append(_f(sum(obj) / len(obj)) if obj else "")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Leaderboard (private test, arXiv 2604.11487): "
                 + "; ".join(f"{k} {a:.4f} / {c:.4f}" for k, (a, c) in LEADERBOARD.items())
                 + " (robust / clean).")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scores_dir")
    ap.add_argument("--pattern", action="append", default=None,
                    help="glob under scores_dir; repeatable (default all "
                         "*_metrics.json)")
    ap.add_argument("--splits", default=None, help="comma list; default: all seen")
    ap.add_argument("--out", default=None, help="write the markdown here too")
    a = ap.parse_args(argv)
    rows = load(a.scores_dir, a.pattern or "*_metrics.json")
    text = render(rows, a.splits.split(",") if a.splits else None)
    sys.stdout.write(text)
    if a.out:
        tmp = a.out + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
