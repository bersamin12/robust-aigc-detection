#!/usr/bin/env python3
"""Score the unfreeze ladder: each depth on the bank ITS OWN tower produced.

Why this is not `run_ablation.py`. Every rung there is scored on one shared
eval bank, and `assert_banks_comparable` enforces that -- correctly, because a
table whose rows were evaluated differently compares evaluations rather than
rungs. The unfreeze ladder cannot satisfy it: a tower whose weights moved does
not produce the features in the frozen bank, so every depth necessarily has its
own. That is the same structural problem rung A5 has with two backbones, and
rather than widen the shared guard for a second exemption this script states
the comparison it is making and checks it directly.

What it checks, before reading a single score: every bank must agree on the
manifest, the condition axis, the canonicalisation policy, the row count and
the row ORDER, and must differ ONLY in `tower_sha256`. That is the whole claim
of a depth ladder -- same evaluation, different tower -- and each half of it
fails silently on its own. A bank over a different subsample has the right
shape; a bank in a different row order joins scores to the wrong labels.

D0's bank is the frozen one and carries no `tower_sha256`, which is correct and
is why the key is compared as "the set of distinct values has one entry per
bank" rather than "present everywhere".
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from aigcdet.eval.errors import (
    SELECTION_METRIC, SELECTION_POPULATION, SELECTION_SPLITS,
    SELECTION_TARGET_FPR, heldout_robust_tpr,
)
from aigcdet.eval.grid import score_grid
from aigcdet.features.bank import FeatureBank
from aigcdet.models.heads import Detector

#: Axis keys every bank in the ladder must agree on. `backbone` is included:
#: unfreezing changes a tower's WEIGHTS, never its architecture, so a depth
#: whose bank names a different backbone is a different experiment.
_AXIS = ("manifest_sha256", "conditions", "canon_policy", "backbone", "n_views")


def _selection_summary(scores, splits) -> dict:
    """The §6.4 metric with the declarations that make it checkable.

    Same shape `run_ablation._selection_summary` produces, and deliberately so:
    `errors._check_provenance` can only refuse a contaminated result that SAYS
    what population, split set and operating point it came from, and a ladder
    written to a different shape would be exempt from that check by accident.
    `target_fpr` is passed explicitly so a change to the operating point shows
    up in a diff and contradicts the declaration two lines below it.
    """
    return {
        SELECTION_METRIC: heldout_robust_tpr(
            scores, splits, target_fpr=SELECTION_TARGET_FPR),
        "target_fpr": SELECTION_TARGET_FPR,
        "population": SELECTION_POPULATION,
        "splits": list(SELECTION_SPLITS),
    }


def _load_head(ckpt: str, device: str):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = ck["config"]
    head = Detector(dim_feat=ck["dim_feat"], use_recon=False,
                    use_film=cfg.get("use_film", False),
                    hidden=cfg.get("head_hidden", 512)).to(device)
    head.load_state_dict(ck["state_dict"])
    head.eval()
    return head, ck


def assert_ladder_comparable(banks: dict) -> None:
    names = list(banks)
    ref = banks[names[0]]
    for n in names[1:]:
        b = banks[n]
        for key in _AXIS:
            if ref.config.get(key) != b.config.get(key):
                raise ValueError(
                    f"{n} and {names[0]} disagree on {key!r}: "
                    f"{b.config.get(key)!r} against {ref.config.get(key)!r}. A "
                    "depth ladder must vary the tower and nothing else, or its "
                    "rows compare evaluations rather than depths.")
        if len(b.meta) != len(ref.meta):
            raise ValueError(
                f"{n} holds {len(b.meta)} rows and {names[0]} holds "
                f"{len(ref.meta)}: the depths were scored on different "
                "subsamples.")
        if not np.array_equal(b.meta["row_id"].to_numpy(),
                              ref.meta["row_id"].to_numpy()):
            raise ValueError(
                f"{n} and {names[0]} hold the same rows in a different ORDER. "
                "Scores are joined positionally, so a depth's row would carry "
                "another image's label.")
    towers = {n: b.config.get("tower_sha256") for n, b in banks.items()}
    distinct = len({v for v in towers.values()})
    if distinct != len(towers):
        dupes = [n for n in towers if list(towers.values()).count(towers[n]) > 1]
        raise ValueError(
            f"two or more depths were scored on banks with the SAME tower "
            f"weights ({dupes}). Either a bank was reused across depths or a "
            "training run did not move the tower -- both make the ladder "
            "report that depth does not help, from a table that looks fine.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", action="append", required=True,
                    metavar="NAME=CHECKPOINT:BANK",
                    help="repeatable, e.g. d0=outputs/unfreeze/d0/checkpoint.pt:"
                         "data/banks/eval_d0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    rungs = {}
    for spec in a.rung:
        name, rest = spec.split("=", 1)
        ckpt, bank = rest.rsplit(":", 1)
        rungs[name] = (ckpt, bank)

    banks = {n: FeatureBank.open(b) for n, (_, b) in rungs.items()}
    assert_ladder_comparable(banks)
    print("ladder comparability: OK "
          f"({len(banks)} depths, same manifest / conditions / policy / rows, "
          "distinct towers)")

    report = {"metric": SELECTION_METRIC, "rungs": {}}
    for name, (ckpt, bank_dir) in rungs.items():
        bank = banks[name]
        head, ck = _load_head(ckpt, a.device)
        scores = score_grid(head, bank, use_recon=False, device=a.device)
        splits = bank.meta["split"].to_numpy()
        summary = _selection_summary(scores, splits)
        u = ck.get("unfrozen") or {}
        report["rungs"][name] = summary | {
            "depth": u.get("depth"),
            "trainable_params": u.get("trainable_params"),
            "tower_sha256": bank.config.get("tower_sha256"),
            "checkpoint": ckpt, "eval_bank": bank_dir,
        }
        print(f"{name}: depth={u.get('depth')} "
              f"trainable={u.get('trainable_params', 0):,} "
              f"{SELECTION_METRIC}={summary[SELECTION_METRIC]:.4f}")

    base = report["rungs"].get("d0", {}).get(SELECTION_METRIC)
    if base is not None:
        print(f"\nread against D0 ({base:.4f}):")
        for n, r in report["rungs"].items():
            if n == "d0":
                continue
            print(f"  {n}: {r[SELECTION_METRIC] - base:+.4f}")
        report["deltas_vs_d0"] = {
            n: r[SELECTION_METRIC] - base
            for n, r in report["rungs"].items() if n != "d0"}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
