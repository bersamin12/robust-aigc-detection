#!/usr/bin/env python3
"""Amendment 7, re-analysis R2 (supplement to Test 4): JPEG vs JPEG-free gain.

    PYTHONPATH=src:scripts python scripts/amend5_test4_supp_a7.py

Test 4 (committed, 4652e27) counts a row in every family its chain contains,
so its family gains overlap. R2 splits the distorted rows into two DISJOINT
groups by Test 4's own family map:

  JPEG     the row's chain contains at least one JPEG-family token
           (jpeg, jpeg_ai, jpeg_recompression_*, quantization)
  no-JPEG  every other distorted row

Gain, as in Test 4: robust AUC(squish) - AUC(centre) within the group's
distorted rows. Reported per branch (A2, B2swa) on ntire_val_hard (primary,
fit split) and ntire_val (centre arm = as-trained score, plan decision D14).

Bootstrap: paper_analyses.paired_diff's index scheme (seed 20260827,
default_rng, cells from np.unique(strata), per draw concatenate
c[rng.integers(0, len(c), len(c))]), 2000 draws, nanquantile 2.5/97.5.
Strata: label x is_distorted x group. Both group gains, their difference
and the pooled gain are computed on each draw. The pooled gain's interval
is asserted equal to paired_diff's on the same rows and strata.

It also checks Test 4 section 4.3's statement that `joint_gain_draws` uses a
scheme "identical to paired_diff" (see `check_test4_resampler`).

Reads only saved per-image logits; loads no model, reads no image, never
reads ntire_test. Writes docs/ntire/amend5/test4_supp_a7.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from amend5_test4 import FAMILIES, joint_gain_draws, load  # noqa: E402
from paired_bootstrap import auc  # noqa: E402
from paper_analyses import MANIFEST, SEED, _tokens, commit, paired_diff  # noqa: E402

OUT = "docs/ntire/amend5/test4_supp_a7.json"
SPLITS = ("ntire_val_hard", "ntire_val")
BRANCHES = {"A2": "A", "B2swa": "B"}  # parquet name -> column prefix used by Test 4's load()
N_BOOT = 2000
JPEG_TOKENS = FAMILIES["JPEG"]


def assert_no_ntire_test(frame: pd.DataFrame, split: str) -> None:
    """Every row must be in the named split of the manifest, none in ntire_test."""
    manifest = pd.read_parquet(MANIFEST, columns=["path", "split", "distortions"]).drop_duplicates("path")
    joined = frame[["path", "distortions"]].merge(manifest, on="path", how="left",
                                                  suffixes=("", "_manifest"))
    assert joined["split"].notna().all(), f"{split}: rows missing from the manifest"
    assert not joined["split"].eq("ntire_test").any(), "R2 never reads ntire_test"
    assert joined["split"].eq(split).all(), f"{split}: rows from another split"
    same_chain = joined["distortions"].apply(_tokens) == joined["distortions_manifest"].apply(_tokens)
    assert same_chain.all(), f"{split}: score parquet and manifest disagree on distortions"


def add_group(frame: pd.DataFrame) -> pd.DataFrame:
    """Disjoint group label from Test 4's parsed tokens and family map."""
    contains_jpeg = frame["tokens"].apply(lambda tokens: any(t in JPEG_TOKENS for t in tokens))
    families_say_jpeg = frame["families"].apply(lambda families: "JPEG" in families)
    assert (contains_jpeg == families_say_jpeg).all(), "group disagrees with Test 4's family column"
    return frame.assign(group=np.where(contains_jpeg, "JPEG", "no-JPEG"))


def make_strata(frame: pd.DataFrame) -> np.ndarray:
    """label x is_distorted x group, as one string per row."""
    label = frame["label"].astype(int).astype(str)
    distorted = frame["is_distorted"].astype(int).astype(str)
    return (label + "|" + distorted + "|" + frame["group"]).to_numpy()


def draw_indices(strata: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    """The resample indices paired_diff draws, in the same order, one array per draw."""
    rng = np.random.default_rng(seed)
    cells = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    for _ in range(n_boot):
        yield np.concatenate([c[rng.integers(0, len(c), len(c))] for c in cells])


def gain(y: np.ndarray, center: np.ndarray, squish: np.ndarray) -> float:
    """Test 4's gain: AUC(squish) - AUC(centre)."""
    return auc(y, squish) - auc(y, center)


def statistics(y, center, squish, is_jpeg, index=None) -> dict[str, float]:
    """Group gains, their difference and the pooled gain on one set of rows."""
    if index is None:
        index = np.arange(len(y))
    y_draw, center_draw, squish_draw = y[index], center[index], squish[index]
    jpeg_draw = is_jpeg[index]
    gain_jpeg = gain(y_draw[jpeg_draw], center_draw[jpeg_draw], squish_draw[jpeg_draw])
    gain_free = gain(y_draw[~jpeg_draw], center_draw[~jpeg_draw], squish_draw[~jpeg_draw])
    return {"gain_jpeg": gain_jpeg, "gain_no_jpeg": gain_free,
            "diff_jpeg_minus_no_jpeg": gain_jpeg - gain_free,
            "gain_pooled": gain(y_draw, center_draw, squish_draw)}


def branch_result(frame: pd.DataFrame, prefix: str) -> dict:
    """Point estimates and intervals for one branch on one split."""
    y = frame["label"].to_numpy().astype(int)
    center = frame[f"{prefix}_center"].to_numpy()
    squish = frame[f"{prefix}_squish"].to_numpy()
    is_jpeg = (frame["group"] == "JPEG").to_numpy()
    strata = make_strata(frame)

    point = statistics(y, center, squish, is_jpeg)
    draws = [statistics(y, center, squish, is_jpeg, k) for k in draw_indices(strata)]
    result = {}
    for name, value in point.items():
        values = np.array([draw[name] for draw in draws])
        lo, hi = np.nanquantile(values, [0.025, 0.975])
        result[name] = {"point": float(value), "ci": [float(lo), float(hi)]}

    # The index scheme must be paired_diff's: same pooled gain interval on the same strata.
    reference = paired_diff(y, center, squish, strata)
    assert np.isclose(reference["diff"], result["gain_pooled"]["point"]), (reference, result["gain_pooled"])
    assert reference["ci"] == result["gain_pooled"]["ci"], (reference["ci"], result["gain_pooled"]["ci"])
    result["pooled_matches_paired_diff"] = True
    return result


def counts(frame: pd.DataFrame) -> dict:
    """Rows per group, split by label, plus the JPEG rows that also contain downscale."""
    out = {}
    for group, rows in frame.groupby("group"):
        has_downscale = rows["tokens"].apply(lambda tokens: "downscale" in tokens)
        out[group] = {"n": int(len(rows)), "n_real": int((rows["label"] == 0).sum()),
                      "n_fake": int((rows["label"] == 1).sum()),
                      "n_with_downscale": int(has_downscale.sum()),
                      "short_p50": float(rows["short"].median())}
    return out


def check_test4_resampler(frame: pd.DataFrame) -> dict:
    """Test 4 section 4.3 says joint_gain_draws's scheme is identical to paired_diff's.

    joint_gain_draws builds its cells from ALL distorted rows (strata = label),
    draws once, then keeps the drawn rows that belong to the family.
    paired_diff, as Test 4 calls it per family, builds its cells from the
    FAMILY's rows only (strata = label within the family). Two cases:

      all-rows cell: a pseudo-family holding every distorted row. Both
        methods should then give the same draws, bit for bit.
      JPEG family:   the real case. Draws are compared directly, plus the
        resampled family size, which paired_diff holds fixed.
    """
    y = frame["label"].to_numpy().astype(int)
    out = {}
    for parquet_name, prefix in BRANCHES.items():
        center = frame[f"{prefix}_center"].to_numpy()
        squish = frame[f"{prefix}_squish"].to_numpy()
        with_all = frame.assign(families=frame["families"].apply(lambda fams: list(fams) + ["ALL"]))
        test4_draws = joint_gain_draws(with_all, ["ALL", "JPEG"], prefix)

        # paired_diff's draws over all rows, strata = label (assert we have its exact draws).
        all_draws = np.array([gain(y[k], center[k], squish[k]) for k in draw_indices(y)])
        assert paired_diff(y, center, squish, y)["ci"] == [float(q) for q in np.nanquantile(all_draws, [.025, .975])]

        # paired_diff's draws as Test 4 calls it for the JPEG row of table (a).
        jpeg_rows = frame[frame["families"].apply(lambda fams: "JPEG" in fams)]
        y_j = jpeg_rows["label"].to_numpy().astype(int)
        c_j, s_j = jpeg_rows[f"{prefix}_center"].to_numpy(), jpeg_rows[f"{prefix}_squish"].to_numpy()
        jpeg_draws_pd = np.array([gain(y_j[k], c_j[k], s_j[k]) for k in draw_indices(y_j)])
        assert paired_diff(y_j, c_j, s_j, y_j)["ci"] == [float(q) for q in np.nanquantile(jpeg_draws_pd, [.025, .975])]

        # Size of the JPEG family inside each joint_gain_draws resample.
        member = frame["families"].apply(lambda fams: "JPEG" in fams).to_numpy()
        family_sizes = np.array([int(member[k].sum()) for k in draw_indices(y)])

        out[parquet_name] = {
            "all_rows_cell_identical": bool(np.array_equal(test4_draws["ALL"], all_draws)),
            "jpeg_family_identical": bool(np.array_equal(test4_draws["JPEG"], jpeg_draws_pd)),
            "jpeg_family_max_abs_draw_diff": float(np.max(np.abs(test4_draws["JPEG"] - jpeg_draws_pd))),
            "jpeg_ci_joint_gain_draws": [float(q) for q in np.nanquantile(test4_draws["JPEG"], [.025, .975])],
            "jpeg_ci_paired_diff": [float(q) for q in np.nanquantile(jpeg_draws_pd, [.025, .975])],
            "jpeg_n_fixed_in_paired_diff": int(len(jpeg_rows)),
            "jpeg_n_per_draw_in_joint_gain_draws": {"min": int(family_sizes.min()),
                                                    "max": int(family_sizes.max()),
                                                    "sd": float(family_sizes.std())},
        }
    return out


def main() -> int:
    report = {"reanalysis": "R2", "test": 4, "amendment": 7, "commit": commit(), "seed": SEED,
              "n_boot": N_BOOT, "jpeg_tokens": sorted(JPEG_TOKENS), "splits": {}}
    for split in SPLITS:
        assert split != "ntire_test"
        frame = add_group(load(split))
        assert_no_ntire_test(frame, split)
        entry = {"n_distorted": int(len(frame)), "counts": counts(frame), "branches": {}}
        for parquet_name, prefix in BRANCHES.items():
            entry["branches"][parquet_name] = branch_result(frame, prefix)
            r = entry["branches"][parquet_name]
            print(f"{split} {parquet_name}: JPEG {r['gain_jpeg']['point']:+.4f} "
                  f"no-JPEG {r['gain_no_jpeg']['point']:+.4f} "
                  f"diff {r['diff_jpeg_minus_no_jpeg']['point']:+.4f} {r['diff_jpeg_minus_no_jpeg']['ci']}",
                  flush=True)
        if split == "ntire_val_hard":
            entry["test4_resampler_check"] = check_test4_resampler(frame)
            print("resampler check:", json.dumps(entry["test4_resampler_check"]), flush=True)
        report["splits"][split] = entry
    with open(OUT, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"R2 -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
