#!/usr/bin/env python3
"""Amendment 11, Test 19, analysis: does enlarging the window hurt, at equal view?

    python scripts/amend5_test19_analysis.py                      # the registered analysis
    python scripts/amend5_test19_analysis.py --write-smoke-manifest PATH
    python scripts/amend5_test19_analysis.py --smoke              # smoke rows only, not results

Three proxies on Branch A's tower, one seed each, centre window at test:

  crop200@512   200 px window enlarged to 512 (and 518 in the tower)   Test 15's run
  native512     512 px window at 512                                   Test 15's run
  crop196@196   196 px window at 196, no resampling                    this test's run

crop196@196 is the registered substitute for crop200@200 (amendment 11):
196 = 14 x 14 patches, while 200 would be squished by the tower.

Registered statistics (robust AUC = AUC on the distorted rows):

  E = AUC(crop196@196) - AUC(crop200@512)   same view, enlarged vs not: > 0 means
                                             enlarging hurts (stretch penalty)
  V = AUC(native512) - AUC(crop196@196)     both unenlarged: the view (and tokens)
  T = AUC(native512) - AUC(crop200@512)     Test 15's effect; T = V + E

Intervals: paired_diff (2,000 draws, 95% percentile, seed 20260827), strata
label x is_distorted. E, V and T on the same rows in the same order call
paired_diff with the same seed and strata, so all three share one set of
bootstrap indices. Rows are kept in the order of Test 15's crop200 parquet, so
T's interval on all rows is Test 15's own.

Registered condition on E, ntire_val_hard, distorted rows with short side >= 512:
  PENALTY    E's CI above zero
  BENEFIT    E's CI below zero
  NONE       E's CI within [-0.01, +0.01] (and so including zero)
  UNRESOLVED otherwise

Writes docs/ntire/amend5/test19_stretch.json (smoke: outputs/test19/smoke/).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_bootstrap import auc  # noqa: E402
from paper_analyses import SEED, commit, paired_diff  # noqa: E402

TEST15_SCORES = "/scratch/dion_r13/dion_extensions/outputs/test15"
TEST15_JSON = "/scratch/dion_r13/dion_extensions/wt/test15/docs/ntire/amend5/test15_native512-proxy.json"
TEST19_SCORES = "/scratch/dion_r13/dion_extensions/outputs/test19"
MANIFEST = "data/manifest_ntire.parquet"
OUT = "docs/ntire/amend5/test19_stretch.json"
SMOKE_OUT = f"{TEST19_SCORES}/smoke/test19_stretch_smoke.json"

#: arm -> (parquet file stem, window side). The first two are Test 15's runs.
ARMS = {
    "crop200@512": (f"{TEST15_SCORES}/abl512_crop200_s1", 200),
    "native512": (f"{TEST15_SCORES}/abl512_native512_s1", 512),
    "crop196@196": ("abl196_crop196_s1", 196),   # relative to the test 19 scores dir
}
SPLITS = ("ntire_val_hard", "ntire_val")
PRIMARY_SPLIT = "ntire_val_hard"
#: Primary rows: short side >= 512, where no arm clamps (every window fits).
PRIMARY_MIN_SHORT = 512
#: NONE's band for E, registered.
NONE_MARGIN = 0.01
BUCKETS = ((80, 336), (352, 512), (528, 736), (752, 4016))  # Test 15's (ntire_crop_probe.BUCKETS)
#: Patch tokens each arm's tower sees: 518 / 14 = 37 per side, 196 / 14 = 14.
TOKENS = {"crop200@512": 37 * 37, "native512": 37 * 37, "crop196@196": 14 * 14}


def arm_parquet(arm: str, split: str, scores_dir: str) -> str:
    stem, _ = ARMS[arm]
    stem = stem if stem.startswith("/") else f"{scores_dir}/{stem}"
    return f"{stem}_{split}.parquet"


def load(split: str, scores_dir: str, smoke: bool) -> pd.DataFrame:
    """The three arms' logits on one split, joined on path, with the native short side.

    Rows keep the order of Test 15's crop200 parquet. Full run: asserts the three
    arms scored exactly the same set of paths. Smoke run: the new arm scored a
    handful of rows, so Test 15's arms are cut down to those rows, and the
    assertion is that they are a subset of Test 15's.
    """
    assert split != "ntire_test", "Test 19 never reads ntire_test"
    frames = {arm: pd.read_parquet(arm_parquet(arm, split, scores_dir)) for arm in ARMS}
    for arm, frame in frames.items():
        assert frame["path"].is_unique, f"{arm} {split}: duplicate paths"
    test15_paths = set(frames["crop200@512"]["path"])
    assert set(frames["native512"]["path"]) == test15_paths, "Test 15's two arms scored different rows"
    new_paths = set(frames["crop196@196"]["path"])
    if smoke:
        assert new_paths and new_paths <= test15_paths, "smoke rows are not Test 15 rows"
    else:
        assert new_paths == test15_paths, f"{split}: crop196@196 scored a different row set from Test 15"

    base = frames["crop200@512"].loc[frames["crop200@512"]["path"].isin(new_paths)]
    joined = base[["path", "label", "is_distorted"]].reset_index(drop=True)
    for arm, frame in frames.items():
        by_path = frame.set_index("path").loc[joined["path"]]
        assert (by_path["label"].to_numpy() == joined["label"].to_numpy()).all(), f"{arm}: labels disagree"
        assert (by_path["is_distorted"].to_numpy() == joined["is_distorted"].to_numpy()).all(), \
            f"{arm}: is_distorted disagrees"
        joined[f"logit_{arm}"] = by_path["logit"].to_numpy()

    sizes = pd.read_parquet(MANIFEST, columns=["path", "width", "height"]).set_index("path")
    found = sizes.reindex(joined["path"])
    assert found.notna().all().all(), f"{split}: rows missing from the manifest"
    joined["short_side"] = np.minimum(found["width"].to_numpy(), found["height"].to_numpy())
    return joined


def contrasts(rows: pd.DataFrame) -> dict:
    """AUC per arm and E, V, T with paired intervals on these rows.

    Each paired_diff call reseeds with SEED and gets the same strata in the same
    row order, so E, V and T are evaluated on one shared set of bootstrap draws.
    """
    labels = rows["label"].to_numpy().astype(int)
    strata = labels * 2 + rows["is_distorted"].to_numpy().astype(int)   # label x is_distorted
    logit = {arm: rows[f"logit_{arm}"].to_numpy() for arm in ARMS}
    # paired_diff(y, a, b, strata) returns AUC(b) - AUC(a).
    stretch = paired_diff(labels, logit["crop200@512"], logit["crop196@196"], strata)   # E
    view = paired_diff(labels, logit["crop196@196"], logit["native512"], strata)        # V
    total = paired_diff(labels, logit["crop200@512"], logit["native512"], strata)       # T
    assert abs(total["diff"] - (view["diff"] + stretch["diff"])) < 1e-12, "T != V + E"
    return {
        "n": int(len(rows)),
        "n_real": int((labels == 0).sum()), "n_fake": int((labels == 1).sum()),
        "auc": {"crop200@512": total["a"], "native512": total["b"], "crop196@196": stretch["b"]},
        "E": {"diff": stretch["diff"], "ci": stretch["ci"]},
        "V": {"diff": view["diff"], "ci": view["ci"]},
        "T": {"diff": total["diff"], "ci": total["ci"]},
        # Descriptions, not tests (T can be near zero on small subsets).
        "E_over_T": stretch["diff"] / total["diff"] if total["diff"] else None,
        "V_over_T": view["diff"] / total["diff"] if total["diff"] else None,
    }


def verdict(ci: list[float]) -> str:
    """The registered reading of E's interval."""
    lower, upper = ci
    if lower > 0:
        return "PENALTY"
    if upper < 0:
        return "BENEFIT"
    if lower >= -NONE_MARGIN and upper <= NONE_MARGIN:
        return "NONE"
    return "UNRESOLVED"


def test15_reproduced() -> dict:
    """T on all distorted rows, from Test 15's own parquets, against Test 15's json (4 dp).

    Run on the full Test 15 rows in both modes, so the smoke run checks it too.
    """
    report = json.load(open(TEST15_JSON))
    checks = {}
    for split in SPLITS:
        crop = pd.read_parquet(arm_parquet("crop200@512", split, "")).set_index("path")
        native = pd.read_parquet(arm_parquet("native512", split, "")).set_index("path").loc[crop.index]
        distorted = crop["is_distorted"].to_numpy() == 1
        labels = crop["label"].to_numpy().astype(int)[distorted]
        recomputed = (auc(labels, native["logit"].to_numpy()[distorted])
                      - auc(labels, crop["logit"].to_numpy()[distorted]))
        registered = report["splits"][split]["robust"]["diff"]
        assert round(recomputed, 4) == round(registered, 4), \
            f"{split}: T on all rows {recomputed:+.4f} != Test 15's {registered:+.4f}"
        checks[split] = {"recomputed": recomputed, "test15": registered}
    return checks


def split_report(rows: pd.DataFrame) -> dict:
    """Primary-size and all rows, robust and clean, and robust by size bucket."""
    distorted = rows["is_distorted"] == 1
    large = rows["short_side"] >= PRIMARY_MIN_SHORT
    entry = {
        "short_ge_512": {"robust": contrasts(rows[distorted & large]),
                         "clean": contrasts(rows[~distorted & large])},
        "all_rows": {"robust": contrasts(rows[distorted]), "clean": contrasts(rows[~distorted])},
        "clamped": {arm: int((rows["short_side"] < side).sum()) for arm, (_, side) in ARMS.items()},
        "clamped_short_ge_512": {arm: int((large & (rows["short_side"] < side)).sum())
                                 for arm, (_, side) in ARMS.items()},
        "robust_by_bucket": {},
    }
    for low, high in BUCKETS:
        in_bucket = distorted & rows["short_side"].between(low, high)
        bucket_labels = rows.loc[in_bucket, "label"]
        if bucket_labels.nunique() == 2:      # a bucket AUC needs both labels
            entry["robust_by_bucket"][f"{low}-{high}"] = contrasts(rows[in_bucket])
    assert all(n == 0 for n in entry["clamped_short_ge_512"].values()), "an arm clamps a primary row"
    return entry


def check_against_test15(split: str, entry: dict) -> None:
    """Full run only: the joined rows reproduce Test 15's own row counts and T values."""
    test15 = json.load(open(TEST15_JSON))["splits"][split]
    all_robust = entry["all_rows"]["robust"]
    assert all_robust["n"] == test15["robust"]["n"], "distorted row count differs from Test 15"
    assert round(all_robust["T"]["diff"], 4) == round(test15["robust"]["diff"], 4), "T differs from Test 15"
    if "robust_by_bucket" in test15:
        for bucket, numbers in test15["robust_by_bucket"].items():
            assert entry["robust_by_bucket"][bucket]["n"] == numbers["n"], f"bucket {bucket} row count differs"
            assert round(entry["robust_by_bucket"][bucket]["T"]["diff"], 4) == round(numbers["diff"], 4)
        assert entry["clamped"]["crop200@512"] == test15["clamped"]["crop200"]
        assert entry["clamped"]["native512"] == test15["clamped"]["native512"]
    # Same rows, order and seed as Test 15, so its interval should come back exactly.
    entry["T_ci_equals_test15"] = bool(np.allclose(all_robust["T"]["ci"], test15["robust"]["ci"]))


def write_smoke_manifest(path: str) -> None:
    """A handful of rows per split for the smoke run: two per label x is_distorted
    with short side >= 512, and one per label below 512 (seed 20260827)."""
    manifest = pd.read_parquet(MANIFEST)
    manifest["short_side"] = manifest[["width", "height"]].min(axis=1)
    picked = []
    for split in SPLITS:
        rows = manifest[manifest["split"] == split]
        for (label, distorted), cell in rows[rows["short_side"] >= 512].groupby(["label", "is_distorted"]):
            picked.append(cell.sample(2, random_state=SEED))
        for label, cell in rows[rows["short_side"] < 512].groupby("label"):
            picked.append(cell.sample(1, random_state=SEED))
    smoke_rows = pd.concat(picked).drop(columns="short_side")
    assert "ntire_test" not in set(smoke_rows["split"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    smoke_rows.to_parquet(path)
    print(f"smoke manifest: {len(smoke_rows)} rows -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--smoke", action="store_true", help="analyse the smoke scores (not results)")
    parser.add_argument("--write-smoke-manifest", metavar="PATH")
    args = parser.parse_args()
    if args.write_smoke_manifest:
        write_smoke_manifest(args.write_smoke_manifest)
        return 0

    scores_dir = f"{TEST19_SCORES}/smoke" if args.smoke else TEST19_SCORES
    out = SMOKE_OUT if args.smoke else OUT
    report = {"test": 19, "smoke": args.smoke, "commit": commit(), "seed": SEED, "n_boot": 2000,
              "input_size_option": "substitute n = 196, window 196 (registered crop200@200 not run)",
              "tower_patch_tokens": TOKENS,
              "token_ratio_enlarged_vs_not": TOKENS["crop200@512"] / TOKENS["crop196@196"],
              "checks": {"test15_T_all_rows_4dp": test15_reproduced()},
              "splits": {}}
    for split in SPLITS:
        rows = load(split, scores_dir, args.smoke)
        entry = split_report(rows)
        if not args.smoke:
            check_against_test15(split, entry)
        report["splits"][split] = entry
        primary = entry["short_ge_512"]["robust"]
        print(f"{split} robust, short >= 512 (n {primary['n']}): "
              + "  ".join(f"{k} {primary[k]['diff']:+.4f} [{primary[k]['ci'][0]:+.4f}, {primary[k]['ci'][1]:+.4f}]"
                          for k in ("E", "V", "T")), flush=True)

    primary_e = report["splits"][PRIMARY_SPLIT]["short_ge_512"]["robust"]["E"]
    report["verdict"] = verdict(primary_e["ci"])
    report["checks"].update({"paths_joined_equal_row_sets": True, "T_equals_V_plus_E": True})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        json.dump(report, handle, indent=1)
    label = "SMOKE (not a result)" if args.smoke else "Test 19"
    print(f"{label}: E's CI {primary_e['ci']} -> {report['verdict']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
