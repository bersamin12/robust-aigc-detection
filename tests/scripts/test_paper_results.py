import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import paper_results as pr  # noqa: E402


def _arm(path, n, seed, robust_shift=0.0, clean_shift=0.0, base_seed=0):
    """Same rows and same base noise for every arm of a seed pair."""
    rng = np.random.default_rng(base_seed)
    y = rng.integers(0, 2, n)
    dist = (np.arange(n) % 2).astype(int)
    z = 1.2 * y + rng.normal(0, 1, n)
    z = z + y * (robust_shift * dist + clean_shift * (1 - dist))
    pd.DataFrame({"path": [f"/i{k}.png" for k in range(n)], "label": y,
                  "is_distorted": dist, "logit": z}).to_parquet(path)


def _proxy(tmp, robust_shift, clean_shift=0.0):
    for i in pr.SEEDS:
        for sp in ("ntire_val_hard", "ntire_val"):
            _arm(tmp / f"abl_crop_s{i}_{sp}.parquet", 1500, i, base_seed=i)
            _arm(tmp / f"abl_squish_s{i}_{sp}.parquet", 1500, i,
                 robust_shift, clean_shift, base_seed=i)


def test_g1_passes_on_a_clear_robust_gain(tmp_path):
    _proxy(tmp_path, robust_shift=1.5)
    r, ok = pr.g1(str(tmp_path), n_boot=200)
    assert ok and r["passed"]
    assert r["mean_robust_diff"] > 0.01 and r["n_seed_ci_above_zero"] == 3
    assert "G1 PASSED" in pr.g1_markdown(r)


def test_g1_fails_on_no_gain(tmp_path):
    _proxy(tmp_path, robust_shift=0.0)
    r, ok = pr.g1(str(tmp_path), n_boot=200)
    assert not ok and not r["checks"]["mean_robust_gain_gt_0.01"]


def test_g1_fails_when_the_gain_costs_clean_auc(tmp_path):
    _proxy(tmp_path, robust_shift=1.5, clean_shift=-1.0)
    r, ok = pr.g1(str(tmp_path), n_boot=200)
    assert not ok and not r["checks"]["mean_clean_loss_at_most_0.005"]


def test_the_cli_exit_code_is_the_gate(tmp_path):
    _proxy(tmp_path, robust_shift=0.0)
    rc = pr.main(["g1", "--scores", str(tmp_path), "--n-boot", "100",
                  "--out", str(tmp_path / "gate")])
    assert rc == 3
    assert json.loads((tmp_path / "gate.json").read_text())["passed"] is False
    assert (tmp_path / "gate.md").read_text().count("FAILED") >= 1


def test_within_decile_auc_removes_a_pure_resolution_reader(tmp_path):
    """A 'model' whose score IS native size separates a size-confounded set
    pooled, and does nothing within deciles; a real detector keeps its edge."""
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 2, n)
    short = np.where(y == 1, rng.integers(300, 900, n), rng.integers(500, 1500, n))
    paths = [f"/e{k}.png" for k in range(n)]
    pd.DataFrame({"path": paths, "width": short, "height": short + 10}).to_parquet(
        tmp_path / "m.parquet")
    base = dict(path=paths, label=y)
    pd.DataFrame({**base, "logit": -short.astype(float)}).to_parquet(tmp_path / "size.parquet")
    pd.DataFrame({**base, "logit": y * 1.5 + rng.normal(0, 1, n)}).to_parquet(
        tmp_path / "det.parquet")
    r = pr.within_decile_boot({"size": str(tmp_path / "size.parquet"),
                               "det": str(tmp_path / "det.parquet")},
                              str(tmp_path / "m.parquet"), n_boot=100)
    assert r["models"]["size"]["value"] < 0.75          # pooled it is ~0.8+
    assert r["models"]["det"]["value"] > 0.8
    assert r["vs_baseline"]["det"]["excludes_zero"]
