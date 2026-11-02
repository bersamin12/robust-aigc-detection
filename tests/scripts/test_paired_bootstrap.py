import importlib.util
import json
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

_SPEC = importlib.util.spec_from_file_location(
    "paired_bootstrap", os.path.join(os.path.dirname(__file__), "..", "..",
                                     "scripts", "paired_bootstrap.py"))
pb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pb)


def _frame(n=600, seed=0, shift=0.0, noise=1.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    dist = rng.integers(0, 2, n)
    base = y * 1.0 + rng.normal(0, noise, n)
    return pd.DataFrame({"path": [f"img{i}.png" for i in range(n)], "label": y,
                         "is_distorted": dist, "logit": base + shift * y * dist})


def test_auc_matches_sklearn_including_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.normal(size=500) + y, 1)       # rounding forces ties
    assert pb.auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_identical_models_differ_by_exactly_zero():
    d = _frame()
    df = d.assign(a=d["logit"], b=d["logit"])
    res = pb.paired_bootstrap(df, ["a", "b"], "a", n_boot=200)
    for k in ("auc", "robust_auc", "clean_auc"):
        assert res["vs_baseline"]["b"][k]["diff"] == 0.0
        assert res["vs_baseline"]["b"][k]["ci"] == [0.0, 0.0]


def test_a_real_robust_gain_is_detected_and_does_not_leak_into_clean():
    d = _frame(n=2000, seed=3)
    better = _frame(n=2000, seed=3, shift=1.5)
    df = d.assign(base=d["logit"], better=better["logit"])
    res = pb.paired_bootstrap(df, ["base", "better"], "base", n_boot=300)
    rob = res["vs_baseline"]["better"]["robust_auc"]
    assert rob["diff"] > 0 and rob["excludes_zero"]
    # the shift only touches distorted rows, so clean is unchanged exactly
    assert res["vs_baseline"]["better"]["clean_auc"]["diff"] == 0.0


def test_the_cli_joins_on_path_and_refuses_disagreeing_labels(tmp_path):
    a, b = _frame(seed=5), _frame(seed=5, shift=0.5)
    a.sample(frac=1, random_state=0).to_parquet(tmp_path / "a.parquet")
    b.to_parquet(tmp_path / "b.parquet")
    out = tmp_path / "res.json"
    pb.main(["--model", f"a={tmp_path / 'a.parquet'}",
             "--model", f"b={tmp_path / 'b.parquet'}:logit",
             "--baseline", "a", "--n-boot", "50", "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["n"] == len(a)
    assert (tmp_path / "res.md").exists()
    bad = b.copy()
    bad.loc[0, "label"] = 1 - bad.loc[0, "label"]
    bad.to_parquet(tmp_path / "bad.parquet")
    with pytest.raises(SystemExit, match="disagrees"):
        pb.main(["--model", f"a={tmp_path / 'a.parquet'}",
                 "--model", f"bad={tmp_path / 'bad.parquet'}",
                 "--out", str(tmp_path / "x.json")])


def test_a_split_without_is_distorted_reports_pooled_auc_only():
    d = _frame().drop(columns="is_distorted")
    df = d.assign(a=d["logit"], b=d["logit"] * 2)
    res = pb.paired_bootstrap(df, ["a", "b"], "a", n_boot=50)
    assert set(res["models"]["a"]) == {"auc"}
