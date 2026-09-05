"""`split_metrics` and the policy override in `scripts/score_plan_splits.py`.

Pure functions only: no tower, no GPU. The scoring loop itself is exercised
through `tests/scripts/test_predict.py`'s stubs.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import score_plan_splits as sps  # noqa: E402


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    dist = rng.integers(0, 2, n)
    logit = 3.0 * y + rng.normal(0, 1.5, n)
    toks = ["[]" if not d else ("['jpeg']" if i % 3 else "['jpeg', 'blur']")
            for i, d in enumerate(dist)]
    return pd.DataFrame({"path": [f"/p{i}" for i in range(n)], "label": y,
                         "source": ["ntire_val"] * n, "is_distorted": dist,
                         "distortions": toks, "logit": logit,
                         "prob": 1 / (1 + np.exp(-logit))})


def test_robust_and_clean_auc_split_on_is_distorted():
    from sklearn.metrics import roc_auc_score
    d = _frame()
    m = sps.split_metrics(d)
    y, p, dist = d.label.to_numpy(), d.prob.to_numpy(), d.is_distorted.to_numpy()
    assert m["robust_auc"] == pytest.approx(roc_auc_score(y[dist == 1], p[dist == 1]))
    assert m["clean_auc"] == pytest.approx(roc_auc_score(y[dist == 0], p[dist == 0]))
    assert m["n_robust"] == int((dist == 1).sum()) and m["n_clean"] == int((dist == 0).sum())
    # Flipping a clean row's label cannot move the robust number.
    d2 = d.copy()
    i = int(np.flatnonzero(dist == 0)[0])
    d2.loc[i, "label"] = 1 - d2.loc[i, "label"]
    assert sps.split_metrics(d2)["robust_auc"] == m["robust_auc"]
    assert sps.split_metrics(d2)["clean_auc"] != m["clean_auc"]


def test_per_distortion_auc_pits_that_token_against_every_clean_row():
    from sklearn.metrics import roc_auc_score
    d = _frame()
    m = sps.split_metrics(d)
    assert set(m["by_distortion"]) == {"jpeg", "blur"}
    toks = [sps._distortion_tokens(s) for s in d.distortions]
    has = np.array(["blur" in t for t in toks])
    mask = has | (d.is_distorted.to_numpy() == 0)
    want = roc_auc_score(d.label.to_numpy()[mask], d.prob.to_numpy()[mask])
    assert m["by_distortion"]["blur"]["auc"] == pytest.approx(want)
    assert m["by_distortion"]["blur"]["n"] == int(has.sum())


def test_metrics_without_is_distorted_are_unchanged_in_shape():
    d = _frame().drop(columns=["is_distorted", "distortions"])
    m = sps.split_metrics(d)
    assert "robust_auc" not in m and "by_distortion" not in m
    assert set(m) == {"n", "auc", "tpr_at_1pct_fpr", "acc_at_0.5", "by_source"}


def test_single_class_half_reports_none():
    d = _frame()
    d.loc[d.is_distorted == 1, "label"] = 1
    assert sps.split_metrics(d)["robust_auc"] is None


def test_distortion_tokens_parse_the_csv_list_strings():
    assert sps._distortion_tokens("[]") == []
    assert sps._distortion_tokens("['jpeg']") == ["jpeg"]
    assert sps._distortion_tokens("['a', 'b']") == ["a", "b"]
    assert sps._distortion_tokens(None) == []
    assert sps._distortion_tokens(float("nan")) == []
    assert sps._distortion_tokens("plain") == ["plain"]


def test_crop_clamp_override_forces_the_flag_without_touching_the_rest():
    cfg = {"policy_mode": "crop", "crop_side": 200, "crop_clamp": False,
           "nominal_side": 224}
    p = sps._policy_from_config(cfg)
    q = sps._policy_from_config(cfg, True)
    assert not p.crop_clamp and q.crop_clamp
    assert p.crop_side == q.crop_side == 200
    assert p.nominal_side == q.nominal_side == 224
    assert sps._policy_from_config(cfg, None).crop_clamp is False


def test_out_columns_keep_logit_and_prob_and_the_distortion_metadata():
    assert {"logit", "prob", "is_distorted", "distortions", "split"} <= set(sps.OUT_COLUMNS)


def test_load_models_rebuilds_the_pooler_from_the_config(monkeypatch):
    import torch
    from aigcdet.features.backbones import BackboneSpec
    from aigcdet.models.heads import Detector
    from aigcdet.models.pooling import TokenPooler, build_pooler

    spec = BackboneSpec("fake", "none", image_size=64, dim=8,
                        num_prefix_tokens=3, params=0)
    monkeypatch.setattr(sps, "load_backbone",
                        lambda name, device: (torch.nn.Linear(8, 8), spec))
    head = Detector(dim_feat=16, pooler=build_pooler("cls_mean", [spec]))
    ck = {"backbone": "fake", "tower_state_dict": torch.nn.Linear(8, 8).state_dict(),
          "state_dict": head.state_dict(), "dim_feat": 16, "swa_n": 0,
          "config": {"token_pool": "cls_mean"}}
    towers, specs, got, tag = sps._load_models(ck, "cpu", use_swa=False)
    assert isinstance(got.pooler, TokenPooler) and got.pooler.mode == "cls_mean"
    assert tag == "final" and len(towers) == 1
    # A legacy checkpoint: no token_pool key, dim_feat == the tower's width.
    legacy = Detector(dim_feat=8)
    ck2 = {**ck, "state_dict": legacy.state_dict(), "dim_feat": 8, "config": {}}
    assert sps._load_models(ck2, "cpu", use_swa=False)[2].pooler is None
    # A width that does not match the declared readout is refused outright.
    with pytest.raises(SystemExit, match="dim_feat"):
        sps._load_models({**ck, "dim_feat": 8}, "cpu", use_swa=False)
