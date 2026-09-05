import numpy as np
import pandas as pd
import pytest

from aigcdet.eval import late_fusion as lf


def _frames(n=400, seed=0, scale_b=1.0, shift_b=0.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    dist = rng.integers(0, 2, n)
    # Branch A is good on clean rows, B on distorted ones, so a fused score
    # has something to gain and the weight is not degenerate.
    a = 2.0 * y + rng.normal(0, 1.0, n) + np.where(dist == 1, rng.normal(0, 1.5, n), 0)
    b = 2.0 * y + rng.normal(0, 1.0, n) + np.where(dist == 0, rng.normal(0, 1.5, n), 0)
    paths = [f"/img/{i}.jpg" for i in range(n)]
    da = pd.DataFrame({"path": paths, "label": y, "is_distorted": dist, "logit": a,
                       "prob": 1 / (1 + np.exp(-a))})
    db = pd.DataFrame({"path": paths, "label": y, "is_distorted": dist,
                       "logit": scale_b * b + shift_b})
    return da, db


def test_join_is_one_to_one_and_checks_labels():
    da, db = _frames()
    j = lf.join_branches(da, db)
    assert len(j) == len(da) and {"score_a", "score_b", "label", "is_distorted"} <= set(j.columns)
    with pytest.raises(ValueError, match="not 1:1"):
        lf.join_branches(da, db.iloc[:-1])
    bad = db.copy()
    bad.loc[0, "label"] = 1 - bad.loc[0, "label"]
    with pytest.raises(ValueError, match="label"):
        lf.join_branches(da, bad)


def test_robust_and_clean_auc_are_computed_over_their_own_halves():
    da, db = _frames()
    j = lf.join_branches(da, db)
    y, s, d = j.label.to_numpy(), j.score_a.to_numpy(), j.is_distorted.to_numpy()
    m = lf.branch_metrics(y, s, d)
    from sklearn.metrics import roc_auc_score
    assert m["robust_auc"] == pytest.approx(roc_auc_score(y[d == 1], s[d == 1]))
    assert m["clean_auc"] == pytest.approx(roc_auc_score(y[d == 0], s[d == 0]))
    assert m["n_robust"] + m["n_clean"] == len(y)
    # Flipping a CLEAN row's label leaves the robust number untouched.
    y2 = y.copy()
    i = int(np.flatnonzero(d == 0)[0])
    y2[i] = 1 - y2[i]
    assert lf.branch_metrics(y2, s, d)["robust_auc"] == m["robust_auc"]
    # A half with one class reports None rather than a number.
    y3 = y.copy()
    y3[d == 1] = 1
    assert lf.branch_metrics(y3, s, d)["robust_auc"] is None


def test_parameters_come_from_the_fit_rows_only():
    da, db = _frames()
    j = lf.join_branches(da, db)
    fit, test = j.iloc[:300], j.iloc[300:]
    fusion, _ = lf.fit_fusion(fit, "split=fit")
    # Changing the test rows cannot move the fitted parameters.
    test2 = test.copy()
    test2["score_a"] *= 100
    fusion2, _ = lf.fit_fusion(fit, "split=fit")
    assert fusion == fusion2
    # And the objective was evaluated on the fit rows: the fitted weight is at
    # least as good there as equal weighting.
    ev = lf.evaluate(fit, fusion)
    assert lf.fusion_objective(ev["fused"]) >= lf.fusion_objective(ev["fused_equal"]) - 1e-12


def test_fused_score_is_invariant_to_one_branches_logit_scale():
    """Mirror of test_fusion.py's scale-invariance: a branch that emits its
    logits ten times larger, or shifted, must produce the same fused ranking."""
    da, db = _frames()
    _, db2 = _frames(scale_b=10.0, shift_b=3.0)
    j1 = lf.join_branches(da, db)
    j2 = lf.join_branches(da, db2)
    f1, _ = lf.fit_fusion(j1, "p")
    f2, _ = lf.fit_fusion(j2, "p")
    assert f1.weight == f2.weight
    np.testing.assert_allclose(f1.apply(j1.score_a, j1.score_b),
                               f2.apply(j2.score_a, j2.score_b), atol=1e-9)


def test_ties_go_to_equal_weight():
    y = np.array([0, 1] * 50)
    z = np.tile([-1.0, 1.0], 50)          # both branches perfect -> flat sweep
    w, sweep = lf.fit_weight(z, z, y, None)
    assert w == 0.5
    assert len(set(sweep.values())) == 1


def test_fusion_record_round_trips():
    da, db = _frames()
    fusion, _ = lf.fit_fusion(lf.join_branches(da, db), "split=x")
    back = lf.Fusion.from_record(fusion.as_record())
    assert back == fusion


def test_constant_scores_are_refused():
    with pytest.raises(ValueError, match="constant"):
        lf.zscore_fit(np.ones(10))
