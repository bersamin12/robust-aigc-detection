"""Tests for the robustness table, heatmap and degradation-head validation.

Everything here is hermetic: no GPU, no weights, no downloads. The two tests
that need real `FeatureBank`s build tiny ones from a dummy manifest with the
backbone stubbed out, exactly as tests/eval/test_grid.py does.
"""
import io

import numpy as np
import pandas as pd
import pytest

from aigcdet.augment.recipes import FAMILIES, N_FAMILIES
from aigcdet.augment.scenarios import (
    CORE_CONDITIONS, EVAL_GRID, HELDOUT_SEVERITY_CONDITIONS,
)
from aigcdet.eval.metrics import (
    accuracy_at_threshold, expected_calibration_error, tpr_at_fpr,
)
from aigcdet.eval.report import (
    BANKS_NOT_VERIFIED, HELDOUT_MARK, METRIC_COLUMNS, NOT_FOR_PUBLICATION,
    PROXIED_FAMILIES, TIER_CONDITIONS, UNVERIFIED_BANNER, condition_metrics,
    robustness_table, save_heatmap, to_markdown, validate_degradation_head,
)

#: Most tests here are not about bank-level comparability, so they take the
#: explicit opt-out. That `banks` is REQUIRED is asserted on its own below.
_NOBANK = BANKS_NOT_VERIFIED


#: Bootstrap resamples for the tests that are not about the bootstrap. The
#: production default is 1000 (spec §6.1); at 20 conditions x ~30 calls that is
#: nine minutes of resampling to assert things the resample count has no
#: bearing on. The two tests that ARE about the resample count leave it alone.
_NB = 100


def _scores(n=120, seed=0, sep=2.0, conditions=None):
    rng = np.random.default_rng(seed)
    rows = []
    for cond in (EVAL_GRID if conditions is None else conditions):
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        # Harsher conditions get less separation, as reality does.
        s = rng.normal(y * sep * (0.4 if "jpeg_q30" in cond else 1.0), 1.0)
        rows.append(pd.DataFrame({"condition": cond, "image_idx": np.arange(n),
                                  "label": y, "generator": "g", "source": "s",
                                  "score": s}))
    return pd.concat(rows, ignore_index=True)


# --- condition_metrics -----------------------------------------------------

def test_condition_metrics_has_a_row_per_condition_with_cis():
    m = condition_metrics(_scores(), seed=0, n_boot=_NB)
    assert len(m) == len(EVAL_GRID)
    assert (m["auc_lo"] <= m["auc"]).all() and (m["auc"] <= m["auc_hi"]).all()
    assert {"tpr_at_1pct", "acc_oracle", "acc_fixed", "n"} <= set(m.columns)


def test_heldout_severity_conditions_are_flagged_in_the_table():
    m = condition_metrics(_scores(), seed=0, n_boot=_NB).set_index("condition")
    for c in HELDOUT_SEVERITY_CONDITIONS:
        assert bool(m.loc[c, "heldout_severity"]) is True
    assert bool(m.loc["jpeg_q30", "heldout_severity"]) is False


def test_heldout_severity_conditions_are_the_four_the_bands_imply():
    """The flag must mark exactly these four and nothing else.

    Kills a mutant that widens or narrows the held-out bands in
    augment.recipes, or that flags every composite scenario: `screenshot`
    (jpeg q50) and `messaging_app` (q30) are composites and are NOT unseen
    severities, while `social_repost` and `filtered_upload` (both q70) are.
    """
    assert set(HELDOUT_SEVERITY_CONDITIONS) == {
        "blur_s1.0", "jpeg_q70", "social_repost", "filtered_upload"}
    m = condition_metrics(_scores(), seed=0, n_boot=_NB).set_index("condition")
    flagged = set(m.index[m["heldout_severity"].to_numpy(dtype=bool)])
    assert flagged == set(HELDOUT_SEVERITY_CONDITIONS)


def test_fixed_threshold_accuracy_never_exceeds_oracle():
    m = condition_metrics(_scores(), seed=0, n_boot=_NB)
    assert (m["acc_fixed"] <= m["acc_oracle"] + 1e-9).all()


def test_fixed_threshold_is_frozen_from_clean_and_actually_costs_accuracy():
    """`acc_fixed` must use the clean threshold, not a per-condition refit.

    Kills the mutant that computes `acc_fixed` with `_best_threshold(y, s)`
    (i.e. the same call `acc_oracle` makes): under that mutant the two columns
    are identical everywhere and the reported score drift vanishes. The
    strict inequality on at least one degraded condition is the part that
    fails, so it is asserted rather than merely `<=`.
    """
    m = condition_metrics(_scores(), seed=0, n_boot=_NB).set_index("condition")
    degraded = m.drop(index="clean")
    assert (degraded["acc_fixed"] < degraded["acc_oracle"] - 1e-9).any()
    # And an explicitly supplied threshold is honoured verbatim.
    fixed = condition_metrics(_scores(), clean_threshold=1e6, seed=0, n_boot=_NB)
    assert (fixed["acc_fixed"] == 0.5).all()   # rejects everything; half the rows are real


def test_oracle_threshold_is_exact_even_with_many_distinct_scores():
    """The oracle must maximise over ALL thresholds, not a 512-point subsample.

    Kills the quantile-subsampling shortcut: with >512 distinct scores it can
    miss the optimum, which both understates `acc_oracle` and can invert the
    `acc_fixed <= acc_oracle` invariant the two columns are read against.
    Brute force over every distinct score is the ground truth here.
    """
    rng = np.random.default_rng(3)
    n = 1400
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    s = rng.normal(y * 0.7, 1.0)
    assert len(np.unique(s)) > 512
    df = pd.DataFrame({"condition": "clean", "image_idx": np.arange(n),
                       "label": y, "score": s})
    got = float(condition_metrics(df, seed=0, n_boot=_NB).loc[0, "acc_oracle"])
    brute = max(accuracy_at_threshold(y, s, t)
                for t in np.append(np.unique(s), s.max() + 1.0))
    assert got == pytest.approx(brute)


def test_condition_metrics_refuses_to_invent_a_clean_threshold():
    df = _scores(conditions=["jpeg_q50", "blur_s2.0"])
    with pytest.raises(ValueError, match="clean"):
        condition_metrics(df, seed=0, n_boot=_NB)
    # ...but an explicit threshold is enough to proceed without a clean block.
    assert len(condition_metrics(df, clean_threshold=0.0, seed=0, n_boot=_NB)) == 2


def test_harsher_condition_scores_lower_auc():
    m = condition_metrics(_scores(), seed=0, n_boot=_NB).set_index("condition")
    assert m.loc["jpeg_q30", "auc"] < m.loc["clean", "auc"]


def test_bootstrap_ci_is_reproducible_and_records_its_seed():
    """Kills a mutant that ignores `seed`/`n_boot` or hardcodes the interval.

    Same seed must give the identical interval; a different seed must give a
    different one (the CI is a resampling estimate, not a closed form); and
    both settings must appear in the table so the interval can be reproduced
    from what was published.
    """
    few = _scores(conditions=["clean", "jpeg_q30", "blur_s1.0"])
    a = condition_metrics(few, seed=7).set_index("condition")   # default n_boot
    b = condition_metrics(few, seed=7).set_index("condition")
    c = condition_metrics(few, seed=8).set_index("condition")
    assert (a["auc_lo"] == b["auc_lo"]).all() and (a["auc_hi"] == b["auc_hi"]).all()
    assert (a["auc_lo"] != c["auc_lo"]).any()
    assert (a["boot_seed"] == 7).all() and (a["boot_n"] == 1000).all()
    assert float(a.loc["clean", "auc_hi"] - a.loc["clean", "auc_lo"]) > 0.0


def test_ece_is_nan_without_probabilities_and_a_number_with_them():
    df = _scores(n=200, seed=2)
    assert condition_metrics(df, seed=0, n_boot=_NB)["ece"].isna().all()
    probs = 1.0 / (1.0 + np.exp(-df["score"].to_numpy()))
    with_probs = condition_metrics(df, probs=pd.Series(probs), seed=0, n_boot=_NB)
    assert with_probs["ece"].notna().all()
    assert (with_probs["ece"] >= 0.0).all()
    with pytest.raises(ValueError, match="align"):
        condition_metrics(df, probs=probs[:10], seed=0, n_boot=_NB)


# --- robustness_table ------------------------------------------------------

def test_robustness_table_rows_are_rungs_and_has_robust_auc():
    t = robustness_table({"a0": _scores(seed=1), "a3": _scores(seed=2)}, tier="ablation", n_boot=_NB, banks=_NOBANK)
    assert set(t.index) == {"a0", "a3"}
    assert "robust_auc" in t.columns and "clean" in t.columns
    # robust_auc excludes the clean column by construction
    assert t.loc["a0", "robust_auc"] <= 1.0


def test_robust_auc_is_the_mean_over_degraded_conditions_excluding_clean():
    """Kills the mutant that leaves `clean` in the robust mean.

    Clean AUC is the highest number in the row, so including it inflates the
    robust figure -- the one §6.1 defines as the mean over TRANSFORMED
    conditions, and the one the report headlines.
    """
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB, banks=_NOBANK)
    degraded = [c for c in EVAL_GRID if c != "clean"]
    assert t.loc["a0", "robust_auc"] == pytest.approx(
        float(t.loc["a0", degraded].astype(float).mean()))
    with_clean = float(t.loc["a0", list(EVAL_GRID)].astype(float).mean())
    assert t.loc["a0", "robust_auc"] != pytest.approx(with_clean)


def test_heldout_and_seen_means_split_the_degraded_conditions_correctly():
    """The unseen-severity summary must average exactly the four held-out
    conditions, and `seen_auc` exactly the other degraded ones.

    Kills a mutant that swaps the two column definitions, that lets `clean`
    into `seen_auc`, or that averages all degraded conditions into both.
    """
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB, banks=_NOBANK)
    heldout = [c for c in EVAL_GRID if c in HELDOUT_SEVERITY_CONDITIONS]
    seen = [c for c in EVAL_GRID
            if c != "clean" and c not in HELDOUT_SEVERITY_CONDITIONS]
    assert len(heldout) == 4 and len(seen) == 15
    assert t.loc["a0", "heldout_auc"] == pytest.approx(
        float(t.loc["a0", heldout].astype(float).mean()))
    assert t.loc["a0", "seen_auc"] == pytest.approx(
        float(t.loc["a0", seen].astype(float).mean()))
    assert t.loc["a0", "heldout_auc"] != pytest.approx(t.loc["a0", "seen_auc"])


def test_summary_column_is_named_after_the_metric_it_averages():
    """A TPR mean must not be published in a column called `robust_auc`.

    Kills the mutant that hardcodes the aggregate column name: under it §6.1's
    reported robust TPR @ 1% FPR would be presented as an AUC. (It is a
    REPORTING metric over every scored row, not the §6.4 selection rule -- that
    one is `errors.SELECTION_METRIC` and has its own column here.)
    """
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", metric="tpr_at_1pct",
                         n_boot=_NB, banks=_NOBANK)
    assert "robust_tpr_at_1pct" in t.columns and "robust_auc" not in t.columns
    assert "heldout_tpr_at_1pct" in t.columns
    per_condition = condition_metrics(_scores(seed=1), n_boot=_NB).set_index("condition")
    assert t.loc["a0", "clean"] == pytest.approx(
        float(per_condition.loc["clean", "tpr_at_1pct"]))


def test_robustness_table_carries_its_tier_as_a_column_not_only_in_attrs(tmp_path):
    """`DataFrame.attrs` does not survive a reshape or a CSV round trip.

    Kills the mutant that records the tier only in `attrs`: after a round trip
    the table would render with no tier at all, and an unlabelled table is one
    whose selection-tier numbers can be quoted as final-report numbers.
    """
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB, banks=_NOBANK)
    assert (t["tier"] == "ablation").all()
    buf = io.StringIO()
    t.to_csv(buf)
    buf.seek(0)
    round_tripped = pd.read_csv(buf, index_col=0)
    assert not round_tripped.attrs.get("tier")      # attrs did not survive
    assert (round_tripped["tier"] == "ablation").all()
    # ...and the round-tripped table still renders, tier stated and unseen
    # severities marked, which is the point of not relying on attrs.
    out = tmp_path / "round_tripped.md"
    to_markdown(round_tripped, tier="ablation", path=str(out))
    assert "**Evaluation tier:** ablation" in out.read_text()
    assert "jpeg_q70 (unseen)" in out.read_text()


def test_robustness_table_records_its_row_count_and_bootstrap_settings():
    """Provenance the tier claim is auditable against.

    `n_images` distinguishes a 5k selection-tier run from a 13.8k final-report
    one at a glance, and the bootstrap settings make a published CI
    reproducible. Both must track what was actually used, not a constant --
    hence one call at the defaults and one at an overridden resample count.
    """
    t = robustness_table({"a0": _scores(n=400, seed=1)}, tier="ablation",
                         n_boot=_NB, seed=99, banks=_NOBANK)
    assert int(t.loc["a0", "n_images"]) == 400
    assert int(t.loc["a0", "boot_seed"]) == 99
    assert int(t.loc["a0", "boot_n"]) == _NB
    # At its defaults the table really does resample 1000 times (spec §6.1).
    d = robustness_table({"a0": _scores(n=20, seed=1, conditions=list(CORE_CONDITIONS))},
                         tier="final_report", banks=_NOBANK)
    assert int(d.loc["a0", "boot_n"]) == 1000
    assert int(d.loc["a0", "boot_seed"]) == 20260827
    assert int(d.loc["a0", "n_images"]) == 20


@pytest.mark.parametrize("tier", ["", "Ablation", "final", "selection", None])
def test_robustness_table_rejects_an_unlabelled_or_unknown_tier(tier):
    with pytest.raises(ValueError, match="tier"):
        robustness_table({"a0": _scores(seed=1)}, tier=tier, banks=_NOBANK)


def test_robustness_table_rejects_a_tier_that_contradicts_the_coverage():
    """The 15-core-condition final-report tier cannot be claimed for a
    20-condition selection-tier run, or the other way round.

    Kills the mutant that accepts any tier string it recognises without
    checking what was actually evaluated -- the mechanism that makes a
    *wrongly* labelled table impossible rather than merely discouraged.
    """
    full = {"a0": _scores(seed=1)}
    with pytest.raises(ValueError, match="final_report"):
        robustness_table(full, tier="final_report", n_boot=_NB, banks=_NOBANK)
    core = {"a0": _scores(seed=1, conditions=list(CORE_CONDITIONS))}
    with pytest.raises(ValueError, match="ablation"):
        robustness_table(core, tier="ablation", n_boot=_NB, banks=_NOBANK)
    # Each is fine at its own tier.
    assert len(robustness_table(core, tier="final_report", n_boot=_NB, banks=_NOBANK).columns) > 15
    assert set(TIER_CONDITIONS["final_report"]) == set(CORE_CONDITIONS)


def test_robustness_table_rejects_rungs_with_different_condition_coverage():
    """CARRY C-A / R24 at the frame level: differing view coverage between
    compared rungs makes the table a comparison of augmentation budgets."""
    short = list(EVAL_GRID)[:10]
    with pytest.raises(ValueError, match="different conditions"):
        robustness_table({"a0": _scores(seed=1),
                          "a3": _scores(seed=2, conditions=short)},
                         tier="ablation", n_boot=_NB, banks=_NOBANK)


def test_robustness_table_rejects_rungs_whose_conditions_are_merely_reordered():
    """Same twenty conditions in a different order is still not comparable
    when it comes from a differently ordered bank; the error must say so
    rather than silently aligning by name."""
    names = list(EVAL_GRID)
    reordered = [names[0]] + names[1:][::-1]
    with pytest.raises(ValueError, match="different order"):
        robustness_table({"a0": _scores(seed=1),
                          "a3": _scores(seed=2, conditions=reordered)},
                         tier="ablation", n_boot=_NB, banks=_NOBANK)


def test_robustness_table_rejects_rungs_scored_on_different_images():
    a = _scores(n=400, seed=1)
    b = _scores(n=400, seed=2)
    b["image_idx"] = b["image_idx"] + 1000
    with pytest.raises(ValueError, match="different images"):
        robustness_table({"a0": a, "a3": b}, tier="ablation", n_boot=_NB, banks=_NOBANK)


def test_robustness_table_rejects_an_empty_comparison():
    with pytest.raises(ValueError, match="nothing to compare"):
        robustness_table({}, tier="ablation", n_boot=_NB, banks=_NOBANK)


# --- CARRY C-A / R24: the bank-level comparability guard -------------------

def _eval_bank(tmp_path, monkeypatch, name, df, backbone="fake", conditions=None):
    """A real (tiny) eval bank with the backbone stubbed; no GPU, no weights."""
    from aigcdet.eval import grid
    from aigcdet.features.backbones import BackboneSpec
    from aigcdet.features.bank import FeatureBank
    spec = BackboneSpec("fake", "none", 64, 4, 1, 0)
    monkeypatch.setattr(grid, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(grid, "embed", lambda m, s, imgs, device, batch_size=16:
                        np.stack([np.full(s.dim, float(i.mean()), np.float32)
                                  for i in imgs]))
    return FeatureBank.open(grid.extract_eval_bank(
        df, backbone, str(tmp_path / name), conditions=conditions, device="cpu"))


def test_robustness_table_accepts_banks_from_one_comparable_extraction(
        tmp_path, monkeypatch):
    """The positive control for the guard below.

    Without it, the rejection tests could all pass because the bank path
    rejects everything -- which would make the guard useless rather than
    strict.
    """
    from aigcdet.data.manifest import make_dummy_manifest
    df = make_dummy_manifest(4, str(tmp_path / "src"), np.random.default_rng(9))
    a = _eval_bank(tmp_path, monkeypatch, "ba", df)
    b = _eval_bank(tmp_path, monkeypatch, "bb", df)
    t = robustness_table({"a0": _scores(n=4, seed=1), "a3": _scores(n=4, seed=2)},
                         tier="ablation", n_boot=_NB,
                         banks={"a0": a, "a3": b})
    assert set(t.index) == {"a0", "a3"}


@pytest.mark.parametrize("differ", ["conditions", "backbone", "manifest_sha256"])
def test_robustness_table_refuses_to_build_across_incomparable_banks(
        tmp_path, monkeypatch, differ):
    """Kills the mutant that drops the `assert_banks_comparable` call.

    Task 1 shipped that guard with no caller; without this wiring a table
    could be built from rungs scored on different backbones, condition orders
    or image sets, and would look entirely plausible while comparing the
    evaluation rather than the models (spec §6.4 kill criterion).
    """
    from aigcdet.data.manifest import make_dummy_manifest
    df = make_dummy_manifest(4, str(tmp_path / f"s{differ}"), np.random.default_rng(9))
    a = _eval_bank(tmp_path, monkeypatch, f"{differ}a", df)
    if differ == "conditions":
        names = list(EVAL_GRID)
        reordered = {k: EVAL_GRID[k] for k in [names[0]] + names[1:][::-1]}
        b = _eval_bank(tmp_path, monkeypatch, f"{differ}b", df, conditions=reordered)
    elif differ == "backbone":
        b = _eval_bank(tmp_path, monkeypatch, f"{differ}b", df, backbone="other")
    else:
        other = make_dummy_manifest(4, str(tmp_path / f"s{differ}2"),
                                    np.random.default_rng(11))
        b = _eval_bank(tmp_path, monkeypatch, f"{differ}b", other)
    scores = {"a0": _scores(n=4, seed=1), "a3": _scores(n=4, seed=2)}
    with pytest.raises(ValueError, match="not comparable"):
        robustness_table(scores, tier="ablation", n_boot=_NB,
                         banks={"a0": a, "a3": b})


def test_robustness_table_rejects_banks_that_record_no_manifest_fingerprint(
        tmp_path, monkeypatch):
    """Stricter than `assert_banks_comparable`, deliberately.

    That guard compares `manifest_sha256` for equality, so `None == None`
    counts as agreement -- absence of evidence, not evidence of sameness.
    Banks index the manifest positionally, so two unfingerprinted extractions
    taken either side of a re-split misalign labels without changing a single
    shape. Kills the mutant that drops this extra check: both banks below pass
    `assert_banks_comparable` unchanged.
    """
    import json
    from aigcdet.data.manifest import make_dummy_manifest
    from aigcdet.eval.grid import assert_banks_comparable
    from aigcdet.features.bank import FeatureBank
    df = make_dummy_manifest(4, str(tmp_path / "sfp"), np.random.default_rng(9))
    paths = []
    for name in ("fpa", "fpb"):
        bank = _eval_bank(tmp_path, monkeypatch, name, df)
        cfg_path = f"{bank.path}/config.json"
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        cfg["manifest_sha256"] = None
        with open(cfg_path, "w") as fh:
            json.dump(cfg, fh)
        paths.append(bank.path)
    a, b = (FeatureBank.open(p) for p in paths)
    assert_banks_comparable([a, b])          # the shipped guard is satisfied
    with pytest.raises(ValueError, match="manifest_sha256"):
        robustness_table({"a0": _scores(n=4, seed=1), "a3": _scores(n=4, seed=2)},
                         tier="ablation", n_boot=_NB,
                         banks={"a0": a, "a3": b})


def test_robustness_table_rejects_banks_that_do_not_belong_to_the_scores(
        tmp_path, monkeypatch):
    """Passing comparable-but-unrelated banks must not launder the check."""
    from aigcdet.data.manifest import make_dummy_manifest
    df = make_dummy_manifest(4, str(tmp_path / "sbel"), np.random.default_rng(9))
    subset = {k: EVAL_GRID[k] for k in list(CORE_CONDITIONS)}
    a = _eval_bank(tmp_path, monkeypatch, "bela", df, conditions=subset)
    b = _eval_bank(tmp_path, monkeypatch, "belb", df, conditions=subset)
    with pytest.raises(ValueError, match="do not belong"):
        robustness_table({"a0": _scores(n=4, seed=1), "a3": _scores(n=4, seed=2)},
                         tier="ablation", n_boot=_NB,
                         banks={"a0": a, "a3": b})
    with pytest.raises(ValueError, match="one bank per rung"):
        robustness_table({"a0": _scores(n=4, seed=1), "a3": _scores(n=4, seed=2)},
                         tier="ablation", n_boot=_NB, banks={"a0": a})


# --- markdown --------------------------------------------------------------

def test_markdown_output_names_its_tier(tmp_path):
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB, banks=_NOBANK)
    p = tmp_path / "t.md"
    to_markdown(t, tier="ablation", path=str(p))
    text = p.read_text()
    assert "ablation" in text and "a0" in text


def test_markdown_marks_every_heldout_severity_condition_and_only_those(tmp_path):
    """The unseen-severity marking must survive the reshape into markdown.

    Kills the mutant that renders bare column names: the four unseen-severity
    conditions would then be indistinguishable from the fifteen the sampler
    could have drawn, which is the single distinction a reader most needs.
    Checked in both directions -- exactly the four marked, none of the others.
    """
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB, banks=_NOBANK)
    p = tmp_path / "marked.md"
    to_markdown(t, tier="ablation", path=str(p))
    header = [c.strip() for c in p.read_text().splitlines()
              if c.startswith("| rung |")][0].split("|")
    marked = {c.strip().removesuffix(" (unseen)")
              for c in header if c.strip().endswith("(unseen)")}
    assert marked == set(HELDOUT_SEVERITY_CONDITIONS)
    plain = {c.strip() for c in header if c.strip() and not c.strip().endswith("(unseen)")}
    assert plain & set(HELDOUT_SEVERITY_CONDITIONS) == set()
    for cond in HELDOUT_SEVERITY_CONDITIONS:
        assert cond in p.read_text()


def test_markdown_refuses_a_tier_label_the_table_was_not_built_at(tmp_path):
    """Relabelling at write time is the exact way a selection-tier number is
    published as a final-report one; it must be an error, not an override."""
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB, banks=_NOBANK)
    with pytest.raises(ValueError, match="built as"):
        to_markdown(t, tier="final_report", path=str(tmp_path / "x.md"))
    with pytest.raises(ValueError, match="tier"):
        to_markdown(t, tier="whatever", path=str(tmp_path / "x.md"))
    assert not (tmp_path / "x.md").exists()


def test_markdown_refuses_a_table_whose_unseen_severity_columns_were_dropped(tmp_path):
    """Kills the mutant that renders whatever columns it is handed.

    Dropping `jpeg_q70` and friends from a twenty-condition table leaves a
    plausible-looking report that quietly omits the unseen severities.
    """
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB, banks=_NOBANK)
    dropped = t.drop(columns=list(HELDOUT_SEVERITY_CONDITIONS))
    with pytest.raises(ValueError, match="unseen-severity"):
        to_markdown(dropped, tier="ablation", path=str(tmp_path / "d.md"))


def test_markdown_refuses_a_table_that_states_no_tier(tmp_path):
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB, banks=_NOBANK)
    naked = t.drop(columns=["tier"])
    naked.attrs.clear()
    with pytest.raises(ValueError, match="states no evaluation tier"):
        to_markdown(naked, tier="ablation", path=str(tmp_path / "n.md"))


def test_markdown_records_the_bootstrap_provenance(tmp_path):
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", seed=1234, n_boot=_NB,
                         banks=_NOBANK)
    p = tmp_path / "prov.md"
    to_markdown(t, tier="ablation", path=str(p))
    text = p.read_text()
    assert f"{_NB} resamples" in text and "seed 1234" in text
    assert "5k internal validation" in text        # the tier's row budget


def test_markdown_values_match_the_table(tmp_path):
    """The rendered numbers must be the table's own, not a re-derivation."""
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB, banks=_NOBANK)
    p = tmp_path / "v.md"
    to_markdown(t, tier="ablation", path=str(p))
    row = [line for line in p.read_text().splitlines() if line.startswith("| a0 |")][0]
    cells = [c.strip() for c in row.split("|")[1:-1]]
    assert cells[0] == "a0"
    assert cells[1] == f"{float(t.loc['a0', 'clean']):.4f}"
    assert "ablation" in cells


# --- heatmap ---------------------------------------------------------------

def test_save_heatmap_writes_a_png_and_ignores_the_summary_columns(tmp_path):
    """Kills the mutant that plots every column but `robust_auc`.

    The table carries a string `tier` column and integer provenance columns;
    casting those to float raises, and plotting a count on a 0.5-1.0 colour
    scale would wash the matrix out. The file must also land under `tmp_path`
    and nowhere else.
    """
    t = robustness_table({"a0": _scores(seed=1), "a3": _scores(seed=2)},
                         tier="ablation", n_boot=_NB, banks=_NOBANK)
    assert not pd.api.types.is_numeric_dtype(t["tier"])
    p = tmp_path / "heat.png"
    save_heatmap(t, str(p))
    assert p.exists() and p.stat().st_size > 1000
    assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    import matplotlib
    assert matplotlib.get_backend().lower() == "agg"


def test_heatmap_scale_does_not_flatten_the_reported_tpr(tmp_path):
    """TPR @ 1% FPR -- §6.1's reported TPR, not §6.4's selection rule -- lives
    well below 0.5.

    Kills the mutant that hardcodes `vmin=0.5` for every metric: under it two
    tables whose only difference is sub-0.5 TPR values render byte-identical,
    because every one of those cells clamps to the same colour and the figure
    hides the differences it exists to show. AUC keeps its 0.5 chance floor.
    """
    from aigcdet.eval.report import heatmap_limits
    assert heatmap_limits("auc") == (0.5, 1.0)
    assert heatmap_limits("tpr_at_1pct") == (0.0, 1.0)

    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", metric="tpr_at_1pct",
                         n_boot=_NB, banks=_NOBANK)
    # Rendered from a CSV round trip, so `attrs` -- and with it any record of
    # the metric held only there -- is gone. The metric must still be
    # recoverable from the `robust_<metric>` column name, or the figure
    # silently reverts to the AUC scale.
    buf = io.StringIO()
    t.to_csv(buf)
    buf.seek(0)
    t = pd.read_csv(buf, index_col=0)
    assert not t.attrs
    conditions = [c for c in t.columns if c in EVAL_GRID]
    low, lower = t.copy(), t.copy()
    low[conditions] = 0.30
    lower[conditions] = 0.05
    for frame, name in ((low, "low.png"), (lower, "lower.png")):
        save_heatmap(frame, str(tmp_path / name))
    assert (tmp_path / "low.png").read_bytes() != (tmp_path / "lower.png").read_bytes()


def test_save_heatmap_refuses_a_table_with_no_conditions(tmp_path):
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB, banks=_NOBANK)
    with pytest.raises(ValueError, match="no condition columns"):
        save_heatmap(t[["robust_auc", "tier"]], str(tmp_path / "empty.png"))


# --- degradation-head validation (spec §3.4) -------------------------------

def _planted(n=500, seed=0, backwards=()):
    """A head that has learned all three proxied families, unless `backwards`
    names a family, whose severity is then inverted."""
    rng = np.random.default_rng(seed)
    sev = {f: rng.uniform(0, 1, n) for f in ("jpeg", "blur", "noise")}
    pred = np.zeros((n, N_FAMILIES), np.float32)
    for f, v in sev.items():
        signed = -v if f in backwards else v
        pred[:, FAMILIES.index(f)] = signed + rng.normal(0, 0.02, n)
    proxies = np.stack([
        100 - sev["jpeg"] * 70,                 # jpeg_quality falls with severity
        500 - sev["blur"] * 450,                # laplacian_var falls with blur
        0.5 + sev["noise"] * 20,                # noise_floor rises with noise
    ], axis=1).astype(np.float32)
    return pred, proxies


def test_degradation_head_validation_finds_a_planted_correlation():
    rng = np.random.default_rng(0)
    n = 500
    true_jpeg_sev = rng.uniform(0, 1, n)
    pred = np.zeros((n, 6), np.float32)
    pred[:, 0] = true_jpeg_sev + rng.normal(0, 0.05, n)
    # Proxy jpeg_quality falls as severity rises, so the correlation is negative
    proxies = np.stack([100 - true_jpeg_sev * 70,
                        rng.normal(size=n), rng.normal(size=n)], axis=1).astype(np.float32)
    out = validate_degradation_head(pred, proxies, families=("jpeg",))
    assert abs(out.loc[0, "spearman"]) > 0.8


def test_expected_sign_is_stated_per_family_and_a_healthy_head_matches_it():
    """jpeg and blur must correlate NEGATIVELY, noise POSITIVELY.

    Kills a mutant that assumes one sign for every family, or that reports
    `abs(rho)`: under the latter the `spearman` column for jpeg and blur would
    come back positive, contradicting the direction the proxies actually move.
    """
    pred, proxies = _planted()
    out = validate_degradation_head(pred, proxies).set_index("family")
    assert list(out["expected_sign"]) == [-1, -1, +1]
    assert list(out["proxy"]) == ["jpeg_quality", "laplacian_var", "noise_floor"]
    assert out.loc["jpeg", "spearman"] < -0.8
    assert out.loc["blur", "spearman"] < -0.8
    assert out.loc["noise", "spearman"] > 0.8
    assert bool(out["sign_ok"].all())
    assert (out["spearman_aligned"] > 0.8).all()


def test_a_backwards_head_is_reported_as_backwards_not_as_strong():
    """The whole reason the correlation is signed.

    A head that learned jpeg severity inverted produces a strong correlation of
    the WRONG sign. Kills the mutant that takes `abs(rho)` (or that drops
    `expected_sign`/`sign_ok`): under it this inverted head is indistinguishable
    from the healthy one above, and the day-4 check silently passes.
    """
    pred, proxies = _planted(backwards=("jpeg",))
    out = validate_degradation_head(pred, proxies).set_index("family")
    assert out.loc["jpeg", "spearman"] > 0.8          # strong, and wrong way round
    assert out.loc["jpeg", "spearman_aligned"] < -0.8
    assert bool(out.loc["jpeg", "sign_ok"]) is False
    # The families that were learned correctly are still reported as fine.
    assert bool(out.loc["blur", "sign_ok"]) is True
    assert bool(out.loc["noise", "sign_ok"]) is True


def test_each_family_reads_its_own_severity_column_not_the_loop_position():
    """Kills the mutant that indexes `pred_severity` by enumerate position.

    `noise` is FAMILIES index 3, not 0. Here only column 3 carries signal and
    the other columns are pure noise, so a positional read would find nothing.
    """
    rng = np.random.default_rng(5)
    n = 400
    sev = rng.uniform(0, 1, n)
    pred = rng.normal(size=(n, N_FAMILIES)).astype(np.float32)
    pred[:, FAMILIES.index("noise")] = sev
    proxies = np.stack([rng.normal(size=n), rng.normal(size=n),
                        0.5 + sev * 20], axis=1).astype(np.float32)
    out = validate_degradation_head(pred, proxies, families=("noise",))
    assert out.loc[0, "spearman"] > 0.9
    assert out.loc[0, "n"] == n


def test_families_without_a_proxy_are_rejected_not_silently_skipped():
    """Kills the mutant that `continue`s past them: the returned frame would
    then read as a complete validation while saying nothing about crop."""
    pred, proxies = _planted(n=50)
    with pytest.raises(ValueError, match="no model-free proxy"):
        validate_degradation_head(pred, proxies, families=("jpeg", "crop"))
    with pytest.raises(ValueError, match="unknown degradation family"):
        validate_degradation_head(pred, proxies, families=("sharpen",))


@pytest.mark.parametrize("bad", ["pred_cols", "proxy_cols", "rows"])
def test_degradation_head_rejects_misshapen_inputs(bad):
    pred, proxies = _planted(n=50)
    if bad == "pred_cols":
        pred = pred[:, :3]
        match = "pred_severity must be"
    elif bad == "proxy_cols":
        proxies = proxies[:, :2]
        match = "proxies must be"
    else:
        proxies = proxies[:20]
        match = "must describe the same views"
    with pytest.raises(ValueError, match=match):
        validate_degradation_head(pred, proxies, families=("jpeg",))


def test_a_constant_prediction_reports_nan_rather_than_warning(recwarn):
    """A head that outputs the same severity for everything has no rank
    correlation to report; it must say so, not emit a RuntimeWarning."""
    pred, proxies = _planted(n=50)
    pred[:, FAMILIES.index("jpeg")] = 0.5
    out = validate_degradation_head(pred, proxies, families=("jpeg",))
    assert np.isnan(out.loc[0, "spearman"])
    assert bool(out.loc[0, "sign_ok"]) is False
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


# --- I1: `banks` is required, and an unverified table says so --------------

def test_robustness_table_will_not_build_without_a_decision_about_banks():
    """Kills the mutant that restores `banks=None` as the default.

    The frame-level checks cover the condition axis, and therefore `n_views`.
    They cannot cover the other two keys `assert_banks_comparable` compares:
    a score frame records no `backbone`, and `image_idx` comes from
    `bank.meta["image_idx"]`, the POSITIONAL manifest index, so two banks over
    two DIFFERENT manifests of equal length produce byte-identical `image_idx`
    sets. Defaulting to "skip the check" makes the misaligned-label comparison
    the path of least resistance, so there is no default.
    """
    with pytest.raises(ValueError, match="requires `banks`"):
        robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB)
    # The error must name the opt-out, or the caller cannot act on it.
    try:
        robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB)
    except ValueError as exc:
        assert "BANKS_NOT_VERIFIED" in str(exc)
    # A near-miss is not the sentinel.
    with pytest.raises(ValueError, match="exact sentinel"):
        robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB,
                         banks="skip")


def test_identical_image_idx_sets_do_not_prove_the_same_manifest():
    """The reason `banks` cannot be optional, asserted directly.

    `image_idx` is positional, so two rungs scored over two entirely different
    manifests of equal length are indistinguishable at the frame level. This
    test pins that the frame-level checks PASS here -- documenting the hole the
    `banks_verified` stamp exists to record, so nobody later mistakes the
    frame-level checks for a manifest check.
    """
    a, b = _scores(n=120, seed=1), _scores(n=120, seed=2)
    assert np.array_equal(np.unique(a["image_idx"]), np.unique(b["image_idx"]))
    t = robustness_table({"a0": a, "a3": b}, tier="ablation", n_boot=_NB,
                         banks=_NOBANK)
    assert not t["banks_verified"].any()


def test_unverified_tables_are_stamped_in_the_table_and_both_renderings(
        tmp_path, monkeypatch):
    """Kills the mutant that drops the stamp, or hardcodes it to True.

    An unverified comparison that renders identically to a verified one is
    worse than no check at all, because it reads as checked.
    """
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB,
                         banks=_NOBANK)
    assert bool(t.loc["a0", "banks_verified"]) is False
    md = tmp_path / "unverified.md"
    to_markdown(t, tier="ablation", path=str(md))
    assert UNVERIFIED_BANNER.upper() in md.read_text()
    captured = _render_and_capture(monkeypatch, t, str(tmp_path / "u.png"))
    assert UNVERIFIED_BANNER.upper() in captured["title"]


def test_a_bank_verified_table_carries_no_unverified_banner(
        tmp_path, monkeypatch):
    """The positive control: the stamp must distinguish, not always fire."""
    from aigcdet.data.manifest import make_dummy_manifest
    df = make_dummy_manifest(4, str(tmp_path / "srcv"), np.random.default_rng(9))
    a = _eval_bank(tmp_path, monkeypatch, "va", df)
    b = _eval_bank(tmp_path, monkeypatch, "vb", df)
    t = robustness_table({"a0": _scores(n=4, seed=1), "a3": _scores(n=4, seed=2)},
                         tier="ablation", n_boot=_NB, banks={"a0": a, "a3": b})
    assert bool(t["banks_verified"].all()) is True
    md = tmp_path / "verified.md"
    to_markdown(t, tier="ablation", path=str(md))
    assert UNVERIFIED_BANNER.upper() not in md.read_text()
    captured = _render_and_capture(monkeypatch, t, str(tmp_path / "v.png"))
    assert UNVERIFIED_BANNER.upper() not in captured["title"]


def test_banks_from_a_different_tier_are_rejected_on_their_row_count(
        tmp_path, monkeypatch):
    """Kills the mutant that checks only the condition axis.

    A 4-image bank matches 120-row score frames on conditions and on nothing
    else; that is a bank from a different evaluation tier wearing the right
    condition list.
    """
    from aigcdet.data.manifest import make_dummy_manifest
    df = make_dummy_manifest(4, str(tmp_path / "srcn"), np.random.default_rng(9))
    a = _eval_bank(tmp_path, monkeypatch, "na", df)
    b = _eval_bank(tmp_path, monkeypatch, "nb", df)
    with pytest.raises(ValueError, match="do not belong to these scores"):
        robustness_table({"a0": _scores(n=120, seed=1),
                          "a3": _scores(n=120, seed=2)},
                         tier="ablation", n_boot=_NB, banks={"a0": a, "a3": b})


# --- I2: the heatmap runs the same gate the markdown runs -----------------

def _render_and_capture(monkeypatch, table, path):
    """Render, and hand back what was actually drawn.

    `save_heatmap` closes its figure, so `plt.close` is stubbed to capture it
    first. No PNG introspection: the ticks, title, colour limits and the
    plotted matrix are all readable off the Figure.
    """
    import matplotlib.pyplot as plt
    captured = {}
    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda fig: captured.setdefault("fig", fig))
    save_heatmap(table, path)
    fig = captured["fig"]
    ax = fig.axes[0]
    out = {"xticks": [t.get_text() for t in ax.get_xticklabels()],
           "yticks": [t.get_text() for t in ax.get_yticklabels()],
           "title": ax.get_title(),
           "clim": ax.images[0].get_clim(),
           "data": np.asarray(ax.images[0].get_array(), dtype=float)}
    real_close(fig)
    return out


@pytest.mark.parametrize("damage", ["dropped_heldout", "relabelled", "junk_tier",
                                    "dropped_condition"])
def test_the_heatmap_refuses_everything_the_markdown_refuses(tmp_path, damage):
    """Kills the mutant that removes `_check_renderable` from `save_heatmap`.

    Before this, the PNG performed NONE of the four checks the markdown
    performs: a 20-condition `ablation` table with every unseen-severity column
    dropped rendered a clean 35 KB figure, and setting `tier` to
    `final_report` -- or to `peer reviewed, tier 1` -- rendered a figure whose
    title said so. `docs/robustness_table.png` is a shipped deliverable and the
    title is what a reader reads, so the figure cannot be the weak path.
    """
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB,
                         banks=_NOBANK)
    if damage == "dropped_heldout":
        t = t.drop(columns=list(HELDOUT_SEVERITY_CONDITIONS))
        match = "unseen-severity"
    elif damage == "relabelled":
        t["tier"] = "final_report"
        match = "final_report"
    elif damage == "junk_tier":
        t["tier"] = "peer reviewed, tier 1"
        match = "unknown evaluation tier"
    else:
        t = t.drop(columns=["jpeg_q90"])       # not held out; reaches the coverage check
        match = "missing"
    out = tmp_path / "damaged.png"
    with pytest.raises(ValueError, match=match):
        save_heatmap(t, str(out))
    assert not out.exists()


def test_to_markdown_also_refuses_a_dropped_non_heldout_condition(tmp_path):
    """Covers `to_markdown`'s coverage check specifically -- the held-out
    marking check sits in front of it and would otherwise mask it."""
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB,
                         banks=_NOBANK)
    with pytest.raises(ValueError, match="missing"):
        to_markdown(t.drop(columns=["jpeg_q90"]), tier="ablation",
                    path=str(tmp_path / "x.md"))


def test_a_table_mixing_tiers_is_refused(tmp_path):
    """Kills the mutant replacing the mixed-tier guard with `if False`."""
    t = robustness_table({"a0": _scores(seed=1), "a3": _scores(seed=2)},
                         tier="ablation", n_boot=_NB, banks=_NOBANK)
    t.loc["a3", "tier"] = "final_report"
    with pytest.raises(ValueError, match="mixes evaluation tiers"):
        to_markdown(t, tier="ablation", path=str(tmp_path / "m.md"))
    with pytest.raises(ValueError, match="mixes evaluation tiers"):
        save_heatmap(t, str(tmp_path / "m.png"))


# --- I2/minor: what the figure actually contains --------------------------

def test_the_heatmap_plots_the_tables_own_numbers_under_their_own_labels(
        tmp_path, monkeypatch):
    """Kills the mutants that transpose or scramble the plotted matrix.

    The reviewer showed both passed the whole file, because nothing asserted
    the figure's CONTENT. Ticks, title, colour limits and the matrix itself are
    all readable off the Figure without touching a pixel.
    """
    t = robustness_table({"a0": _scores(seed=1), "a3": _scores(seed=2)},
                         tier="ablation", n_boot=_NB, banks=_NOBANK)
    conditions = [c for c in t.columns if c in EVAL_GRID]
    got = _render_and_capture(monkeypatch, t, str(tmp_path / "content.png"))

    assert got["yticks"] == ["a0", "a3"]
    assert got["xticks"] == [
        c + HELDOUT_MARK if c in HELDOUT_SEVERITY_CONDITIONS else c
        for c in conditions]
    # Exactly the four unseen severities are marked, in the figure too.
    assert {x.removesuffix(HELDOUT_MARK) for x in got["xticks"]
            if x.endswith(HELDOUT_MARK)} == set(HELDOUT_SEVERITY_CONDITIONS)
    assert np.array_equal(got["data"], t[conditions].to_numpy(dtype=float))
    assert got["clim"] == (0.5, 1.0)
    assert "auc" in got["title"] and "ablation" in got["title"]


# --- I3: markdown alignment, cell by cell, for every rung -----------------

def _md_rows(text):
    return [line for line in text.splitlines() if line.startswith("|")]


def _md_cells(line):
    import re
    return [c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]]


def _expected_cell(value):
    """Format a value the way a reader expects, derived independently of the
    module's own `_format` so a change there cannot silently agree."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "" if np.isnan(value) else f"{float(value):.4f}"
    return str(value)


def test_every_markdown_cell_sits_under_its_own_column_heading(tmp_path):
    """Kills two number-scrambling mutants that passed the whole file before.

    (a) rendering every rung with the FIRST rung's numbers -- previously
    invisible because the values test used a single-rung table; (b) reversing
    the header list so every heading sits over the wrong column -- previously
    invisible because the marking test compared a SET of marked names, which
    reversal preserves, and the values test pinned only one cell that a
    header-only reversal does not move. Either would publish `boot_n` under
    `jpeg_q70 (unseen)`, or a0's row under a3's name.
    """
    t = robustness_table({"a0": _scores(seed=1), "a3": _scores(seed=2, sep=3.0)},
                         tier="ablation", n_boot=_NB, banks=_NOBANK)
    assert len(t) >= 2
    p = tmp_path / "aligned.md"
    to_markdown(t, tier="ablation", path=str(p))
    rows = _md_rows(p.read_text())
    header = _md_cells(rows[0])

    assert header[0] == "rung"
    assert len(header) == len(t.columns) + 1
    for i, col in enumerate(t.columns):
        expected = (str(col) + HELDOUT_MARK
                    if col in HELDOUT_SEVERITY_CONDITIONS else str(col))
        assert header[i + 1] == expected, f"header slot {i + 1} is not {col!r}"

    for r, rung in enumerate(t.index):
        cells = _md_cells(rows[2 + r])
        assert len(cells) == len(header)
        assert cells[0] == rung
        for i, col in enumerate(t.columns):
            assert cells[i + 1] == _expected_cell(t.loc[rung, col]), (
                f"rung {rung!r} column {col!r} (slot {i + 1})")
    # The two rungs really do differ, so "every row is rung 0" cannot pass.
    assert _md_cells(rows[2])[1:] != _md_cells(rows[3])[1:]


def test_a_pipe_in_a_rung_name_cannot_shift_the_columns(tmp_path):
    """Rung names are caller-supplied dict keys.

    Kills the mutant that drops `_escape`: an unescaped `|` adds a cell to the
    row, so every value lands one column left of its heading -- a total,
    silent misattribution of the numbers.
    """
    t = robustness_table({"a0 | evil": _scores(seed=1)}, tier="ablation",
                         n_boot=_NB, banks=_NOBANK)
    p = tmp_path / "pipe.md"
    to_markdown(t, tier="ablation", path=str(p))
    rows = _md_rows(p.read_text())
    header, cells = _md_cells(rows[0]), _md_cells(rows[2])
    assert len(cells) == len(header)
    assert cells[0] == r"a0 \| evil"
    assert cells[1] == _expected_cell(t.loc["a0 | evil", "clean"])


# --- I4: the file is written as UTF-8, not through the locale codec -------

def test_markdown_writes_utf8_under_a_c_locale(tmp_path):
    """Kills the mutant that drops `encoding="utf-8"`.

    The body contains `spec §4.6`. A bare `open(path, "w")` encodes through the
    locale codec; under LC_ALL=C that is ANSI_X3.4-1968 and the write dies with
    UnicodeEncodeError. C/POSIX is the default locale in many container and CI
    images, and Kaggle is in this project's critical path -- the same
    crash-at-write-time class as the missing `tabulate`.
    """
    import os
    import subprocess
    import sys
    script = tmp_path / "probe.py"
    out = tmp_path / "c_locale.md"
    script.write_text(
        "import numpy as np, pandas as pd, locale, sys\n"
        "assert locale.getpreferredencoding(False) == 'ANSI_X3.4-1968', "
        "locale.getpreferredencoding(False)\n"
        "from aigcdet.augment.scenarios import EVAL_GRID\n"
        "from aigcdet.eval.report import (robustness_table, to_markdown,\n"
        "                                 BANKS_NOT_VERIFIED)\n"
        "rng = np.random.default_rng(0)\n"
        "rows = [pd.DataFrame({'condition': c, 'image_idx': np.arange(40),\n"
        "                      'label': np.array([0]*20+[1]*20),\n"
        "                      'score': rng.normal(np.array([0]*20+[1]*20)*2, 1)})\n"
        "        for c in EVAL_GRID]\n"
        "t = robustness_table({'a0': pd.concat(rows, ignore_index=True)},\n"
        "                     tier='ablation', n_boot=5, banks=BANKS_NOT_VERIFIED)\n"
        "to_markdown(t, tier='ablation', path=sys.argv[1])\n")
    env = {k: v for k, v in os.environ.items()
           if k not in ("LANG", "LC_CTYPE", "LC_ALL")}
    env.update(LC_ALL="C", PYTHONUTF8="0", PYTHONCOERCECLOCALE="0")
    done = subprocess.run([sys.executable, str(script), str(out)],
                          capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr
    assert "§4.6" in out.read_text(encoding="utf-8")


# --- I5: the numbers behind the columns, pinned ---------------------------

def test_n_is_the_per_condition_row_count_not_the_whole_frame(tmp_path):
    """`n` is the sample size behind every published CI.

    Kills the mutant reporting `len(df)`, which is 20x the true count and
    would make every interval look 20x better supported than it is.
    """
    df = _scores(n=120, seed=4)
    m = condition_metrics(df, seed=0, n_boot=_NB)
    assert (m["n"] == 120).all()
    assert int(m["n"].sum()) == len(df)
    assert len(df) == 120 * len(EVAL_GRID)     # the value the mutant would report


def test_tpr_at_1pct_is_measured_at_one_percent_fpr():
    """Kills the mutant publishing `tpr_at_fpr(y, s, 0.05)` under this name.

    This is §6.1's REPORTING TPR at the project operating point, computed over
    every scored row -- not the §6.4 selection rule, which restricts to
    val_internal authentic against heldout_generator generated and drops clean
    (`errors.heldout_robust_tpr`). Reporting the 5% figure under the 1% heading
    still misstates the operating point the whole report is written at.
    """
    df = _scores(n=120, seed=6)
    m = condition_metrics(df, seed=0, n_boot=_NB).set_index("condition")
    g = df[df["condition"] == "jpeg_q30"]
    y, s = g["label"].to_numpy(), g["score"].to_numpy()
    at_1 = tpr_at_fpr(y, s, 0.01)
    at_5 = tpr_at_fpr(y, s, 0.05)
    assert m.loc["jpeg_q30", "tpr_at_1pct"] == pytest.approx(at_1)
    assert at_1 != pytest.approx(at_5)         # the two levels really do differ here


def test_ece_is_computed_per_condition_not_over_the_whole_frame():
    """Kills the mutant passing the whole frame's probabilities.

    Under it every condition reports the same pooled number, so a single badly
    calibrated condition is averaged into invisibility.
    """
    df = _scores(n=120, seed=8)
    probs = 1.0 / (1.0 + np.exp(-df["score"].to_numpy()))
    # One condition is deliberately, grossly miscalibrated.
    bad = (df["condition"] == "jpeg_q30").to_numpy()
    probs = probs.copy()
    probs[bad] = 0.99
    m = condition_metrics(df, probs=pd.Series(probs), seed=0,
                          n_boot=_NB).set_index("condition")
    for cond in ("clean", "jpeg_q30", "blur_s2.0"):
        g = df["condition"].to_numpy() == cond
        assert m.loc[cond, "ece"] == pytest.approx(
            expected_calibration_error(df["label"].to_numpy()[g], probs[g]))
    pooled = expected_calibration_error(df["label"].to_numpy(), probs)
    assert m["ece"].nunique() > 1                       # not one pooled number
    assert m.loc["jpeg_q30", "ece"] != pytest.approx(pooled)
    assert m.loc["jpeg_q30", "ece"] > m.loc["clean", "ece"]


def test_the_reject_everything_operating_point_is_reachable():
    """When a condition has destroyed the signal, rejecting everything is the
    best accuracy available, and it is not a threshold in the observed scores.

    Kills the mutant that drops the `k == n` candidate from the sweep.
    """
    rng = np.random.default_rng(11)
    # The signal is not merely absent but inverted: the ten positives hold the
    # ten LOWEST scores, so every threshold in the observed scores does worse
    # than declining to fire at all.
    s = np.sort(rng.normal(0.0, 1.0, 100))
    y = np.zeros(100, dtype=int)
    y[:10] = 1
    df = pd.DataFrame({"condition": "clean", "image_idx": np.arange(100),
                       "label": y, "score": s})
    got = float(condition_metrics(df, seed=0, n_boot=_NB).loc[0, "acc_oracle"])
    reject_all = float((y == 0).mean())
    best_observed = max(accuracy_at_threshold(y, s, t) for t in np.unique(s))
    assert best_observed < reject_all           # unreachable from the scores alone
    assert got == pytest.approx(reject_all)


# --- the smoke tier is first-class, and stamped -------------------------

def test_smoke_is_a_tier_rather_than_a_bypass(tmp_path, monkeypatch):
    """A partial-grid run needs somewhere to go that is not around the checks.

    A bypass is how a three-condition smoke number reaches a results table, so
    `smoke` is a real tier with open coverage whose every rendering carries
    NOT FOR PUBLICATION. Kills the mutant that drops the banner, and the one
    that lets `smoke` render as a publishable tier.
    """
    assert "smoke" in TIER_CONDITIONS and TIER_CONDITIONS["smoke"] is None
    partial = _scores(n=40, seed=1, conditions=["clean", "jpeg_q30", "blur_s2.0"])
    t = robustness_table({"a0": partial}, tier="smoke", n_boot=_NB, banks=_NOBANK)
    assert (t["tier"] == "smoke").all()
    md = tmp_path / "smoke.md"
    to_markdown(t, tier="smoke", path=str(md))
    text = md.read_text()
    assert NOT_FOR_PUBLICATION in text
    assert "must not be quoted" in text
    got = _render_and_capture(monkeypatch, t, str(tmp_path / "smoke.png"))
    assert NOT_FOR_PUBLICATION in got["title"]
    # The same three conditions would be rejected at a publishable tier.
    for tier in ("ablation", "final_report"):
        with pytest.raises(ValueError, match="missing"):
            robustness_table({"a0": partial}, tier=tier, n_boot=_NB, banks=_NOBANK)


def test_publishable_tiers_carry_no_not_for_publication_banner(tmp_path):
    """The positive control for the banner above."""
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB,
                         banks=_NOBANK)
    p = tmp_path / "pub.md"
    to_markdown(t, tier="ablation", path=str(p))
    assert NOT_FOR_PUBLICATION not in p.read_text()


# --- the metric is validated before anything is computed ------------------

def test_the_metric_is_rejected_before_a_single_resample_runs(monkeypatch):
    """Kills the mutant validating `metric` inside the rung loop.

    There it accepted `heldout_severity` -- a bool flag, not a metric, which
    silently tabulated as 0.0/1.0 -- and spent a full 20 x n_boot resampling
    pass before rejecting anything else.

    Asserted by counting calls to `condition_metrics` rather than by timing:
    "nothing was scored" IS the property, and a wall-clock proxy makes the
    mutant hang for the whole resampling pass instead of failing fast.
    """
    from aigcdet.eval import report as report_mod
    calls = []
    real = report_mod.condition_metrics
    monkeypatch.setattr(report_mod, "condition_metrics",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    for bad in ("heldout_severity", "n", "boot_seed", "nonsense"):
        with pytest.raises(ValueError, match="unknown metric"):
            robustness_table({"a0": _scores(seed=1)}, tier="ablation",
                             metric=bad, n_boot=_NB, banks=_NOBANK)
    assert calls == [], "a rejected metric must not resample anything first"

    # The positive control: a valid metric does reach the scoring pass.
    robustness_table({"a0": _scores(seed=1)}, tier="ablation", metric="auc",
                     n_boot=_NB, banks=_NOBANK)
    assert calls == [1]
    assert set(METRIC_COLUMNS) == {"auc", "auc_lo", "auc_hi", "tpr_at_1pct",
                                   "acc_oracle", "acc_fixed", "ece"}


def test_proxied_families_is_exported_and_is_the_default():
    """Callers should not hardcode the triple and drift from the mapping."""
    assert PROXIED_FAMILIES == ("jpeg", "blur", "noise")
    pred, proxies = _planted(n=200)
    default = validate_degradation_head(pred, proxies)
    explicit = validate_degradation_head(pred, proxies, families=PROXIED_FAMILIES)
    assert list(default["family"]) == list(PROXIED_FAMILIES)
    assert default["spearman"].tolist() == explicit["spearman"].tolist()


# --- C1: the table carries the §6.4 number, not just a TPR-shaped column ---

def _split_scores(sep_held, sep_val_fake, seed, n=60, conditions=None,
                  bench_shift=0.0):
    """A score frame over the four populations an eval bank really holds.

    `_scores` above is one undifferentiated block, which is exactly the fixture
    that cannot see the C1 defect: with no splits, §6.1's whole-frame TPR and
    §6.4's held-out-generator TPR are the same number. Here the two are pulled
    apart on purpose -- `sep_held` separates the held-out-generator fakes, which
    §6.4 selects on, and `sep_val_fake` the SEEN-generator fakes, which only the
    whole-frame column sees.

    Returns `(scores, splits)` with `splits` positionally indexed by image_idx,
    exactly as `bank.meta["split"]` is. `bench_shift` moves the BENCHMARK
    authentic rows, the confound that makes a threshold fitted on the scored
    frame differ from one fitted on validation.
    """
    rng = np.random.default_rng(seed)
    splits = np.array(["val_internal"] * n + ["val_internal"] * n
                      + ["heldout_generator"] * n + ["benchmark"] * n)
    labels = np.array([0] * n + [1] * n + [1] * n + [0] * n)
    rows = []
    for cond in (EVAL_GRID if conditions is None else conditions):
        mu = np.where(labels == 1, sep_val_fake, 0.0).astype(float)
        mu[splits == "heldout_generator"] = sep_held
        mu[splits == "benchmark"] += bench_shift
        rows.append(pd.DataFrame({"condition": cond,
                                  "image_idx": np.arange(len(labels)),
                                  "label": labels, "generator": "g",
                                  "source": "src",
                                  "score": rng.normal(mu, 1.0)}))
    return pd.concat(rows, ignore_index=True), splits


def _disagreeing_rungs(conditions=None):
    """Two rungs the §6.1 column and the §6.4 rule rank in OPPOSITE orders.

    `a3` finds the held-out generators and nothing else; `a4` finds only the
    seen ones. The whole-frame TPR column prefers a4 (it has more separable
    positives overall); §6.4, which looks only at held-out-generator fakes
    against val_internal authentics, prefers a3.
    """
    a3, splits = _split_scores(3.5, 0.0, seed=1, conditions=conditions)
    a4, _ = _split_scores(0.0, 6.0, seed=2, conditions=conditions)
    return {"a3": a3, "a4": a4}, splits


def test_the_tables_tpr_column_and_the_64_rule_can_name_different_winners():
    """The premise of the selection column, asserted rather than assumed.

    If this ever stops holding, the fixture has stopped exercising the defect
    and the tests below are vacuous.
    """
    from aigcdet.eval.errors import heldout_robust_tpr

    per_rung, splits = _disagreeing_rungs()
    t = robustness_table(per_rung, tier="ablation", metric="tpr_at_1pct",
                         n_boot=_NB, banks=_NOBANK)
    rule = {r: heldout_robust_tpr(df, splits) for r, df in per_rung.items()}
    assert t["robust_tpr_at_1pct"].idxmax() == "a4"
    assert max(rule, key=rule.get) == "a3"


def test_the_selection_metric_is_carried_as_its_own_column():
    """Kills the mutant that drops `selection` on the floor.

    Without the column a reader picks the headline off `robust_tpr_at_1pct`,
    which on this fixture names the rung §6.4 rejects -- a `robustness_table.md`
    saying A4 next to a `selection.json` saying A3.
    """
    from aigcdet.eval.errors import SELECTION_METRIC, heldout_robust_tpr

    per_rung, splits = _disagreeing_rungs()
    rule = {r: heldout_robust_tpr(df, splits) for r, df in per_rung.items()}
    t = robustness_table(per_rung, tier="ablation", metric="tpr_at_1pct",
                         n_boot=_NB, banks=_NOBANK, selection=rule)
    assert SELECTION_METRIC in t.columns
    for rung, value in rule.items():
        assert float(t.loc[rung, SELECTION_METRIC]) == pytest.approx(value)
    assert t[SELECTION_METRIC].idxmax() == "a3" != t["robust_tpr_at_1pct"].idxmax()
    # And it is a summary column: not plotted, not counted as a condition, so
    # it cannot break the tier's coverage check or wash out the heatmap.
    from aigcdet.eval.report import _condition_columns
    assert SELECTION_METRIC not in _condition_columns(t)


def test_the_markdown_names_the_64_winner_and_not_the_column_winner(tmp_path):
    """Kills the mutant that renders the table without the selection paragraph,
    and the one that reads the headline off `robust_<metric>`."""
    from aigcdet.eval.errors import SELECTION_METRIC, heldout_robust_tpr

    per_rung, splits = _disagreeing_rungs()
    rule = {r: heldout_robust_tpr(df, splits) for r, df in per_rung.items()}
    t = robustness_table(per_rung, tier="ablation", metric="tpr_at_1pct",
                         n_boot=_NB, banks=_NOBANK, selection=rule)
    p = tmp_path / "sel.md"
    to_markdown(t, tier="ablation", path=str(p))
    text = p.read_text()
    assert SELECTION_METRIC in text
    assert "Highest among the eligible rungs in this table: `a3`" in text
    assert "REPORTING" in text                    # the §6.1 disclaimer
    assert f"| {SELECTION_METRIC} |" in text      # a real column in the table


def test_a_table_without_the_selection_column_says_so(tmp_path):
    """The negative case must be stated, not left blank: a reader who cannot
    see a §6.4 column reads the headline off whichever column looks like it."""
    from aigcdet.eval.errors import SELECTION_METRIC

    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB,
                         banks=_NOBANK)
    p = tmp_path / "nosel.md"
    to_markdown(t, tier="ablation", path=str(p))
    text = p.read_text()
    assert "not carried in this table" in text
    assert "selection.json" in text
    assert SELECTION_METRIC in text               # named, so it can be looked up


def test_the_heatmap_states_the_64_headline_too(tmp_path, monkeypatch):
    """The figure is what gets screenshotted, so it must not be the one path
    where a headline can be read off a §6.1 colour scale.

    Kills the mutant that drops the subtitle from `save_heatmap`.
    """
    from aigcdet.eval.errors import heldout_robust_tpr

    per_rung, splits = _disagreeing_rungs()
    rule = {r: heldout_robust_tpr(df, splits) for r, df in per_rung.items()}
    t = robustness_table(per_rung, tier="ablation", metric="tpr_at_1pct",
                         n_boot=_NB, banks=_NOBANK, selection=rule)
    title = _render_and_capture(monkeypatch, t, str(tmp_path / "sel.png"))["title"]
    assert "a3" in title and "§6.4" in title

    bare = robustness_table(per_rung, tier="ablation", metric="tpr_at_1pct",
                            n_boot=_NB, banks=_NOBANK)
    bare_title = _render_and_capture(monkeypatch, bare,
                                     str(tmp_path / "bare.png"))["title"]
    assert "a3" not in bare_title and "selection.json" in bare_title


def test_a_partial_or_unusable_selection_mapping_is_refused():
    """A blank cell in the selection column reads as a rung that LOST."""
    per_rung, splits = _disagreeing_rungs()
    with pytest.raises(ValueError, match="selection covers rungs"):
        robustness_table(per_rung, tier="ablation", n_boot=_NB, banks=_NOBANK,
                         selection={"a3": 0.5})
    with pytest.raises(ValueError, match="non-finite"):
        robustness_table(per_rung, tier="ablation", n_boot=_NB, banks=_NOBANK,
                         selection={"a3": 0.5, "a4": float("nan")})


# --- I1: a metric with no producer is refused, not rendered blank ----------

def test_ece_without_probabilities_is_refused_rather_than_tabulated_as_nan():
    """Kills the mutant that lets `metric="ece"` through to `condition_metrics`
    with no probabilities.

    Every cell is then NaN, which renders as blank markdown cells and a flat
    heatmap -- a table that looks like a result and says nothing.
    """
    with pytest.raises(ValueError, match="calibrated probabilities"):
        robustness_table({"a0": _scores(seed=1)}, tier="ablation", metric="ece",
                         n_boot=_NB, banks=_NOBANK)


def test_ece_is_tabulated_when_probabilities_are_supplied():
    """The positive control: the refusal is about missing inputs, not about the
    metric being unsupported."""
    scores = _scores(n=200, seed=3)
    probs = 1.0 / (1.0 + np.exp(-scores["score"].to_numpy()))
    t = robustness_table({"a0": scores}, tier="ablation", metric="ece",
                         n_boot=_NB, banks=_NOBANK, probs={"a0": probs})
    assert t.loc["a0", "clean"] == pytest.approx(
        expected_calibration_error(
            scores[scores["condition"] == "clean"]["label"].to_numpy(),
            probs[(scores["condition"] == "clean").to_numpy()]))
    assert np.isfinite(t.loc["a0", "robust_ece"])


def test_probabilities_must_cover_every_rung_and_align_row_for_row():
    scores = _scores(n=40, seed=3)
    probs = 1.0 / (1.0 + np.exp(-scores["score"].to_numpy()))
    with pytest.raises(ValueError, match="probs covers rungs"):
        robustness_table({"a0": scores, "a3": _scores(n=40, seed=4)},
                         tier="ablation", metric="ece", n_boot=_NB,
                         banks=_NOBANK, probs={"a0": probs})
    with pytest.raises(ValueError, match="align row for row"):
        robustness_table({"a0": scores}, tier="ablation", metric="ece",
                         n_boot=_NB, banks=_NOBANK, probs={"a0": probs[:10]})


# --- I2: the frozen threshold comes from validation, not from the scored rows

def test_acc_fixed_is_refused_without_a_threshold_from_validation():
    """Kills the mutant that lets `robustness_table` default the threshold.

    Defaulted, it is fitted on the clean rows of the frame being scored: at the
    final_report tier that frame IS the benchmark §6.7 says is touched once,
    and on `clean` it makes acc_fixed a second acc_oracle.
    """
    with pytest.raises(ValueError, match="clean_threshold"):
        robustness_table({"a0": _scores(seed=1)}, tier="ablation",
                         metric="acc_fixed", n_boot=_NB, banks=_NOBANK)


def test_a_supplied_threshold_reopens_the_gap_the_two_columns_exist_to_show():
    """With the threshold fitted on the scored frame's own clean rows, the
    `clean` column's acc_fixed - acc_oracle gap is zero BY CONSTRUCTION -- the
    one condition where the reported score drift cannot be non-zero.

    The fixture shifts the BENCHMARK authentic rows up, which is the confound
    that makes the difference visible: the frame-fitted threshold is tuned to
    them, a validation-fitted one is not. Kills the mutant that ignores a
    supplied `clean_threshold`.
    """
    from aigcdet.eval.report import clean_validation_threshold

    scores, splits = _split_scores(3.5, 2.0, seed=11,
                                   conditions=["clean", "jpeg_q30"],
                                   bench_shift=2.0)
    fitted_here = condition_metrics(scores, seed=0, n_boot=_NB).set_index("condition")
    assert float(fitted_here.loc["clean", "acc_fixed"]) == pytest.approx(
        float(fitted_here.loc["clean", "acc_oracle"]))

    threshold = clean_validation_threshold(scores, splits)
    from_validation = condition_metrics(scores, clean_threshold=threshold,
                                        seed=0, n_boot=_NB).set_index("condition")
    assert float(from_validation.loc["clean", "acc_fixed"]) < float(
        from_validation.loc["clean", "acc_oracle"])
    assert (from_validation["acc_fixed"] <= from_validation["acc_oracle"] + 1e-9).all()


def test_condition_metrics_records_where_its_threshold_came_from():
    """A number whose provenance is not recorded cannot be audited later."""
    df = _scores(n=40, seed=5)
    default = condition_metrics(df, seed=0, n_boot=_NB)
    supplied = condition_metrics(df, clean_threshold=0.25, seed=0, n_boot=_NB)
    assert (default["clean_threshold_source"] == "fitted_on_the_scored_clean_rows").all()
    assert (supplied["clean_threshold_source"] == "supplied").all()
    assert (supplied["clean_threshold"] == 0.25).all()


def test_the_validation_threshold_ignores_benchmark_and_degraded_rows():
    """Kills two mutants: one that fits on every clean row, one that fits on
    every `val_internal` row whatever the condition.

    Both need rows that INTERLEAVE with the decision region to be detectable --
    a block shifted far above every candidate threshold is misclassified at a
    constant rate and moves no optimum, which is how a lenient fixture can hide
    both mutants. So the benchmark authentics sit ON the generated cluster (the
    "benchmark reals look fake" confound this threshold must not be tuned to),
    and the degraded condition is barely separated at all.
    """
    from aigcdet.eval.report import _best_threshold, clean_validation_threshold

    n = 20
    rng = np.random.default_rng(3)
    splits = np.array(["val_internal"] * (2 * n) + ["benchmark"] * (6 * n))
    labels = np.array([0] * n + [1] * n + [0] * (6 * n))
    rows = []
    for cond, mu_fake, sd in (("clean", 3.0, 0.3), ("jpeg_q30", 0.5, 1.0)):
        mu = np.where(labels == 1, mu_fake, 0.0).astype(float)
        if cond == "clean":
            mu = mu + (splits == "benchmark") * 3.0
        rows.append(pd.DataFrame({"condition": cond,
                                  "image_idx": np.arange(len(labels)),
                                  "label": labels, "generator": "g",
                                  "source": "src", "score": rng.normal(mu, sd)}))
    scores = pd.concat(rows, ignore_index=True)
    row_split = splits[scores["image_idx"].to_numpy()]

    got = clean_validation_threshold(scores, splits)
    val_clean = scores[(row_split == "val_internal") & (scores["condition"] == "clean")]
    assert got == pytest.approx(_best_threshold(val_clean["label"].to_numpy(),
                                                val_clean["score"].to_numpy()))
    all_clean = scores[scores["condition"] == "clean"]
    assert got != pytest.approx(_best_threshold(all_clean["label"].to_numpy(),
                                                all_clean["score"].to_numpy()))
    all_val = scores[row_split == "val_internal"]
    assert got != pytest.approx(_best_threshold(all_val["label"].to_numpy(),
                                                all_val["score"].to_numpy()))


def test_the_validation_threshold_refuses_a_benchmark_only_frame():
    """A benchmark-only bank must not quietly supply the deployment threshold."""
    from aigcdet.eval.report import clean_validation_threshold

    scores, splits = _split_scores(3.5, 0.0, seed=8, conditions=["clean"])
    benchmark_only = np.full(len(splits), "benchmark")
    with pytest.raises(ValueError, match="val_internal"):
        clean_validation_threshold(scores, benchmark_only)


def test_a_threshold_reaches_the_table_and_is_recorded_there():
    from aigcdet.eval.report import clean_validation_threshold

    per_rung, splits = _disagreeing_rungs()
    thresholds = {r: clean_validation_threshold(df, splits)
                  for r, df in per_rung.items()}
    t = robustness_table(per_rung, tier="ablation", metric="acc_fixed",
                         n_boot=_NB, banks=_NOBANK, clean_threshold=thresholds)
    for rung, thr in thresholds.items():
        assert float(t.loc[rung, "clean_threshold"]) == pytest.approx(thr)
        expected = condition_metrics(per_rung[rung], clean_threshold=thr,
                                     n_boot=_NB).set_index("condition")
        assert float(t.loc[rung, "clean"]) == pytest.approx(
            float(expected.loc["clean", "acc_fixed"]))
    with pytest.raises(ValueError, match="clean_threshold covers rungs"):
        robustness_table(per_rung, tier="ablation", metric="acc_fixed",
                         n_boot=_NB, banks=_NOBANK, clean_threshold={"a3": 0.0})


# --- I3: the tier line describes THIS table, not the plan ------------------

def test_the_tier_line_reports_what_the_table_actually_covers(tmp_path):
    """Kills the mutant that writes `TIER_DESCRIPTIONS[tier]` alone.

    That sentence asserted "the complete benchmark over the 15 core conditions"
    verbatim above a 40-image table, in the file a report writer quotes from.
    """
    from aigcdet.eval.report import describe_tier

    small = robustness_table(
        {"a0": _scores(n=40, seed=1, conditions=list(CORE_CONDITIONS))},
        tier="final_report", n_boot=_NB, banks=_NOBANK)
    line = describe_tier("final_report", small)
    assert "1 rung(s) x 15 condition(s) over 40 image(s)" in line
    # The plan is still quoted -- explicitly AS the plan, not as this table.
    assert "Planned budget" in line and "13.8k" in line
    assert line.index("Planned budget") < line.index("THIS TABLE")

    p = tmp_path / "obs.md"
    to_markdown(small, tier="final_report", path=str(p))
    assert "over 40 image(s)" in p.read_text()


def test_the_tier_line_moves_with_the_table_it_describes():
    """Two tables at one tier must not get the same composition sentence."""
    from aigcdet.eval.report import describe_tier

    a = robustness_table({"a0": _scores(n=40, seed=1)}, tier="ablation",
                         n_boot=_NB, banks=_NOBANK)
    b = robustness_table({"a0": _scores(n=120, seed=1), "a3": _scores(n=120, seed=2)},
                         tier="ablation", n_boot=_NB, banks=_NOBANK)
    assert "1 rung(s)" in describe_tier("ablation", a)
    assert "40 image(s)" in describe_tier("ablation", a)
    assert "2 rung(s)" in describe_tier("ablation", b)
    assert "120 image(s)" in describe_tier("ablation", b)
    assert describe_tier("ablation", a) != describe_tier("ablation", b)
