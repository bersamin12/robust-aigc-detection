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
from aigcdet.eval.metrics import accuracy_at_threshold
from aigcdet.eval.report import (
    TIER_CONDITIONS, condition_metrics, robustness_table, save_heatmap,
    to_markdown, validate_degradation_head,
)


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
    t = robustness_table({"a0": _scores(seed=1), "a3": _scores(seed=2)}, tier="ablation", n_boot=_NB)
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
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB)
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
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB)
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

    Kills the mutant that hardcodes the aggregate column name: under it the
    §6.4 selection metric (robust TPR @ 1% FPR) would be presented as an AUC.
    """
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation",
                         metric="tpr_at_1pct", n_boot=_NB)
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
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB)
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
                         n_boot=_NB, seed=99)
    assert int(t.loc["a0", "n_images"]) == 400
    assert int(t.loc["a0", "boot_seed"]) == 99
    assert int(t.loc["a0", "boot_n"]) == _NB
    # At its defaults the table really does resample 1000 times (spec §6.1).
    d = robustness_table({"a0": _scores(n=20, seed=1, conditions=list(CORE_CONDITIONS))},
                         tier="final_report")
    assert int(d.loc["a0", "boot_n"]) == 1000
    assert int(d.loc["a0", "boot_seed"]) == 20260827
    assert int(d.loc["a0", "n_images"]) == 20


@pytest.mark.parametrize("tier", ["", "Ablation", "final", "selection", None])
def test_robustness_table_rejects_an_unlabelled_or_unknown_tier(tier):
    with pytest.raises(ValueError, match="tier"):
        robustness_table({"a0": _scores(seed=1)}, tier=tier)


def test_robustness_table_rejects_a_tier_that_contradicts_the_coverage():
    """The 15-core-condition final-report tier cannot be claimed for a
    20-condition selection-tier run, or the other way round.

    Kills the mutant that accepts any tier string it recognises without
    checking what was actually evaluated -- the mechanism that makes a
    *wrongly* labelled table impossible rather than merely discouraged.
    """
    full = {"a0": _scores(seed=1)}
    with pytest.raises(ValueError, match="final_report"):
        robustness_table(full, tier="final_report", n_boot=_NB)
    core = {"a0": _scores(seed=1, conditions=list(CORE_CONDITIONS))}
    with pytest.raises(ValueError, match="ablation"):
        robustness_table(core, tier="ablation", n_boot=_NB)
    # Each is fine at its own tier.
    assert len(robustness_table(core, tier="final_report", n_boot=_NB).columns) > 15
    assert set(TIER_CONDITIONS["final_report"]) == set(CORE_CONDITIONS)


def test_robustness_table_rejects_rungs_with_different_condition_coverage():
    """CARRY C-A / R24 at the frame level: differing view coverage between
    compared rungs makes the table a comparison of augmentation budgets."""
    short = list(EVAL_GRID)[:10]
    with pytest.raises(ValueError, match="different conditions"):
        robustness_table({"a0": _scores(seed=1),
                          "a3": _scores(seed=2, conditions=short)}, tier="ablation", n_boot=_NB)


def test_robustness_table_rejects_rungs_whose_conditions_are_merely_reordered():
    """Same twenty conditions in a different order is still not comparable
    when it comes from a differently ordered bank; the error must say so
    rather than silently aligning by name."""
    names = list(EVAL_GRID)
    reordered = [names[0]] + names[1:][::-1]
    with pytest.raises(ValueError, match="different order"):
        robustness_table({"a0": _scores(seed=1),
                          "a3": _scores(seed=2, conditions=reordered)},
                         tier="ablation", n_boot=_NB)


def test_robustness_table_rejects_rungs_scored_on_different_images():
    a = _scores(n=400, seed=1)
    b = _scores(n=400, seed=2)
    b["image_idx"] = b["image_idx"] + 1000
    with pytest.raises(ValueError, match="different images"):
        robustness_table({"a0": a, "a3": b}, tier="ablation", n_boot=_NB)


def test_robustness_table_rejects_an_empty_comparison():
    with pytest.raises(ValueError, match="nothing to compare"):
        robustness_table({}, tier="ablation", n_boot=_NB)


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
        robustness_table(scores, tier="ablation", banks={"a0": a, "a3": b})


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
        cfg = json.load(open(cfg_path))
        cfg["manifest_sha256"] = None
        json.dump(cfg, open(cfg_path, "w"))
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
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB)
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
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB)
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
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB)
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
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB)
    dropped = t.drop(columns=list(HELDOUT_SEVERITY_CONDITIONS))
    with pytest.raises(ValueError, match="unseen-severity"):
        to_markdown(dropped, tier="ablation", path=str(tmp_path / "d.md"))


def test_markdown_refuses_a_table_that_states_no_tier(tmp_path):
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB)
    naked = t.drop(columns=["tier"])
    naked.attrs.clear()
    with pytest.raises(ValueError, match="states no evaluation tier"):
        to_markdown(naked, tier="ablation", path=str(tmp_path / "n.md"))


def test_markdown_records_the_bootstrap_provenance(tmp_path):
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", seed=1234, n_boot=_NB)
    p = tmp_path / "prov.md"
    to_markdown(t, tier="ablation", path=str(p))
    text = p.read_text()
    assert f"{_NB} resamples" in text and "seed 1234" in text
    assert "5k internal validation" in text        # the tier's row budget


def test_markdown_values_match_the_table(tmp_path):
    """The rendered numbers must be the table's own, not a re-derivation."""
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB)
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
                         tier="ablation", n_boot=_NB)
    assert not pd.api.types.is_numeric_dtype(t["tier"])
    p = tmp_path / "heat.png"
    save_heatmap(t, str(p))
    assert p.exists() and p.stat().st_size > 1000
    assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    import matplotlib
    assert matplotlib.get_backend().lower() == "agg"


def test_heatmap_scale_does_not_flatten_the_selection_metric(tmp_path):
    """TPR @ 1% FPR -- the §6.4 selection metric -- lives well below 0.5.

    Kills the mutant that hardcodes `vmin=0.5` for every metric: under it two
    tables whose only difference is sub-0.5 TPR values render byte-identical,
    because every one of those cells clamps to the same colour and the figure
    hides the differences it exists to show. AUC keeps its 0.5 chance floor.
    """
    from aigcdet.eval.report import heatmap_limits
    assert heatmap_limits("auc") == (0.5, 1.0)
    assert heatmap_limits("tpr_at_1pct") == (0.0, 1.0)

    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation",
                         metric="tpr_at_1pct", n_boot=_NB)
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
    t = robustness_table({"a0": _scores(seed=1)}, tier="ablation", n_boot=_NB)
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
