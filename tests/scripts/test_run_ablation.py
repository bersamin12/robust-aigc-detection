"""The ablation orchestrator, exercised end to end on tiny synthetic banks.

No GPU, no weights, no downloads: the banks are written directly with
`BankWriter`, so no backbone is ever loaded, and every artefact lands under
`tmp_path`.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest
import yaml

from aigcdet.augment.scenarios import EVAL_GRID
from aigcdet.eval.errors import (
    SELECTION_METRIC, SELECTION_POPULATION, SELECTION_SPLITS,
    SELECTION_TARGET_FPR, IneligibleRungWarning,
)
from aigcdet.features.bank import N_FAMILIES, N_VIEWS, BankWriter

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = _ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ra = _load_script("run_ablation")

DIM = 6


def _train_bank(tmp_path, n=48, name="train_bank", seed=0) -> str:
    """A learnable training bank: fakes sit at +1.5, reals at -1.5.

    `name` and `seed` exist for the fusion tests, which need a SECOND,
    independently extracted bank whose head is a different model.
    """
    out = str(tmp_path / name)
    w = BankWriter(out, n, N_VIEWS, DIM, "fake", 0, manifest_sha256="tb")
    rng = np.random.default_rng(seed)
    for i in range(n):
        label = i % 2
        clean = rng.normal(1.5 if label else -1.5, 0.4, DIM)
        feats = np.stack([clean] + [clean + rng.normal(0, 0.5, DIM)
                                    for _ in range(N_VIEWS - 1)]).astype(np.float32)
        presence = np.zeros((N_VIEWS, N_FAMILIES), np.float32)
        presence[1:, 0] = 1.0
        severity = np.zeros((N_VIEWS, N_FAMILIES), np.float32)
        severity[1:, 0] = 0.5
        w.write_image(i, {"path": f"/p{i}.png", "label": label,
                          "generator": f"g{label}", "source": "s",
                          "split": "train" if i < n - 12 else "val_internal"},
                      feats=feats, presence=presence, severity=severity,
                      proxies=np.zeros((N_VIEWS, 3), np.float32),
                      recipes=["[]"] * N_VIEWS)
    w.close()
    return out


def _eval_bank(tmp_path, name="eval_bank", fingerprint="eb", n_per_block=60,
               backbone="fake", seed=1) -> str:
    """An eval bank over the full 20-condition grid.

    Rows come in four blocks so the §6.4 population and the contamination it
    must ignore are both present: val_internal authentic, val_internal
    generated (SEEN generators), heldout_generator generated, benchmark
    authentic.

    The classes overlap here, and the degraded views far more than the clean
    one, so the robust held-out TPR lands well below 1.0 and is distinguishable
    from the clean-view AUC. A perfectly separable eval bank would make the two
    numbers equal and hide a selection rule reading the wrong key.

    `n_per_block` is 60, not a handful, for a second reason: with only ten
    authentic rows per condition the reachable FPRs are so coarse (1/10) that
    TPR@1% and TPR@5% are the SAME NUMBER, and a call site that moved the
    operating point would be invisible to every test here. At 60 they are
    roughly 0.34 against 0.66.
    """
    out = str(tmp_path / name)
    conditions = list(EVAL_GRID)
    blocks = [("val_internal", 0, ""), ("val_internal", 1, "g_seen"),
              ("heldout_generator", 1, "g_held"), ("benchmark", 0, "")]
    n = n_per_block * len(blocks)
    w = BankWriter(out, n, len(conditions), DIM, backbone, 0,
                   manifest_sha256=fingerprint,
                   extra_config={"conditions": conditions})
    rng = np.random.default_rng(seed)
    i = 0
    for split, label, generator in blocks:
        for _ in range(n_per_block):
            base = rng.normal(0.5 if label else -0.5, 0.6, DIM)
            feats = np.stack([base] + [base + rng.normal(0, 1.0, DIM)
                                       for _ in range(len(conditions) - 1)]
                             ).astype(np.float32)
            presence = np.zeros((len(conditions), N_FAMILIES), np.float32)
            presence[1:, 0] = 1.0
            severity = np.zeros((len(conditions), N_FAMILIES), np.float32)
            severity[1:, 0] = 0.5
            w.write_image(i, {"path": f"/e{i}.png", "label": label,
                              "generator": generator, "source": "s", "split": split},
                          feats=feats, presence=presence, severity=severity,
                          proxies=np.zeros((len(conditions), 3), np.float32),
                          recipes=["[]"] * len(conditions))
            i += 1
    w.close()
    return out


_RUNGS = {
    "a0": {"use_augmented": False, "use_consistency": False, "use_degradation": False},
    "a3": {"use_augmented": True, "use_consistency": True, "use_degradation": True},
    # A3 + FiLM, recon off: the rung whose name is not a bare `aN`, so it is
    # also the one that proves the orchestrator keys nothing on that shape.
    "a7_norecon": {"use_augmented": True, "use_consistency": True,
                   "use_degradation": True, "use_recon": False, "use_film": True},
    # A6 is a3's config plus the one inference-only flag. Written out in full
    # rather than derived from _RUNGS["a3"], because the thing under test is
    # that the two are identical everywhere else.
    "a6": {"use_augmented": True, "use_consistency": True,
           "use_degradation": True, "tta": True},
}


def _rung_yaml(tmp_path, name, epochs=2) -> str:
    cfg = {"name": name, "epochs": epochs, "n_src": 8, "m_deg": 2, **_RUNGS[name]}
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def _argv(tmp_path, rungs=("a0", "a3"), **overrides) -> list[str]:
    args = {
        "--bank": _train_bank(tmp_path),
        "--eval-bank": _eval_bank(tmp_path),
        "--tier": "ablation",
        "--out": str(tmp_path / "out" / "robustness_table.md"),
        "--selection": str(tmp_path / "out" / "selection.json"),
        "--out-dir": str(tmp_path / "rungs"),
        "--device": "cpu",
        "--boot-n": "5",
    }
    args.update(overrides)
    argv = [x for kv in args.items() for x in kv]
    return argv + ["--rungs"] + [_rung_yaml(tmp_path, r) for r in rungs]


def _tta_eval_bank(tmp_path, plain_dir, views=("identity", "hflip", "jpeg_95"),
                   name="tta_eval_bank"):
    """A TTA bank built FROM a plain one, so the two are the same evaluation.

    View `j * n + 0` is copied from the plain bank's condition `j` -- which is
    what the identity view means -- and the rest are perturbed. That makes the
    averaged score a genuinely different number from the plain bank's while
    keeping every axis, split, label and fingerprint identical, which is
    exactly the situation A6 is in.
    """
    from aigcdet.features.bank import FeatureBank

    plain = FeatureBank.open(plain_dir)
    conditions = list(plain.config["conditions"])
    n, views = len(views), list(views)
    out = str(tmp_path / name)
    w = BankWriter(out, len(plain.meta), len(conditions) * n, DIM,
                   plain.config["backbone"], 0,
                   manifest_sha256=plain.config["manifest_sha256"],
                   extra_config={"conditions": conditions, "tta_views": views})
    rng = np.random.default_rng(11)
    for i, row in plain.meta.iterrows():
        base = np.asarray(plain.feats[i]).astype(np.float32)
        feats = np.stack([base[j] if k == 0 else base[j] + rng.normal(0, 0.3, DIM)
                          for j in range(len(conditions)) for k in range(n)]
                         ).astype(np.float32)
        presence = np.repeat(np.asarray(plain.presence[i]), n, axis=0)
        severity = np.repeat(np.asarray(plain.severity[i]), n, axis=0)
        w.write_image(int(i), {"path": row["path"], "label": int(row["label"]),
                               "generator": row["generator"],
                               "source": row["source"], "split": row["split"]},
                      feats=feats, presence=presence, severity=severity,
                      proxies=np.zeros((len(conditions) * n, 3), np.float32),
                      recipes=["[]"] * (len(conditions) * n))
    w.close()
    return out


# --- rung A6 ---------------------------------------------------------------

def test_a6_is_scored_from_the_tta_bank_and_lands_in_the_table(tmp_path):
    """The whole point of T2-4: A6 is a ROW, not a footnote."""
    plain = _eval_bank(tmp_path)
    argv = _argv(tmp_path, rungs=("a3", "a6"), **{
        "--eval-bank": plain,
        "--tta-eval-bank": _tta_eval_bank(tmp_path, plain)})
    with _quiet_control_warning():
        ra.main(argv)
    rec = json.loads((tmp_path / "out" / "selection.json").read_text())

    assert "a6" in rec["summary"], "A6 produced no row"
    # a6 is in ELIGIBLE_RUNGS, so it is a CANDIDATE and not an
    # excluded control -- unlike a4vq/a4both/aF, which are not.
    assert "a6" in rec["candidates"]
    assert rec["tta"]["scored_here"] is True
    assert rec["tta"]["scored_rungs"] == ["a6"]
    assert rec["tta"]["cost_multiplier"] == 3
    table = (tmp_path / "out" / "robustness_table.md").read_text()
    assert "a6" in table


def test_a6_is_registered_with_the_bank_that_actually_produced_it(tmp_path):
    """Borrowing the plain bank would make the comparability check pass on an
    evaluation A6 never had, and would state a cost multiplier of one for a
    rung that paid three."""
    plain = _eval_bank(tmp_path)
    tta = _tta_eval_bank(tmp_path, plain)
    argv = _argv(tmp_path, rungs=("a3", "a6"),
                 **{"--eval-bank": plain, "--tta-eval-bank": tta})
    with _quiet_control_warning():
        ra.main(argv)
    rec = json.loads((tmp_path / "out" / "selection.json").read_text())
    assert rec["tta"]["eval_bank"] == tta


def test_a6_scores_differ_from_its_base_rung(tmp_path):
    """Same head, different views: if the two rows were identical the TTA bank
    was not being read at all."""
    plain = _eval_bank(tmp_path)
    argv = _argv(tmp_path, rungs=("a3", "a6"), **{
        "--eval-bank": plain,
        "--tta-eval-bank": _tta_eval_bank(tmp_path, plain)})
    with _quiet_control_warning():
        ra.main(argv)
    rec = json.loads((tmp_path / "out" / "selection.json").read_text())
    assert (rec["summary"]["a3"][SELECTION_METRIC]
            != rec["summary"]["a6"][SELECTION_METRIC])


def test_a6_head_is_bit_identical_to_its_base_rungs_head(tmp_path):
    """The one-flag claim, measured on the WEIGHTS.

    `tta` is inference-only, so an A6 head trained from the same seed on the
    same rows must BE a3's head. If it is not, something in training read the
    flag and A6's score stops being a measurement of test-time augmentation.
    """
    plain = _eval_bank(tmp_path)
    argv = _argv(tmp_path, rungs=("a3", "a6"), **{
        "--eval-bank": plain,
        "--tta-eval-bank": _tta_eval_bank(tmp_path, plain)})
    with _quiet_control_warning():
        ra.main(argv)
    rec = json.loads((tmp_path / "out" / "selection.json").read_text())
    assert rec["tta"]["weight_delta_vs_base"]["a6_vs_a3"] == 0.0


def test_a6_without_a_tta_bank_is_refused_by_name(tmp_path):
    """Scoring it off the plain bank would average over CONDITIONS -- a number
    that looks like a robustness score and is nothing of the kind."""
    with pytest.raises(SystemExit, match="tta-eval-bank"):
        with _quiet_control_warning():
            ra.main(_argv(tmp_path, rungs=("a3", "a6")))


def test_a_tta_bank_from_another_evaluation_is_refused_before_training(tmp_path):
    """Before, not after: the alternative is finding out once every rung in the
    ladder has been fitted."""
    plain = _eval_bank(tmp_path)
    other = _eval_bank(tmp_path, name="other", fingerprint="different")
    argv = _argv(tmp_path, rungs=("a3", "a6"), **{
        "--eval-bank": plain,
        "--tta-eval-bank": _tta_eval_bank(tmp_path, other, name="tta_other")})
    with pytest.raises(ValueError, match="manifest_sha256"):
        ra.main(argv)
    assert not (tmp_path / "rungs" / "a3" / "checkpoint.pt").exists(), (
        "a rung was trained before the TTA bank was checked")


def test_the_a6_temperature_is_refitted_on_the_averaged_logits(tmp_path):
    """The trap named in `aigcdet.eval.tta`'s docstring, discharged.

    A mean of correlated logits has a narrower spread than the single-view
    logits a temperature was fitted on, so A6 must carry its OWN `T`. It is
    recorded and not applied -- selection is invariant to a monotone rescale --
    but the shipped bundle's calibration is not, and this is where that number
    comes from.
    """
    plain = _eval_bank(tmp_path)
    argv = _argv(tmp_path, rungs=("a3", "a6"), **{
        "--eval-bank": plain,
        "--tta-eval-bank": _tta_eval_bank(tmp_path, plain)})
    with _quiet_control_warning():
        ra.main(argv)
    t = json.loads((tmp_path / "out" / "selection.json").read_text())["tta"]["temperature"]["a6"]
    assert t["fit_split"] == "val_internal"
    assert t["base_rung"] == "a3"
    assert t["T"] is not None and t["T"] > 0
    # Fitted on the AVERAGED logits, so it is its own number rather than the
    # base rung's copied across.
    assert t["T"] != t["base_T_single_view"]


def test_a_run_with_no_a6_still_records_the_cost_and_says_it_scored_nothing(tmp_path):
    """The footnote path has to survive A6 becoming real, or every run without
    it starts claiming an A6 row it does not have."""
    with _quiet_control_warning():
        ra.main(_argv(tmp_path, rungs=("a0", "a3")) + ["--tta"])
    rec = json.loads((tmp_path / "out" / "selection.json").read_text())
    assert rec["tta"]["scored_here"] is False
    assert rec["tta"]["scored_rungs"] == []
    assert "a6" not in rec["summary"]
    # The FULL view list, because no bank narrowed it: a run that records a
    # cost it did not pay is the thing this key exists to prevent.
    assert rec["tta"]["cost_multiplier"] == len(ra.TTA_VIEWS)


# --- end to end ------------------------------------------------------------

@contextlib.contextmanager
def _quiet_control_warning():
    """`a0` may or may not outscore `a3` on this synthetic bank, and the
    warning fires only when it does; either outcome is fine in the tests that
    are not about the warning, so it is silenced rather than left as noise."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IneligibleRungWarning)
        yield


def test_run_ablation_writes_the_table_heatmap_and_selection_record(tmp_path):
    with _quiet_control_warning():
        report = ra.main(_argv(tmp_path))

    out = tmp_path / "out"
    assert (out / "robustness_table.md").exists()
    assert (out / "robustness_table.png").exists()
    record = json.loads((out / "selection.json").read_text())
    assert record == report
    assert record["metric"] == SELECTION_METRIC
    assert record["population"] == SELECTION_POPULATION
    assert record["tier"] == "ablation"
    # The table's per-condition metric and the selection metric are different
    # things and are both recorded, so neither can be read as the other.
    assert record["table_metric"] == "auc"
    assert set(record["summary"]) == {"a0", "a3"}
    for rung in record["summary"].values():
        assert 0.0 <= rung[SELECTION_METRIC] <= 1.0
        assert rung["population"] == SELECTION_POPULATION
        assert "val_auc_clean_view_only" in rung
        assert "val_auc" not in rung   # the ambiguous name is never emitted


def test_the_recorded_selection_metric_is_the_robust_heldout_tpr(tmp_path):
    """The number the headline is chosen on is recomputed from scratch here.

    Kills the mutant that records `train_rung`'s `val_auc` under the selection
    metric's key -- the C-C substitution, which is invisible in `selection.json`
    because both are floats in [0, 1] and both look like plausible detector
    quality. The check is an independent recomputation: reload the checkpoint
    the run left behind, rescore the eval bank, and apply the §6.4 metric.
    """
    from aigcdet.eval.errors import heldout_robust_tpr
    from aigcdet.eval.grid import score_grid
    from aigcdet.features.bank import FeatureBank
    from aigcdet.train.train_head import load_detector

    argv = _argv(tmp_path, rungs=("a3",))
    with _quiet_control_warning():
        report = ra.main(argv)

    bank = FeatureBank.open(argv[argv.index("--eval-bank") + 1])
    model, _ = load_detector(str(tmp_path / "rungs" / "a3" / "checkpoint.pt"),
                             device="cpu")
    expected = heldout_robust_tpr(score_grid(model, bank, device="cpu"),
                                  bank.meta["split"].to_numpy())
    recorded = report["summary"]["a3"]
    assert recorded[SELECTION_METRIC] == pytest.approx(expected)
    assert recorded[SELECTION_METRIC] != \
        pytest.approx(recorded["val_auc_clean_view_only"]), \
        "fixture cannot tell the selection metric from the clean-view AUC"


def test_the_metric_is_computed_at_one_percent_fpr_by_the_orchestrator(tmp_path):
    """The operating point is pinned at the CALL SITE, not only in the default.

    `run_ablation` could pass any `target_fpr` and every other assertion in this
    file would still hold: the metric key still reads `_at_1pct`, the value is
    still in [0, 1], and `selection.json` still quotes the rule. So the recorded
    number is compared against BOTH rates, and the test first proves this
    fixture can tell them apart -- on a coarser bank they are the same number
    and this test would silently assert nothing.
    """
    from aigcdet.eval.errors import heldout_robust_tpr
    from aigcdet.eval.grid import score_grid
    from aigcdet.features.bank import FeatureBank
    from aigcdet.train.train_head import load_detector

    argv = _argv(tmp_path, rungs=("a3",))
    with _quiet_control_warning():
        report = ra.main(argv)

    bank = FeatureBank.open(argv[argv.index("--eval-bank") + 1])
    model, _ = load_detector(str(tmp_path / "rungs" / "a3" / "checkpoint.pt"),
                             device="cpu")
    scores = score_grid(model, bank, device="cpu")
    splits = bank.meta["split"].to_numpy()
    at_one = heldout_robust_tpr(scores, splits, target_fpr=0.01)
    at_five = heldout_robust_tpr(scores, splits, target_fpr=0.05)

    assert at_one != pytest.approx(at_five), \
        "fixture cannot tell 1% FPR from 5%; this test would assert nothing"
    assert report["summary"]["a3"][SELECTION_METRIC] == pytest.approx(at_one)
    assert report["summary"]["a3"]["target_fpr"] == SELECTION_TARGET_FPR
    assert report["target_fpr"] == SELECTION_TARGET_FPR


def test_the_headline_is_chosen_from_a3_to_a6_even_when_a_control_wins(tmp_path):
    """The eligibility filter is asserted with the control forced to win.

    `heldout_robust_tpr` is stubbed so `a0` scores 0.99 against `a3`'s 0.10:
    without the §6.4 filter the orchestrator would name `a0` as the headline
    model, which is the failure this test exists to prevent. The metric itself
    is tested for real in tests/eval/test_errors.py.
    """
    forced = [0.99, 0.10]   # rungs are processed in the order they are given

    def fake_metric(scores, splits, target_fpr=0.01):
        return forced.pop(0)

    original = ra.heldout_robust_tpr
    ra.heldout_robust_tpr = fake_metric
    try:
        with pytest.warns(IneligibleRungWarning, match="a0"):
            report = ra.main(_argv(tmp_path))
    finally:
        ra.heldout_robust_tpr = original

    assert report["headline"] == "a3"
    assert report["candidates"] == {"a3": 0.10}
    assert report["excluded_as_ineligible"] == {"a0": 0.99}


def test_a7_norecon_trains_on_a_bank_with_no_recon_and_is_reported_as_ineligible(tmp_path):
    """FiLM asked of the shipping system. `_train_bank` writes no recon.npy, so
    this row can only exist if the FiLM rung is genuinely independent of A4's
    branch -- and its name is not a bare `aN`, so it also proves the orchestrator
    keys checkpoints, scores and the selection record on the YAML's `name`
    rather than on that shape. Like A7 it is a hypothesis test, not a headline
    candidate: it must be scored, tabulated and then excluded under §6.4."""
    forced = [0.10, 0.99]   # a3, then a7_norecon, forced to outscore it

    def fake_metric(scores, splits, target_fpr=0.01):
        return forced.pop(0)

    original = ra.heldout_robust_tpr
    ra.heldout_robust_tpr = fake_metric
    try:
        with pytest.warns(IneligibleRungWarning, match="a7_norecon"):
            report = ra.main(_argv(tmp_path, rungs=("a3", "a7_norecon")))
    finally:
        ra.heldout_robust_tpr = original

    assert set(report["summary"]) == {"a3", "a7_norecon"}
    assert report["headline"] == "a3"
    assert report["excluded_as_ineligible"] == {"a7_norecon": 0.99}
    assert (tmp_path / "rungs" / "a7_norecon" / "checkpoint.pt").exists()
    ckpt = ra.torch.load(tmp_path / "rungs" / "a7_norecon" / "checkpoint.pt",
                         map_location="cpu", weights_only=True)
    assert ckpt["config"]["use_film"] is True and ckpt["config"]["use_recon"] is False
    table = (tmp_path / "out" / "robustness_table.md").read_text(encoding="utf-8")
    assert "a7_norecon" in table


# --- resumability ----------------------------------------------------------

def test_a_second_run_reuses_the_checkpoint_and_says_so(tmp_path, capsys):
    argv = _argv(tmp_path, rungs=("a3",))
    with _quiet_control_warning():
        ra.main(argv)
    checkpoint = tmp_path / "rungs" / "a3" / "checkpoint.pt"
    first = checkpoint.read_bytes()
    capsys.readouterr()

    with _quiet_control_warning():
        second = ra.main(argv)
    printed = capsys.readouterr().out
    assert "SKIP a3" in printed and "NOT retrained" in printed
    assert checkpoint.read_bytes() == first, "the rung was silently retrained"
    assert second["summary"]["a3"]["resumed_from_checkpoint"] is True


def test_force_retrain_overwrites_the_checkpoint(tmp_path, capsys):
    argv = _argv(tmp_path, rungs=("a3",))
    with _quiet_control_warning():
        ra.main(argv)
    capsys.readouterr()
    with _quiet_control_warning():
        report = ra.main(argv + ["--force-retrain"])
    assert "SKIP" not in capsys.readouterr().out
    assert report["summary"]["a3"]["resumed_from_checkpoint"] is False


def test_resume_is_refused_when_the_rung_config_changed(tmp_path):
    """A checkpoint reused under a changed config would put a row in the table
    describing a model that no longer matches its config file."""
    argv = _argv(tmp_path, rungs=("a3",))
    with _quiet_control_warning():
        ra.main(argv)
    config_path = argv[argv.index("--rungs") + 1]
    cfg = yaml.safe_load(open(config_path))
    cfg["lr"] = 0.5
    pathlib.Path(config_path).write_text(yaml.safe_dump(cfg))

    with pytest.raises(ValueError, match="refusing to resume"):
        ra.main(argv)


def test_resume_is_refused_when_the_result_json_is_missing(tmp_path):
    argv = _argv(tmp_path, rungs=("a3",))
    with _quiet_control_warning():
        ra.main(argv)
    (tmp_path / "rungs" / "a3" / "result.json").unlink()
    with pytest.raises(FileExistsError, match="killed between the two writes"):
        ra.main(argv)


def test_config_differences_ignores_only_the_non_model_fields():
    """Where the checkpoint ran is not what it is. Which bank, seed, loss terms
    and learning rate it was trained with are."""
    from dataclasses import asdict

    from aigcdet.train.train_head import RungConfig
    cfg = RungConfig(name="a3", bank_dir="b", out_dir="o", device="cpu")
    stored = asdict(cfg) | {"out_dir": "ELSEWHERE", "device": "cuda",
                            "manifest_path": "m.parquet"}
    assert ra.config_differences(stored, cfg) == {}
    assert set(ra.config_differences(stored | {"lr": 0.5}, cfg)) == {"lr"}
    assert set(ra.config_differences(stored | {"use_recon": True}, cfg)) == \
        {"use_recon"}
    assert set(ra.config_differences(stored | {"bank_dir": "other"}, cfg)) == \
        {"bank_dir"}


def test_a_benchmark_only_eval_bank_is_refused_before_any_rung_trains(tmp_path):
    """Split coverage is knowable from the bank in the first millisecond.

    Kills the mutant that drops the up-front `check_selection_population`: the
    run would then train a rung -- hours of GPU on the real thing -- and only
    discover the bank cannot supply the §6.4 population when it tries to score
    the result. The assertion is that NO checkpoint was written.
    """
    bank = _eval_bank(tmp_path, "benchmark_only", n_per_block=6)
    meta_path = pathlib.Path(bank) / "meta.parquet"
    meta = pd.read_parquet(meta_path)
    meta["split"] = "benchmark"
    meta.to_parquet(meta_path, index=False)

    argv = _argv(tmp_path, rungs=("a3",), **{"--eval-bank": bank})
    with pytest.raises(ValueError, match="demo set"):
        ra.main(argv)
    assert not (tmp_path / "rungs" / "a3" / "checkpoint.pt").exists()


def test_the_exclusion_is_printed_on_stdout_beside_the_headline(tmp_path, capsys):
    """The IneligibleRungWarning goes to stderr, where a multi-hour log buries
    it. The exclusion belongs where the choice it constrains is read."""
    forced = [0.99, 0.10]

    def fake_metric(scores, splits, target_fpr=0.01):
        return forced.pop(0)

    original = ra.heldout_robust_tpr
    ra.heldout_robust_tpr = fake_metric
    try:
        with pytest.warns(IneligibleRungWarning):
            ra.main(_argv(tmp_path))
    finally:
        ra.heldout_robust_tpr = original
    out = capsys.readouterr().out
    assert "headline model: a3" in out
    assert "excluded as ineligible" in out and "a0" in out


def test_every_rung_is_registered_with_the_bank_it_was_scored_on(tmp_path):
    """The invariant Task 8 must preserve when fusion opens a second eval bank.

    A `{rung: eval_bank for rung in per_rung}` comprehension built AFTER the
    loop is behaviourally identical today -- there is one bank -- and becomes a
    false statement the moment a rung is scored somewhere else. What is
    testable now is that the mapping covers exactly the scored rungs, which is
    what `robustness_table` checks against.
    """
    captured = {}
    original = ra.robustness_table

    def spy(per_rung, **kwargs):
        captured["rungs"] = set(per_rung)
        captured["banks"] = kwargs["banks"]
        return original(per_rung, **kwargs)

    ra.robustness_table = spy
    try:
        with _quiet_control_warning():
            ra.main(_argv(tmp_path))
    finally:
        ra.robustness_table = original
    assert set(captured["banks"]) == captured["rungs"] == {"a0", "a3"}
    assert all(hasattr(b, "config") for b in captured["banks"].values())


def test_the_summary_lines_survive_a_c_locale_on_a_pipe(tmp_path):
    """Kills the mutant that drops `_make_stdout_encoding_safe`.

    The exclusion line and `headline_error` both quote `§6.4`. Python encodes
    stdout with the LOCALE codec and `errors="strict"`; under LC_ALL=C on a
    pipe that is ASCII, and the `print` raises UnicodeEncodeError -- after the
    table, the heatmap and `selection.json` are already on disk, so a
    multi-hour ablation exits non-zero having done all of its work.
    """
    import os
    import subprocess
    import sys
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import sys, importlib.util\n"
        "spec = importlib.util.spec_from_file_location('ra', sys.argv[1])\n"
        "ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)\n"
        "ra._make_stdout_encoding_safe()\n"
        "print('excluded as ineligible under \u00a76.4: {\"a0\": 0.99}')\n",
        encoding="utf-8")
    env = {k: v for k, v in os.environ.items()
           if k not in ("LANG", "LC_CTYPE", "LC_ALL")}
    env.update(LC_ALL="C", PYTHONUTF8="0", PYTHONCOERCECLOCALE="0")
    done = subprocess.run(
        [sys.executable, str(probe), str(_ROOT / "scripts" / "run_ablation.py")],
        capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr
    assert "excluded as ineligible under" in done.stdout


def test_footnotes_are_appended_as_utf8_under_a_c_locale(tmp_path):
    """Kills the mutant that drops `encoding="utf-8"` from the append.

    The footnote body contains `spec §6.3`. A bare `open(path, "a")` encodes
    through the locale codec; under LC_ALL=C that is ANSI_X3.4-1968 and the
    append dies with UnicodeEncodeError -- after `to_markdown` has already
    written the table, so the run fails having produced a truncated deliverable.
    C/POSIX is the default locale in many container and CI images.
    """
    import os
    import subprocess
    import sys
    target = tmp_path / "table.md"
    target.write_text("# table\n", encoding="utf-8")
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import locale, sys\n"
        "assert locale.getpreferredencoding(False) == 'ANSI_X3.4-1968', "
        "locale.getpreferredencoding(False)\n"
        "import importlib.util\n"
        "spec = importlib.util.spec_from_file_location('ra', sys.argv[1])\n"
        "ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)\n"
        "ra.append_footnotes(sys.argv[2], ra.baseline_footnotes(['npr']))\n",
        encoding="utf-8")
    env = {k: v for k, v in os.environ.items()
           if k not in ("LANG", "LC_CTYPE", "LC_ALL")}
    env.update(LC_ALL="C", PYTHONUTF8="0", PYTHONCOERCECLOCALE="0")
    done = subprocess.run(
        [sys.executable, str(probe), str(_ROOT / "scripts" / "run_ablation.py"),
         str(target)],
        capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr
    assert "§6.3" in target.read_text(encoding="utf-8")


# --- comparability ---------------------------------------------------------

def test_an_eval_bank_without_a_manifest_fingerprint_is_refused(tmp_path):
    """Proves the rungs' banks reach `robustness_table`.

    `robustness_table(..., banks=...)` is what routes them through
    `assert_banks_comparable` and demands a manifest fingerprint; a call that
    omitted `banks` would build the table happily and the ladder would be
    compared across banks that cannot be shown to index the same manifest.
    """
    argv = _argv(tmp_path, rungs=("a3",),
                 **{"--eval-bank": _eval_bank(tmp_path, "unfingerprinted",
                                              fingerprint=None)})
    with pytest.raises(ValueError, match="records no manifest_sha256"):
        ra.main(argv)


# --- baseline row labels (ruling R38/I3) -----------------------------------

def test_baseline_footnotes_spell_out_the_npr_row_label():
    text = ra.baseline_footnotes(["npr", "a3"])
    assert "NPR-style neighbouring-pixel summary + linear probe" in text
    assert "ResNet-50" in text
    assert "`a3`" not in text


def test_baseline_footnotes_flag_a0_as_univfd_only_on_the_clip_bank():
    text = ra.baseline_footnotes(["a0", "aeroblade"], backbone="clipl")
    assert "UnivFD (rung A0 on the `clipl` bank)" in text
    assert "AEROBLADE (LPIPS round-trip distance, training-free)" in text
    assert "L1 variant" in text


@pytest.mark.parametrize("backbone", [None, "dinov3l", "siglip2l", "fake"])
def test_a0_is_not_called_univfd_on_any_other_bank(backbone):
    """R38/I3 inverted: the footnote must not MANUFACTURE a mislabelling.

    Kills the ungated `RUNG_IS_A_BASELINE` lookup, which fires on any row named
    `a0` and writes "evaluated on the `clipl` bank" into the results file --
    false on the invocation this script's own docstring documents
    (`--bank banks/dinov3l`), in the file a report writer copies numbers from.
    """
    text = ra.baseline_footnotes(["a0"], backbone=backbone)
    assert text == ""
    assert "UnivFD" not in text and "clipl" not in text


def test_the_written_table_makes_no_univfd_claim_about_a_non_clip_bank(tmp_path):
    """The gate has to hold where it matters: in the file on disk."""
    argv = _argv(tmp_path, rungs=("a0",))
    with _quiet_control_warning():
        ra.main(argv)
    written = (tmp_path / "out" / "robustness_table.md").read_text(encoding="utf-8")
    assert "| a0 |" in written              # the row is there
    assert "UnivFD" not in written          # the false claim is not
    assert "clipl" not in written


def test_no_baseline_footnote_is_emitted_when_no_baseline_row_is_present():
    assert ra.baseline_footnotes(["a3", "a4"]) == ""


def test_the_table_carries_the_baseline_footnote_when_a_baseline_row_is_present(
        tmp_path, monkeypatch):
    """The footnote must reach the file, not just the helper."""
    monkeypatch.setitem(_RUNGS, "npr", dict(_RUNGS["a3"]))
    argv = _argv(tmp_path, rungs=("npr",))
    with _quiet_control_warning():
        ra.main(argv)
    written = (tmp_path / "out" / "robustness_table.md").read_text()
    assert "NPR-style neighbouring-pixel summary + linear probe" in written


# --- rung A5: fusion -------------------------------------------------------

def _fusion_argv(tmp_path, rungs=("a3",), fingerprint="eb", backbone="fake",
                 extra=(), **overrides) -> list[str]:
    """`_argv` plus a second, independently extracted training/eval bank pair.

    The partner bank draws different features, so the head trained on it is a
    genuinely different model and the fused row is not a copy of A3's.
    """
    argv = _argv(tmp_path, rungs=rungs, **overrides)
    fuse = ["--fuse-bank", _train_bank(tmp_path, name="fuse_train_bank", seed=5),
            "--fuse-eval-bank", _eval_bank(tmp_path, "fuse_eval_bank",
                                           fingerprint=fingerprint,
                                           backbone=backbone, seed=9)]
    at = argv.index("--rungs")
    return argv[:at] + fuse + list(extra) + argv[at:]


def test_fuse_produces_an_a5_row_from_two_independently_trained_banks(tmp_path):
    argv = _fusion_argv(tmp_path)
    with _quiet_control_warning():
        report = ra.main(argv)

    assert set(report["summary"]) == {"a3", "a5"}
    assert report["candidates"]["a5"] == report["summary"]["a5"][SELECTION_METRIC]
    assert report["fusion"]["run"] is True
    assert report["fusion"]["partner_bank"] == argv[argv.index("--fuse-bank") + 1]
    assert (tmp_path / "rungs" / ra.FUSION_PARTNER_NAME / "checkpoint.pt").exists()
    written = (tmp_path / "out" / "robustness_table.md").read_text(encoding="utf-8")
    assert "| a5 |" in written


def test_the_a5_row_is_registered_with_a_bank_covering_both_parents(tmp_path):
    """Carry from Task 7, made non-equivalent here.

    `banks[rung]` is what `robustness_table` routes through
    `assert_banks_comparable`. Until A5 existed there was one bank and any
    mapping was the same mapping; now the A5 row is scored on a second eval
    bank as well, and the registration has to say so.
    """
    captured = {}
    original = ra.robustness_table

    def spy(per_rung, **kwargs):
        captured["banks"] = kwargs["banks"]
        return original(per_rung, **kwargs)

    argv = _fusion_argv(tmp_path)
    ra.robustness_table = spy
    try:
        with _quiet_control_warning():
            ra.main(argv)
    finally:
        ra.robustness_table = original

    banks = captured["banks"]
    assert set(banks) == {"a3", "a5"}
    assert banks["a5"] is not banks["a3"]
    assert banks["a5"].config["fused_from"] == [
        argv[argv.index("--eval-bank") + 1],
        argv[argv.index("--fuse-eval-bank") + 1]]


def test_the_cross_backbone_a5_row_is_admitted_and_names_both_backbones(
        tmp_path):
    """The cross-backbone A5 is the A5 the spec describes, and it must reach
    the table.

    This test asserted the OPPOSITE until R43: `_COMPARABLE_KEYS` includes
    `backbone`, so a DINOv3+SigLIP2 composite was refused from the results
    entirely -- the spec's own A5 could not appear in its own ablation ladder.
    R43 admits a bank that DECLARES its parents and their backbones and whose
    own `backbone` is exactly their composite; a borrowed parent name is still
    refused, and plain backbones are still compared as a set so a composite
    cannot bridge two single-backbone rungs.

    The kill this test used to carry -- `banks[FUSION_RUNG] = eval_bank`, which
    would make `assert_banks_comparable` compare A3's bank against itself -- is
    NOT lost: `test_the_a5_row_is_registered_with_a_bank_covering_both_parents`
    still fails on exactly that mutation. Verified before this test was
    retired, rather than assumed.
    """
    captured = {}
    original = ra.robustness_table

    def spy(per_rung, **kwargs):
        captured["banks"] = kwargs["banks"]
        return original(per_rung, **kwargs)

    ra.robustness_table = spy
    try:
        with _quiet_control_warning():
            ra.main(_fusion_argv(tmp_path, backbone="other_backbone"))
    finally:
        ra.robustness_table = original

    a5 = captured["banks"][ra.FUSION_RUNG]
    assert a5.config["fused_backbones"] == ["fake", "other_backbone"]
    assert a5.config["backbone"] == "fake+other_backbone"
    assert a5.config["n_images"] == captured["banks"]["a3"].config["n_images"]
    assert "| a5 |" in (tmp_path / "out" / "robustness_table.md").read_text()


def test_a_partner_eval_bank_from_another_manifest_is_refused_before_training(
        tmp_path):
    """Fail fast: the manifest fingerprints are knowable before any GPU."""
    argv = _fusion_argv(tmp_path, fingerprint="a_different_manifest")
    with pytest.raises(ValueError, match="different frozen manifests"):
        ra.main(argv)
    assert not (tmp_path / "rungs" / "a3" / "checkpoint.pt").exists()
    assert not (tmp_path / "rungs" / ra.FUSION_PARTNER_NAME).exists()


def test_a_partner_eval_bank_whose_splits_disagree_is_refused(tmp_path):
    """Which parent's `split` column applies to a fused row is only defined
    when the two agree; the manifest fingerprint covers the path column, so a
    re-split that kept the paths passes it and this check is what catches it."""
    argv = _fusion_argv(tmp_path)
    bank = pathlib.Path(argv[argv.index("--fuse-eval-bank") + 1])
    meta = pd.read_parquet(bank / "meta.parquet")
    meta.loc[0, "split"] = "benchmark"
    meta.to_parquet(bank / "meta.parquet", index=False)

    with pytest.raises(ValueError, match="disagree on the split"):
        ra.main(argv)
    assert not (tmp_path / "rungs" / "a3" / "checkpoint.pt").exists()


def test_fusion_without_an_a3_rung_in_the_ladder_is_refused(tmp_path):
    """A5 is "A3 + a second backbone": there is no A5 without an A3 to fuse."""
    argv = _fusion_argv(tmp_path, rungs=("a0",))
    with pytest.raises(ValueError, match="is named 'a3'"):
        ra.main(argv)
    assert not (tmp_path / "rungs" / "a0" / "checkpoint.pt").exists()


def test_one_fusion_flag_without_the_other_is_refused(tmp_path):
    argv = _argv(tmp_path, rungs=("a3",))
    at = argv.index("--rungs")
    argv = argv[:at] + ["--fuse-bank", str(tmp_path / "nowhere")] + argv[at:]
    with pytest.raises(ValueError, match="go together"):
        ra.main(argv)


def test_the_a5_metric_is_computed_on_the_split_column_the_parents_share(
        tmp_path):
    """Independent recomputation of the fused row's §6.4 number.

    Kills the A5 row that records A3's metric (a fusion that forgot to fuse),
    and the one computed against a split column that is not the parents'
    shared one. The fixture is checked to be able to tell A3 and A5 apart
    before either assertion is trusted.
    """
    from aigcdet.eval.errors import heldout_robust_tpr
    from aigcdet.eval.fusion import (
        FIT_SPLITS_FOR_SELECTION, fuse_scores, fused_splits,
    )
    from aigcdet.eval.grid import score_grid
    from aigcdet.features.bank import FeatureBank
    from aigcdet.train.train_head import load_detector

    argv = _fusion_argv(tmp_path)
    with _quiet_control_warning():
        report = ra.main(argv)

    banks = [FeatureBank.open(argv[argv.index(flag) + 1])
             for flag in ("--eval-bank", "--fuse-eval-bank")]
    names = ("a3", ra.FUSION_PARTNER_NAME)
    frames = []
    for name, bank in zip(names, banks):
        model, _ = load_detector(str(tmp_path / "rungs" / name / "checkpoint.pt"),
                                 device="cpu")
        frames.append(score_grid(model, bank, device="cpu"))

    splits = fused_splits(banks)
    # The SAME declared population the call site fits on. Recomputing with a
    # whole-frame fit would not merely be a different number -- `_eval_bank`
    # deliberately carries a 60-row `benchmark` block, so it is the very
    # contamination the declaration exists to exclude.
    expected = heldout_robust_tpr(
        fuse_scores(frames, splits=splits,
                    fit_splits=FIT_SPLITS_FOR_SELECTION),
        splits)
    a3_alone = heldout_robust_tpr(frames[0], splits)

    assert expected != pytest.approx(a3_alone), \
        "fixture cannot tell the fused row from the a3 row it was built on"
    assert report["summary"]["a5"][SELECTION_METRIC] == pytest.approx(expected)
    assert report["summary"]["a3"][SELECTION_METRIC] == pytest.approx(a3_alone)


def test_the_a5_row_declares_the_population_it_was_selected_on(tmp_path):
    """`select_headline` can only refuse a contaminated result that says where
    it came from, so the fused row must declare the same block as a trained
    one rather than arriving as a bare float."""
    with _quiet_control_warning():
        report = ra.main(_fusion_argv(tmp_path))
    row = report["summary"]["a5"]
    assert row["population"] == SELECTION_POPULATION
    assert row["target_fpr"] == SELECTION_TARGET_FPR
    assert row["splits"] == list(SELECTION_SPLITS)
    assert row["fused_from"] == ["a3", ra.FUSION_PARTNER_NAME]


def test_a5_is_a_real_candidate_for_the_headline(tmp_path):
    """§6.4's eligible range is a3-a6; before this task it could only ever
    resolve to a3 or a4. Forced metrics put a5 top and it must be chosen."""
    forced = [0.10, 0.90]          # a3 first, then the fused a5 row

    def fake_metric(scores, splits, target_fpr=0.01):
        return forced.pop(0)

    original = ra.heldout_robust_tpr
    ra.heldout_robust_tpr = fake_metric
    try:
        report = ra.main(_fusion_argv(tmp_path))
    finally:
        ra.heldout_robust_tpr = original
    assert report["headline"] == "a5"
    assert report["candidates"] == {"a3": 0.10, "a5": 0.90}


def test_the_partner_head_is_resumed_rather_than_retrained(tmp_path, capsys):
    argv = _fusion_argv(tmp_path)
    with _quiet_control_warning():
        ra.main(argv)
    checkpoint = tmp_path / "rungs" / ra.FUSION_PARTNER_NAME / "checkpoint.pt"
    first = checkpoint.read_bytes()
    capsys.readouterr()
    with _quiet_control_warning():
        report = ra.main(argv)
    assert f"SKIP {ra.FUSION_PARTNER_NAME}" in capsys.readouterr().out
    assert checkpoint.read_bytes() == first
    assert report["summary"]["a5"]["resumed_from_checkpoint"] is True


# --- rung A6: test-time augmentation ---------------------------------------

def test_selection_json_records_a5_and_a6_as_not_run_rather_than_omitting_them(
        tmp_path):
    """Plan 3's completion criterion: a rung skipped for time is recorded as
    skipped, so an absent A5/A6 row is never read as an A5/A6 that lost."""
    with _quiet_control_warning():
        report = ra.main(_argv(tmp_path, rungs=("a3",)))
    assert "a5" not in report["summary"] and "a6" not in report["summary"]
    assert report["fusion"]["run"] is False
    assert "--fuse-bank" in report["fusion"]["reason"]
    assert report["tta"]["requested"] is False
    assert report["tta"]["scored_here"] is False


def test_tta_records_its_cost_multiplier_and_the_tier_it_applies_to(
        tmp_path, capsys):
    """No silent caps: the multiplier and the tier are in the record and on
    stdout, and the multiplier is the real view count, not a literal."""
    from aigcdet.eval.tta import TTA_VIEWS

    argv = _argv(tmp_path, rungs=("a3",))
    at = argv.index("--rungs")
    with _quiet_control_warning():
        report = ra.main(argv[:at] + ["--tta"] + argv[at:])

    assert report["tta"]["requested"] is True
    assert report["tta"]["scored_here"] is False
    assert report["tta"]["cost_multiplier"] == len(TTA_VIEWS) == 8
    assert report["tta"]["views"] == list(TTA_VIEWS)
    assert report["tta"]["tier"] == "ablation"
    printed = capsys.readouterr().out
    assert f"{len(TTA_VIEWS)}x" in printed and "ablation" in printed
    record = json.loads((tmp_path / "out" / "selection.json").read_text())
    assert record["tta"] == report["tta"]


# --- the table and selection.json cannot disagree -------------------------

def _table_row(text, rung):
    """`{column: cell}` for one rung of the written markdown table."""
    header = [c.strip() for c in
              [line for line in text.splitlines() if line.startswith("| rung |")][0]
              .split("|")[1:-1]]
    row = [c.strip() for c in
           [line for line in text.splitlines() if line.startswith(f"| {rung} |")][0]
           .split("|")[1:-1]]
    return dict(zip(header[1:], row[1:]))


def test_the_written_table_carries_the_same_selection_number_as_selection_json(
        tmp_path):
    """Kills the mutant that builds the table without `selection=`.

    Without it the shipped `robustness_table.md` holds only §6.1 columns
    computed over EVERY scored row -- benchmark and seen-generator fakes
    included -- while `selection.json` holds the §6.4 number over a different
    population. A reader picks the headline off the table's best column, and
    the two artefacts name different rungs.
    """
    with _quiet_control_warning():
        report = ra.main(_argv(tmp_path))

    text = (tmp_path / "out" / "robustness_table.md").read_text(encoding="utf-8")
    for rung in ("a0", "a3"):
        cell = float(_table_row(text, rung)[SELECTION_METRIC])
        assert cell == pytest.approx(report["summary"][rung][SELECTION_METRIC],
                                     abs=5e-5)
    # And the rule, the disclaimer and the chosen rung are all in the file.
    assert f"Highest among the eligible rungs in this table: `{report['headline']}`" \
        in text
    assert "REPORTING" in text
    assert report["headline"] == "a3"


def test_the_table_states_what_it_actually_covers_rather_than_the_plan(tmp_path):
    """The tier sentence used to assert the plan's row budget as this table's.

    Kills the mutant that writes `TIER_DESCRIPTIONS[tier]` alone: on this
    fixture that sentence claims a 5k+5k evaluation above a 240-image one.
    """
    with _quiet_control_warning():
        ra.main(_argv(tmp_path))
    text = (tmp_path / "out" / "robustness_table.md").read_text(encoding="utf-8")
    assert "THIS TABLE covers 2 rung(s) x 20 condition(s) over 240 image(s)" in text
    assert "Planned budget" in text


def test_ece_is_refused_by_the_orchestrator_rather_than_tabulated_blank(
        tmp_path, capsys):
    """`--metric ece` was reachable and produced an all-NaN table and a blank
    heatmap: this script has no producer of calibrated probabilities.

    Refused at PARSE time, so the ladder is not trained first. Kills the mutant
    that leaves `ece` in the accepted choices and lets `robustness_table` raise
    after every rung has trained.
    """
    with pytest.raises(SystemExit):
        ra.main(_argv(tmp_path, **{"--metric": "ece"}))
    assert "ece" in capsys.readouterr().err
    assert not (tmp_path / "rungs").exists(), "a rejected metric trained a rung"
    assert not (tmp_path / "out" / "robustness_table.md").exists()


def test_acc_fixed_uses_a_threshold_from_internal_validation(tmp_path):
    """`--metric acc_fixed` must not fit its frozen threshold on the rows it
    scores -- at the final-report tier those are the external benchmark.

    Kills the mutant that drops `clean_threshold=` from the orchestrator's
    `robustness_table` call (the table then refuses) and the one that fits it
    on every clean row rather than the val_internal ones.
    """
    from aigcdet.eval.grid import score_grid
    from aigcdet.eval.report import clean_validation_threshold
    from aigcdet.features.bank import FeatureBank
    from aigcdet.train.train_head import load_detector

    argv = _argv(tmp_path, rungs=("a3",), **{"--metric": "acc_fixed"})
    with _quiet_control_warning():
        report = ra.main(argv)
    text = (tmp_path / "out" / "robustness_table.md").read_text(encoding="utf-8")
    assert report["table_metric"] == "acc_fixed"

    bank = FeatureBank.open(argv[argv.index("--eval-bank") + 1])
    model, _ = load_detector(str(tmp_path / "rungs" / "a3" / "checkpoint.pt"),
                             device="cpu")
    scores = score_grid(model, bank, device="cpu")
    splits = bank.meta["split"].to_numpy()
    expected = clean_validation_threshold(scores, splits)
    assert float(_table_row(text, "a3")["clean_threshold"]) == pytest.approx(
        expected, abs=5e-5)
    # What this fixture CAN tell apart: fitting on the clean view only versus
    # on every view of the validation rows. It cannot tell "val_internal clean"
    # from "all clean" -- its benchmark block is cleanly separated, so both fits
    # land on the same score -- and that discrimination is pinned instead by
    # tests/eval/test_report.py's purpose-built fixture, whose benchmark
    # authentics sit on the generated cluster.
    from aigcdet.eval.report import _best_threshold
    val_all_views = scores[(splits[scores["image_idx"].to_numpy()] == "val_internal")]
    assert expected != pytest.approx(
        _best_threshold(val_all_views["label"].to_numpy(),
                        val_all_views["score"].to_numpy()))


# --- the organisers' announced score ---------------------------------------

def test_the_announced_score_is_recorded_beside_the_headline_not_instead_of_it(
        tmp_path):
    """0.50 x AUC_clean + 0.50 x AUC_robust is what the judges compute, and
    §6.4's rule is what this project selects on. Both go in the record. Letting
    the announced score quietly become the selection rule would discard the
    deployment argument the whole calibration branch is built on; leaving it
    out entirely means discovering the ranking disagreement on submission
    night."""
    with _quiet_control_warning():
        report = ra.main(_argv(tmp_path))

    cs = report["challenge_score"]
    assert cs["weights"] == {"clean": 0.5, "robust": 0.5}
    assert set(cs["per_rung"]) == {"a0", "a3"}
    for row in cs["per_rung"].values():
        assert set(row) == {"auc_clean", "auc_robust", "challenge_score"}
        assert row["challenge_score"] == pytest.approx(
            0.5 * row["auc_clean"] + 0.5 * row["auc_robust"])
    assert cs["best"] in cs["per_rung"]

    # ... and the selection rule is untouched by it.
    assert report["metric"] == SELECTION_METRIC
    assert report["population"] == SELECTION_POPULATION


def test_the_recorded_robust_conditions_are_the_briefs_not_our_scenarios(
        tmp_path):
    """The composed conditions (`social_repost`, `messaging_app`, ...) are
    this project's invention and the judges do not score them. Recording the
    list is what lets a reader verify the number was averaged over the right
    fourteen rather than take it on trust."""
    from aigcdet.augment.scenarios import CORE_CONDITIONS

    with _quiet_control_warning():
        report = ra.main(_argv(tmp_path))

    got = report["challenge_score"]["robust_conditions"]
    assert set(got) == set(CORE_CONDITIONS) - {"clean"}
    ours = [c for c in EVAL_GRID if c not in CORE_CONDITIONS]
    assert ours and not (set(got) & set(ours))


def test_a_non_auc_table_records_why_the_score_is_absent(tmp_path):
    """A missing key reads as an oversight; a recorded reason reads as a
    measurement that was not taken, and says how to take it."""
    with _quiet_control_warning():
        report = ra.main(_argv(tmp_path, **{"--metric": "tpr_at_1pct"}))

    cs = report["challenge_score"]
    assert cs["computed"] is False
    assert "auc" in cs["reason"].lower()
    assert "per_rung" not in cs


def test_the_score_is_printed_beside_the_headline(tmp_path, capsys):
    """The selection log is the only place most readers look. A number that
    exists solely in selection.json is a number nobody reads on the day."""
    with _quiet_control_warning():
        report = ra.main(_argv(tmp_path))
    out = capsys.readouterr().out

    assert "challenge score" in out
    best = report["challenge_score"]["best"]
    assert best in out
    if best != report["headline"]:
        assert "NOTE:" in out and report["headline"] in out
