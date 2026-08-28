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
    SELECTION_METRIC, SELECTION_POPULATION, SELECTION_TARGET_FPR,
    IneligibleRungWarning,
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


def _train_bank(tmp_path, n=48) -> str:
    """A learnable training bank: fakes sit at +1.5, reals at -1.5."""
    out = str(tmp_path / "train_bank")
    w = BankWriter(out, n, N_VIEWS, DIM, "fake", 0, manifest_sha256="tb")
    rng = np.random.default_rng(0)
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


def _eval_bank(tmp_path, name="eval_bank", fingerprint="eb", n_per_block=60) -> str:
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
    w = BankWriter(out, n, len(conditions), DIM, "fake", 0,
                   manifest_sha256=fingerprint,
                   extra_config={"conditions": conditions})
    rng = np.random.default_rng(1)
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
