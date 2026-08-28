"""The error-sheet CLI, on a tiny synthetic bank with real image files.

No GPU, no weights, no downloads; every file read or written lives under
`tmp_path`.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from aigcdet.features.bank import N_FAMILIES, BankWriter

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = _ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mes = _load_script("make_error_sheet")

CONDITIONS = ["clean", "jpeg_q50"]
N = 12
DIM = 4


def _bank_with_images(tmp_path) -> str:
    """Half authentic (source `coco`), half generated (source `dalle`)."""
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    out = str(tmp_path / "eval_bank")
    w = BankWriter(out, N, len(CONDITIONS), DIM, "fake", 0, manifest_sha256="eb",
                   extra_config={"conditions": CONDITIONS})
    for i in range(N):
        label = int(i >= N // 2)
        path = images / f"im{i}.png"
        Image.new("RGB", (16, 16), (8 * i, 40, 90)).save(path)
        presence = np.zeros((len(CONDITIONS), N_FAMILIES), np.float32)
        presence[1:, 0] = 1.0
        w.write_image(i, {"path": str(path), "label": label,
                          "generator": "g" if label else "",
                          "source": "dalle" if label else "coco",
                          "split": "val_internal"},
                      feats=np.zeros((len(CONDITIONS), DIM), np.float32),
                      presence=presence,
                      severity=np.zeros((len(CONDITIONS), N_FAMILIES), np.float32),
                      proxies=np.zeros((len(CONDITIONS), 3), np.float32),
                      recipes=["[]"] * len(CONDITIONS))
    w.close()
    return out


def _scores_parquet(tmp_path) -> str:
    """Deliberately imperfect scores: two authentic images score high (false
    positives) and two generated ones score low (false negatives)."""
    rows = []
    for cond in CONDITIONS:
        score = np.linspace(-3.0, 3.0, N)
        score[[0, 1]] = [2.5, 2.4]      # authentic, confidently wrong
        score[[N - 1, N - 2]] = [-2.5, -2.4]   # generated, confidently wrong
        rows.append(pd.DataFrame({
            "condition": cond, "image_idx": np.arange(N),
            "label": (np.arange(N) >= N // 2).astype(int),
            "generator": ["" if i < N // 2 else "g" for i in range(N)],
            "source": ["coco" if i < N // 2 else "dalle" for i in range(N)],
            "score": score}))
    path = str(tmp_path / "scores.parquet")
    pd.concat(rows, ignore_index=True).to_parquet(path, index=False)
    return path


def _argv(tmp_path, **overrides) -> list[str]:
    args = {"--scores": _scores_parquet(tmp_path),
            "--eval-bank": _bank_with_images(tmp_path),
            "--out": str(tmp_path / "errors"),
            "--k": "4"}
    args.update(overrides)
    return [x for kv in args.items() for x in kv]


def test_it_writes_both_sheets_and_the_per_source_table(tmp_path):
    mes.main(_argv(tmp_path))
    out = tmp_path / "errors"
    assert (out / "clean_fp.png").stat().st_size > 0
    assert (out / "clean_fn.png").stat().st_size > 0
    assert (out / "fp_by_source.md").exists()
    with Image.open(out / "clean_fp.png") as im:
        assert im.size[0] > 0


def test_the_per_source_table_names_its_threshold_and_disclaims_deployment(tmp_path):
    """The number in this file is a diagnostic, and the file has to say so:
    quoted as a deployed false-positive rate it would be wrong, since the
    deployment operating point is fitted by `calibrate.policy` on internal
    validation (ruling R33's discipline, applied to the figure this script
    does emit)."""
    mes.main(_argv(tmp_path))
    text = (tmp_path / "errors" / "fp_by_source.md").read_text()
    assert "Diagnostic threshold:" in text
    assert "NOT the deployment operating point" in text
    assert "calibrate.policy" in text
    assert "**Rows by split:**" in text and "val_internal" in text


def test_the_per_source_table_reports_every_source_and_leaves_empty_rates_blank(
        tmp_path):
    mes.main(_argv(tmp_path))
    text = (tmp_path / "errors" / "fp_by_source.md").read_text()
    assert "| coco |" in text
    assert "| dalle |" in text
    # `dalle` contributed no authentic image, so its rate is blank, not 0.0000.
    dalle = [line for line in text.splitlines() if line.startswith("| dalle |")][0]
    assert dalle.rstrip().endswith("|  |")


def test_an_explicit_threshold_is_used_and_recorded(tmp_path):
    mes.main(_argv(tmp_path, **{"--threshold": "1.25"}))
    text = (tmp_path / "errors" / "fp_by_source.md").read_text()
    assert "1.250000" in text
    assert "supplied on the command line" in text


def test_only_the_chosen_conditions_rows_reach_the_sheets_and_the_table(tmp_path):
    """Kills the mutant that drops the `--condition` filter in `main`.

    The previous version of this test filtered the frame ITSELF and called
    `top_errors` directly, so it could not see the filter in `main` at all and
    the mutant survived it. What the mutant actually does is put both
    conditions' rows through: every image then appears twice on the contact
    sheet, `n_images` in `fp_by_source.md` doubles, and the file still says
    `**Condition:** clean`. The row counts are asserted here, on `main`'s own
    output.
    """
    mes.main(_argv(tmp_path))
    text = (tmp_path / "errors" / "fp_by_source.md").read_text(encoding="utf-8")
    assert "**Condition:** `clean`" in text
    rows = {line.split("|")[1].strip(): line.split("|")[2].strip()
            for line in text.splitlines() if line.startswith("| coco |")
            or line.startswith("| dalle |")}
    # N // 2 per source for ONE condition; both conditions would give N.
    assert rows == {"coco": str(N // 2), "dalle": str(N // 2)}


def test_top_errors_over_one_condition_picks_the_planted_mistakes(tmp_path):
    """The fixture's planted errors, so the sheet's contents are pinned too."""
    from aigcdet.eval.errors import top_errors
    scores = pd.read_parquet(_scores_parquet(tmp_path))
    clean = scores[scores["condition"] == "clean"]
    assert top_errors(clean, k=4, kind="fp")["image_idx"].tolist() == [0, 1, 5, 4]
    assert top_errors(clean, k=4, kind="fn")["image_idx"].tolist() == [11, 10, 6, 7]


def test_the_target_fpr_flag_moves_the_threshold_and_is_recorded(tmp_path):
    """`--target-fpr` was unpinned: nothing asserted that it reached
    `threshold_at_fpr`, so a mutant ignoring it and hardcoding 0.01 survived."""
    mes.main(_argv(tmp_path, **{"--target-fpr": "0.5"}))
    loose = (tmp_path / "errors" / "fp_by_source.md").read_text(encoding="utf-8")
    assert "50.0%" in loose

    mes.main(_argv(tmp_path, **{"--target-fpr": "0.01"}))
    tight = (tmp_path / "errors" / "fp_by_source.md").read_text(encoding="utf-8")
    assert "1.0%" in tight

    def threshold_of(text):
        line = [x for x in text.splitlines()
                if x.startswith("**Diagnostic threshold:**")][0]
        return float(line.split("**Diagnostic threshold:**")[1].split("--")[0])

    # A looser FPR budget buys a lower threshold, hence more false positives.
    assert threshold_of(loose) < threshold_of(tight)


def test_the_table_says_its_threshold_was_fitted_on_the_rows_it_reports(tmp_path):
    """The aggregate rate measures the threshold, not the detector; only the
    relative concentration across sources carries information, and the file
    must say so rather than leave a reader to infer it."""
    mes.main(_argv(tmp_path))
    text = (tmp_path / "errors" / "fp_by_source.md").read_text(encoding="utf-8")
    assert "fitted on the very rows tabulated below" in text
    assert "measures the threshold rather than the detector" in text
    assert "RELATIVE concentration" in text


def test_the_realised_aggregate_rate_is_printed_rather_than_asserted(tmp_path):
    """The file used to claim the aggregate `fp_rate` was `--target-fpr` "by
    construction". It is not, and cannot be: `threshold_at_fpr` returns the
    lowest threshold whose FPR does not EXCEED the target, so the realised rate
    is bounded above by it and quantised to 1/n_authentic. On this fixture's 6
    authentic rows a 1% target can only realise 0%.

    Kills the mutant that restores the "by construction" sentence, and the one
    that prints the target as if it were the realised rate.
    """
    mes.main(_argv(tmp_path))
    text = (tmp_path / "errors" / "fp_by_source.md").read_text(encoding="utf-8")
    assert "Realised aggregate FP rate:** 0.00% (0 of 6 authentic rows)" in text
    assert "against a 1.0% target" in text
    assert "16.67%" in text          # the finest rate 6 authentic rows express
    assert "by construction" not in text or "never equal to the target" in text


def test_the_realised_rate_is_computed_not_copied_from_the_target(tmp_path):
    """A looser target really does produce false positives, and the file
    reports the number it GOT rather than the number it asked for.

    40% is chosen because 6 authentic rows cannot express it: the realised rate
    lands on 33.33%, two ticks of the 1/6 grid below the target. At 50% the two
    would coincide and the mutant that prints the target survives.
    """
    mes.main(_argv(tmp_path, **{"--target-fpr": "0.4"}))
    text = (tmp_path / "errors" / "fp_by_source.md").read_text(encoding="utf-8")
    assert "Realised aggregate FP rate:** 33.33% (2 of 6 authentic rows)" in text
    assert "against a 40.0% target" in text


def test_realised_fp_rate_aggregates_over_every_source():
    """Kills a mutant that averages the per-source rates instead of pooling the
    counts: with an empty-denominator source in the frame the mean is NaN, and
    with unequal sources it is not the aggregate rate at all."""
    by_source = pd.DataFrame({"source": ["coco", "dalle", "laion"],
                              "n_images": [10, 10, 90],
                              "n_authentic": [10, 0, 90],
                              "n_fp": [1, 0, 9],
                              "fp_rate": [0.1, float("nan"), 0.1]})
    assert mes.realised_fp_rate(by_source) == (10, 100, 0.1)


def test_the_table_is_written_as_utf8_under_a_c_locale(tmp_path):
    """Kills the mutant that drops `encoding="utf-8"`.

    The body contains `spec §6.6`, and a source name may be non-ASCII too. A
    bare `open(path, "w")` encodes through the locale codec; under LC_ALL=C
    that is ANSI_X3.4-1968 and the write dies -- after both contact sheets have
    already been rendered.
    """
    import os
    import subprocess
    import sys
    out = tmp_path / "c_locale.md"
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import locale, sys, importlib.util, pandas as pd\n"
        "assert locale.getpreferredencoding(False) == 'ANSI_X3.4-1968', "
        "locale.getpreferredencoding(False)\n"
        "spec = importlib.util.spec_from_file_location('mes', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "df = pd.DataFrame({'source': ['caf\u00e9'], 'n_images': [1],\n"
        "                   'n_authentic': [1], 'n_fp': [0], 'fp_rate': [0.0]})\n"
        "m.write_fp_by_source(sys.argv[2], df, 'clean', 0.5, 'a probe',\n"
        "                     {'val_internal': 1}, 0.01)\n",
        encoding="utf-8")
    env = {k: v for k, v in os.environ.items()
           if k not in ("LANG", "LC_CTYPE", "LC_ALL")}
    env.update(LC_ALL="C", PYTHONUTF8="0", PYTHONCOERCECLOCALE="0")
    done = subprocess.run(
        [sys.executable, str(probe),
         str(_ROOT / "scripts" / "make_error_sheet.py"), str(out)],
        capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr
    body = out.read_text(encoding="utf-8")
    assert "§6.6" in body and "café" in body


def test_an_unknown_condition_is_refused_before_anything_is_written(tmp_path):
    out = tmp_path / "errors"
    with pytest.raises(ValueError, match="is not in the scores"):
        mes.main(_argv(tmp_path, **{"--condition": "jpeg_q99"}))
    assert list(out.iterdir()) == []


def test_scores_from_another_bank_are_refused(tmp_path):
    """An image_idx with no row in the bank means the two artefacts do not
    belong together; rendering the rows that DID match would produce a sheet
    silently missing its worst errors."""
    scores = pd.read_parquet(_scores_parquet(tmp_path))
    scores.loc[scores.index[0], "image_idx"] = 999
    path = str(tmp_path / "wrong.parquet")
    scores.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="do not belong together"):
        mes.main(_argv(tmp_path, **{"--scores": path}))


def test_markdown_table_renders_nan_as_blank_without_tabulate():
    df = pd.DataFrame({"source": ["a", "b"], "n_fp": [1, 0],
                       "fp_rate": [0.5, float("nan")]})
    text = mes.markdown_table(df)
    assert text.splitlines()[0] == "| source | n_fp | fp_rate |"
    assert text.splitlines()[2] == "| a | 1 | 0.5000 |"
    assert text.splitlines()[3] == "| b | 0 |  |"
