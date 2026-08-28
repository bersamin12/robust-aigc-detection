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
    images.mkdir()
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


def test_the_sheet_shows_the_most_confident_mistakes_of_the_chosen_condition(
        tmp_path):
    """The rendered rows come from `top_errors` over one condition only.

    Kills a mutant that forgets the condition filter: with two identical
    conditions in the frame, the unfiltered version renders each image twice
    and the four tiles cover only two distinct images.
    """
    from aigcdet.eval.errors import top_errors
    scores = pd.read_parquet(_scores_parquet(tmp_path))
    clean = scores[scores["condition"] == "clean"]
    assert top_errors(clean, k=4, kind="fp")["image_idx"].tolist() == [0, 1, 5, 4]
    assert top_errors(clean, k=4, kind="fn")["image_idx"].tolist() == [11, 10, 6, 7]


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
