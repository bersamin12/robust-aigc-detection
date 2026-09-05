import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import metrics_table as mt  # noqa: E402


def _write(d, name, epoch, pool, hard, val=None):
    m = {"epoch": epoch, "weights": "final", "token_pool": pool, "dim_feat": 1024,
         "splits": {"ntire_val_hard": {"auc": hard[2], "robust_auc": hard[0],
                                       "clean_auc": hard[1]}}}
    if val:
        m["splits"]["ntire_val"] = {"auc": val[2], "robust_auc": val[0], "clean_auc": val[1]}
    with open(d / f"{name}_metrics.json", "w") as f:
        json.dump(m, f)


def test_rows_sort_by_epoch_within_a_run_and_carry_the_objective(tmp_path):
    _write(tmp_path, "b_ep10", 10, "mean", (0.8, 0.9, 0.85))
    _write(tmp_path, "b_ep2", 2, "mean", (0.7, 0.8, 0.75), val=(0.6, 0.7, 0.65))
    _write(tmp_path, "a_ep1", 1, "attn", (0.5, 0.6, 0.55))
    rows = mt.load(str(tmp_path))
    assert [r["name"] for r in rows] == ["a_ep1", "b_ep2", "b_ep10"]
    text = mt.render(rows)
    lines = [ln for ln in text.splitlines() if ln.startswith("| b_ep2")]
    assert len(lines) == 1
    cells = [c.strip() for c in lines[0].strip("|").split("|")]
    # objective = mean over the ntire_val* splits of mean(robust, clean)
    assert cells[-1] == f"{((0.7 + 0.8) / 2 + (0.6 + 0.7) / 2) / 2:.4f}"
    assert cells[3] == "mean" and cells[1] == "2"
    assert "attn" in text and "Leaderboard" in text


def test_missing_split_leaves_blank_cells_and_no_objective(tmp_path):
    _write(tmp_path, "x", 1, "cls", (None, 0.9, 0.85))
    text = mt.render(mt.load(str(tmp_path)))
    row = [ln for ln in text.splitlines() if ln.startswith("| x")][0]
    assert "n/a" in row and row.rstrip().endswith("|  |")


def test_empty_dir_renders_a_placeholder(tmp_path):
    assert "no metrics" in mt.render(mt.load(str(tmp_path)))
