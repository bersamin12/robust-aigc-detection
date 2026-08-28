"""Every committed `.ipynb` opens cleanly, and says what it has to say.

Cheap, and it catches the failure that is most expensive to discover socially:
a teammate on a free Kaggle account, at the start of their one GPU session,
being handed a notebook that will not open. The structural half is nbformat's
schema; the rest are project rules a notebook can violate silently (a pasted
token, a committed output, a `pip install torch`).
"""
from __future__ import annotations

import ast
import glob
import json
import os

import pytest

# tests/notebooks/<this file> -> tests/notebooks -> tests -> repo root.
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NOTEBOOK_PATHS = sorted(glob.glob(os.path.join(REPO_ROOT, "notebooks", "*.ipynb")))
#: Notebooks this project commits to shipping. Named rather than globbed, so
#: DELETING one is a test failure too -- a teammate's bookmarked link going
#: dead is exactly as bad as a corrupt file.
EXPECTED_NOTEBOOKS = ("kaggle_stage_a.ipynb", "kaggle_merge_train.ipynb")


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _source(cell) -> str:
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def _code_cells(nb):
    return [c for c in nb["cells"] if c.get("cell_type") == "code"]


def _all_source(nb) -> str:
    return "\n".join(_source(c) for c in nb["cells"])


def test_the_expected_notebooks_exist():
    assert {os.path.basename(p) for p in NOTEBOOK_PATHS} == set(EXPECTED_NOTEBOOKS)


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=os.path.basename)
def test_notebook_is_valid_json(path):
    _load(path)


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=os.path.basename)
def test_notebook_has_the_nbformat_4_schema(path):
    nb = _load(path)
    assert nb["nbformat"] == 4
    assert isinstance(nb.get("nbformat_minor"), int)
    assert isinstance(nb["cells"], list) and nb["cells"]
    assert nb["metadata"]["kernelspec"]["language"] == "python"


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=os.path.basename)
def test_every_cell_has_the_keys_a_reader_requires(path):
    for i, cell in enumerate(_load(path)["cells"]):
        assert cell["cell_type"] in ("code", "markdown"), (path, i)
        assert "source" in cell and "metadata" in cell, (path, i)
        if cell["cell_type"] == "code":
            # jupyter refuses to render a code cell missing either of these.
            assert "outputs" in cell and isinstance(cell["outputs"], list), (path, i)
            assert "execution_count" in cell, (path, i)


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=os.path.basename)
def test_notebook_opens_under_nbformat_if_it_is_installed(path):
    """The authoritative schema check, when the library is available. Skipped
    rather than vendored: the structural assertions above are the floor that
    holds in this project's dependency set."""
    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.read(path, as_version=4))


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=os.path.basename)
def test_every_code_cell_is_parseable_python(path):
    """A cell that does not parse is a cell that fails the moment it is run,
    which on Kaggle is after the clone, the install and the model download."""
    for i, cell in enumerate(_code_cells(_load(path))):
        src = _source(cell)
        if any(line.lstrip().startswith(("!", "%")) for line in src.splitlines()):
            continue          # IPython magics are not Python; none are used today
        ast.parse(src)


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=os.path.basename)
def test_no_cell_uses_ipython_magics_or_shell_escapes(path):
    """`!pip install ...` is a shell string: a path with a space in it silently
    becomes two arguments. Every command in these notebooks goes through an
    argv list instead."""
    for cell in _code_cells(_load(path)):
        for line in _source(cell).splitlines():
            assert not line.lstrip().startswith(("!", "%")), line


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=os.path.basename)
def test_no_outputs_are_committed(path):
    """A committed output is a diff nobody can read and, worse, a place a
    printed secret would come to rest in a public repo."""
    for cell in _code_cells(_load(path)):
        assert cell["outputs"] == [], path
        assert cell["execution_count"] is None, path


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=os.path.basename)
def test_no_credential_is_pasted_into_a_cell(path):
    """This repo is public and a notebook is committed with its cell source."""
    text = _all_source(_load(path))
    for marker in ("hf_", "sk-", "ghp_", "AKIA"):
        for line in text.splitlines():
            if marker in line:
                # A mention is fine; an assignment of a literal is not.
                assert f'= "{marker}' not in line and f"= '{marker}" not in line, line


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=os.path.basename)
def test_no_notebook_installs_or_upgrades_torch(path):
    """The single most expensive mistake available on Kaggle: replacing the
    session's driver-matched torch.

    CODE cells only. The markdown is required to say `pip install torch` --
    telling a teammate not to do it is the point of the warning."""
    text = "\n".join(_source(c) for c in _code_cells(_load(path)))
    for bad in ("pip install torch", "pip install -U torch",
                'install", "torch', "'install', 'torch'"):
        assert bad not in text, (path, bad)


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=os.path.basename)
def test_every_cell_has_a_unique_id(path):
    """nbformat 4.5 requires one, and warns today where it will error later."""
    ids = [c.get("id") for c in _load(path)["cells"]]
    assert all(isinstance(i, str) and i for i in ids), path
    assert len(set(ids)) == len(ids), path


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=os.path.basename)
def test_no_notebook_enables_the_gpu_test_escape_hatch(path):
    """`AIGCDET_ALLOW_GPU_TESTS` makes the test suite download model weights.
    A notebook is not where that decision belongs."""
    assert "AIGCDET_ALLOW_GPU_TESTS" not in _all_source(_load(path))


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=os.path.basename)
def test_no_notebook_offers_to_rebuild_the_frozen_manifest(path):
    """`build_dataset` re-splits the data. Re-splitting after banks exist
    silently misaligns labels against cached features. It may be NAMED (both
    notebooks warn about it) but never invoked."""
    for cell in _code_cells(_load(path)):
        assert "build_dataset.py" not in _source(cell), path


# --- the Stage A notebook's specific obligations ----------------------------

@pytest.fixture
def stage_a():
    return _load(os.path.join(REPO_ROOT, "notebooks", "kaggle_stage_a.ipynb"))


def test_stage_a_installs_the_project_with_no_deps(stage_a):
    assert "--no-deps" in _all_source(stage_a)


def test_stage_a_reads_the_token_from_kaggle_secrets(stage_a):
    text = _all_source(stage_a)
    assert "UserSecretsClient" in text and "kb.hf_token" in text


def test_stage_a_verifies_before_it_extracts(stage_a):
    """Order is the safety property: the gate must be produced before the cell
    that spends the GPU consumes it."""
    sources = [_source(c) for c in stage_a["cells"]]
    verify_at = next(i for i, s in enumerate(sources)
                     if "open_verified_manifest" in s)
    extract_at = next(i for i, s in enumerate(sources)
                      if "run_shard_argv" in s)
    assert verify_at < extract_at


def test_stage_a_passes_the_gate_into_every_expensive_call(stage_a):
    """`GATE` is a required positional argument, so a teammate who skips the
    verification cell gets a NameError in seconds rather than a bad bank in
    ten hours."""
    for cell in _code_cells(stage_a):
        src = _source(cell)
        for call in ("kb.run_shard_argv(", "kb.shard_frame(", "kb.shard_plan("):
            if call in src:
                assert f"{call}\n" in src or f"{call}GATE" in src or \
                    src.split(call, 1)[1].lstrip().startswith("GATE"), src


def test_stage_a_extracts_both_splits_stage_b_needs(stage_a):
    assert '"train,val_internal"' in _all_source(stage_a)


def test_stage_a_writes_the_bank_where_kaggle_persists_it(stage_a):
    """A bank in /kaggle/temp is discarded at the end of the session, which is
    precisely the event --resume exists to survive."""
    out = next(s for s in (_source(c) for c in stage_a["cells"])
               if "OUT_DIR =" in s)
    assert "/kaggle/working" in out and "/kaggle/temp" not in out.split("OUT_DIR")[1]


def test_stage_a_resumes_rather_than_restarting(stage_a):
    assert "resume=True" in _all_source(stage_a)


def test_stage_a_has_a_smoke_path_before_the_real_run(stage_a):
    text = _all_source(stage_a)
    assert "SMOKE" in text and "marginal_rate" in text


def test_stage_a_documents_what_a_session_timeout_costs(stage_a):
    assert "CHECKPOINT_EVERY" in _all_source(stage_a)


def test_stage_a_carries_the_2am_playbook(stage_a):
    """Markdown a stressed teammate can act on: which errors are fatal, which
    are retryable, and what never to do."""
    text = _all_source(stage_a).lower()
    for needle in ("retryable", "fatal", "out of memory", "gated repo",
                   "cannot resume", "factory reset", "frozen"):
        assert needle in text, needle
    assert "kb.explain" in _all_source(stage_a)


def test_stage_a_warns_against_resetting_the_manifest_index(stage_a):
    """The mutation that silently re-keys every view's RNG."""
    assert "reset_index" in _all_source(stage_a)


# --- the merge notebook's specific obligations ------------------------------

@pytest.fixture
def merge_nb():
    return _load(os.path.join(REPO_ROOT, "notebooks", "kaggle_merge_train.ipynb"))


def test_merge_notebook_orders_shards_numerically(merge_nb):
    """String order gives shard0, shard1, shard10, shard2 -- and merge_banks
    concatenates in the order it is given."""
    assert "sorted_shard_dirs" in _all_source(merge_nb)


def test_merge_notebook_verifies_the_merged_bank(merge_nb):
    assert "verify_merged_bank" in _all_source(merge_nb)


def test_merge_notebook_refuses_a_partial_shard(merge_nb):
    text = _all_source(merge_nb)
    assert "read_resume_state" in text and "INCOMPLETE" in text


def test_merge_notebook_does_not_ask_for_a_gpu(merge_nb):
    """Stage B trains ~1M-parameter heads on cached vectors; asking for a GPU
    only shortens the weekly budget the shards need."""
    assert merge_nb["metadata"].get("accelerator") == "None"
    assert '"cpu"' in _all_source(merge_nb)


def test_stage_a_does_ask_for_a_gpu(stage_a):
    assert stage_a["metadata"].get("accelerator") == "GPU"


def test_merge_notebook_explains_the_train_rung_manifest_trap(merge_nb):
    """`train_rung --manifest` reads the WHOLE manifest while a training bank
    covers only train,val_internal, so it rejects a good bank."""
    text = _all_source(merge_nb)
    assert "--manifest" in text and "whole" in text.lower()
