"""Make `notebooks/` importable, the same way the notebooks themselves do.

`notebooks/` is deliberately not a package under `src/`: it is not part of the
installed `aigcdet` distribution, and a Kaggle session reaches it by path after
cloning the repo. The tests reach it the same way, so what is tested is what
runs.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# tests/notebooks/conftest.py -> tests/notebooks -> tests -> repo root.
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NOTEBOOKS_DIR = os.path.join(REPO_ROOT, "notebooks")
assert os.path.isdir(NOTEBOOKS_DIR), NOTEBOOKS_DIR

if NOTEBOOKS_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOKS_DIR)


@pytest.fixture
def repo_root() -> str:
    return REPO_ROOT


@pytest.fixture
def frozen_manifest(tmp_path):
    """A small real manifest, frozen to parquet with byte digests, plus the
    root its images live under.

    Uses the project's own `make_dummy_manifest` / `write_manifest` rather
    than a hand-built frame, so the identity columns (`rel_path`,
    `content_sha256`) are the ones `verify_images` and `manifest_fingerprint`
    actually read.
    """
    from aigcdet.data.manifest import make_dummy_manifest, write_manifest

    images = tmp_path / "data" / "normalized"
    df = make_dummy_manifest(80, str(images), np.random.default_rng(7))
    path = tmp_path / "manifest.parquet"
    write_manifest(df, str(path), root=str(images))
    return {"path": str(path), "root": str(images), "df": df}
