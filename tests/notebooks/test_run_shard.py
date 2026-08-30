"""Tests for `notebooks/run_shard.py`, the sharded Stage A entry point.

Everything here runs through `--dry-run`, which stops immediately before the
backbone is loaded. That is not a testing compromise: `--dry-run` exists so a
teammate can check the shard, the gate and the resume in seconds rather than
after a 1.2 GB gated download, and these tests exercise exactly the path they
would.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import kaggle_bootstrap as kb
import run_shard


def _argv(frozen_manifest, sha, **over):
    a = {"--manifest": frozen_manifest["path"], "--root": frozen_manifest["root"],
         "--backbone": "dinov3l", "--split": "train,val_internal",
         "--shard": "0", "--n-shards": "4",
         "--expect-manifest-sha256": sha, "--out": over.pop("out", "/tmp/unused")}
    a.update({k: str(v) for k, v in over.items()})
    argv = [x for kv in a.items() for x in kv] + ["--dry-run"]
    return argv


@pytest.fixture
def sha(frozen_manifest):
    from aigcdet.data.manifest import read_manifest
    from aigcdet.features.bank import manifest_fingerprint

    return manifest_fingerprint(
        read_manifest(frozen_manifest["path"], root=frozen_manifest["root"]))


def test_dry_run_resolves_the_shard_without_loading_a_backbone(
        frozen_manifest, sha, tmp_path, capsys):
    rc = run_shard.main(_argv(frozen_manifest, sha, out=str(tmp_path / "b")))
    assert rc == 0
    out = capsys.readouterr().out
    assert "shard 0/4" in out and "row_id" in out
    assert "stopping before the backbone" in out
    assert not os.path.exists(tmp_path / "b")


def test_the_gate_crosses_the_process_boundary(frozen_manifest, tmp_path):
    """The notebook verified the data and holds the fingerprint; this process
    reads the manifest for itself and must refuse if it is not the same one."""
    with pytest.raises(ValueError, match="not the one that was verified"):
        run_shard.main(_argv(frozen_manifest, "0" * 64, out=str(tmp_path / "b")))


def test_an_empty_shard_is_refused_rather_than_written(frozen_manifest, sha, tmp_path):
    """A zero-row bank merges silently and contributes nothing."""
    with pytest.raises(SystemExit, match="is empty"):
        run_shard.main(_argv(frozen_manifest, sha, out=str(tmp_path / "b"),
                             **{"--n-shards": 10_000, "--shard": 9_999}))


def test_shards_resolved_by_the_cli_match_what_the_notebook_planned(
        frozen_manifest, sha, tmp_path, capsys):
    """The CLI re-derives the slice from (shard, n_shards) instead of trusting
    a row range typed into a notebook; the two must agree."""
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    gate = kb.carried_gate(sha, frozen_manifest["root"])
    for k in range(4):
        run_shard.main(_argv(frozen_manifest, sha, out=str(tmp_path / f"b{k}"),
                             **{"--shard": k}))
        want = kb.shard_frame(gate, df, k, 4, splits="train,val_internal")
        line = capsys.readouterr().out.splitlines()[0]
        assert f"{len(want)} rows" in line
        assert f"row_id {int(want.index[0])}..{int(want.index[-1])}" in line


def test_limit_narrows_this_shard_not_the_whole_manifest(
        frozen_manifest, sha, tmp_path, capsys):
    """`--limit` is the smoke path. Applied to the shard, so the bank's
    recorded fingerprint covers exactly the rows it holds -- handing it to
    `extract_bank` instead would fingerprint the full shard and then write a
    shorter bank, and every later resume would be refused."""
    run_shard.main(_argv(frozen_manifest, sha, out=str(tmp_path / "b"),
                         **{"--limit": 3}))
    assert "3 rows" in capsys.readouterr().out.splitlines()[0]


def test_a_mismatched_resume_is_caught_before_the_backbone(
        frozen_manifest, sha, tmp_path):
    """The commonest Kaggle accident -- change SHARD_INDEX, forget --out --
    costs a second here instead of a gated model download first."""
    from aigcdet.data.manifest import read_manifest
    from aigcdet.features.bank import BankWriter, manifest_fingerprint

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    gate = kb.carried_gate(sha, frozen_manifest["root"])
    other = kb.shard_frame(gate, df, 3, 4, splits="train,val_internal")
    out = str(tmp_path / "bank")
    BankWriter(out, len(other), 11, 8, "dinov3l", 20260827,
               manifest_sha256=manifest_fingerprint(other)).close()

    with pytest.raises(ValueError, match="SHARD_INDEX"):
        run_shard.main(_argv(frozen_manifest, sha, out=out, **{"--shard": 0}))


def test_the_cli_defaults_to_the_projects_seed_and_both_splits():
    """A shard extracted under a different seed is not merge-compatible, and a
    train-only bank is unusable in Stage B. Both are defaults so that getting
    them right requires typing nothing."""
    a = run_shard.build_parser().parse_args(
        ["--manifest", "m", "--root", "/r", "--backbone", "dinov3l",
         "--out", "/o", "--shard", "0", "--n-shards", "5",
         "--expect-manifest-sha256", "x"])
    assert a.seed == 20260827
    assert a.split == "train,val_internal"
    assert a.resume is False        # opt-in, so a fresh run is never a surprise


def test_the_cli_requires_the_gate_fingerprint():
    """Not optional: without it the child process could extract from a copy
    the notebook never checked."""
    with pytest.raises(SystemExit):
        run_shard.build_parser().parse_args(
            ["--manifest", "m", "--root", "/r", "--backbone", "dinov3l",
             "--out", "/o", "--shard", "0", "--n-shards", "5"])


def test_run_shard_is_guarded_by_a_main_block(repo_root):
    """Load-bearing, not conventional: --workers > 0 spawns subprocesses that
    re-import this module, and without the guard each would start its own
    extraction."""
    src = open(os.path.join(repo_root, "notebooks", "run_shard.py")).read()
    assert 'if __name__ == "__main__":' in src
    assert src.index('if __name__ == "__main__":') < src.index("sys.exit(main())")


# --- standardisation policy -------------------------------------------------

def test_the_cli_defaults_to_band_so_the_frozen_stream_is_unchanged():
    a = run_shard.build_parser().parse_args(
        ["--manifest", "m", "--root", "/r", "--backbone", "dinov3l",
         "--out", "/o", "--shard", "0", "--n-shards", "1",
         "--expect-manifest-sha256", "x"])
    assert a.canon_mode == "band"
    assert a.geometric is False


def test_the_cli_accepts_the_crop_policy():
    a = run_shard.build_parser().parse_args(
        ["--manifest", "m", "--root", "/r", "--backbone", "dinov3l",
         "--out", "/o", "--shard", "0", "--n-shards", "1",
         "--expect-manifest-sha256", "x",
         "--canon-mode", "crop", "--crop-side", "200", "--geometric"])
    assert (a.canon_mode, a.crop_side, a.geometric) == ("crop", 200, True)


def test_geometric_under_band_mode_exits_before_the_shard_is_resolved(
        frozen_manifest, sha, tmp_path):
    """The guard must fire on the CONFIGURATION, before any work: on Kaggle the
    next step is a multi-GB download inside a metered session."""
    argv = _argv(frozen_manifest, sha, out=str(tmp_path / "b"))
    argv = [x for x in argv if x != "--dry-run"] + ["--geometric"]
    with pytest.raises(SystemExit, match="--canon-mode crop"):
        run_shard.main(argv)


def test_the_crop_policy_reaches_extract_bank(frozen_manifest, sha, tmp_path,
                                              monkeypatch):
    """The whole point of the plumbing: a band bank over crop-stream data is
    not an error and not empty, just silently wrong, so this asserts the
    policy actually arrives rather than that the run merely succeeds."""
    seen = {}

    def fake_extract_bank(df, backbone, out, **kw):
        seen.update(kw)

    import aigcdet.features.extract as ex
    monkeypatch.setattr(ex, "extract_bank", fake_extract_bank)
    argv = [x for x in _argv(frozen_manifest, sha, out=str(tmp_path / "b"))
            if x != "--dry-run"]
    run_shard.main(argv + ["--canon-mode", "crop", "--crop-side", "200",
                           "--geometric"])
    assert seen["geometric"] is True
    assert seen["policy"].mode == "crop"
    assert seen["policy"].crop_side == 200
    assert seen["policy"].is_square
