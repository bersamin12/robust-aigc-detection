"""Tests for `notebooks/kaggle_bootstrap.py`.

The module exists because notebook cells cannot be tested; these are the tests
that buys. The expensive failures it guards against all share a shape -- they
are silent, and they surface hours or days later on someone else's machine --
so most of what is asserted here is an INVARIANT (the shards partition the
manifest exactly; the index labels survive; no pip command ever names torch)
rather than a return value.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

import kaggle_bootstrap as kb


# --- version comparison ----------------------------------------------------

def test_version_tuple_is_numeric_not_lexical():
    """The whole reason this function exists: `"4.9" > "4.53"` as strings, so
    a string comparison would wave through a transformers too old to have
    DINOv3 -- and the failure would land after the gated 1.2 GB download."""
    assert kb.version_tuple("4.9") < kb.version_tuple("4.53")
    assert kb.version_tuple("4.9") == (4, 9)
    assert "4.9" > "4.53"          # the trap, stated


@pytest.mark.parametrize("text,want", [
    ("2.10.0+cu126", (2, 10, 0)),
    ("4.53.0.dev0", (4, 53, 0)),
    ("1.26.4rc1", (1, 26, 4)),
    ("3.11.11", (3, 11, 11)),
    ("", ()),
])
def test_version_tuple_tolerates_real_world_suffixes(text, want):
    assert kb.version_tuple(text) == want


def test_transformers_floor_matches_pyproject(repo_root):
    """A floor that drifts from pyproject is worse than none: the notebook
    would pass an environment the package refuses to run in."""
    deps = kb.read_dependencies(os.path.join(repo_root, "pyproject.toml"))
    line = next(d for d in deps if kb.requirement_name(d) == "transformers")
    assert kb.version_tuple(line.split(">=")[1]) == kb.TRANSFORMERS_FLOOR


def test_python_floor_matches_pyproject(repo_root):
    import tomllib

    with open(os.path.join(repo_root, "pyproject.toml"), "rb") as f:
        requires = tomllib.load(f)["project"]["requires-python"]
    assert kb.version_tuple(requires.lstrip(">=")) == kb.PYTHON_FLOOR


# --- install plan: never touch the session's torch --------------------------

def test_requirement_name_strips_specifiers_and_extras():
    assert kb.requirement_name("opencv-python-headless>=4.9") == "opencv-python-headless"
    assert kb.requirement_name("torch>=2.0") == "torch"
    assert kb.requirement_name("uvicorn[standard]==0.1") == "uvicorn"


def test_missing_requirements_never_reports_host_pinned_packages():
    """torch is absent from this hypothetical image, and the answer is still
    'do not install it' -- installing it is the failure mode, not the fix.

    `transformers` is deliberately NOT host-pinned: it is a pure-Python
    package that does not depend on torch, the project needs >=4.53 for
    DINOv3, and Kaggle images move. So it stays installable while every
    CUDA-matched package is dropped."""
    reqs = ["torch>=2.0", "torchvision>=0.1", "triton>=3", "nvidia-cudnn-cu12",
            "transformers>=4.53", "numpy>=1.26"]
    missing = kb.missing_requirements(reqs, is_present=lambda m: False)
    assert missing == ["transformers>=4.53", "numpy>=1.26"]


def test_host_pinned_covers_every_cuda_matched_package_in_the_image():
    assert set(kb.HOST_PINNED) >= {"torch", "torchvision", "triton", "nvidia"}
    assert "transformers" not in kb.HOST_PINNED


def test_missing_requirements_checks_the_import_name_not_the_distribution():
    """Kaggle ships `opencv-python`, which provides `cv2`. Installing
    `opencv-python-headless` on top of it is a known way to break cv2, so the
    presence check must ask about `cv2`."""
    seen = []

    def is_present(module):
        seen.append(module)
        return module == "cv2"

    assert kb.missing_requirements(["opencv-python-headless>=4.9"], is_present) == []
    assert seen == ["cv2"]


def test_missing_requirements_keeps_the_declared_version_bound():
    got = kb.missing_requirements(["scikit-learn>=1.4"], lambda m: False)
    assert got == ["scikit-learn>=1.4"]


def test_install_plan_installs_the_project_with_no_deps(repo_root):
    """`pip install -e .` hands pip `torch>=2.0` and invites it to resolve a
    wheel for the wrong CUDA. `--no-deps` makes the editable install the pure
    path registration it is only ever needed for."""
    plan = kb.install_plan(os.path.join(repo_root, "pyproject.toml"), "/repo",
                           is_present=lambda m: True)
    assert plan[0][-3:] == ["--no-deps", "-e", "/repo"]


def test_install_plan_never_names_a_host_pinned_package_except_transformers(repo_root):
    plan = kb.install_plan(os.path.join(repo_root, "pyproject.toml"), "/repo",
                           is_present=lambda m: False,
                           transformers_version="4.60.0")
    words = [w for cmd in plan for w in cmd]
    assert not any(w.startswith(("torch", "torchvision", "triton", "nvidia"))
                   for w in words), words


def test_install_plan_upgrades_a_too_old_transformers_with_no_deps(repo_root):
    plan = kb.install_plan(os.path.join(repo_root, "pyproject.toml"), "/repo",
                           is_present=lambda m: True,
                           transformers_version="4.44.0")
    upgrade = plan[-1]
    assert "transformers>=4.53" in upgrade
    assert "--no-deps" in upgrade and "-U" in upgrade


def test_install_plan_leaves_a_new_enough_transformers_alone(repo_root):
    plan = kb.install_plan(os.path.join(repo_root, "pyproject.toml"), "/repo",
                           is_present=lambda m: True,
                           transformers_version="4.53.0")
    assert not any("transformers" in w for cmd in plan for w in cmd)


def test_install_plan_commands_are_argv_lists_not_shell_strings(repo_root):
    """A Kaggle path can contain a space; a shell string would split it."""
    plan = kb.install_plan(os.path.join(repo_root, "pyproject.toml"),
                           "/kaggle/working/my repo", is_present=lambda m: True)
    assert all(isinstance(cmd, list) for cmd in plan)
    assert "/kaggle/working/my repo" in plan[0]


# --- environment problems ---------------------------------------------------

def test_environment_is_clean_when_everything_meets_the_floors():
    assert kb.environment_problems("3.11.11", "2.6.0+cu124", "4.53.0") == []


def test_environment_flags_old_python():
    problems = kb.environment_problems("3.10.14", "2.6.0", "4.53.0")
    assert len(problems) == 1 and "3.10.14" in problems[0]


def test_environment_flags_missing_torch_and_says_not_to_install_it():
    problems = kb.environment_problems("3.11.11", None, "4.53.0")
    assert "do NOT try to pip install torch".lower() in problems[0].lower()


def test_environment_flags_a_transformers_below_the_dinov3_floor():
    problems = kb.environment_problems("3.11.11", "2.6.0", "4.9.0")
    assert len(problems) == 1 and "4.9.0" in problems[0]


# --- HuggingFace auth -------------------------------------------------------

class _Secrets:
    def __init__(self, value=None, raises=False):
        self.value, self.raises = value, raises

    def get_secret(self, name):
        if self.raises:
            raise RuntimeError("secret not attached to this notebook")
        return self.value


def test_hf_token_prefers_kaggle_secrets():
    got = kb.hf_token(_Secrets("  hf_secret  "), environ={"HF_TOKEN": "hf_env"})
    assert got == "hf_secret"


def test_hf_token_falls_back_to_the_environment_when_the_secret_is_absent():
    assert kb.hf_token(_Secrets(None), environ={"HF_TOKEN": "hf_env"}) == "hf_env"


def test_hf_token_survives_an_unattached_secret_rather_than_raising():
    """A raise here produces a traceback with the word 'token' in it, which is
    the cell a teammate screenshots into a group chat."""
    assert kb.hf_token(_Secrets(raises=True), environ={"HF_TOKEN": "hf_env"}) == "hf_env"


def test_hf_token_is_none_when_nothing_provides_one():
    assert kb.hf_token(_Secrets(None), environ={}) is None


def test_hf_token_ignores_a_whitespace_only_secret():
    assert kb.hf_token(_Secrets("   "), environ={}) is None


def test_hf_advice_says_the_licence_is_per_account_even_when_a_token_exists():
    """The two failures are indistinguishable from the notebook (both are a
    401/403) and only one is fixed by adding a token."""
    for token in (None, "hf_abc"):
        text = " ".join(kb.hf_auth_advice(token, "facebook/dinov3-vitl16"))
        assert "per account" in text.lower()


def test_hf_advice_never_echoes_the_token():
    text = " ".join(kb.hf_auth_advice("hf_SUPERSECRET", "facebook/dinov3-vitl16"))
    assert "hf_SUPERSECRET" not in text


def test_hf_advice_without_a_token_tells_you_where_secrets_live():
    text = " ".join(kb.hf_auth_advice(None, "facebook/dinov3-vitl16"))
    assert "Secrets" in text and kb.HF_SECRET_NAME in text


# --- mounts -----------------------------------------------------------------

def test_content_root_descends_kaggles_single_wrapper_directory(tmp_path):
    deep = tmp_path / "slug" / "normalized" / "wildfake"
    deep.mkdir(parents=True)
    (deep / "a.png").write_bytes(b"x")
    (deep.parent / "coco").mkdir()
    assert kb.content_root(str(tmp_path / "slug")) == str(deep.parent)


def test_content_root_stops_where_the_manifest_says_the_root_is(tmp_path):
    """The blind descent overshoots a genuine root that holds a single source
    directory -- and 'one level too deep' and 'one level too high' produce the
    identical symptom (every row missing) with opposite fixes. The manifest's
    own top-level names remove the guess."""
    root = tmp_path / "slug" / "normalized"
    (root / "wildfake" / "a").mkdir(parents=True)
    # Unguided, it runs all the way to the bottom of the single-child chain.
    assert kb.content_root(str(tmp_path / "slug")) == str(root / "wildfake" / "a")
    assert kb.content_root(str(tmp_path / "slug"), {"wildfake"}) == str(root)


def test_top_level_names_reads_the_first_component_of_each_rel_path():
    df = pd.DataFrame({"rel_path": ["wildfake/x/a.png", "coco/b.png",
                                    "wildfake/y/c.png"]})
    assert kb.top_level_names(df) == {"wildfake", "coco"}


def test_top_level_names_refuses_a_manifest_with_no_portable_identity():
    with pytest.raises(ValueError, match="no rel_path"):
        kb.top_level_names(pd.DataFrame({"path": ["/abs/a.png"]}))


def test_content_root_stops_at_a_directory_with_several_children(tmp_path):
    root = tmp_path / "slug"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir()
    assert kb.content_root(str(root)) == str(root)


def test_content_root_stops_when_the_only_entry_is_a_file(tmp_path):
    root = tmp_path / "slug"
    root.mkdir()
    (root / "only.png").write_bytes(b"x")
    assert kb.content_root(str(root)) == str(root)


#: What the frozen manifest says its dataset root contains.
EXPECTED = {f"part{i}" for i in range(6)}


def _mount(base, slug, *children):
    """A Kaggle mount: `<slug>/normalized/<source-dir>/img.png`, i.e. the
    published Dataset wrapped in one extra directory level."""
    d = base / slug / "normalized"
    d.mkdir(parents=True)
    for c in children:
        (d / c).mkdir()
        (d / c / "img.png").write_bytes(b"x")
    return str(base / slug)


def test_unify_mounts_presents_five_datasets_as_one_tree(tmp_path):
    """A manifest describes ONE tree; Kaggle mounts five Datasets at five
    paths. Without this the manifest cannot be rebased onto anything."""
    mounts = [_mount(tmp_path / "in", f"slug{i}", f"part{i}") for i in range(5)]
    unified = kb.unify_mounts(mounts, str(tmp_path / "temp" / "dataset"), EXPECTED)
    assert sorted(os.listdir(unified.root)) == [f"part{i}" for i in range(5)]
    assert os.path.isfile(os.path.join(unified.root, "part3", "img.png"))


def test_unify_mounts_marks_the_result_as_a_symlink_farm(tmp_path):
    """`verify_images`' extra-file scan uses `os.walk`, which does not follow
    symlinked directories -- so 'extra files: 0' from a farm means 'not
    checked'. The flag is what lets the caller refuse to quote it."""
    mounts = [_mount(tmp_path / "in", "slug0", "part0")]
    unified = kb.unify_mounts(mounts, str(tmp_path / "temp" / "d"), EXPECTED)
    assert unified.linked is True


def test_unify_mounts_refuses_two_datasets_claiming_the_same_directory(tmp_path):
    """A collision means the wrong Datasets are attached, or one twice.
    Letting the first win leaves rows missing for a reason no error names."""
    mounts = [_mount(tmp_path / "in", "slug0", "part0"),
              _mount(tmp_path / "in", "slug1", "part0")]
    with pytest.raises(ValueError, match="more than one attached Dataset"):
        kb.unify_mounts(mounts, str(tmp_path / "temp" / "d"), EXPECTED)


def test_unify_mounts_is_idempotent_across_a_rerun_of_the_cell(tmp_path):
    mounts = [_mount(tmp_path / "in", "slug0", "part0")]
    target = str(tmp_path / "temp" / "d")
    kb.unify_mounts(mounts, target, EXPECTED)
    kb.unify_mounts(mounts, target, EXPECTED)          # must not raise "collision"
    assert os.listdir(target) == ["part0"]


def test_unify_mounts_records_the_content_roots_it_descended_to(tmp_path):
    mounts = [_mount(tmp_path / "in", "slug0", "part0")]
    unified = kb.unify_mounts(mounts, str(tmp_path / "temp" / "d"), EXPECTED)
    assert unified.sources == (os.path.join(mounts[0], "normalized"),)


# --- the verification gate --------------------------------------------------

def _gate(df, root="/root"):
    from aigcdet.features.bank import manifest_fingerprint

    return kb.VerifyGate(manifest_sha256=manifest_fingerprint(df), root=root,
                         n_rows=len(df), digest_kind="bytes",
                         n_digested=len(df), sampled=False, extra_checked=True)


def test_require_gate_rejects_anything_that_is_not_a_gate(frozen_manifest):
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    for impostor in (None, True, "verified", {"ok": True}):
        with pytest.raises(TypeError, match="VerifyGate"):
            kb.require_gate(impostor, df)


def test_require_gate_rejects_a_manifest_that_moved_since_verification(frozen_manifest):
    """The exact accident it exists for: the manifest is reloaded or
    re-filtered in a cell run AFTER the verify cell, so what is about to be
    extracted is not what was checked."""
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    gate = _gate(df)
    with pytest.raises(ValueError, match="not the one that was verified"):
        kb.require_gate(gate, df.iloc[:-1])


def test_require_gate_accepts_the_frame_it_was_made_from(frozen_manifest):
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    assert kb.require_gate(_gate(df), df).manifest_sha256


def test_open_verified_manifest_raises_when_a_file_is_missing(frozen_manifest):
    """The gate must be unobtainable when the Dataset is incomplete -- that is
    the only thing standing between a bad mount and ten GPU-hours."""
    victim = sorted(os.listdir(frozen_manifest["root"]))[0]
    os.remove(os.path.join(frozen_manifest["root"], victim))
    with pytest.raises(ValueError, match="verify_images: FAILED"):
        kb.open_verified_manifest(frozen_manifest["path"], frozen_manifest["root"])


def test_open_verified_manifest_raises_when_the_pixels_changed(frozen_manifest):
    from PIL import Image

    victim = os.path.join(frozen_manifest["root"],
                          sorted(os.listdir(frozen_manifest["root"]))[0])
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(victim)
    with pytest.raises(ValueError, match="verify_images: FAILED"):
        kb.open_verified_manifest(frozen_manifest["path"], frozen_manifest["root"])


def test_open_verified_manifest_returns_a_usable_gate_on_clean_data(frozen_manifest):
    from aigcdet.features.bank import manifest_fingerprint

    df, gate = kb.open_verified_manifest(frozen_manifest["path"],
                                         frozen_manifest["root"])
    assert gate.manifest_sha256 == manifest_fingerprint(df)
    assert gate.n_rows == len(df) and gate.digest_kind == "bytes"
    assert gate.warnings == ()


def test_a_sampled_verification_is_recorded_as_a_warning(frozen_manifest):
    """'It said OK' is what a teammate remembers; the report has to keep
    saying that a sample is evidence, not proof."""
    _, gate = kb.open_verified_manifest(frozen_manifest["path"],
                                        frozen_manifest["root"], sample=5)
    assert gate.sampled is True
    assert any("evidence, not proof" in w for w in gate.warnings)
    assert "evidence, not proof" in kb.describe_gate(gate)


def test_skipping_the_extra_scan_is_recorded_rather_than_reported_as_clean(frozen_manifest):
    _, gate = kb.open_verified_manifest(frozen_manifest["path"],
                                        frozen_manifest["root"],
                                        check_extra=False)
    assert any("was not\nchecked" in w or "not checked" in w.replace("\n", " ")
               for w in gate.warnings)


def test_describe_gate_never_claims_more_than_was_checked(frozen_manifest):
    _, gate = kb.open_verified_manifest(frozen_manifest["path"],
                                        frozen_manifest["root"], sample=5)
    text = kb.describe_gate(gate)
    assert "WARNING" in text and gate.manifest_sha256[:16] in text


# --- sharding: the property the whole five-account plan rests on ------------

@pytest.mark.parametrize("n,k", [(100, 5), (103, 5), (7, 7), (3, 5), (0, 4),
                                 (1, 1), (999983, 5)])
def test_shard_bounds_partition_every_row_exactly_once(n, k):
    bounds = kb.shard_bounds(n, k)
    covered = [i for a, b in bounds for i in range(a, b)] if n < 10_000 else None
    assert len(bounds) == k
    assert sum(b - a for a, b in bounds) == n
    assert bounds[0][0] == 0 and bounds[-1][1] == n
    assert all(bounds[i][1] == bounds[i + 1][0] for i in range(k - 1))
    if covered is not None:
        assert covered == list(range(n))


@pytest.mark.parametrize("n,k", [(100, 5), (103, 5), (7, 3), (999983, 5)])
def test_shard_bounds_are_balanced_to_within_one_row(n, k):
    sizes = [b - a for a, b in kb.shard_bounds(n, k)]
    assert max(sizes) - min(sizes) <= 1


def test_shard_bounds_give_the_remainder_to_the_first_shards():
    assert kb.shard_bounds(103, 5) == [(0, 21), (21, 42), (42, 63),
                                       (63, 83), (83, 103)]


def test_shard_bounds_are_contiguous_not_strided():
    """Strided slices (`iloc[k::5]`) preserve index labels and produce
    identical pixels, and still corrupt the bank: `merge_banks` concatenates
    in shard order, so a strided split yields rows 0,5,10,...,1,6,11,... --
    a permutation of the manifest that every positional reader misreads."""
    first = kb.shard_bounds(50, 5)[0]
    assert first == (0, 10)
    assert list(range(*first)) == list(range(10))   # not [0, 5, 10, ...]


def test_shard_bounds_rejects_a_nonsense_shard_count():
    with pytest.raises(ValueError, match="n_shards must be"):
        kb.shard_bounds(10, 0)


def test_shard_frame_preserves_the_frozen_manifests_index_labels(frozen_manifest):
    """The single most destructive available mutation. `extract_bank` keys
    every view's RNG on the index label, so a `reset_index` would restart each
    shard's keys at 0, collide five shards in key space, and give the same
    physical image different pixels depending on who extracted it -- with no
    error anywhere."""
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    gate = _gate(df)
    labels = []
    for k in range(4):
        part = kb.shard_frame(gate, df, k, 4)
        labels += list(part.index)
    assert labels == list(df.index)
    assert kb.shard_frame(gate, df, 2, 4).index[0] != 0


def test_shard_frames_never_overlap_and_cover_everything(frozen_manifest):
    """`merge_banks` raises on overlapping row_ids -- after five people have
    each paid for a session. This is the same check, before."""
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    gate = _gate(df)
    seen = [set(kb.shard_frame(gate, df, k, 5).index) for k in range(5)]
    assert set().union(*seen) == set(df.index)
    assert sum(len(s) for s in seen) == len(df)


def test_shard_frame_filters_splits_before_slicing(frozen_manifest):
    """Slicing the whole manifest into five would give shard 0 nothing but
    `train` rows and shard 4 nothing but `benchmark` ones."""
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    gate = _gate(df)
    parts = [kb.shard_frame(gate, df, k, 4, splits="train,val_internal")
             for k in range(4)]
    total = sum(len(p) for p in parts)
    assert total == int(df["split"].isin(["train", "val_internal"]).sum())
    assert total < len(df)
    assert all(set(p["split"]) <= {"train", "val_internal"} for p in parts)


def test_shard_frame_requires_the_verification_gate(frozen_manifest):
    """Ergonomics as a safety property: you cannot get a shard without having
    verified the data, because the gate is a required positional argument."""
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    with pytest.raises(TypeError, match="VerifyGate"):
        kb.shard_frame(None, df, 0, 5)


def test_shard_frame_rejects_an_out_of_range_shard_index(frozen_manifest):
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    gate = _gate(df)
    for bad in (-1, 5, 99):
        with pytest.raises(ValueError, match="shard must be in range"):
            kb.shard_frame(gate, df, bad, 5)


def test_select_splits_is_the_scripts_own_filter_not_a_copy(frozen_manifest):
    """Reused so the notebook and the CLI cannot drift on what an unknown
    split name does. It raises -- the alternative is an empty bank, discovered
    after the extraction is paid for."""
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    with pytest.raises(ValueError, match="which the manifest does not contain"):
        kb.select_splits(df, "trian,val_internal")


def test_select_splits_accepts_a_list_as_well_as_a_string(frozen_manifest):
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    assert len(kb.select_splits(df, ["train", "val_internal"])) == \
        len(kb.select_splits(df, "train,val_internal"))


def test_shard_plan_reports_disjoint_ranges_summing_to_the_whole(frozen_manifest):
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    rows = kb.shard_plan(_gate(df), df, 5, splits="train,val_internal")
    assert [r["shard"] for r in rows] == [0, 1, 2, 3, 4]
    assert sum(r["rows"] for r in rows) == \
        int(df["split"].isin(["train", "val_internal"]).sum())
    assert all(r["row_id_first"] <= r["row_id_last"] for r in rows)


# --- sizing and scheduling --------------------------------------------------

def test_bank_bytes_matches_the_on_disk_layout():
    """feats float16 (dim), presence/severity float32 (6 each), proxies
    float32 (3) -- per view, per image."""
    assert kb.bank_bytes(1, 1024, n_views=1) == 1024 * 2 + 6 * 4 + 6 * 4 + 3 * 4
    assert kb.bank_bytes(10, 1024, n_views=11) == 10 * 11 * (2048 + 60)


def test_bank_bytes_reproduces_the_specs_23gb_figure_for_100k_images():
    """The spec states 'roughly 2.3 GB per backbone' for 100k x 11 x 1024 fp16;
    a formula that does not land there is measuring something else."""
    gb = kb.bank_bytes(100_000, 1024) / 1000**3
    assert 2.2 < gb < 2.4


def test_bank_bytes_counts_recon_only_when_asked():
    plain = kb.bank_bytes(100, 1024)
    assert kb.bank_bytes(100, 1024, with_recon=True) == plain + 100 * 11 * 48


def test_a_realistic_shard_fits_the_working_quota():
    ok, text = kb.fits_in_working(20_000, 1024)
    assert ok and "fits" in text


def test_an_oversized_shard_is_refused_with_a_shard_count_to_use():
    """`BankWriter` preallocates every .npy at FULL size before the first
    image, so a shard that will not fit fails at the end with nothing saved --
    not gradually, and not at 50%."""
    ok, text = kb.fits_in_working(50_000_000, 1024)
    assert not ok and "DOES NOT FIT" in text and "raise n_shards" in text


def test_measure_rate_is_seconds_per_image():
    assert kb.measure_rate(32, 16.0) == 0.5


@pytest.mark.parametrize("n,seconds", [(0, 10.0), (10, 0.0), (10, -1.0)])
def test_measure_rate_refuses_a_degenerate_measurement(n, seconds):
    with pytest.raises(ValueError):
        kb.measure_rate(n, seconds)


def test_session_plan_converts_a_measured_rate_into_hours():
    plan = kb.session_plan(20_000, 0.36)
    assert plan.hours == pytest.approx(2.0)
    assert plan.fits_session and plan.sessions_needed == 1 and plan.notes == ()


def test_session_plan_says_how_many_sessions_an_oversized_shard_needs():
    plan = kb.session_plan(200_000, 0.36, usable_hours=10.0)
    assert plan.hours == pytest.approx(20.0)
    assert not plan.fits_session and plan.sessions_needed == 2
    assert any("RESUME=True" in n for n in plan.notes)


def test_session_plan_quantifies_what_a_session_timeout_costs():
    """The honest answer to 'what happens when Kaggle kills it': the images
    since the last metadata checkpoint, and nothing more."""
    plan = kb.session_plan(20_000, 0.36, checkpoint_every=500)
    assert plan.minutes_at_risk == pytest.approx(3.0)


def test_session_plan_suggests_a_tighter_checkpoint_when_too_much_is_at_risk():
    plan = kb.session_plan(20_000, 5.0, checkpoint_every=500)
    assert any("CHECKPOINT_EVERY" in n for n in plan.notes)


# --- resume -----------------------------------------------------------------

def _write_bank(path, manifest_df, *, backbone="dinov3l", seed=20260827,
                dim=8, n_views=11, n_done=None):
    """A tiny real bank, written through the project's own `BankWriter`.

    Through the real writer rather than a hand-made directory: what
    `check_resume` reads is `config.json` as `BankWriter` writes it, and a
    fixture that fabricated that file would test the fixture.
    """
    from aigcdet.features.bank import BankWriter, manifest_fingerprint
    from aigcdet.data.manifest import dataset_root

    n = len(manifest_df)
    n_done = n if n_done is None else n_done
    w = BankWriter(path, n, n_views, dim, backbone, seed,
                   manifest_sha256=manifest_fingerprint(manifest_df),
                   manifest_root=dataset_root(manifest_df))
    rng = np.random.default_rng(0)
    for i, (row_id, row) in enumerate(manifest_df.iterrows()):
        if i >= n_done:
            break
        w.write_image(
            i, {"path": row["path"], "label": int(row["label"]),
                "generator": row["generator"], "source": row["source"],
                "split": row["split"]},
            feats=rng.standard_normal((n_views, dim)).astype(np.float16),
            presence=np.zeros((n_views, 6), np.float32),
            severity=np.zeros((n_views, 6), np.float32),
            proxies=np.zeros((n_views, 3), np.float32),
            recipes=["[]"] * n_views, row_id=int(row_id))
    w.close()
    return path


def test_read_resume_state_reports_a_fresh_directory_as_nothing_to_resume(tmp_path):
    state = kb.read_resume_state(str(tmp_path / "nope"))
    assert state.exists is False and state.n_done == 0


def test_read_resume_state_counts_the_images_already_written(frozen_manifest, tmp_path):
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    bank = _write_bank(str(tmp_path / "bank"), df, n_done=30)
    state = kb.read_resume_state(bank)
    assert state.exists and state.n_images == len(df) and state.n_done == 30
    assert state.n_remaining == len(df) - 30
    assert state.fraction_done == pytest.approx(30 / len(df))


def test_read_resume_state_treats_a_config_only_directory_as_zero_done(tmp_path):
    """`config.json` is written before the first image, so this is a session
    that died inside its first checkpoint interval -- not an error."""
    d = tmp_path / "bank"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(
        {"backbone": "dinov3l", "n_images": 100, "seed": 1, "n_views": 11}))
    state = kb.read_resume_state(str(d))
    assert state.exists and state.n_done == 0 and state.n_images == 100


def test_check_resume_accepts_a_genuine_continuation(frozen_manifest, tmp_path):
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    shard = kb.shard_frame(_gate(df), df, 1, 4)
    bank = _write_bank(str(tmp_path / "bank"), shard, n_done=5)
    state = kb.check_resume(bank, shard, backbone="dinov3l", seed=20260827)
    assert state.n_done == 5


def test_check_resume_refuses_a_different_shard_in_the_same_directory(
        frozen_manifest, tmp_path):
    """The commonest Kaggle accident: change SHARD_INDEX, forget to change
    --out, re-run with RESUME=True. Caught here from config.json alone, before
    the 1.2 GB gated download that BankWriter's own check would follow."""
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    gate = _gate(df)
    bank = _write_bank(str(tmp_path / "bank"), kb.shard_frame(gate, df, 0, 4))
    with pytest.raises(ValueError, match="SHARD_INDEX"):
        kb.check_resume(bank, kb.shard_frame(gate, df, 1, 4),
                        backbone="dinov3l", seed=20260827)


@pytest.mark.parametrize("kwargs,needle", [
    ({"backbone": "siglip2l"}, "backbone"),
    ({"seed": 1}, "seed"),
    ({"n_views": 5}, "n_views"),
])
def test_check_resume_refuses_a_changed_extraction_parameter(
        frozen_manifest, tmp_path, kwargs, needle):
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    shard = kb.shard_frame(_gate(df), df, 0, 4)
    bank = _write_bank(str(tmp_path / "bank"), shard)
    call = {"backbone": "dinov3l", "seed": 20260827, **kwargs}
    with pytest.raises(ValueError, match=needle):
        kb.check_resume(bank, shard, **call)


def test_check_resume_never_suggests_deleting_completed_work(
        frozen_manifest, tmp_path):
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    gate = _gate(df)
    bank = _write_bank(str(tmp_path / "bank"), kb.shard_frame(gate, df, 0, 4))
    with pytest.raises(ValueError) as exc:
        kb.check_resume(bank, kb.shard_frame(gate, df, 1, 4),
                        backbone="dinov3l", seed=20260827)
    assert "Do NOT delete" in str(exc.value)


def test_check_resume_treats_a_missing_directory_as_a_first_session(
        frozen_manifest, tmp_path):
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    state = kb.check_resume(str(tmp_path / "fresh"), df,
                            backbone="dinov3l", seed=20260827)
    assert state.exists is False


# --- command construction ---------------------------------------------------

def test_run_shard_argv_requires_a_gate():
    with pytest.raises(TypeError, match="VerifyGate"):
        kb.run_shard_argv(None, manifest_path="m", root="/r", backbone="dinov3l",
                          out_dir="/o", splits="train", shard=0, n_shards=5)


def test_run_shard_argv_carries_the_verified_fingerprint_into_the_child(frozen_manifest):
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    gate = _gate(df)
    argv = kb.run_shard_argv(gate, manifest_path="m", root="/r",
                             backbone="dinov3l", out_dir="/o",
                             splits="train,val_internal", shard=2, n_shards=5)
    assert argv[argv.index("--expect-manifest-sha256") + 1] == gate.manifest_sha256
    assert argv[argv.index("--shard") + 1] == "2"
    assert argv[argv.index("--split") + 1] == "train,val_internal"


def test_run_shard_argv_is_a_list_so_a_comma_split_stays_one_argument(frozen_manifest):
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    argv = kb.run_shard_argv(_gate(df), manifest_path="m", root="/r",
                             backbone="dinov3l", out_dir="/o",
                             splits="train,val_internal", shard=0, n_shards=5)
    assert isinstance(argv, list)
    assert all(isinstance(a, str) for a in argv)
    assert "train,val_internal" in argv


def test_run_shard_argv_omits_resume_when_not_asked(frozen_manifest):
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    kw = dict(manifest_path="m", root="/r", backbone="dinov3l", out_dir="/o",
              splits="train", shard=0, n_shards=5)
    assert "--resume" in kb.run_shard_argv(_gate(df), resume=True, **kw)
    assert "--resume" not in kb.run_shard_argv(_gate(df), resume=False, **kw)


def test_run_shard_argv_passes_limit_only_for_the_smoke_path(frozen_manifest):
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    kw = dict(manifest_path="m", root="/r", backbone="dinov3l", out_dir="/o",
              splits="train", shard=0, n_shards=5)
    assert "--limit" not in kb.run_shard_argv(_gate(df), **kw)
    assert kb.run_shard_argv(_gate(df), limit=8, **kw)[-3:-1] == ["--limit", "8"] \
        or "--limit" in kb.run_shard_argv(_gate(df), limit=8, **kw)


def test_sorted_shard_dirs_orders_numerically_not_lexically():
    """Lexical order gives shard0, shard1, shard10, shard2 -- and
    `merge_banks` concatenates in the order it is given, so a ten-shard bank
    would come out permuted."""
    got = kb.sorted_shard_dirs([f"banks/d_shard{i}" for i in (2, 10, 0, 1, 9)])
    assert got == [f"banks/d_shard{i}" for i in (0, 1, 2, 9, 10)]


def test_sorted_shard_dirs_puts_unrecognised_names_last_rather_than_guessing():
    got = kb.sorted_shard_dirs(["banks/extra", "banks/d_shard1", "banks/d_shard0"])
    assert got == ["banks/d_shard0", "banks/d_shard1", "banks/extra"]


def test_merge_argv_preserves_the_shard_order_it_is_given():
    argv = kb.merge_argv("/out", ["/s0", "/s1", "/s2"], repo_dir="/repo")
    assert argv[-3:] == ["/s0", "/s1", "/s2"]
    assert argv[argv.index("--out") + 1] == "/out"
    assert argv[1] == os.path.join("/repo", "scripts", "merge_banks.py")


def test_merge_argv_refuses_an_empty_shard_list():
    with pytest.raises(ValueError, match="at least one shard"):
        kb.merge_argv("/out", [])


# --- the merged bank --------------------------------------------------------

def test_verify_merged_bank_accepts_a_bank_covering_the_extracted_splits(
        frozen_manifest, tmp_path):
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    sub = kb.select_splits(df, "train,val_internal")
    bank = _write_bank(str(tmp_path / "merged"), sub)
    text = kb.verify_merged_bank(bank, frozen_manifest["path"],
                                 frozen_manifest["root"], "train,val_internal")
    assert f"{len(sub)} rows" in text and "val_internal rows" in text


def test_verify_merged_bank_catches_a_bank_built_from_different_rows(
        frozen_manifest, tmp_path):
    """A re-split manifest, or shards merged out of order: both land here."""
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    sub = kb.select_splits(df, "train,val_internal")
    bank = _write_bank(str(tmp_path / "merged"), sub.iloc[::-1])
    with pytest.raises(ValueError, match="not the manifest the bank was built from"):
        kb.verify_merged_bank(bank, frozen_manifest["path"],
                              frozen_manifest["root"], "train,val_internal")


def test_verify_merged_bank_rejects_a_bank_with_no_val_internal_rows(
        frozen_manifest, tmp_path):
    """Stage B evaluates on the bank's own val_internal rows, so a train-only
    bank is 8-13 h that has to be paid again."""
    from aigcdet.data.manifest import read_manifest

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    sub = kb.select_splits(df, "train")
    bank = _write_bank(str(tmp_path / "merged"), sub)
    with pytest.raises(ValueError, match="no val_internal rows"):
        kb.verify_merged_bank(bank, frozen_manifest["path"],
                              frozen_manifest["root"], "train")


# --- the child process's gate ----------------------------------------------

def test_resolve_shard_refuses_a_manifest_that_is_not_the_verified_one(
        frozen_manifest):
    """The gate crossing the process boundary: `run_shard.py` gets only a
    fingerprint, and must refuse if the manifest it reads is not that one."""
    with pytest.raises(ValueError, match="not the one that was verified"):
        kb.resolve_shard(frozen_manifest["path"], frozen_manifest["root"],
                         "train,val_internal", 0, 5, "0" * 64)


def test_resolve_shard_returns_the_same_slice_the_notebook_planned(frozen_manifest):
    from aigcdet.data.manifest import read_manifest
    from aigcdet.features.bank import manifest_fingerprint

    df = read_manifest(frozen_manifest["path"], root=frozen_manifest["root"])
    want = kb.shard_frame(_gate(df), df, 3, 5, splits="train,val_internal")
    _, got = kb.resolve_shard(frozen_manifest["path"], frozen_manifest["root"],
                              "train,val_internal", 3, 5,
                              manifest_fingerprint(df))
    assert list(got.index) == list(want.index)


# --- the 2am playbook -------------------------------------------------------

@pytest.mark.parametrize("text,kind,fatal", [
    ("OSError: You are trying to access a gated repo", "hf-gated", True),
    ("401 Client Error: Unauthorized for url", "hf-gated", True),
    ("ValueError: cannot resume the bank at /kaggle/working/b", "resume-mismatch", True),
    ("ValueError: this is not the manifest the bank was built from", "manifest-drift", True),
    ("ValueError: shards overlap: 3 row_id(s)", "shard-overlap", True),
    ("ValueError: bank has no val_internal rows", "missing-split", True),
    ("verify_images: FAILED  root=/kaggle/temp/d", "data-mismatch", True),
    ("torch.OutOfMemoryError: CUDA out of memory.", "oom", False),
    ("OSError: [Errno 28] No space left on device", "disk-full", True),
    ("requests.exceptions.ReadTimeout: Read timed out", "network", False),
    ("ModuleNotFoundError: No module named 'aigcdet'", "not-installed", True),
    ("RuntimeError: could not start the worker pool", "spawn", True),
    ("KeyError: 'wobble'", "unknown", True),
])
def test_diagnose_names_the_kind_and_whether_a_rerun_can_help(text, kind, fatal):
    d = kb.diagnose(text)
    assert (d.kind, d.fatal) == (kind, fatal)
    assert d.action.strip()


def test_diagnose_accepts_an_exception_object_not_only_a_string():
    d = kb.diagnose(ValueError("bank has no val_internal rows, so the val AUC"))
    assert d.kind == "missing-split"


def test_every_fatal_diagnosis_names_what_to_change():
    """A fatal verdict with no instruction is worse than none: the reader
    re-runs it anyway, at an hour of a 30 h weekly budget per attempt."""
    for _pattern, _kind, fatal, action in kb._DIAGNOSES:
        if fatal:
            assert len(action) > 60, action


def test_the_oom_playbook_says_to_resume_rather_than_restart():
    assert "RESUME=True" in kb.diagnose("CUDA out of memory").action


def test_the_torch_playbook_forbids_pip_installing_torch():
    """The one instruction that turns a recoverable session into a lost one."""
    action = kb.diagnose("ImportError: libcudart.so.12: cannot open").action
    assert "Do NOT pip install torch".lower() in action.lower()


def test_the_manifest_drift_playbook_forbids_rerunning_build_dataset():
    """The manifest is frozen. Re-splitting after banks exist silently
    misaligns labels against features."""
    action = kb.diagnose("manifest/bank row 41 misaligned").action
    assert "build_dataset" in action and "Never re-run" in action


def test_explain_states_the_verdict_and_the_action():
    text = kb.explain("CUDA out of memory")
    assert "RETRYABLE" in text and "BATCH_SIZE" in text
    assert "FATAL" in kb.explain("KeyError: 'x'")


def test_diagnosis_patterns_are_ordered_specific_before_general():
    """`_DIAGNOSES` is first-match-wins, so a general pattern placed above a
    specific one would swallow it."""
    assert kb.diagnose(
        "OSError: [Errno 28] No space left on device while writing "
        "feats.npy").kind == "disk-full"
    assert kb.diagnose(
        "ValueError: cannot resume the bank at /b: config.json disagrees "
        "on manifest_sha256").kind == "resume-mismatch"


# --- constants that encode a Kaggle fact ------------------------------------

def test_the_bank_is_written_where_kaggle_persists_it():
    """`/kaggle/temp` is faster and free of the quota, and is DISCARDED at the
    end of the session -- which is precisely the event --resume exists for."""
    assert kb.WORKING_DIR == "/kaggle/working"
    assert kb.TEMP_DIR == "/kaggle/temp"
    assert kb.USABLE_SESSION_HOURS < kb.SESSION_LIMIT_HOURS


# --- measuring the real rate, and streaming a long job ----------------------

def test_marginal_rate_differences_out_the_fixed_startup_cost():
    """One timed smoke run measures mostly the model download, the CUDA
    context and the process start. 8 images in 100 s and 40 in 116 s is
    0.5 s/image of real work behind a 96 s constant -- not 12.5 s/image."""
    assert kb.marginal_rate(8, 100.0, 40, 116.0) == pytest.approx(0.5)


def test_marginal_rate_refuses_runs_that_cannot_be_differenced():
    with pytest.raises(ValueError, match="strictly larger"):
        kb.marginal_rate(40, 100.0, 8, 116.0)
    with pytest.raises(ValueError, match="no slower"):
        kb.marginal_rate(8, 116.0, 40, 100.0)


def test_run_streaming_returns_the_exit_code_and_echoes_each_line():
    """Streamed, not captured: the thing being run prints a tqdm bar for the
    next several hours."""
    lines = []
    rc = kb.run_streaming(
        [sys.executable, "-c", "print('alpha'); print('beta')"], lines.append)
    assert rc == 0 and lines == ["alpha", "beta"]


def test_run_streaming_surfaces_a_failing_command_rather_than_raising():
    """The notebook decides what to do with a non-zero exit (usually: print
    the playbook), so this must not raise past it."""
    lines = []
    rc = kb.run_streaming(
        [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"],
        lines.append)
    assert rc == 3 and lines == ["boom"]


def test_run_streaming_merges_stderr_into_the_stream():
    """A traceback goes to stderr; losing it would leave the playbook nothing
    to diagnose."""
    lines = []
    kb.run_streaming(
        [sys.executable, "-c", "import sys; print('err', file=sys.stderr)"],
        lines.append)
    assert lines == ["err"]


def test_read_resume_state_survives_a_bank_that_wrote_no_images(tmp_path):
    """A `BankWriter.close()` with nothing written leaves an EMPTY meta.parquet
    -- a frame with no columns at all. Asking it for `image_idx` by name
    raises, and progress reporting must never be the thing that fails a
    resume."""
    from aigcdet.features.bank import BankWriter

    out = str(tmp_path / "bank")
    BankWriter(out, 10, 11, 8, "dinov3l", 20260827, manifest_sha256="x").close()
    state = kb.read_resume_state(out)
    assert state.exists and state.n_done == 0 and state.n_images == 10
    assert state.n_remaining == 10 and state.fraction_done == 0.0


def test_version_tuple_stops_at_a_prerelease_suffix():
    """A component like `3rc1` ends the release segment; anything after it is
    prerelease metadata, not a fourth release number. Without the explicit
    stop, `1.2.3rc1.4` would read as (1, 2, 3, 4) -- a different version."""
    assert kb.version_tuple("1.2.3rc1.4") == (1, 2, 3)
    assert kb.version_tuple("2.10.0+cu126.1") == (2, 10, 0)


def test_session_plan_rounds_the_session_count_up_not_down():
    """15 h against a 10 h budget is two sessions. Truncating says one, and a
    teammate plans a single sitting for work that cannot finish in it."""
    plan = kb.session_plan(150_000, 0.36, usable_hours=10.0)
    assert plan.hours == pytest.approx(15.0)
    assert plan.sessions_needed == 2 and not plan.fits_session


def test_session_plan_needs_no_second_session_for_an_exact_fit():
    assert kb.session_plan(100_000, 0.36, usable_hours=10.0).sessions_needed == 1


def test_script_modules_are_loaded_once(repo_root):
    """`scripts/extract_features.py` imports torch on the way in, and
    `select_splits` calls it per shard."""
    kb._SCRIPT_MODULES.clear()
    first = kb.load_script_module("extract_features", repo_root)
    assert kb.load_script_module("extract_features", repo_root) is first


def test_load_script_module_reuses_the_cli_not_a_copy_of_it(repo_root):
    """If these two ever became separate implementations, the notebook and the
    CLI could disagree about what an unknown split name does."""
    mod = kb.load_script_module("extract_features", repo_root)
    assert kb.select_splits.__module__ == kb.__name__
    assert callable(mod.select_splits)
