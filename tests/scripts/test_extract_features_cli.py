"""Stage A CLI: the --split contract, and the --shard split the docstring promised.

`scripts/extract_features.py` documented `--split train`, but Stage B's
`train_rung` evaluates on the bank's own `val_internal` rows, so that bank is
rejected -- on Kaggle, after 8-13 hours of extraction. These tests pin the
comma-separated form and the error that names what a bank actually holds.

The same docstring promised "shards -- disjoint slices of the SAME manifest,
extracted in separate sessions" while the CLI offered no way to ask for rows
40000..60000 at all: `--limit` takes a prefix and nothing else selects a
range. The whole delivery plan is five teammates each extracting a fifth on a
free Kaggle account, so the rest of this file is `--shard I/N` and the four
properties that make it true, every one of which fails SILENTLY:

1. The blocks are CONTIGUOUS. A strided `iloc[k::n]` keeps the index labels
   and so draws byte-identical PIXELS -- no pixel check can see it -- but
   merges into a bank whose rows are a permutation of the manifest.
2. The shard frame keeps the frozen manifest's INDEX LABELS. That label is
   the per-view RNG key, so a `reset_index()` makes the same physical image
   carry different pixels depending on who extracted it. Asserted over
   pixels, not over labels alone.
3. The blocks TILE: exhaustive and disjoint. `merge_banks` checks shards for
   overlap but never for coverage, so a dropped remainder is invisible.
4. The order of operations is `--split`, then `--limit`, then `--shard`, and
   the plausible alternatives give different, non-overlapping answers.

Property 3's block arithmetic is also pinned against
`notebooks/kaggle_bootstrap.shard_bounds`: `notebooks/run_shard.py` and this
script cut shards of the SAME training bank and those shards get merged, so
the two partitions must agree row for row.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from aigcdet.data.manifest import MANIFEST_COLUMNS, read_manifest, write_manifest
from aigcdet.features.bank import N_VIEWS, BankWriter, FeatureBank, merge_banks
from aigcdet.train.train_head import RungConfig, train_rung

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = _ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ef = _load_script("extract_features")


def _kaggle_bootstrap():
    """`notebooks/kaggle_bootstrap`, reached the way the notebooks reach it.

    `notebooks/` is not a package under `src/`; `tests/notebooks/conftest.py`
    puts it on `sys.path` for its own directory, and this file needs it too
    when run on its own.
    """
    nb = str(_ROOT / "notebooks")
    if nb not in sys.path:
        sys.path.insert(0, nb)
    import kaggle_bootstrap

    return kaggle_bootstrap


def _manifest(n=12):
    splits = ["train", "val_internal", "heldout_generator", "benchmark"]
    return pd.DataFrame({
        "path": [f"/p{i}.png" for i in range(n)],
        "label": [i % 2 for i in range(n)],
        "generator": ["" if i % 2 == 0 else "g" for i in range(n)],
        "source": ["s"] * n,
        "split": [splits[i % len(splits)] for i in range(n)],
    })


def test_split_accepts_a_comma_separated_list():
    df = _manifest()
    out = ef.select_splits(df, "train,val_internal")
    assert sorted(out["split"].unique()) == ["train", "val_internal"]
    assert len(out) == 6


def test_split_accepts_a_single_name_and_tolerates_whitespace():
    df = _manifest()
    assert set(ef.select_splits(df, "train")["split"]) == {"train"}
    assert sorted(ef.select_splits(df, " train , val_internal ")["split"].unique()) == \
        ["train", "val_internal"]


def test_empty_split_keeps_every_row():
    df = _manifest()
    assert ef.select_splits(df, "") is df


def test_split_preserves_the_frozen_manifest_index_labels():
    """extract_bank keys each view's RNG on the index label, so the filter
    must never reset it (a reset would change which views get drawn and break
    shard consistency)."""
    df = _manifest()
    out = ef.select_splits(df, "train,val_internal")
    assert out.index.tolist() == [0, 1, 4, 5, 8, 9]


def test_unknown_split_name_fails_before_any_extraction():
    df = _manifest()
    with pytest.raises(ValueError, match="does not contain"):
        ef.select_splits(df, "train,val")


def test_documented_command_in_the_module_docstring_names_both_splits():
    """The docstring is the command a human copies onto Kaggle. It must name
    the combination that actually produces a trainable bank."""
    assert "--split train,val_internal" in ef.__doc__
    assert "--split train " not in ef.__doc__


def _bank_without_val(tmp_path, n=8, dim=4):
    w = BankWriter(str(tmp_path / "b"), n, N_VIEWS, dim, "t", 0)
    for i in range(n):
        pres = np.zeros((N_VIEWS, 6), np.float32); pres[1:, 0] = 1.0
        w.write_image(i, {"path": f"/p{i}", "label": i % 2, "generator": "",
                          "source": "s", "split": "train"},
                      feats=np.zeros((N_VIEWS, dim), np.float32),
                      presence=pres, severity=np.zeros((N_VIEWS, 6), np.float32),
                      proxies=np.zeros((N_VIEWS, 3), np.float32),
                      recipes=["[]"] * N_VIEWS)
    w.close()
    return str(tmp_path / "b")


def test_train_rung_names_the_splits_the_bank_actually_contains(tmp_path):
    cfg = RungConfig(name="a0", bank_dir=_bank_without_val(tmp_path), epochs=1,
                     out_dir=str(tmp_path / "out"))
    with pytest.raises(ValueError) as exc:
        train_rung(cfg)
    msg = str(exc.value)
    assert "val_internal" in msg
    assert "'train': 8" in msg            # names what the bank DOES contain
    assert "--split train,val_internal" in msg


# --- fixtures for the sharding tests ---------------------------------------

#: The rows this file's split filter keeps, in manifest order. Stated as a
#: literal rather than recomputed from `select_splits`, so an expectation is
#: never derived from the object under test.
WANTED_SPLITS = "train,val_internal"
SELECTED_LABELS = [0, 1, 2, 3, 4, 10, 11, 12, 13]


def _tree(root: str) -> pd.DataFrame:
    """A 14-row dataset on disk whose split layout is CLUSTERED, not interleaved.

    Row layout, in manifest order:

        0..4    train           (5)
        5..9    benchmark       (5)   <- never selected; the gap in the middle
        10..13  val_internal    (4)

    Three fixture properties are load-bearing, and each of them is a
    documented way these tests could have passed against broken code:

    * The unselected `benchmark` block sits BETWEEN the two selected ones. If
      the splits were interleaved (`splits[i % 4]`, as `_manifest` above does)
      then "filter, then shard" and "shard, then filter" would return the same
      rows for this manifest and the order-of-operations mutation would be
      undetectable.
    * 14 is divisible by neither 3 nor 5, so a partition that drops the
      remainder is visible. A row count divisible by the shard count could not
      detect that at all.
    * 3 shards, not 2: with two shards a strided split and a contiguous one
      differ only in the interleaving of two blocks, which several natural
      orderings still satisfy.
    """
    rng = np.random.default_rng(0)
    rows = []
    layout = [("train", 5), ("benchmark", 5), ("val_internal", 4)]
    k = 0
    for split, n in layout:
        for _ in range(n):
            label = k % 2
            gen = "" if label == 0 else f"g{k % 3}"
            p = os.path.abspath(os.path.join(root, split, f"{split}_{k:03d}.png"))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(p)
            rows.append({"path": p, "label": label, "generator": gen,
                         "source": "s", "licence": "CC0", "width": 32,
                         "height": 32, "split": split})
            k += 1
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


@pytest.fixture
def frozen(tmp_path):
    """(manifest path, frozen frame) -- with the `rel_path` identity columns,
    so a merged bank fingerprints to what the manifest fingerprints to."""
    root = str(tmp_path / "normalized")
    path = str(tmp_path / "manifest.parquet")
    write_manifest(_tree(root), path, root=root)
    return path, read_manifest(path)


def _fake_backbone(monkeypatch, dim=4):
    """No GPU, no weights, and an 'embedding' that is a function of the PIXELS.

    `sum % 2003` stays inside float16's exactly-representable integer range,
    so a bank round-trip survives equality comparison: two runs whose cached
    feats are equal really did see the same pixels. An image MEAN would not --
    float16 rounds it, and two different views could compare equal.
    """
    from aigcdet.features import extract
    from aigcdet.features.backbones import BackboneSpec

    spec = BackboneSpec("fake", "none", 32, dim, 1, 0)
    monkeypatch.setattr(extract, "load_backbone", lambda n, device: (None, spec))
    monkeypatch.setattr(
        extract, "embed",
        lambda m, s, imgs, device, batch_size=16:
            np.stack([np.full(s.dim,
                              float(np.asarray(v, np.float64).sum() % 2003),
                              np.float32) for v in imgs]))
    return extract


def _run(argv, monkeypatch):
    _fake_backbone(monkeypatch)
    return ef.main(argv)


def _common(manifest):
    return ["--manifest", manifest, "--backbone", "fake",
            "--split", WANTED_SPLITS, "--device", "cpu"]


# --- shard_frame: the partition itself -------------------------------------

def test_the_split_filter_selects_the_rows_these_tests_assume(frozen):
    """Guard on the fixture: if the layout ever changes, every expectation
    below is derived from a stale literal. Fail here instead."""
    _, df = frozen
    assert len(df) == 14
    assert ef.select_splits(df, WANTED_SPLITS).index.tolist() == SELECTED_LABELS
    # ...and the unselected block really is in the MIDDLE, which is what makes
    # the filter/shard order-of-operations mutation detectable at all.
    assert df.loc[5:9, "split"].tolist() == ["benchmark"] * 5


@pytest.mark.parametrize("n_shards", [3, 5])
def test_shards_are_contiguous_disjoint_and_exhaustive(frozen, n_shards):
    _, df = frozen
    assert len(df) % n_shards != 0, "a divisible fixture cannot see a dropped remainder"

    seen: list[int] = []
    sizes: list[int] = []
    for i in range(n_shards):
        part = ef.shard_frame(df, f"{i}/{n_shards}")
        labels = part.index.tolist()
        # Contiguous: a run of consecutive labels, not a stride.
        assert labels == list(range(labels[0], labels[0] + len(labels)))
        seen += labels
        sizes.append(len(labels))

    assert len(set(seen)) == len(seen), "shards overlap"          # disjoint
    assert sorted(seen) == df.index.tolist()                      # exhaustive
    assert seen == df.index.tolist()                              # in order
    assert max(sizes) - min(sizes) <= 1                           # balanced


def test_the_union_of_every_shard_is_exactly_the_unsharded_frame(frozen):
    """Same rows, same order, same index labels -- asserted on the frame, not
    on the labels alone, so a shard that silently dropped or duplicated a
    column would show up too."""
    _, df = frozen
    rebuilt = pd.concat([ef.shard_frame(df, f"{i}/4") for i in range(4)])
    pd.testing.assert_frame_equal(rebuilt, df)


def test_shard_frame_preserves_the_frozen_manifest_index_labels(frozen):
    _, df = frozen
    sel = ef.select_splits(df, WANTED_SPLITS)
    part = ef.shard_frame(sel, "2/3")
    assert part.index.tolist() == SELECTED_LABELS[6:9] == [11, 12, 13]
    # Not vacuous: a reset index would have produced range(len(part)) instead.
    assert part.index.tolist() != list(range(len(part)))


@pytest.mark.parametrize("bad", ["3/3", "5/3", "-1/5", "1", "a/b", "1/0",
                                 "1/2/3", "1/-2", "", "/5", "5/"])
def test_shard_frame_rejects_a_malformed_or_out_of_range_spec(frozen, bad):
    _, df = frozen
    if bad == "":
        assert ef.shard_frame(df, bad) is df       # optional flag, not an error
        return
    with pytest.raises(ValueError, match="I/N"):
        ef.shard_frame(df, bad)


def test_no_shard_returns_the_frame_untouched(frozen):
    _, df = frozen
    assert ef.shard_frame(df, None) is df


def test_shard_blocks_agree_with_the_kaggle_bootstrap_partition(frozen):
    """`notebooks/run_shard.py` and this script cut shards of the SAME
    training bank and `merge_banks` glues them together, so the two block
    partitions must be identical -- not merely both contiguous.

    Two rules that each look correct in isolation are not interchangeable:
    "remainder to the first shards" and `np.linspace(0, n, k+1).astype(int)`
    (which `scripts/extract_eval_bank.py` uses, for the separate EVAL bank)
    put every boundary one row apart at 120,001 rows over 5 shards. A fleet
    mixing the two entry points would produce shards overlapping by one image
    and `merge_banks` would refuse the lot -- after five people had each paid
    for a session.
    """
    from aigcdet.features.bank import manifest_fingerprint
    from aigcdet.features.extract import shard_bounds

    kb = _kaggle_bootstrap()
    for n in list(range(0, 25)) + [120000, 120001, 137_842]:
        for k in (1, 2, 3, 5, 7):
            assert shard_bounds(n, k) == kb.shard_bounds(n, k), (n, k)

    # And end to end on a real frame, through both public entry points.
    _, df = frozen
    sel = ef.select_splits(df, WANTED_SPLITS)
    gate = kb.carried_gate(manifest_fingerprint(sel), "/unused")
    for i in range(3):
        assert (ef.shard_frame(sel, f"{i}/3").index.tolist()
                == kb.shard_frame(gate, sel, i, 3).index.tolist())

    # The linspace rule really is a different partition, or the check above
    # is asserting that two identical things are identical.
    linspace = np.linspace(0, 120001, 6).astype(int).tolist()
    assert [b[0] for b in shard_bounds(120001, 5)] != linspace[:5]


# --- order of operations ---------------------------------------------------

def test_the_split_filter_runs_before_the_shard(tmp_path, frozen, monkeypatch):
    """`--split` then `--shard`, so the N shards partition the rows that will
    actually be extracted. Sharding the whole manifest first would hand shard
    0 nothing but `train` and shard 4 nothing but `benchmark`.

    Driven through the CLI, because the order lives in `main()`. With this
    fixture the two orders are observably different: filter-first yields
    3/3/3 over the nine selected rows; shard-first makes shard 0 all-`train`
    (which `select_splits` rejects outright, mid-run) and shard 1 the
    `benchmark` block, which filters down to nothing.
    """
    manifest, df = frozen
    got = []
    for i in range(3):
        out = str(tmp_path / f"sf{i}")
        _run(_common(manifest) + ["--out", out, "--shard", f"{i}/3"], monkeypatch)
        got.append(FeatureBank.open(out).row_ids.tolist())

    assert got == [[0, 1, 2], [3, 4, 10], [11, 12, 13]]
    assert [len(g) for g in got] == [3, 3, 3]
    assert sum(got, []) == SELECTED_LABELS

    # The other order, stated explicitly, so this test could actually fail if
    # the two steps were swapped rather than merely describing the good one.
    with pytest.raises(ValueError, match="does not contain"):
        ef.select_splits(ef.shard_frame(df, "0/3"), WANTED_SPLITS)
    assert ef.shard_frame(df, "1/3")["split"].unique().tolist() == ["benchmark"]


def test_limit_truncates_before_the_shard_so_the_shards_still_tile(
        tmp_path, frozen, monkeypatch):
    """`--limit` is a prefix of the SELECTION, and the shards then tile that
    prefix. Applied after the shard instead, each of the three shards would
    keep its own first 7 rows and the union would be all 9 selected rows --
    neither the first 7 nor a contiguous block.
    """
    manifest, df = frozen
    ids: list[int] = []
    for i in range(3):
        out = str(tmp_path / f"lim{i}")
        _run(_common(manifest) + ["--out", out, "--limit", "7",
                                  "--shard", f"{i}/3"], monkeypatch)
        ids += FeatureBank.open(out).row_ids.tolist()

    assert ids == SELECTED_LABELS[:7]
    assert len(ids) == 7 != len(SELECTED_LABELS)   # the inverted order gives 9


def test_an_empty_shard_is_refused_rather_than_written(tmp_path, frozen,
                                                      monkeypatch):
    """A zero-row bank merges silently and contributes nothing, so the session
    that would have produced it must not start."""
    manifest, _ = frozen
    with pytest.raises(SystemExit):
        _run(_common(manifest) + ["--out", str(tmp_path / "empty"),
                                  "--shard", "12/13"], monkeypatch)


# --- pixels: what the index labels are FOR ---------------------------------

def _extract(manifest, out, monkeypatch, *extra):
    _run(_common(manifest) + ["--out", out, *extra], monkeypatch)
    return FeatureBank.open(out)


def test_a_shards_pixels_are_the_pixels_the_whole_run_would_have_drawn(
        tmp_path, frozen, monkeypatch):
    """The test that can tell `df.iloc[a:b]` from `df.iloc[a:b].reset_index()`.

    Both hold the same rows in the same order and differ only in the index
    label -- which is the RNG key every augmented view is drawn from. So this
    compares the CACHED FEATURES of the images the two banks share: under a
    reset index a shard's views 1..K are drawn from different keys and its
    pixels, and therefore its embeddings, differ.
    """
    manifest, _ = frozen
    whole = _extract(manifest, str(tmp_path / "whole"), monkeypatch)
    by_path = {p: i for i, p in enumerate(whole.meta["path"])}

    for i in range(3):
        part = _extract(manifest, str(tmp_path / f"p{i}"), monkeypatch,
                        "--shard", f"{i}/3")
        assert len(part.meta) > 0
        for j, path in enumerate(part.meta["path"]):
            np.testing.assert_array_equal(
                np.asarray(part.feats[j]),
                np.asarray(whole.feats[by_path[path]]))

    # Not vacuous: the augmented views really are key-dependent, so a wrong
    # row_id would have changed the numbers just compared.
    assert not np.array_equal(np.asarray(whole.feats[:, 0, :]),
                              np.asarray(whole.feats[:, 1, :]))
    assert len(set(np.asarray(whole.feats[:, 1, 0]).tolist())) > 1


def test_a_reset_index_in_the_shard_would_have_broken_that(
        tmp_path, frozen, monkeypatch):
    """The guard on the guard: prove the pixel comparison above is capable of
    failing, so a `reset_index()` in `shard_frame` is a killable mutation
    rather than an invisible one."""
    from aigcdet.features.extract import extract_bank

    _, df = frozen
    sel = ef.select_splits(df, WANTED_SPLITS)
    part = ef.shard_frame(sel, "2/3")
    assert part.index.tolist() == [11, 12, 13]

    _fake_backbone(monkeypatch)
    good = FeatureBank.open(extract_bank(part, "fake", str(tmp_path / "good"),
                                         seed=20260827, device="cpu"))
    bad = FeatureBank.open(extract_bank(part.reset_index(drop=True), "fake",
                                        str(tmp_path / "bad"), seed=20260827,
                                        device="cpu"))

    assert list(good.meta["path"]) == list(bad.meta["path"])      # same images
    assert good.row_ids.tolist() != bad.row_ids.tolist()          # different keys
    assert not np.array_equal(np.asarray(good.feats[:, 1:, :]),
                              np.asarray(bad.feats[:, 1:, :]))
    # ...and only the augmented views moved; view 0 is the clean one and is
    # key-independent, which is what makes the comparison specific.
    np.testing.assert_array_equal(np.asarray(good.feats[:, 0, :]),
                                  np.asarray(bad.feats[:, 0, :]))


# --- end to end: shards merge into the bank one run would have written ------

def test_shards_merge_into_the_bank_the_unsharded_run_would_have_written(
        tmp_path, frozen, monkeypatch):
    manifest, df = frozen
    whole = _extract(manifest, str(tmp_path / "whole"), monkeypatch)

    parts = []
    for i in range(3):
        p = str(tmp_path / f"shard{i}")
        _extract(manifest, p, monkeypatch, "--shard", f"{i}/3")
        parts.append(p)
    merged = FeatureBank.open(merge_banks(parts, str(tmp_path / "merged")))

    assert merged.row_ids.tolist() == whole.row_ids.tolist() == SELECTED_LABELS
    assert merged.meta["path"].tolist() == whole.meta["path"].tolist()
    np.testing.assert_array_equal(np.asarray(merged.feats),
                                  np.asarray(whole.feats))
    np.testing.assert_array_equal(np.asarray(merged.proxies),
                                  np.asarray(whole.proxies))
    np.testing.assert_array_equal(np.asarray(merged.presence),
                                  np.asarray(whole.presence))
    for i in range(len(SELECTED_LABELS)):
        for v in range(N_VIEWS):
            assert merged.recipe_json(i, v) == whole.recipe_json(i, v)
    # The merged fingerprint is taken over the CONCATENATED rel_paths, so this
    # is what a strided or reordered partition breaks.
    assert merged.config["manifest_sha256"] == whole.config["manifest_sha256"]
    merged.verify_against_manifest(ef.select_splits(df, WANTED_SPLITS))


def test_workers_still_produce_a_bit_identical_shard(tmp_path, frozen,
                                                     monkeypatch):
    """`--workers` predates `--shard`; the CPU pool must still be
    bit-identical to the inline path now that the frame it is handed is a
    slice rather than the whole manifest."""
    manifest, _ = frozen
    serial = _extract(manifest, str(tmp_path / "ser"), monkeypatch,
                      "--shard", "1/3", "--workers", "0")
    parallel = _extract(manifest, str(tmp_path / "par"), monkeypatch,
                        "--shard", "1/3", "--workers", "2")

    assert serial.row_ids.tolist() == parallel.row_ids.tolist() == [3, 4, 10]
    np.testing.assert_array_equal(np.asarray(parallel.feats),
                                  np.asarray(serial.feats))
    np.testing.assert_array_equal(np.asarray(parallel.proxies),
                                  np.asarray(serial.proxies))
    for i in range(3):
        for v in range(N_VIEWS):
            assert parallel.recipe_json(i, v) == serial.recipe_json(i, v)


# --- the docstring is the command a human copies onto Kaggle ---------------

def test_the_docstring_documents_the_sharding_it_now_actually_has():
    """It promised sharding for months while the CLI had no way to express
    it. Pin the promise to the flag, and to the two properties that make the
    promise true."""
    doc = ef.__doc__
    assert "--shard I/N" in doc
    assert "--shard 0/5" in doc
    assert "merge_banks.py" in doc
    assert "read_manifest -> --split -> --limit -> --shard" in doc
    assert "CONTIGUOUS" in doc and "INDEX LABELS" in doc
    # The flag the docstring documents must exist and be spelled that way.
    assert "--shard" in {a.option_strings[0]
                         for a in ef.build_parser()._actions if a.option_strings}
