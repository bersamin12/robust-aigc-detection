"""The combined manifest the ablation-tier eval bank is extracted from.

The ablation tier spans `val_internal`, `heldout_generator` and `benchmark`
(`extract_eval_bank.TIERS`), but those live in two separate frozen manifests
and `--manifest` takes one. Nothing joined them, so the tier could not be
extracted at all -- the same shape as the three missing producers already found
in this project.

Joining them is not a concatenation. Two things break silently first:

* **Two dataset roots.** The training tree is `data/normalized`, the benchmark
  tree is `data/demo`. `dataset_root` refuses a frame implying both, so the
  combined frame has to be re-rooted onto their common ancestor -- which
  rewrites every `rel_path`, i.e. every row's portable identity.
* **Colliding index labels.** Both manifests are indexed 0..N. The index label
  is the per-view RNG key, so a naive concat gives two different images the
  same key. Freezing through `write_manifest` (which writes `index=False`)
  is what makes the labels unique, and it must happen exactly once, here.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from aigcdet.data.manifest import dataset_root, read_manifest, write_manifest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = str(REPO / "scripts" / "build_eval_manifest.py")


@pytest.fixture()
def bem():
    spec = importlib.util.spec_from_file_location("build_eval_manifest", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree(root: pathlib.Path, rows: list[tuple[str, str]], rng) -> pd.DataFrame:
    """`rows` is [(split, subdir)]; one small PNG per row."""
    recs = []
    for i, (split, sub) in enumerate(rows):
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{i:05d}.png"
        Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)).save(p)
        recs.append({"path": str(p), "label": i % 2, "generator": f"g{i % 3}",
                     "source": sub, "licence": "L", "width": 8, "height": 8,
                     "split": split})
    return pd.DataFrame(recs)


@pytest.fixture()
def two_manifests(tmp_path):
    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    train_df = _tree(data / "normalized",
                     [("train", "sid_set")] * 4
                     + [("val_internal", "sid_set")] * 3
                     + [("heldout_generator", "wildfake")] * 2, rng)
    bench_df = _tree(data / "demo",
                     [("benchmark", "coco_val2017")] * 3
                     + [("benchmark", "dalle_advanced")] * 2, rng)
    tm = str(tmp_path / "train_manifest.parquet")
    bm = str(tmp_path / "bench_manifest.parquet")
    write_manifest(train_df, tm, root=str(data / "normalized"))
    write_manifest(bench_df, bm, root=str(data / "demo"))
    return tm, bm, str(data)


# --- what it produces -------------------------------------------------------

def test_combines_only_the_eval_splits_from_both_manifests(bem, tmp_path, two_manifests):
    tm, bm, _ = two_manifests
    out = str(tmp_path / "eval_manifest.parquet")
    bem.main(["--manifest", tm, "--benchmark-manifest", bm, "--out", out])

    df = read_manifest(out)
    assert df["split"].value_counts().to_dict() == {
        "benchmark": 5, "val_internal": 3, "heldout_generator": 2}
    assert "train" not in set(df["split"])


def test_the_result_has_one_dataset_root(bem, tmp_path, two_manifests):
    """`dataset_root` raises on a frame implying two trees, and
    `extract_eval_bank` calls it. A plain concat of the two manifests would
    die there -- after the backbone had been loaded."""
    tm, bm, data = two_manifests
    out = str(tmp_path / "eval_manifest.parquet")
    bem.main(["--manifest", tm, "--benchmark-manifest", bm, "--out", out])

    df = read_manifest(out)
    assert dataset_root(df) == data
    tops = {r.split("/")[0] for r in df["rel_path"]}
    assert tops == {"normalized", "demo"}


def test_index_labels_are_unique(bem, tmp_path, two_manifests):
    """Both sources are indexed 0..N. The index label keys every view's RNG,
    so colliding labels would give two different images identical pixels --
    silently, and only in the shard that happened to hold both."""
    tm, bm, _ = two_manifests
    out = str(tmp_path / "eval_manifest.parquet")
    bem.main(["--manifest", tm, "--benchmark-manifest", bm, "--out", out])

    df = read_manifest(out)
    assert df.index.is_unique
    assert df.index.tolist() == list(range(len(df)))


def test_row_order_is_identical_across_runs(bem, tmp_path, two_manifests):
    """Row order IS the key space. A manifest built twice in two orders gives
    two banks that look identical and are not interchangeable."""
    tm, bm, _ = two_manifests
    outs = []
    for i in range(2):
        o = str(tmp_path / f"m{i}.parquet")
        bem.main(["--manifest", tm, "--benchmark-manifest", bm, "--out", o])
        outs.append(read_manifest(o)["rel_path"].tolist())
    assert outs[0] == outs[1]


def test_the_frozen_digests_carry_through(bem, tmp_path, two_manifests):
    tm, bm, _ = two_manifests
    out = str(tmp_path / "eval_manifest.parquet")
    bem.main(["--manifest", tm, "--benchmark-manifest", bm, "--out", out])

    df = read_manifest(out)
    assert (df["content_sha256"].str.len() == 64).all()
    src = pd.concat([read_manifest(tm), read_manifest(bm)])
    by_name = dict(zip(src["path"], src["content_sha256"]))
    for path, digest in zip(df["path"], df["content_sha256"]):
        assert digest == by_name[path]


# --- what it refuses --------------------------------------------------------

def test_a_file_that_changed_since_its_manifest_was_frozen_is_reported(
        bem, tmp_path, two_manifests):
    """Re-digesting is the point of doing this work here rather than copying
    the columns across: the eval bank is about to be extracted from these
    exact files, and a file that no longer matches what was frozen produces
    features that do not correspond to its recorded label."""
    tm, bm, _ = two_manifests
    victim = read_manifest(bm)["path"].iloc[0]
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(victim)

    with pytest.raises(SystemExit, match="differ|digest"):
        bem.main(["--manifest", tm, "--benchmark-manifest", bm,
                  "--out", str(tmp_path / "m.parquet")])


def test_refuses_when_a_requested_split_is_absent(bem, tmp_path, two_manifests):
    tm, bm, _ = two_manifests
    with pytest.raises(SystemExit, match="nonesuch"):
        bem.main(["--manifest", tm, "--benchmark-manifest", bm,
                  "--out", str(tmp_path / "m.parquet"),
                  "--splits", "val_internal,nonesuch"])


def test_refuses_a_benchmark_manifest_holding_non_benchmark_rows(
        bem, tmp_path, two_manifests):
    """The guard against being handed the training manifest twice, which
    would double-count val_internal and quietly halve the eval bank's
    effective sample size."""
    tm, bm, _ = two_manifests
    with pytest.raises(SystemExit, match="benchmark"):
        bem.main(["--manifest", tm, "--benchmark-manifest", tm,
                  "--out", str(tmp_path / "m.parquet")])


def test_refuses_when_an_image_appears_in_both_manifests(bem, tmp_path, two_manifests):
    """COCO val2017 is in the benchmark and must never be in training. If a
    row ever appears on both sides, the eval bank would score the same image
    twice under two splits."""
    tm, bm, _ = two_manifests
    train = read_manifest(tm)
    bench = read_manifest(bm)
    leaked = pd.concat([train, bench.iloc[:1].assign(split="val_internal")],
                       ignore_index=True)
    tm2 = str(tmp_path / "leaky.parquet")
    leaked.to_parquet(tm2, index=False)
    with pytest.raises(SystemExit, match="both"):
        bem.main(["--manifest", tm2, "--benchmark-manifest", bm,
                  "--out", str(tmp_path / "m.parquet")])


# --- the lineage holdout ----------------------------------------------------

def test_extra_generators_are_promoted_from_train_to_heldout(bem, tmp_path,
                                                             two_manifests):
    """A lineage holdout needs the siblings on BOTH sides.

    The manifest's pinned pair holds out `SDwithAdaptor_controlnet` while
    `_lora` and `_lycris` stay in training -- same decoder, different adapter,
    so "unseen generator" is an easier question than it sounds. Asking the
    harder one means excluding the siblings from training AND having them in
    the eval manifest to score. They sit in `train`, so nothing puts them here
    by default and this flag is the only way in.
    """
    tm, bm, _ = two_manifests
    out = str(tmp_path / "e.parquet")
    bem.main(["--manifest", tm, "--benchmark-manifest", bm, "--out", out,
              "--extra-heldout-generators", "g0"])

    df = read_manifest(out)
    counts = df["split"].value_counts().to_dict()
    # g0 is the only family in `train` that is not already held out: train rows
    # are g0,g1,g2,g0 and the heldout rows are g1,g2.
    assert counts["heldout_generator"] == 4      # 2 already + 2 promoted
    assert "train" not in set(df["split"])
    promoted = df[(df["split"] == "heldout_generator") & (df["generator"] == "g0")]
    assert len(promoted) == 2


def test_promoted_rows_keep_the_manifest_row_order(bem, tmp_path, two_manifests):
    """Row order IS the key space -- the index label keys every view's RNG. A
    concat of two selections would append the promoted rows after the ones
    they precede in the manifest, so the same corpus built with and without
    the flag would disagree about which pixels a row has."""
    tm, bm, _ = two_manifests
    out = str(tmp_path / "e.parquet")
    bem.main(["--manifest", tm, "--benchmark-manifest", bm, "--out", out,
              "--extra-heldout-generators", "g0"])

    df = read_manifest(out)
    # Train-manifest row 0 (a `train` row of g0) precedes rows 4-8, so the
    # promoted row must be FIRST, not appended after them.
    non_bench = df[~df["rel_path"].str.startswith("demo/")]
    assert non_bench.iloc[0]["generator"] == "g0"
    assert non_bench.iloc[0]["split"] == "heldout_generator"


def test_a_family_that_matches_no_train_row_raises(bem, tmp_path, two_manifests):
    """It would contribute zero rows and the number would still be reported as
    a lineage-holdout score."""
    tm, bm, _ = two_manifests
    with pytest.raises(SystemExit, match="matches nothing|no `train` row"):
        bem.main(["--manifest", tm, "--benchmark-manifest", bm,
                  "--out", str(tmp_path / "e.parquet"),
                  "--extra-heldout-generators", "not_a_family"])


def test_a_family_already_held_out_raises(bem, tmp_path, two_manifests):
    """The fixture's heldout_generator rows carry g1 and g2. Naming one here
    would claim the two manifests disagree about its split when they agree."""
    tm, bm, _ = two_manifests
    held = set(read_manifest(tm).query("split == 'heldout_generator'")["generator"])
    name = sorted(held)[0]
    with pytest.raises(SystemExit, match="ALREADY holds out"):
        bem.main(["--manifest", tm, "--benchmark-manifest", bm,
                  "--out", str(tmp_path / "e.parquet"),
                  "--extra-heldout-generators", name])


def test_without_the_flag_nothing_changes(bem, tmp_path, two_manifests):
    """The default must be byte-identical to before, or every manifest already
    frozen would need re-freezing."""
    tm, bm, _ = two_manifests
    a_out = str(tmp_path / "a.parquet")
    bem.main(["--manifest", tm, "--benchmark-manifest", bm, "--out", a_out])
    df = read_manifest(a_out)
    assert df["split"].value_counts().to_dict() == {
        "benchmark": 5, "val_internal": 3, "heldout_generator": 2}
