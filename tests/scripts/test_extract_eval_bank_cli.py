"""The eval-bank producer: the tier, the subsample, and what makes them safe.

`run_ablation.py --eval-bank` documented a bank "written by
`eval.grid.extract_eval_bank`" that no code path could produce, and
`docs/robustness_table.md` is presented as an artefact built from it. These
tests pin the four properties the missing producer has to have, each of which
fails silently rather than loudly if it is wrong:

1. The §4.4a subsample runs on the MANIFEST, before extraction, and only over
   the split it is budgeted for. Subsampling the whole frame starves the
   held-out generator families, and `errors.heldout_robust_tpr` then refuses
   the rung.
2. It is applied as `df.iloc[idx]` with NO `reset_index()`. The index label is
   the per-view RNG key, so a reset makes a subsampled bank's pixels differ
   from the full run's and makes two shards of one subsample disagree. That is
   asserted over PIXELS, not over index labels alone.
3. The bank records which subsample was taken. `manifest_fingerprint` already
   makes a subsampled bank incomparable with a full one -- correctly -- but
   mutely.
4. The bank is portable: a shard extracted from a Kaggle mount merges with one
   extracted elsewhere and still verifies against the frozen manifest.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import shutil

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from aigcdet.augment.scenarios import CORE_CONDITIONS, EVAL_GRID
from aigcdet.data.manifest import MANIFEST_COLUMNS, read_manifest, write_manifest
from aigcdet.features.bank import FeatureBank, manifest_fingerprint, merge_banks

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = _ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eb = _load_script("extract_eval_bank")

#: Two conditions, one of which consumes randomness. `noise_s0.05` is what
#: makes the RNG-key tests non-vacuous: every other condition in the grid is a
#: deterministic function of the pixels, so a wrong row_id would be invisible.
TWO_CONDITIONS = {"clean": EVAL_GRID["clean"],
                  "noise_s0.05": EVAL_GRID["noise_s0.05"]}


# --- fixtures --------------------------------------------------------------

def _tree(root: str) -> pd.DataFrame:
    """A dataset on disk whose splits and strata exercise the real cap.

    Row layout, in manifest order -- deliberately NOT benchmark-first, so a
    subsample that kept `df.iloc[:n]` or that reset the index is visible in
    the index labels alone:

        0..5    train              (must never reach the bank)
        6..11   val_internal       3 authentic, 3 fake over two generators
        12..17  heldout_generator  6 fakes of the two held-out families
        18..57  benchmark          20 authentic, 20 fake over two generators
    """
    rng = np.random.default_rng(0)
    rows = []

    def add(n, split, label_of, gen_of, source):
        for k in range(n):
            label = label_of(k)
            gen = gen_of(k)
            bucket = f"{split}/{gen or 'real'}"
            p = os.path.abspath(os.path.join(root, bucket, f"{split}_{k:03d}.png"))
            os.makedirs(os.path.dirname(p), exist_ok=True)
            arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(p)
            rows.append({"path": p, "label": label, "generator": gen,
                         "source": source, "licence": "CC0", "width": 64,
                         "height": 64, "split": split})

    add(6, "train", lambda k: k % 2, lambda k: "" if k % 2 == 0 else "trg", "wf")
    add(6, "val_internal", lambda k: k % 2,
        lambda k: "" if k % 2 == 0 else ("vg1" if k < 3 else "vg2"), "wf")
    add(6, "heldout_generator", lambda k: 1,
        lambda k: "hg1" if k < 3 else "hg2", "wf")
    add(40, "benchmark", lambda k: k % 2,
        lambda k: "" if k % 2 == 0 else ("bg1" if k < 20 else "bg2"), "bench")
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def _frozen(tmp_path, name="normalized"):
    """(root, manifest path, frozen manifest frame) -- with `rel_path`."""
    root = str(tmp_path / name)
    write_manifest(_tree(root), str(tmp_path / f"{name}.parquet"))
    m = str(tmp_path / f"{name}.parquet")
    return root, m, read_manifest(m)


def _fake_backbone(monkeypatch, dim=4, record=None):
    """No GPU, no weights, and an embedding that is a function of the PIXELS.

    `sum % 2003` stays under float16's exactly-representable range, so a bank
    round-trip can be compared with equality: two runs whose cached feats are
    equal really did see the same pixels.
    """
    from aigcdet.eval import grid
    from aigcdet.features.backbones import BackboneSpec

    spec = BackboneSpec("fake", "none", 64, dim, 1, 0)
    monkeypatch.setattr(grid, "load_backbone", lambda n, device: (None, spec))

    def _embed(m, s, imgs, device, batch_size=16):
        if record is not None:
            record.append([np.asarray(v).copy() for v in imgs])
        return np.stack([np.full(s.dim, float(np.asarray(v, np.float64).sum() % 2003),
                                 np.float32) for v in imgs])

    monkeypatch.setattr(grid, "embed", _embed)
    return grid


def _run(argv, monkeypatch, record=None):
    _fake_backbone(monkeypatch, record=record)
    return eb.main(argv)


# --- tier -> condition axis ------------------------------------------------

def test_the_tier_decides_the_condition_axis():
    assert list(eb.resolve_conditions("ablation")) == list(EVAL_GRID)
    assert list(eb.resolve_conditions("final_report")) == list(CORE_CONDITIONS)
    # The two tiers are genuinely different axes, or this test proves nothing.
    assert list(EVAL_GRID) != list(CORE_CONDITIONS)


def test_smoke_tier_must_name_its_conditions():
    """`TIER_CONDITIONS['smoke']` is None ('any subset'), so there is nothing
    to default to and guessing one is how a three-condition number reaches a
    results table."""
    with pytest.raises(ValueError, match="no fixed condition coverage"):
        eb.resolve_conditions("smoke")
    got = eb.resolve_conditions("smoke", "clean,noise_s0.05")
    assert list(got) == ["clean", "noise_s0.05"]


def test_conditions_that_disagree_with_the_tier_are_refused_before_extraction():
    """report._check_banks compares the bank's condition list against the
    tier's coverage for LIST equality, so a bank over a subset could never be
    rendered as that tier -- which is discoverable now, or after the GPU."""
    subset = ",".join(list(EVAL_GRID)[:5])
    with pytest.raises(ValueError, match="could never be rendered"):
        eb.resolve_conditions("ablation", subset)
    reordered = ",".join([list(EVAL_GRID)[1], *list(EVAL_GRID)[2:], "clean"])
    with pytest.raises(ValueError, match="could never be rendered"):
        eb.resolve_conditions("ablation", reordered)


def test_unknown_condition_names_are_refused():
    with pytest.raises(ValueError, match="not in the evaluation"):
        eb.resolve_conditions("smoke", "clean,jpeg_q99")


# --- split selection -------------------------------------------------------

def test_the_ablation_tier_carries_both_selection_splits(tmp_path):
    """§6.4's population is val_internal authentic vs heldout_generator fakes
    (`errors.SELECTION_SPLITS`); run_ablation refuses a bank missing either."""
    from aigcdet.eval.errors import SELECTION_SPLITS

    _, _, df = _frozen(tmp_path)
    kept = eb.select_splits(df, ",".join(eb.TIER_PLANS["ablation"].splits))
    present = set(kept["split"])
    assert set(SELECTION_SPLITS) <= present
    assert "train" not in present
    assert "benchmark" in present


def test_the_final_report_tier_is_the_benchmark_alone(tmp_path):
    _, _, df = _frozen(tmp_path)
    kept = eb.select_splits(df, ",".join(eb.TIER_PLANS["final_report"].splits))
    assert set(kept["split"]) == {"benchmark"}


def test_split_selection_keeps_the_frozen_manifest_index_labels(tmp_path):
    _, _, df = _frozen(tmp_path)
    kept = eb.select_splits(df, "heldout_generator")
    assert kept.index.tolist() == list(range(12, 18))


def test_unknown_split_name_fails_before_any_extraction(tmp_path):
    _, _, df = _frozen(tmp_path)
    with pytest.raises(ValueError, match="does not contain"):
        eb.select_splits(df, "benchmark,val")


# --- the subsample ---------------------------------------------------------

def test_the_budget_caps_only_the_split_it_names(tmp_path):
    """Not the whole frame. A single `stratified_subsample` over everything
    balances the held-out families against the benchmark's strata and takes
    most of them away -- and `heldout_robust_tpr` then refuses the rung for a
    condition with only one class left."""
    _, _, df = _frozen(tmp_path)
    selected = eb.select_splits(df, "val_internal,heldout_generator,benchmark")
    out, kept = eb.subsample_manifest(selected, {"benchmark": 10}, seed=7)

    counts = out["split"].value_counts().to_dict()
    assert counts["benchmark"] == 10
    assert kept == {"benchmark": 10}
    # Untouched, down to the row: these are the §6.4 population.
    assert counts["val_internal"] == 6
    assert counts["heldout_generator"] == 6
    assert out[out["split"] == "heldout_generator"].index.tolist() == list(range(12, 18))


def test_the_benchmark_cap_stays_stratified(tmp_path):
    _, _, df = _frozen(tmp_path)
    bench = df[df["split"] == "benchmark"]
    out, _ = eb.subsample_manifest(bench, {"benchmark": 8}, seed=7)
    assert len(out) == 8
    assert (out["label"] == 1).sum() == 4          # class balance preserved
    assert set(out["generator"]) == {"", "bg1", "bg2"}


def test_the_subsample_is_reproducible_from_the_seed_and_seed_sensitive(tmp_path):
    _, _, df = _frozen(tmp_path)
    bench = df[df["split"] == "benchmark"]
    a, _ = eb.subsample_manifest(bench, {"benchmark": 8}, seed=7)
    b, _ = eb.subsample_manifest(bench, {"benchmark": 8}, seed=7)
    c, _ = eb.subsample_manifest(bench, {"benchmark": 8}, seed=8)
    assert a.index.tolist() == b.index.tolist()
    assert a.index.tolist() != c.index.tolist()


def test_the_subsample_keeps_the_manifests_row_order(tmp_path):
    """The bank is aligned to the manifest positionally and its fingerprint is
    order-sensitive, so a subsample that emitted its rows grouped by split
    would produce a different bank from the same seed.

    The frame here INTERLEAVES the budgeted split with the un-budgeted one on
    purpose. With every benchmark row after every internal row -- which is how
    a real manifest and the rest of this file's fixture are laid out --
    "concatenate the kept blocks" and "sort" are the same list, and the
    property is unfalsifiable.
    """
    n = 24
    interleaved = pd.DataFrame({
        "path": [f"/p{i}.png" for i in range(n)],
        "label": [i % 2 for i in range(n)],
        "generator": ["" if i % 2 == 0 else f"g{i % 4}" for i in range(n)],
        "source": ["s"] * n,
        "split": ["benchmark" if i % 2 == 0 else "val_internal" for i in range(n)],
    })
    out, _ = eb.subsample_manifest(interleaved, {"benchmark": 6}, seed=7)

    assert out.index.tolist() == sorted(out.index.tolist())
    assert list(out["split"]) != sorted(out["split"])      # really interleaved
    assert len(out) == 6 + 12

    # ...and the same property on the file's own fixture, for the real shape.
    _, _, df = _frozen(tmp_path)
    selected = eb.select_splits(df, "val_internal,heldout_generator,benchmark")
    picked, _ = eb.subsample_manifest(selected, {"benchmark": 10}, seed=7)
    assert picked.index.tolist() == [i for i in selected.index
                                     if i in set(picked.index)]


def test_the_subsample_does_not_reset_the_index(tmp_path):
    """The index label is the per-view RNG key. This is the cheap half of the
    check; `test_a_subsampled_bank_reproduces_the_full_runs_pixels` is the
    half that a `.reset_index(drop=True)` cannot survive."""
    _, _, df = _frozen(tmp_path)
    bench = df[df["split"] == "benchmark"]
    out, _ = eb.subsample_manifest(bench, {"benchmark": 8}, seed=7)
    assert out.index.min() >= 18                       # never renumbered from 0
    assert out.index.tolist() != list(range(len(out)))
    assert set(out.index) <= set(bench.index)


def test_a_budget_that_matches_no_row_is_a_typo_not_a_no_op(tmp_path):
    _, _, df = _frozen(tmp_path)
    bench = df[df["split"] == "benchmark"]
    with pytest.raises(ValueError, match="silently caps nothing"):
        eb.subsample_manifest(bench, {"val_internal": 4}, seed=7)


def test_no_budgets_returns_the_frame_untouched(tmp_path):
    _, _, df = _frozen(tmp_path)
    out, kept = eb.subsample_manifest(df, {}, seed=7)
    assert out is df and kept == {}


def test_parse_budgets_reads_split_equals_n():
    assert eb.parse_budgets(["benchmark=5000"]) == {"benchmark": 5000}
    assert eb.parse_budgets(None) == {}
    with pytest.raises(ValueError, match="SPLIT=N"):
        eb.parse_budgets(["benchmark"])
    with pytest.raises(ValueError, match="must be an integer"):
        eb.parse_budgets(["benchmark=all"])
    with pytest.raises(ValueError, match="must be positive"):
        eb.parse_budgets(["benchmark=0"])
    with pytest.raises(ValueError, match="not a manifest"):
        eb.parse_budgets(["bench=10"])


def test_the_ablation_tier_caps_the_benchmark_and_nothing_else():
    """§4.4a's cap, as a fact about the shipped default rather than a flag a
    human has to remember."""
    assert eb.TIER_PLANS["ablation"].subsample == {"benchmark": 5000}
    assert eb.TIER_PLANS["final_report"].subsample == {}


# --- the property the index labels exist for -------------------------------

def test_a_subsampled_bank_reproduces_the_full_runs_pixels(tmp_path, monkeypatch):
    """The test that can tell `df.iloc[idx]` from `df.iloc[idx].reset_index()`.

    Both look identical in every column; they differ only in the index label,
    which is the RNG key every noise view is drawn from. So compare the CACHED
    FEATURES of the images the two banks share: under a reset index the
    subsampled bank's `noise_s0.05` view is drawn from a different key and its
    pixels -- and therefore its embedding -- differ.
    """
    from aigcdet.eval.grid import extract_eval_bank

    _, _, df = _frozen(tmp_path)
    bench = df[df["split"] == "benchmark"]
    sub, _ = eb.subsample_manifest(bench, {"benchmark": 8}, seed=7)

    _fake_backbone(monkeypatch)
    full_bank = FeatureBank.open(extract_eval_bank(
        bench, "fake", str(tmp_path / "full"), conditions=TWO_CONDITIONS,
        device="cpu"))
    sub_bank = FeatureBank.open(extract_eval_bank(
        sub, "fake", str(tmp_path / "sub"), conditions=TWO_CONDITIONS,
        device="cpu"))

    full_by_id = {int(r): i for i, r in enumerate(full_bank.row_ids)}
    assert len(sub_bank.row_ids) == 8
    for i, row_id in enumerate(sub_bank.row_ids):
        j = full_by_id[int(row_id)]
        np.testing.assert_array_equal(np.asarray(sub_bank.feats[i]),
                                      np.asarray(full_bank.feats[j]))

    # Not vacuous: the noise view really is key-dependent, so a wrong row_id
    # would have changed the number just compared.
    noise = np.asarray(full_bank.feats[:, 1, 0])
    assert len(set(noise.tolist())) > 1
    assert not np.array_equal(np.asarray(full_bank.feats[:, 0, :]),
                              np.asarray(full_bank.feats[:, 1, :]))


def test_a_reset_index_would_have_broken_that(tmp_path, monkeypatch):
    """The guard on the guard: prove the comparison above is capable of
    failing, so a `reset_index()` in `subsample_manifest` is a killable
    mutation rather than an invisible one."""
    from aigcdet.eval.grid import extract_eval_bank

    _, _, df = _frozen(tmp_path)
    bench = df[df["split"] == "benchmark"]
    sub, _ = eb.subsample_manifest(bench, {"benchmark": 8}, seed=7)

    _fake_backbone(monkeypatch)
    good = FeatureBank.open(extract_eval_bank(
        sub, "fake", str(tmp_path / "good"), conditions=TWO_CONDITIONS,
        device="cpu"))
    bad = FeatureBank.open(extract_eval_bank(
        sub.reset_index(drop=True), "fake", str(tmp_path / "bad"),
        conditions=TWO_CONDITIONS, device="cpu"))

    assert list(good.meta["path"]) == list(bad.meta["path"])      # same images
    assert good.row_ids.tolist() != bad.row_ids.tolist()          # different keys
    assert not np.array_equal(np.asarray(good.feats[:, 1, :]),
                              np.asarray(bad.feats[:, 1, :]))
    # ...and only the randomised view moved; the clean view is key-independent.
    np.testing.assert_array_equal(np.asarray(good.feats[:, 0, :]),
                                  np.asarray(bad.feats[:, 0, :]))


# --- what the bank records -------------------------------------------------

def test_the_bank_records_its_tier_and_its_subsample(tmp_path, monkeypatch):
    root, m, _ = _frozen(tmp_path)
    out = str(tmp_path / "eb_ab")
    _run(["--manifest", m, "--backbone", "fake", "--out", out,
          "--tier", "smoke", "--conditions", "clean,noise_s0.05",
          "--split", "benchmark", "--subsample", "benchmark=8",
          "--subsample-seed", "7", "--device", "cpu"], monkeypatch)

    cfg = FeatureBank.open(out).config
    assert cfg["tier"] == "smoke"
    assert cfg["subsample"] == {"seed": 7, "budgets": {"benchmark": 8},
                                "kept": {"benchmark": 8}}
    assert cfg["n_images"] == 8
    assert list(cfg["conditions"]) == ["clean", "noise_s0.05"]


def test_no_subsample_overrides_the_tiers_own_budget(tmp_path, monkeypatch):
    """The tier's budget is a default, and `--no-subsample` has to beat it.

    The tier is given a budget that actually bites: the shipped ablation cap
    is 5,000 rows, which on any fixture small enough to extract in a test is
    indistinguishable from no cap at all -- so a `--no-subsample` that did
    nothing would pass unnoticed.
    """
    root, m, _ = _frozen(tmp_path)
    monkeypatch.setitem(eb.TIER_PLANS, "smoke",
                        eb.TierPlan(splits=("benchmark",),
                                    subsample={"benchmark": 8}))
    common = ["--manifest", m, "--backbone", "fake", "--tier", "smoke",
              "--conditions", "clean,noise_s0.05", "--device", "cpu"]

    _run(common + ["--out", str(tmp_path / "eb_capped")], monkeypatch)
    capped = FeatureBank.open(str(tmp_path / "eb_capped")).config
    assert capped["subsample"]["budgets"] == {"benchmark": 8}
    assert capped["n_images"] == 8

    _run(common + ["--out", str(tmp_path / "eb_full"), "--no-subsample"],
         monkeypatch)
    cfg = FeatureBank.open(str(tmp_path / "eb_full")).config
    assert cfg["subsample"]["budgets"] == {} and cfg["subsample"]["kept"] == {}
    assert cfg["n_images"] == 40


def test_a_subsampled_bank_is_not_comparable_with_a_full_one(tmp_path, monkeypatch):
    """The refusal is `manifest_fingerprint`'s and it already worked; what is
    new is that the bank now says WHICH subsample made it different."""
    from aigcdet.eval.grid import assert_banks_comparable

    root, m, _ = _frozen(tmp_path)
    common = ["--manifest", m, "--backbone", "fake", "--tier", "smoke",
              "--conditions", "clean,noise_s0.05", "--split", "benchmark",
              "--device", "cpu"]
    _run(common + ["--out", str(tmp_path / "b_sub"),
                   "--subsample", "benchmark=8"], monkeypatch)
    _run(common + ["--out", str(tmp_path / "b_all"), "--no-subsample"], monkeypatch)

    sub = FeatureBank.open(str(tmp_path / "b_sub"))
    full = FeatureBank.open(str(tmp_path / "b_all"))
    assert sub.config["manifest_sha256"] != full.config["manifest_sha256"]
    assert sub.config["subsample"] != full.config["subsample"]
    with pytest.raises(ValueError, match="not comparable"):
        assert_banks_comparable([full, sub])


def test_a_bank_written_for_a_tier_passes_the_reports_bank_check(tmp_path,
                                                                 monkeypatch):
    """The end the whole script exists for: `report._check_banks` accepts a
    bank this script produced, for the tier it was produced as."""
    from aigcdet.eval.report import TIER_CONDITIONS, _check_banks

    root, m, _ = _frozen(tmp_path)
    out = str(tmp_path / "eb_final")
    _run(["--manifest", m, "--backbone", "fake", "--out", out,
          "--tier", "final_report", "--device", "cpu"], monkeypatch)

    bank = FeatureBank.open(out)
    # The tier decides the ROWS too, not only the conditions: `final_report` is
    # the external benchmark alone (§4.4a). Asserted against the fixture's own
    # counts rather than against the bank's self-report, which would match
    # itself whatever it held.
    assert set(bank.meta["split"]) == {"benchmark"}
    assert bank.config["n_images"] == 40
    scores = pd.DataFrame({"image_idx": np.arange(bank.config["n_images"])})
    _check_banks({"a3": bank}, {"a3": scores},
                 list(TIER_CONDITIONS["final_report"]))
    # ...and it is refused for the tier it is NOT.
    with pytest.raises(ValueError, match="do not belong to these scores"):
        _check_banks({"a3": bank}, {"a3": scores},
                     list(TIER_CONDITIONS["ablation"]))


def test_shard_and_limit_are_not_recorded_in_the_bank_config(tmp_path, monkeypatch):
    """They differ between shards of one bank, and `merge_banks` requires every
    unrecognised config key to agree -- so recording them would make the merge
    refuse exactly the shards it exists to combine."""
    root, m, _ = _frozen(tmp_path)
    out = str(tmp_path / "eb_shard")
    _run(["--manifest", m, "--backbone", "fake", "--out", out,
          "--tier", "smoke", "--conditions", "clean", "--split", "benchmark",
          "--shard", "1/4", "--device", "cpu"], monkeypatch)
    cfg = FeatureBank.open(out).config
    assert "shard" not in cfg and "limit" not in cfg
    assert cfg["n_images"] == 10


# --- sharding --------------------------------------------------------------

def test_shards_are_contiguous_disjoint_and_exhaustive(tmp_path):
    _, _, df = _frozen(tmp_path)
    seen: list[int] = []
    for i in range(5):
        part = eb.shard_frame(df, f"{i}/5")
        labels = part.index.tolist()
        assert labels == list(range(labels[0], labels[0] + len(labels)))
        seen += labels
    assert seen == df.index.tolist()


def test_shard_frame_preserves_index_labels(tmp_path):
    _, _, df = _frozen(tmp_path)
    part = eb.shard_frame(df, "2/4")
    assert part.index.tolist() != list(range(len(part)))
    assert part.index.tolist() == df.index.tolist()[len(df) // 2:3 * len(df) // 4]


def test_shard_frame_rejects_a_malformed_spec(tmp_path):
    _, _, df = _frozen(tmp_path)
    for bad in ("5/5", "-1/5", "1", "a/b", "1/0"):
        with pytest.raises(ValueError, match="I/N"):
            eb.shard_frame(df, bad)
    assert eb.shard_frame(df, None) is df


def test_shards_merge_into_the_bank_the_full_run_would_have_written(
        tmp_path, monkeypatch):
    root, m, _ = _frozen(tmp_path)
    common = ["--manifest", m, "--backbone", "fake", "--tier", "smoke",
              "--conditions", "clean,noise_s0.05", "--split", "benchmark",
              "--subsample", "benchmark=12", "--subsample-seed", "7",
              "--device", "cpu"]
    _run(common + ["--out", str(tmp_path / "whole")], monkeypatch)
    parts = []
    for i in range(3):
        p = str(tmp_path / f"part{i}")
        _run(common + ["--out", p, "--shard", f"{i}/3"], monkeypatch)
        parts.append(p)

    merged = FeatureBank.open(merge_banks(parts, str(tmp_path / "merged")))
    whole = FeatureBank.open(str(tmp_path / "whole"))
    assert merged.row_ids.tolist() == whole.row_ids.tolist()
    np.testing.assert_array_equal(np.asarray(merged.feats), np.asarray(whole.feats))
    assert merged.config["manifest_sha256"] == whole.config["manifest_sha256"]
    assert merged.config["subsample"] == whole.config["subsample"]


def test_shards_of_different_subsamples_refuse_to_merge(tmp_path, monkeypatch):
    """Two teammates who ran different `--subsample` produce disjoint row_ids
    and would otherwise merge into a bank of no stated tier at all."""
    root, m, _ = _frozen(tmp_path)
    common = ["--manifest", m, "--backbone", "fake", "--tier", "smoke",
              "--conditions", "clean", "--split", "benchmark", "--device", "cpu"]
    _run(common + ["--out", str(tmp_path / "s_a"), "--subsample", "benchmark=8",
                   "--shard", "0/2"], monkeypatch)
    _run(common + ["--out", str(tmp_path / "s_b"), "--subsample", "benchmark=12",
                   "--shard", "1/2"], monkeypatch)
    with pytest.raises(ValueError, match="not part of the same bank"):
        merge_banks([str(tmp_path / "s_a"), str(tmp_path / "s_b")],
                    str(tmp_path / "s_m"))


# --- portability -----------------------------------------------------------

def test_a_shard_extracted_from_another_mount_merges_and_verifies(
        tmp_path, monkeypatch):
    """The Kaggle case: one frozen manifest, the same images attached under two
    different roots, one shard extracted from each. The merged bank must
    fingerprint to what the FROZEN manifest fingerprints to."""
    from aigcdet.eval.grid import extract_eval_bank

    root_a, m, df = _frozen(tmp_path)
    root_b = str(tmp_path / "kaggle_mount")
    shutil.copytree(root_a, root_b)
    df_b = read_manifest(m, root=root_b)
    # The fixture is only meaningful if the second mount really is a second
    # mount: same rows, same identity, different absolute paths.
    assert list(df_b["path"]) != list(df["path"])
    assert list(df_b["rel_path"]) == list(df["rel_path"])

    _fake_backbone(monkeypatch)
    s0 = extract_eval_bank(df.iloc[:20], "fake", str(tmp_path / "p0"),
                           conditions=TWO_CONDITIONS, device="cpu")
    s1 = extract_eval_bank(df_b.iloc[20:], "fake", str(tmp_path / "p1"),
                           conditions=TWO_CONDITIONS, device="cpu")

    b0, b1 = FeatureBank.open(s0), FeatureBank.open(s1)
    assert b0.config["manifest_root"] == os.path.abspath(root_a)
    assert b1.config["manifest_root"] == os.path.abspath(root_b)
    assert all(not os.path.isabs(r) for r in b0.rel_paths + b1.rel_paths)

    merged = FeatureBank.open(merge_banks([s0, s1], str(tmp_path / "pm")))
    assert merged.config["manifest_sha256"] == manifest_fingerprint(df)
    merged.verify_against_manifest(df)                 # must not raise


def test_root_rebases_the_manifest_onto_this_machines_mount(tmp_path, monkeypatch):
    root_a, m, df = _frozen(tmp_path)
    root_b = str(tmp_path / "elsewhere")
    shutil.move(root_a, root_b)
    assert not os.path.exists(root_a)      # the old mount is really gone

    out = str(tmp_path / "eb_root")
    _run(["--manifest", m, "--backbone", "fake", "--out", out, "--root", root_b,
          "--tier", "smoke", "--conditions", "clean", "--split", "benchmark",
          "--device", "cpu"], monkeypatch)
    bank = FeatureBank.open(out)
    assert bank.config["manifest_root"] == os.path.abspath(root_b)
    assert bank.config["manifest_sha256"] == manifest_fingerprint(
        df[df["split"] == "benchmark"])


def test_the_data_root_env_var_is_the_fallback(tmp_path, monkeypatch):
    root_a, m, df = _frozen(tmp_path)
    root_b = str(tmp_path / "env_mount")
    shutil.move(root_a, root_b)
    monkeypatch.setenv("AIGCDET_DATA_ROOT", root_b)

    out = str(tmp_path / "eb_env")
    _run(["--manifest", m, "--backbone", "fake", "--out", out,
          "--tier", "smoke", "--conditions", "clean", "--split", "benchmark",
          "--device", "cpu"], monkeypatch)
    assert FeatureBank.open(out).config["manifest_root"] == os.path.abspath(root_b)


# --- the pre-flight --------------------------------------------------------

def test_a_wrong_mount_is_caught_before_a_single_forward(tmp_path, monkeypatch):
    """The Kaggle failure that actually happens: the Dataset is attached under
    another slug. It must cost a stat call, not an hour of GPU."""
    root, m, df = _frozen(tmp_path)
    os.remove(str(df.loc[20, "path"]))

    out = str(tmp_path / "eb_missing")
    with pytest.raises(ValueError, match="missing"):
        _run(["--manifest", m, "--backbone", "fake", "--out", out,
              "--tier", "smoke", "--conditions", "clean", "--split", "benchmark",
              "--device", "cpu"], monkeypatch)
    assert not os.path.exists(out)


def test_the_preflight_can_be_escalated_to_the_manifests_digests(tmp_path):
    root, m, df = _frozen(tmp_path)
    assert "content_sha256" in df.columns
    eb.preflight(df, digest="auto")                    # clean copy: no raise

    victim = str(df.loc[0, "path"])
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(victim)
    with pytest.raises(ValueError, match="divergent"):
        eb.preflight(df, digest="auto")
    # ...and the presence-only default cannot see it, which is why --verify exists.
    eb.preflight(df, digest=None)


def test_no_verify_skips_the_preflight(tmp_path, monkeypatch):
    root, m, df = _frozen(tmp_path)
    out = str(tmp_path / "eb_noverify")
    _run(["--manifest", m, "--backbone", "fake", "--out", out,
          "--tier", "smoke", "--conditions", "clean", "--split", "benchmark",
          "--no-verify", "--device", "cpu"], monkeypatch)
    assert FeatureBank.open(out).config["n_images"] == 40


# --- resume ----------------------------------------------------------------

def test_resume_continues_an_interrupted_extraction(tmp_path, monkeypatch):
    """Kaggle sessions are killed. A resumed bank must be the bank an
    uninterrupted run would have written, and must not re-embed what is
    already on disk."""
    from aigcdet.eval import grid
    from aigcdet.eval.grid import extract_eval_bank

    _, _, df = _frozen(tmp_path)
    bench = df[df["split"] == "benchmark"].iloc[:6]

    _fake_backbone(monkeypatch)
    reference = FeatureBank.open(extract_eval_bank(
        bench, "fake", str(tmp_path / "ref"), conditions=TWO_CONDITIONS,
        device="cpu", checkpoint_every=1))

    calls = {"n": 0}
    real_embed = grid.embed

    def flaky(m, s, imgs, device, batch_size=16):
        calls["n"] += 1
        if calls["n"] > 3:
            raise RuntimeError("session killed")
        return real_embed(m, s, imgs, device, batch_size=batch_size)

    monkeypatch.setattr(grid, "embed", flaky)
    out = str(tmp_path / "resumed")
    with pytest.raises(RuntimeError, match="session killed"):
        extract_eval_bank(bench, "fake", out, conditions=TWO_CONDITIONS,
                          device="cpu", checkpoint_every=1)

    monkeypatch.setattr(grid, "embed", real_embed)
    calls["n"] = 0
    extract_eval_bank(bench, "fake", out, conditions=TWO_CONDITIONS,
                      device="cpu", checkpoint_every=1, resume=True)
    assert calls["n"] == 0                      # counter is no longer wired in
    resumed = FeatureBank.open(out)
    assert resumed.row_ids.tolist() == reference.row_ids.tolist()
    np.testing.assert_array_equal(np.asarray(resumed.feats),
                                  np.asarray(reference.feats))


def test_resume_does_not_re_embed_the_rows_already_written(tmp_path, monkeypatch):
    from aigcdet.eval import grid
    from aigcdet.eval.grid import extract_eval_bank

    _, _, df = _frozen(tmp_path)
    bench = df[df["split"] == "benchmark"].iloc[:6]

    _fake_backbone(monkeypatch)
    real_embed = grid.embed
    seen: list[int] = []

    def counting(m, s, imgs, device, batch_size=16):
        seen.append(1)
        if len(seen) > 4:
            raise RuntimeError("session killed")
        return real_embed(m, s, imgs, device, batch_size=batch_size)

    monkeypatch.setattr(grid, "embed", counting)
    out = str(tmp_path / "resumed2")
    with pytest.raises(RuntimeError):
        extract_eval_bank(bench, "fake", out, conditions=TWO_CONDITIONS,
                          device="cpu", checkpoint_every=1)
    assert len(seen) == 5                       # 4 written, the 5th died

    seen.clear()
    monkeypatch.setattr(grid, "embed", real_embed)
    calls: list[int] = []

    def tallying(m, s, imgs, device, batch_size=16):
        calls.append(1)
        return real_embed(m, s, imgs, device, batch_size=batch_size)

    monkeypatch.setattr(grid, "embed", tallying)
    extract_eval_bank(bench, "fake", out, conditions=TWO_CONDITIONS,
                      device="cpu", checkpoint_every=1, resume=True)
    assert len(calls) == 2                      # only rows 4 and 5
    assert FeatureBank.open(out).config["n_images"] == 6


def test_resume_refuses_a_different_subsample(tmp_path, monkeypatch):
    """A resume must continue the SAME extraction. `extra_config` is merged
    into the config precisely so the subsample takes part in that check."""
    from aigcdet.eval.grid import extract_eval_bank

    _, _, df = _frozen(tmp_path)
    bench = df[df["split"] == "benchmark"].iloc[:4]
    _fake_backbone(monkeypatch)
    out = str(tmp_path / "eb_resume_cfg")
    extract_eval_bank(bench, "fake", out, conditions=TWO_CONDITIONS,
                      device="cpu", extra_config={"tier": "smoke",
                                                  "subsample": {"kept": {"benchmark": 4}}})
    with pytest.raises(ValueError, match="cannot resume"):
        extract_eval_bank(bench, "fake", out, conditions=TWO_CONDITIONS,
                          device="cpu", resume=True,
                          extra_config={"tier": "smoke",
                                        "subsample": {"kept": {"benchmark": 5}}})


def test_extra_config_may_not_shadow_the_condition_list(tmp_path, monkeypatch):
    from aigcdet.eval.grid import extract_eval_bank

    _, _, df = _frozen(tmp_path)
    _fake_backbone(monkeypatch)
    with pytest.raises(ValueError, match="may not shadow"):
        extract_eval_bank(df.iloc[:2], "fake", str(tmp_path / "eb_shadow"),
                          conditions=TWO_CONDITIONS, device="cpu",
                          extra_config={"conditions": ["clean"]})


# --- the plan a teammate reads before paying -------------------------------

def test_dry_run_prints_the_plan_and_writes_nothing(tmp_path, monkeypatch, capsys):
    root, m, _ = _frozen(tmp_path)
    out = str(tmp_path / "eb_dry")
    # No backbone is patched: --dry-run must return before load_backbone.
    eb.main(["--manifest", m, "--backbone", "fake", "--out", out,
             "--tier", "ablation", "--subsample", "benchmark=10", "--dry-run"])
    text = capsys.readouterr().out
    assert not os.path.exists(out)
    assert "tier               ablation" in text
    assert f"forwards           {22 * len(EVAL_GRID)}" in text
    assert "'benchmark': 10" in text


def test_the_subsample_seed_flag_reaches_the_subsample(tmp_path, monkeypatch):
    """Recording a seed the extraction did not use would make the bank's own
    provenance a lie -- and §4.4a's "the subsample seed is fixed and
    committed" unverifiable."""
    root, m, _ = _frozen(tmp_path)
    common = ["--manifest", m, "--backbone", "fake", "--tier", "smoke",
              "--conditions", "clean", "--split", "benchmark",
              "--subsample", "benchmark=8", "--device", "cpu"]
    _run(common + ["--out", str(tmp_path / "sd7"), "--subsample-seed", "7"],
         monkeypatch)
    _run(common + ["--out", str(tmp_path / "sd8"), "--subsample-seed", "8"],
         monkeypatch)
    a = FeatureBank.open(str(tmp_path / "sd7"))
    b = FeatureBank.open(str(tmp_path / "sd8"))
    assert a.config["subsample"]["seed"] == 7
    assert b.config["subsample"]["seed"] == 8
    assert a.row_ids.tolist() != b.row_ids.tolist()


def test_limit_truncates_after_the_subsample_and_before_the_shard(tmp_path,
                                                                  monkeypatch):
    root, m, _ = _frozen(tmp_path)
    out = str(tmp_path / "eb_limit")
    _run(["--manifest", m, "--backbone", "fake", "--out", out,
          "--tier", "smoke", "--conditions", "clean", "--split", "benchmark",
          "--limit", "5", "--device", "cpu"], monkeypatch)
    bank = FeatureBank.open(out)
    assert bank.config["n_images"] == 5
    assert bank.row_ids.tolist() == list(range(18, 23))
