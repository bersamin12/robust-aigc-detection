"""`scripts/build_ntire_manifest.py` over a miniature of every tree it reads."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import build_ntire_manifest as bnm  # noqa: E402

from aigcdet.data.manifest import MANIFEST_COLUMNS  # noqa: E402


def _img(path, size=(64, 48), seed=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


def _manifest_rows(paths, labels, gens, source, split, root):
    rows = []
    for p, y, g in zip(paths, labels, gens):
        rows.append({"path": p, "label": y, "generator": g, "source": source,
                     "licence": "x", "width": 64, "height": 48, "split": split,
                     "rel_path": os.path.relpath(p, root), "content_sha256": "",
                     "pixel_sha256": ""})
    return rows


@pytest.fixture
def world(tmp_path):
    """exp/ holds every tree; repo/ holds the two frozen manifests."""
    exp, repo = tmp_path / "exp", tmp_path / "repo"
    repo.mkdir()
    union_root = exp / "normalized_union"
    rows = []
    # Union: ntire shards 0-2 (3 real + 3 fake), coco reals, open_images reals,
    # plus sid_set/wildfake rows whose files do NOT exist (deleted from disk).
    for i in range(3):
        rows += _manifest_rows([_img(str(union_root / "ntire" / "real" / f"{i:07d}.png"))],
                               [0], [""], "ntire", "train" if i < 2 else "val_internal", union_root)
        rows += _manifest_rows([_img(str(union_root / "ntire" / "ntire" / f"{i:07d}.png"))],
                               [1], ["ntire"], "ntire", "train" if i < 2 else "val_internal", union_root)
    rows += _manifest_rows([_img(str(union_root / "coco_train2017" / "real" / "0000000.png"))],
                           [0], [""], "coco_train2017", "train", union_root)
    rows += _manifest_rows([_img(str(union_root / "open_images" / "portrait" / "0000000.png"))],
                           [0], [""], "open_images", "train", union_root)
    rows += _manifest_rows([str(union_root / "sid_set" / "real" / "0000000.png")],
                           [0], [""], "sid_set", "train", union_root)
    rows += _manifest_rows([str(union_root / "wildfake" / "VQGAN" / "0000000.png")],
                           [1], ["VQGAN"], "wildfake", "heldout_generator", union_root)
    # Rebase onto a foreign root, as the real manifests are.
    u = pd.DataFrame(rows)
    u["path"] = "/mnt/elsewhere/normalized_union/" + u.rel_path
    u.to_parquet(repo / "manifest_union.parquet", index=False)

    # OV7 11,978 build: two pairs, one train one heldout, one val.
    ov_root = exp / "normalized_ov7" / "open_images_v7"
    orows = []
    for i, split in enumerate(("train", "heldout_generator", "val_internal")):
        orows += _manifest_rows([_img(str(ov_root / "sdxl_t2i" / f"{i:07d}.png"))],
                                [1], ["sdxl_t2i"], "open_images_v7", split, ov_root)
        orows += _manifest_rows([_img(str(ov_root / "real" / f"{i:07d}.png"))],
                                [0], [""], "open_images_v7", split, ov_root)
    o = pd.DataFrame(orows)
    o["path"] = "/mnt/elsewhere/normalized_ov7/open_images_v7/" + o.rel_path
    o.to_parquet(repo / "manifest_ov7.parquet", index=False)

    # Sidecar for shard 3: 10 real + 10 fake.
    side = []
    for i in range(20):
        y = i % 2
        b = "ntire" if y else "real"
        p = _img(str(union_root / "ntire" / b / f"s3_{i:07d}.png"))
        side.append({"path": p, "label": y, "generator": "ntire" if y else "",
                     "source": "ntire", "licence": "x", "width": 64, "height": 48,
                     "shard": 3, "src_name": f"n{i}.jpg", "content_sha256": "a" * 64})
    pd.DataFrame(side).to_parquet(exp / "ntire_shard3_rows.parquet", index=False)

    # OV7 42k raw pairs: 10 pairs across two families.
    k_root = exp / "raw_ov7_42k"
    prs = []
    for i in range(10):
        fam = "sd35_t2i" if i % 2 else "flux_t2i"
        _img(str(k_root / "open_images_v7" / fam / f"id{i}.jpg"))
        _img(str(k_root / "open_images_v7" / "real" / f"id{i}.jpg"))
        prs.append({"image_id": f"id{i}", "family": fam,
                    "fake_rel": f"open_images_v7/{fam}/id{i}.jpg",
                    "real_rel": f"open_images_v7/real/id{i}.jpg",
                    "width": 64, "height": 48})
    pd.DataFrame(prs).to_csv(k_root / "pairs.csv", index=False)

    # Commercial-API pairs: 4 reals, each with 2 fakes from different APIs.
    c_root = exp / "raw_commercial_api"
    crows = []
    for i in range(4):
        iid = f"c{i}"
        _img(str(c_root / "reals" / f"{iid}.jpg"))
        crows.append({"path": f"reals/{iid}.jpg", "label": 0, "generator": "",
                      "source": "open_images_v7_validation", "licence": "CC BY 2.0",
                      "split": "heldout_generator", "image_id": iid, "family": "",
                      "redistributable": True, "kind": "real"})
        for api in ("ideogram_40_turbo", "openai_gpt_image_2"):
            _img(str(c_root / api / f"{iid}.jpg"))
            crows.append({"path": f"{api}/{iid}.jpg", "label": 1, "generator": api,
                          "source": "commercial_api", "licence": "API terms",
                          "split": "heldout_generator", "image_id": iid, "family": api,
                          "redistributable": False, "kind": api})
    pd.DataFrame(crows).to_csv(c_root / "manifest.csv", index=False)

    # Organisers' sets, in both zip layouts.
    def _set(csv_rel, stem, nested, n=8):
        names, labels, dist = [], [], []
        d = exp / stem / os.path.basename(stem) if nested else exp / stem
        for i in range(n):
            nm = f"{os.path.basename(stem)}_{i}.jpg"
            _img(str(d / nm), size=(300, 200) if i % 4 else (150, 120))
            names.append(nm)
            labels.append(i % 2)
            dist.append(1 if i >= n // 2 else 0)
        pd.DataFrame({"image_name": names, "label": labels,
                      "distortions": ["[]" if not x else "['jpeg']" for x in dist],
                      "distortion_scales": ["[]"] * n,
                      "is_distorted": dist}).to_csv(exp / csv_rel, index=False)
    _set("raw_ntire_val/val_labels.csv", "raw_ntire_val/val_images", nested=True)
    _set("raw_ntire_val/val_hard_labels.csv", "raw_ntire_val/val_images_hard", nested=False)
    _set("raw_ntire_test_public/test_labels.csv", "raw_ntire_test_public/test_images", nested=True)
    return str(exp), str(repo)


def _build(world, **kw):
    exp, repo = world
    args = dict(exp=exp, repo=repo, sidecar_dir=exp, shards=(3,), seed=1,
                val_frac=0.2, ov7_42k_root=f"{exp}/raw_ov7_42k",
                commercial_root=f"{exp}/raw_commercial_api")
    args.update(kw)
    return bnm.build(**args)


def test_commercial_pairs_train_with_their_real_and_never_straddle(world):
    df = _build(world)
    c = df[df.variant == "commercial_api"]
    assert len(c) == 12
    assert set(c.source) == {"commercial_api", "open_images_v7_validation"}
    assert set(c.split) == {"train", "val"}
    assert (c.groupby("image_id").split.nunique() == 1).all()
    assert (c[c.label == 1].family != "").all() and (c[c.label == 0].family == "").all()
    assert c.assign_rule.str.startswith("commercial_pair_").all()
    # Its csv's own held-out split is ignored: nothing lands outside train/val.
    exp, _ = world
    with pytest.raises(SystemExit, match="no manifest.csv"):
        _build(world, commercial_root=f"{exp}/nope")
    d2 = _build(world, commercial_root=f"{exp}/nope", require_commercial=False)
    assert not (d2.variant == "commercial_api").any()


def test_splits_and_sources_land_where_the_docstring_says(world):
    df = _build(world)
    assert set(df.columns) == set(bnm.COLS)
    assert set(df.split) == {"train", "val", "ntire_val", "ntire_val_hard", "ntire_test"}
    # Deleted sources are gone by name; every surviving path exists.
    assert not df.source.isin(["sid_set", "wildfake"]).any()
    assert all(os.path.exists(p) for p in df.path)
    # Union val_internal -> val, train -> train.
    u = df[df.assign_rule.str.startswith("union_")]
    assert set(u.split) == {"train", "val"}
    # The organisers' sets sit in their fixed splits and nowhere else.
    for src, sp in (("ntire_val", "ntire_val"), ("ntire_val_hard", "ntire_val_hard"),
                    ("ntire_test_public", "ntire_test")):
        assert set(df[df.source == src].split) == {sp}
        assert (df[df.split == sp].source == src).all()
    # No never-train source in train/val.
    assert not df[df.split.isin(("train", "val"))].source.isin(
        ["ntire_val", "ntire_val_hard", "ntire_test_public"]).any()


def test_rel_path_is_relative_to_exp_for_every_tree(world):
    exp, _ = world
    df = _build(world)
    assert not df.rel_path.str.startswith("/").any()
    assert all(os.path.join(exp, r) == p for r, p in zip(df.rel_path, df.path))
    heads = {r.split(os.sep)[0] for r in df.rel_path}
    assert heads == {"normalized_union", "normalized_ov7", "raw_ov7_42k",
                     "raw_commercial_api", "raw_ntire_val", "raw_ntire_test_public"}


def test_is_distorted_and_distortions_are_carried_for_the_eval_sets(world):
    df = _build(world)
    ev = df[df.split.isin(bnm.FIXED_SPLITS)]
    assert set(ev.is_distorted) == {0, 1}
    assert (ev[ev.is_distorted == 1].distortions == "['jpeg']").all()
    assert (df[~df.split.isin(bnm.FIXED_SPLITS)].is_distorted == 0).all()
    assert (ev.content_sha256.str.len() == 64).all()


def test_ov7_42k_pairs_never_straddle_train_and_val_and_val_draw_is_seeded(world):
    a = _build(world)
    b = _build(world)
    pd.testing.assert_frame_equal(a, b)
    k = a[a.variant == "open_images_v7_42k"]
    assert len(k) == 20
    assert (k.groupby("image_id").split.nunique() == 1).all()
    assert set(k.split) == {"train", "val"}
    assert (k.groupby("image_id").label.sum() == 1).all()   # one real, one fake
    c = _build(world, seed=2)
    assert not a.split.equals(c.split)


def test_sidecar_shards_get_a_per_label_val_slice(world):
    df = _build(world)
    s = df[df.assign_rule.str.startswith("ntire_shard_")]
    assert len(s) == 20
    for y in (0, 1):
        assert (s[s.label == y].split == "val").sum() == 2      # 20% of 10


def test_ov7_small_folds_heldout_into_train(world):
    df = _build(world)
    o = df[df.variant == "open_images_v7_11978"]
    assert o.split.value_counts().to_dict() == {"train": 4, "val": 2}


def test_missing_42k_is_refused_unless_told_otherwise(world):
    exp, _ = world
    with pytest.raises(SystemExit, match="no pairs.csv"):
        _build(world, ov7_42k_root=f"{exp}/nope")
    df = _build(world, ov7_42k_root=f"{exp}/nope", require_ov7_42k=False)
    assert not (df.variant == "open_images_v7_42k").any()


def test_a_never_train_row_in_train_is_caught(world):
    df = _build(world)
    df.loc[df.source == "ntire_val", "split"] = "train"
    with pytest.raises(SystemExit, match="never-train"):
        bnm.check(df)


def test_a_one_class_distorted_half_is_caught(world):
    df = _build(world)
    m = (df.split == "ntire_test") & (df.is_distorted == 1)
    df.loc[m, "label"] = 1
    with pytest.raises(SystemExit, match="lacks a class"):
        bnm.check(df)


def test_cli_refuses_to_overwrite_and_writes_provenance(world, tmp_path):
    exp, repo = world
    out = tmp_path / "m.parquet"
    js = tmp_path / "splits.json"
    argv = ["--exp", exp, "--repo", repo, "--shards", "3", "--out", str(out),
            "--splits-json", str(js), "--seed", "1", "--val-frac", "0.2"]
    assert bnm.main(argv) == 0
    rec = json.load(open(js))
    assert rec["rows"] == len(pd.read_parquet(out))
    assert set(rec["split_totals"]) == {"train", "val", "ntire_val", "ntire_val_hard", "ntire_test"}
    assert "ntire_test" in rec["eval_short_side_by_is_distorted"]
    assert rec["eval_short_side_by_is_distorted"]["ntire_test"]["0"]["n_below_200"] >= 1
    with pytest.raises(SystemExit, match="exists"):
        bnm.main(argv)
    assert bnm.main(argv + ["--force"]) == 0


def test_manifest_columns_are_a_superset_of_the_frozen_schema(world):
    df = _build(world)
    assert set(MANIFEST_COLUMNS) <= set(df.columns)
