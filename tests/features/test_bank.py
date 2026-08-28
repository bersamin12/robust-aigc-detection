import numpy as np
import pandas as pd
import pytest

from aigcdet.augment.recipes import Recipe
from aigcdet.features.bank import N_VIEWS, BankWriter, FeatureBank


def _build(tmp_path, n=4, dim=8, out_name="bank"):
    """Build a bank and return (bank, written) so tests can assert real
    values were round-tripped, not just array shapes."""
    w = BankWriter(str(tmp_path / out_name), n_images=n, n_views=N_VIEWS,
                   dim=dim, backbone="test", seed=0)
    rng = np.random.default_rng(0)
    written = []
    for i in range(n):
        presence = np.zeros((N_VIEWS, 6), np.float32)
        severity = np.zeros((N_VIEWS, 6), np.float32)
        presence[1:, 0] = 1.0                       # view 0 stays clean
        severity[1:, 0] = 0.5
        meta_row = {"path": f"/x/{i}.png", "label": i % 2, "generator": "g",
                    "source": "s", "split": "train"}
        feats = rng.normal(size=(N_VIEWS, dim)).astype(np.float32)
        proxies = rng.normal(size=(N_VIEWS, 3)).astype(np.float32)
        recipes = ["[]"] + ['[{"name": "jpeg", "params": {"quality": 50}}]'] * (N_VIEWS - 1)
        w.write_image(i, meta_row, feats=feats, presence=presence,
                      severity=severity, proxies=proxies, recipes=recipes)
        written.append({"meta_row": meta_row, "feats": feats, "presence": presence,
                        "severity": severity, "proxies": proxies, "recipes": recipes})
    w.close()
    return FeatureBank.open(str(tmp_path / out_name)), written


def test_bank_roundtrips_all_arrays(tmp_path):
    b, written = _build(tmp_path)
    assert b.feats.shape == (4, N_VIEWS, 8)
    assert b.presence.shape == (4, N_VIEWS, 6)
    assert b.severity.shape == (4, N_VIEWS, 6)
    assert b.proxies.shape == (4, N_VIEWS, 3)
    assert b.recon is None
    assert len(b.meta) == 4 and b.meta["label"].tolist() == [0, 1, 0, 1]
    assert b.config["backbone"] == "test"

    # Real value assertions: the shape checks above would pass even if
    # write_image stored zeros, so pin the actual bytes/values round-trip.
    for i, rec in enumerate(written):
        np.testing.assert_array_equal(np.asarray(b.feats[i]), rec["feats"].astype(np.float16))
        np.testing.assert_array_equal(np.asarray(b.presence[i]), rec["presence"])
        np.testing.assert_array_equal(np.asarray(b.severity[i]), rec["severity"])
        np.testing.assert_array_equal(np.asarray(b.proxies[i]), rec["proxies"])
        # the degraded views (1..N_VIEWS-1) must actually carry non-zero
        # jpeg-family presence, not merely have the right shape
        assert np.asarray(b.presence[i, 1:, 0]).sum() > 0.0


def test_meta_dtypes_survive_the_parquet_roundtrip(tmp_path):
    b, written = _build(tmp_path)
    for i, rec in enumerate(written):
        row = b.meta[b.meta["image_idx"] == i].iloc[0]
        assert row["path"] == rec["meta_row"]["path"]
        assert isinstance(row["label"], (int, np.integer))
        assert int(row["label"]) == rec["meta_row"]["label"]
        assert row["generator"] == rec["meta_row"]["generator"]
        assert row["source"] == rec["meta_row"]["source"]
        assert row["split"] == rec["meta_row"]["split"]


def test_view_zero_is_the_clean_view(tmp_path):
    b, _ = _build(tmp_path)
    b.check_invariants()
    assert b.presence[:, 0, :].sum() == 0.0
    # the invariant is only meaningful if the rest of the bank isn't also
    # all-zero -- confirm degraded views genuinely carry presence
    assert b.presence[:, 1:, 0].sum() > 0.0


def test_check_invariants_rejects_a_degraded_view_zero(tmp_path):
    b, _ = _build(tmp_path)
    # presence is opened read-only in production; build a writable copy to
    # simulate corruption rather than relying on open() handing back a
    # writable handle onto the on-disk array.
    b.presence = np.array(b.presence)
    b.presence[0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="view 0"):
        b.check_invariants()


def test_check_invariants_rejects_desynced_recipe_for_view_zero(tmp_path):
    # presence says every view (including view 0) is clean, but view 0's
    # recipe_json encodes a real degradation -- the two stored encodings of
    # "what happened to view 0" disagree, which check_invariants must catch
    # even though the presence-only check above would pass.
    w = BankWriter(str(tmp_path / "desynced"), n_images=1, n_views=N_VIEWS,
                   dim=4, backbone="test", seed=0)
    presence = np.zeros((N_VIEWS, 6), np.float32)
    severity = np.zeros((N_VIEWS, 6), np.float32)
    feats = np.zeros((N_VIEWS, 4), np.float32)
    proxies = np.zeros((N_VIEWS, 3), np.float32)
    recipes = ['[{"name": "jpeg", "params": {"quality": 50}}]'] * N_VIEWS
    w.write_image(0, {"path": "/x/0.png", "label": 0, "generator": "g",
                      "source": "s", "split": "train"},
                 feats=feats, presence=presence, severity=severity,
                 proxies=proxies, recipes=recipes)
    w.close()
    b = FeatureBank.open(str(tmp_path / "desynced"))
    assert b.presence[:, 0, :].sum() == 0.0     # presence alone looks clean
    with pytest.raises(ValueError, match="view 0"):
        b.check_invariants()


def test_attach_recon_persists_and_reloads(tmp_path):
    b, _ = _build(tmp_path)
    r = np.arange(4 * N_VIEWS * 12, dtype=np.float32).reshape(4, N_VIEWS, 12)
    b.attach_recon(r)
    b2 = FeatureBank.open(b.path)
    assert b2.recon is not None and b2.recon.shape == (4, N_VIEWS, 12)
    np.testing.assert_allclose(b2.recon[1, 2], r[1, 2])


def test_recipes_are_recoverable_per_view(tmp_path):
    b, _ = _build(tmp_path)
    assert Recipe.from_json(b.recipe_json(0, 0)).ops == ()
    assert Recipe.from_json(b.recipe_json(0, 1)).ops[0].name == "jpeg"


def test_verify_against_manifest_accepts_a_matching_manifest(tmp_path):
    b, written = _build(tmp_path)
    manifest = pd.DataFrame([w["meta_row"] for w in written])
    b.verify_against_manifest(manifest)  # must not raise


def test_verify_against_manifest_rejects_a_row_count_mismatch(tmp_path):
    b, written = _build(tmp_path)
    manifest = pd.DataFrame([w["meta_row"] for w in written][:-1])
    with pytest.raises(ValueError, match="rows"):
        b.verify_against_manifest(manifest)


def test_verify_against_manifest_rejects_a_reordered_manifest(tmp_path):
    b, written = _build(tmp_path)
    rows = [w["meta_row"] for w in written]
    rows[1], rows[2] = rows[2], rows[1]      # re-split-style row shuffle
    manifest = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="row 1"):
        b.verify_against_manifest(manifest)


# --- H1: row_id is stored, because it is the replay key --------------------

def test_row_id_defaults_to_the_write_position(tmp_path):
    b, _ = _build(tmp_path)
    np.testing.assert_array_equal(b.row_ids, np.arange(4))


def test_row_id_round_trips_a_non_contiguous_manifest_label(tmp_path):
    w = BankWriter(str(tmp_path / "rid"), n_images=2, n_views=N_VIEWS, dim=3,
                   backbone="test", seed=0)
    for i, rid in enumerate((17, 5000)):
        pres = np.zeros((N_VIEWS, 6), np.float32); pres[1:, 0] = 1.0
        w.write_image(i, {"path": f"/p{i}", "label": 0, "generator": "",
                          "source": "s", "split": "train"},
                      feats=np.zeros((N_VIEWS, 3), np.float32), presence=pres,
                      severity=np.zeros((N_VIEWS, 6), np.float32),
                      proxies=np.zeros((N_VIEWS, 3), np.float32),
                      recipes=["[]"] * N_VIEWS, row_id=rid)
    w.close()
    b = FeatureBank.open(str(tmp_path / "rid"))
    np.testing.assert_array_equal(b.row_ids, [17, 5000])
    b.check_invariants()


def test_check_invariants_rejects_duplicate_row_ids(tmp_path):
    """Two rows sharing a row_id means two images replay to the same pixels,
    since every view's RNG is keyed on (seed, row_id, view_idx)."""
    w = BankWriter(str(tmp_path / "dup"), n_images=2, n_views=N_VIEWS, dim=3,
                   backbone="test", seed=0)
    for i in range(2):
        pres = np.zeros((N_VIEWS, 6), np.float32); pres[1:, 0] = 1.0
        w.write_image(i, {"path": f"/p{i}", "label": 0, "generator": "",
                          "source": "s", "split": "train"},
                      feats=np.zeros((N_VIEWS, 3), np.float32), presence=pres,
                      severity=np.zeros((N_VIEWS, 6), np.float32),
                      proxies=np.zeros((N_VIEWS, 3), np.float32),
                      recipes=["[]"] * N_VIEWS, row_id=9)
    w.close()
    with pytest.raises(ValueError, match="duplicate row_id"):
        FeatureBank.open(str(tmp_path / "dup")).check_invariants()


# --- M3: config.json links the bank to its manifest ------------------------

def test_manifest_fingerprint_depends_on_paths_and_their_order():
    from aigcdet.features.bank import manifest_fingerprint

    a = pd.DataFrame({"path": ["/a.png", "/b.png", "/c.png"]})
    assert manifest_fingerprint(a) == manifest_fingerprint(a.copy())
    reordered = pd.DataFrame({"path": ["/a.png", "/c.png", "/b.png"]})
    assert manifest_fingerprint(a) != manifest_fingerprint(reordered)
    subset = pd.DataFrame({"path": ["/a.png", "/b.png"]})
    assert manifest_fingerprint(a) != manifest_fingerprint(subset)
    # A reset index must NOT change it: the fingerprint is about which rows,
    # in which order, not about index labels.
    assert manifest_fingerprint(a) == manifest_fingerprint(
        a.set_index(pd.Index([9, 8, 7])))


def _build_with_fingerprint(tmp_path, paths, out_name="fp"):
    from aigcdet.features.bank import manifest_fingerprint

    manifest = pd.DataFrame({"path": paths, "label": 0, "generator": "",
                             "source": "s", "split": "train"})
    w = BankWriter(str(tmp_path / out_name), n_images=len(paths), n_views=N_VIEWS,
                   dim=3, backbone="test", seed=0,
                   manifest_sha256=manifest_fingerprint(manifest))
    for i, p in enumerate(paths):
        pres = np.zeros((N_VIEWS, 6), np.float32); pres[1:, 0] = 1.0
        w.write_image(i, {"path": p, "label": 0, "generator": "",
                          "source": "s", "split": "train"},
                      feats=np.zeros((N_VIEWS, 3), np.float32), presence=pres,
                      severity=np.zeros((N_VIEWS, 6), np.float32),
                      proxies=np.zeros((N_VIEWS, 3), np.float32),
                      recipes=["[]"] * N_VIEWS, row_id=i)
    w.close()
    return FeatureBank.open(str(tmp_path / out_name)), manifest


def test_config_records_the_manifest_fingerprint(tmp_path):
    b, manifest = _build_with_fingerprint(tmp_path, ["/a.png", "/b.png"])
    from aigcdet.features.bank import manifest_fingerprint

    assert b.config["manifest_sha256"] == manifest_fingerprint(manifest)
    b.verify_against_manifest(manifest)  # must not raise


def test_verify_against_manifest_rejects_a_different_manifest_by_fingerprint(tmp_path):
    """Same row count, same order, different files -- the positional path
    check catches this too, but the fingerprint names it as a manifest
    identity mismatch and fails before the row loop."""
    b, _ = _build_with_fingerprint(tmp_path, ["/a.png", "/b.png"])
    other = pd.DataFrame({"path": ["/x.png", "/y.png"], "label": 0,
                          "generator": "", "source": "s", "split": "train"})
    with pytest.raises(ValueError, match="not the manifest the bank was built from"):
        b.verify_against_manifest(other)


def test_banks_without_a_recorded_fingerprint_still_verify(tmp_path):
    """manifest_sha256 is optional: a bank written without one falls back to
    the positional path check rather than refusing to verify at all."""
    b, written = _build(tmp_path, out_name="nofp")
    assert b.config["manifest_sha256"] is None
    b.verify_against_manifest(pd.DataFrame([w["meta_row"] for w in written]))


# --- M2: the recipe lookup is built on first use, not on open --------------

def test_open_does_not_build_the_recipe_lookup(tmp_path):
    """At 100k images x 11 views the lookup is a 1.1M-entry dict costing ~10 s
    and hundreds of MB, and it is paid on EVERY open -- including train_rung's,
    which needs only recipe_json(i, 0)."""
    b, _ = _build(tmp_path, out_name="lazy")
    assert b._recipe_lookup is None
    assert Recipe.from_json(b.recipe_json(0, 0)).ops == ()
    assert b._recipe_lookup is not None
    # Second call reuses it and still returns the right thing.
    assert Recipe.from_json(b.recipe_json(0, 1)).ops[0].name == "jpeg"


# --- H2: shard merge -------------------------------------------------------

def _shard(tmp_path, name, row_ids, backbone="test", seed=0, n_views=N_VIEWS, dim=3,
           extra_config=None):
    from aigcdet.features.bank import manifest_fingerprint

    paths = [f"/img{r}.png" for r in row_ids]
    fp = manifest_fingerprint(pd.DataFrame({"path": paths}))
    w = BankWriter(str(tmp_path / name), n_images=len(row_ids), n_views=n_views,
                   dim=dim, backbone=backbone, seed=seed, manifest_sha256=fp,
                   extra_config=extra_config)
    for i, r in enumerate(row_ids):
        pres = np.zeros((n_views, 6), np.float32); pres[1:, 0] = 1.0
        sev = np.zeros((n_views, 6), np.float32); sev[1:, 0] = 0.3
        recipes = ["[]"] + ['[{"name": "blur", "params": {"sigma": 0.5}}]'] * (n_views - 1)
        w.write_image(i, {"path": paths[i], "label": int(r) % 2,
                          "generator": "g", "source": "s", "split": "train"},
                      feats=np.full((n_views, dim), float(r), np.float32),
                      presence=pres, severity=sev,
                      proxies=np.full((n_views, 3), float(r), np.float32),
                      recipes=recipes, row_id=int(r))
    w.close()
    return str(tmp_path / name)


def test_merge_banks_concatenates_shards_in_order(tmp_path):
    from aigcdet.features.bank import manifest_fingerprint, merge_banks

    a = _shard(tmp_path, "s0", [10, 11])
    b = _shard(tmp_path, "s1", [20, 21, 22])
    out = merge_banks([a, b], str(tmp_path / "merged"))

    m = FeatureBank.open(out)
    m.check_invariants()
    assert len(m.meta) == 5
    np.testing.assert_array_equal(m.row_ids, [10, 11, 20, 21, 22])
    assert m.meta["image_idx"].tolist() == [0, 1, 2, 3, 4]
    for pos, rid in enumerate([10, 11, 20, 21, 22]):
        np.testing.assert_array_equal(np.asarray(m.feats[pos]),
                                       np.full((N_VIEWS, 3), float(rid), np.float16))
        assert m.recipe_json(pos, 0) == "[]"
        assert "blur" in m.recipe_json(pos, 1)
    assert m.config["backbone"] == "test" and m.config["seed"] == 0
    assert m.config["n_images"] == 5
    # The merged bank is aligned to the concatenation of the shards' rows.
    assert m.config["manifest_sha256"] == manifest_fingerprint(
        pd.DataFrame({"path": [f"/img{r}.png" for r in [10, 11, 20, 21, 22]]}))


def test_merge_banks_rejects_overlapping_row_ids(tmp_path):
    from aigcdet.features.bank import merge_banks

    a = _shard(tmp_path, "o0", [1, 2, 3])
    b = _shard(tmp_path, "o1", [3, 4])
    with pytest.raises(ValueError, match="shards overlap"):
        merge_banks([a, b], str(tmp_path / "bad"))


@pytest.mark.parametrize("kwargs", [
    {"backbone": "other"}, {"seed": 99}, {"dim": 5}, {"n_views": N_VIEWS - 1},
])
def test_merge_banks_rejects_shards_from_different_extractions(tmp_path, kwargs):
    from aigcdet.features.bank import merge_banks

    a = _shard(tmp_path, "d0", [1, 2])
    b = _shard(tmp_path, "d1", [3, 4], **kwargs)
    with pytest.raises(ValueError, match="not part of the same bank"):
        merge_banks([a, b], str(tmp_path / "bad2"))


def test_merge_banks_carries_recon_when_every_shard_has_it(tmp_path):
    from aigcdet.features.bank import merge_banks

    a, b = _shard(tmp_path, "r0", [1, 2]), _shard(tmp_path, "r1", [3])
    for d, val in ((a, 1.0), (b, 2.0)):
        bank = FeatureBank.open(d)
        bank.attach_recon(np.full((len(bank.meta), N_VIEWS, 12), val, np.float32))
    m = FeatureBank.open(merge_banks([a, b], str(tmp_path / "merged_recon")))
    assert m.recon is not None and m.recon.shape == (3, N_VIEWS, 12)
    np.testing.assert_allclose(np.asarray(m.recon[:2]), 1.0)
    np.testing.assert_allclose(np.asarray(m.recon[2]), 2.0)


def test_merge_banks_rejects_partial_recon_coverage(tmp_path):
    from aigcdet.features.bank import merge_banks

    a, b = _shard(tmp_path, "p0", [1, 2]), _shard(tmp_path, "p1", [3])
    bank = FeatureBank.open(a)
    bank.attach_recon(np.zeros((2, N_VIEWS, 12), np.float32))
    with pytest.raises(ValueError, match="some shards have recon"):
        merge_banks([a, b], str(tmp_path / "bad3"))


def test_merge_banks_needs_at_least_one_shard(tmp_path):
    from aigcdet.features.bank import merge_banks

    with pytest.raises(ValueError, match="at least one"):
        merge_banks([], str(tmp_path / "nothing"))


# --- extra_config ----------------------------------------------------------

def test_extra_config_is_recorded_in_config_json(tmp_path):
    w = BankWriter(str(tmp_path / "xc"), n_images=1, n_views=2, dim=3,
                   backbone="test", seed=0,
                   extra_config={"conditions": ["clean", "jpeg_q90"]})
    w.write_image(0, {"path": "/a.png", "label": 0, "generator": "", "source": "s",
                      "split": "train"},
                  feats=np.zeros((2, 3), np.float32),
                  presence=np.zeros((2, 6), np.float32),
                  severity=np.zeros((2, 6), np.float32),
                  proxies=np.zeros((2, 3), np.float32), recipes=["[]", "[]"])
    w.close()
    assert FeatureBank.open(str(tmp_path / "xc")).config["conditions"] == \
        ["clean", "jpeg_q90"]


def test_resume_rejects_a_changed_extra_config(tmp_path):
    """extra_config participates in the resume equality check.

    A resume that quietly accepted a different condition list would continue
    one bank's rows into another bank's view axis.
    """
    out = str(tmp_path / "xcr")
    BankWriter(out, n_images=1, n_views=2, dim=3, backbone="test", seed=0,
               extra_config={"conditions": ["clean", "jpeg_q90"]}).close()
    with pytest.raises(ValueError, match="cannot resume"):
        BankWriter(out, n_images=1, n_views=2, dim=3, backbone="test", seed=0,
                   extra_config={"conditions": ["clean", "jpeg_q70"]}, resume=True)
    # the identical list still resumes
    BankWriter(out, n_images=1, n_views=2, dim=3, backbone="test", seed=0,
               extra_config={"conditions": ["clean", "jpeg_q90"]}, resume=True)


def test_resume_rejects_dropping_an_extra_config_key(tmp_path):
    """"A different condition list" includes "no condition list".

    A resume that omits `extra_config` used to be accepted -- the equality
    check iterated only the REQUESTED keys -- and the next checkpoint then
    rewrote config.json without `conditions`, erasing what makes the bank an
    eval bank at all.
    """
    out = str(tmp_path / "xcd")
    w = BankWriter(out, n_images=1, n_views=2, dim=3, backbone="test", seed=0,
                   extra_config={"conditions": ["clean", "jpeg_q90"]})
    w.write_image(0, {"path": "/a.png", "label": 0, "generator": "", "source": "s",
                      "split": "train"},
                  feats=np.zeros((2, 3), np.float32),
                  presence=np.zeros((2, 6), np.float32),
                  severity=np.zeros((2, 6), np.float32),
                  proxies=np.zeros((2, 3), np.float32), recipes=["[]", "[]"])
    w.close()
    with pytest.raises(ValueError, match="cannot resume"):
        BankWriter(out, n_images=1, n_views=2, dim=3, backbone="test", seed=0,
                   resume=True)
    # and the bank on disk is untouched by the refusal
    assert FeatureBank.open(out).config["conditions"] == ["clean", "jpeg_q90"]


def test_extra_config_may_not_shadow_a_core_config_key(tmp_path):
    with pytest.raises(ValueError, match="reserved"):
        BankWriter(str(tmp_path / "xcs"), n_images=1, n_views=2, dim=3,
                   backbone="test", seed=0, extra_config={"backbone": "other"})


def test_merge_banks_carries_extra_config_through(tmp_path):
    from aigcdet.features.bank import merge_banks

    extra = {"conditions": ["clean", "jpeg_q90"]}
    a = _shard(tmp_path, "e0", [1, 2], n_views=2, extra_config=extra)
    b = _shard(tmp_path, "e1", [3], n_views=2, extra_config=extra)
    m = FeatureBank.open(merge_banks([a, b], str(tmp_path / "merged_extra")))
    assert m.config["conditions"] == ["clean", "jpeg_q90"]


def test_merge_banks_rejects_shards_with_different_extra_config(tmp_path):
    from aigcdet.features.bank import merge_banks

    a = _shard(tmp_path, "e2", [1, 2], n_views=2,
               extra_config={"conditions": ["clean", "jpeg_q90"]})
    b = _shard(tmp_path, "e3", [3], n_views=2,
               extra_config={"conditions": ["clean", "jpeg_q70"]})
    with pytest.raises(ValueError, match="not part of the same bank"):
        merge_banks([a, b], str(tmp_path / "bad_extra"))
