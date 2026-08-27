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
