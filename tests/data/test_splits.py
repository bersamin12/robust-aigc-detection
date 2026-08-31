import numpy as np
import pandas as pd
import pytest

from aigcdet.data.manifest import MANIFEST_COLUMNS
from aigcdet.data.splits import assign_splits, choose_heldout_generators, split_report


def _df(n_per_gen=250, gens=("g1", "g2", "g3", "g4")):  # >=200/gen: see MIN_HELDOUT_IMAGES
    rows = []
    for g in gens:
        for i in range(n_per_gen):
            rows.append({"path": f"/f/{g}/{i}.png", "label": 1, "generator": g,
                         "source": "wildfake", "licence": "x", "width": 512,
                         "height": 512, "split": ""})
    for i in range(n_per_gen * len(gens)):
        rows.append({"path": f"/r/{i}.png", "label": 0, "generator": "",
                     "source": "sid_set", "licence": "x", "width": 512,
                     "height": 512, "split": ""})
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def _df_with_pseudo_generator(n_per_gen=250):
    """Two real generator families plus a dataset-level pseudo-generator that
    clears MIN_HELDOUT_IMAGES easily, so it is rejected on eligibility rather
    than on count."""
    df = _df(n_per_gen=n_per_gen, gens=("g1", "g2"))
    pseudo = pd.DataFrame([
        {"path": f"/s/{i}.png", "label": 1, "generator": "sid_set",
         "source": "sid_set", "licence": "x", "width": 512, "height": 512,
         "split": ""}
        for i in range(n_per_gen * 4)
    ], columns=MANIFEST_COLUMNS)
    return pd.concat([df, pseudo], ignore_index=True)


@pytest.mark.parametrize("seed", range(12))
def test_dataset_level_pseudo_generator_is_never_held_out(seed):
    """Holding out "sid_set" removes an entire SOURCE, measuring dataset
    shift rather than unseen-generator generalisation (spec §4.6). It is the
    largest family here, so an unfiltered draw would return it often."""
    held = choose_heldout_generators(_df_with_pseudo_generator(), n=2, seed=seed)
    assert set(held) == {"g1", "g2"}


def test_raises_when_only_pseudo_generators_are_large_enough():
    df = _df_with_pseudo_generator()
    df = df[df["generator"] != "g2"]
    with pytest.raises(ValueError, match="pseudo-generators"):
        choose_heldout_generators(df, n=2, seed=0)


def test_choose_heldout_generators_is_deterministic_and_returns_two():
    df = _df()
    a = choose_heldout_generators(df, n=2, seed=1)
    b = choose_heldout_generators(df, n=2, seed=1)
    assert a == b and len(a) == 2 and set(a) <= {"g1", "g2", "g3", "g4"}


def test_heldout_generator_images_never_land_in_train():
    df = _df()
    out = assign_splits(df, heldout_generators=["g1", "g2"])
    assert set(out[out["generator"].isin(["g1", "g2"])]["split"]) == {"heldout_generator"}
    assert "g1" not in set(out[out["split"] == "train"]["generator"])


def test_splits_are_exhaustive_and_disjoint():
    out = assign_splits(_df(), heldout_generators=["g1"])
    assert (out["split"] != "").all()
    assert set(out["split"]) <= {"train", "val_internal", "heldout_generator"}
    assert len(out) == len(_df())


def test_validation_fraction_is_approximately_respected():
    out = assign_splits(_df(), heldout_generators=["g1"], val_fraction=0.1)
    pool = out[out["split"].isin(["train", "val_internal"])]
    frac = (pool["split"] == "val_internal").mean()
    assert frac == pytest.approx(0.1, abs=0.03)


def test_assignment_is_reproducible_with_the_same_seed():
    a = assign_splits(_df(), ["g1"], seed=7)["split"].tolist()
    b = assign_splits(_df(), ["g1"], seed=7)["split"].tolist()
    assert a == b


def test_split_report_counts_by_split_and_label():
    rep = split_report(assign_splits(_df(), ["g1"]))
    assert {"split", "label", "n"} <= set(rep.columns)
    assert rep["n"].sum() == len(_df())


def test_raises_when_a_heldout_generator_is_absent():
    with pytest.raises(ValueError, match="not present"):
        assign_splits(_df(), heldout_generators=["nope"])


# --- group_key: paired corpora (AI-OV7) ------------------------------------
# A fake generated FROM a real carries that real's ImageID on both rows. Drawn
# per row, the pair straddles the val boundary ~18% of the time and the model
# trains on a scene it is validated against under the other label.

def _paired(n=400, heldout_family="klein4b_t2i"):
    import pandas as pd
    ids = [f"img{i:05d}" for i in range(n)]
    fams = [heldout_family if i % 5 == 0 else "sdxl_t2i" for i in range(n)]
    rows = ([{"generator": "", "image_id": i, "label": 0} for i in ids]
            + [{"generator": f, "image_id": i, "label": 1}
               for i, f in zip(ids, fams)])
    return pd.DataFrame(rows)


def test_group_key_keeps_a_real_and_its_fake_on_the_same_side():
    from aigcdet.data.splits import assign_splits
    df = _paired()
    out = assign_splits(df, ["klein4b_t2i"], group_key=df["image_id"])
    spans = out.groupby("image_id")["split"].nunique()
    assert (spans == 1).all(), "a pair straddled a split boundary"


def test_without_a_group_key_pairs_do_straddle():
    """The bug this argument exists for; asserted so the fix cannot be quietly
    reverted to 'it was fine anyway'."""
    from aigcdet.data.splits import assign_splits
    df = _paired()
    out = assign_splits(df, ["klein4b_t2i"])
    assert (out.groupby("image_id")["split"].nunique() > 1).any()


def test_heldout_membership_propagates_to_the_paired_real():
    """Otherwise the held-out rung evaluates on scenes the model already
    memorised under the other label."""
    from aigcdet.data.splits import assign_splits
    df = _paired()
    out = assign_splits(df, ["klein4b_t2i"], group_key=df["image_id"])
    held_ids = set(out.loc[out["generator"] == "klein4b_t2i", "image_id"])
    reals = out.loc[(out["label"] == 0) & (out["image_id"].isin(held_ids))]
    assert (reals["split"] == "heldout_generator").all()
    assert len(reals) == len(held_ids)


def test_group_key_none_preserves_the_original_rng_stream():
    """Manifests frozen before this argument existed must rebuild
    byte-identically -- every feature bank on disk fingerprints its manifest."""
    import numpy as np
    import pandas as pd
    from aigcdet.data.splits import assign_splits, DEFAULT_SEED
    df = _paired()
    out = assign_splits(df, ["klein4b_t2i"])
    held = df["generator"].isin(["klein4b_t2i"])
    expected = np.where(
        np.random.default_rng(DEFAULT_SEED).random(int((~held).sum())) < 0.1,
        "val_internal", "train")
    assert list(out.loc[~held, "split"]) == list(expected)


def test_blank_group_entries_raise_rather_than_orphan_a_row():
    import pandas as pd
    from aigcdet.data.splits import assign_splits
    df = _paired()
    keys = df["image_id"].copy()
    keys.iloc[3] = ""
    with pytest.raises(ValueError, match="blank"):
        assign_splits(df, ["klein4b_t2i"], group_key=keys)
