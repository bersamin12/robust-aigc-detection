"""Pairing and eligibility.

The property that matters most here is prefix stability: raising `--total`
must resume the earlier run rather than reshuffle it. With contiguous
per-family slices it does not, and every image already generated is orphaned.
"""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from aigcdet.generate.pool import _strata, build_pool, select
from aigcdet.generate.registry import SUITE, resolve_suite

SHARES = {k: v.share for k, v in SUITE.items()}


def _pool(n=4000, eligible_frac=1.0):
    rng = np.random.default_rng(0)
    ids = [f"{i:016x}" for i in range(n)]
    elig = rng.random(n) < eligible_frac
    return pd.DataFrame({
        "image_id": ids, "path": [f"/x/{i}.jpg" for i in ids],
        "width": 416, "height": 640, "crop_l": 0, "crop_t": 0,
        "crop_r": 416, "crop_b": 640, "subsampling": 0,
        "jpeg_quality": 90.0, "mode": "RGB", "eligible": elig,
        "reason": np.where(elig, "", "nope")})


def test_strata_realise_the_shares_exactly():
    pat = _strata(SHARES)
    assert len(pat) == 100
    for name, share in SHARES.items():
        assert abs(pat.count(name) / 100 - share) < 0.01


def test_a_smaller_total_is_a_strict_prefix_of_a_larger_one():
    """The resume property. Without it, `--total 10000` after `--total 2000`
    reassigns reals that already have fakes on disk, and the run restarts."""
    pool = _pool()
    small = select(pool, resolve_suite(700), seed=7, shares=SHARES)
    big = select(pool, resolve_suite(3500), seed=7, shares=SHARES)
    assigned = dict(zip(big.image_id, big.family))
    assert all(assigned.get(i) == f for i, f in zip(small.image_id, small.family))


def test_pairing_is_disjoint():
    """One real, one fake, one family. The same real across several families
    would put one scene in the corpus five times on the fake side against one
    real, and any content the model memorises then carries a label prior."""
    sel = select(_pool(), resolve_suite(2000), seed=7, shares=SHARES)
    assert not sel.image_id.duplicated().any()
    assert len(sel) == 2000


def test_every_prefix_is_balanced_across_families():
    """A run cut short must still cover every family, not stop having produced
    only the ones that happened to sort first."""
    sel = select(_pool(), resolve_suite(2000), seed=7, shares=SHARES)
    counts = sel.family.value_counts()
    for name, share in SHARES.items():
        assert abs(counts[name] / 2000 - share) < 0.01


def test_assignment_is_stable_under_reordering_of_the_pool():
    pool = _pool()
    a = select(pool, resolve_suite(500), seed=7, shares=SHARES)
    b = select(pool.sample(frac=1.0, random_state=3), resolve_suite(500),
               seed=7, shares=SHARES)
    assert dict(zip(a.image_id, a.family)) == dict(zip(b.image_id, b.family))


def test_shards_never_hand_the_same_real_to_two_workers():
    pool = _pool()
    seen = [set(select(pool, resolve_suite(300), seed=7, shard=i, n_shards=4,
                       shares=SHARES).image_id) for i in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not seen[i] & seen[j]


def test_a_pool_too_small_says_so_instead_of_returning_short():
    with pytest.raises(ValueError, match="eligible reals"):
        select(_pool(n=50), resolve_suite(2000), seed=7, shares=SHARES)


def test_ineligible_reals_are_never_selected():
    pool = _pool(eligible_frac=0.5)
    sel = select(pool, resolve_suite(300), seed=7, shares=SHARES)
    assert pool.set_index("image_id").loc[sel.image_id, "eligible"].all()


def test_build_pool_records_a_reason_rather_than_filtering_silently(tmp_path):
    d = tmp_path / "portrait"
    d.mkdir()
    rng = np.random.default_rng(0)

    def w(name, size, mode="RGB", **kw):
        a = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        Image.fromarray(a).convert(mode).save(d / name, "JPEG", **kw)

    w("aaaaaaaaaaaaaaa1.jpg", (416, 640), subsampling=0, quality=90)  # good
    w("aaaaaaaaaaaaaaa2.jpg", (100, 120), subsampling=0, quality=90)  # tiny
    w("aaaaaaaaaaaaaaa3.jpg", (416, 640), mode="L", quality=90)       # grayscale

    attr = tmp_path / "attribution.csv"
    pd.DataFrame({
        "ImageID": [f"aaaaaaaaaaaaaaa{i}" for i in (1, 2, 3)],
        "Author": ["a"] * 3, "OriginalURL": ["u"] * 3,
        "License": ["https://creativecommons.org/licenses/by/2.0/"] * 3,
    }).to_csv(attr, index=False)

    pool = build_pool(d, attr, min_side=320)
    by_id = pool.set_index("image_id")
    assert by_id.loc["aaaaaaaaaaaaaaa1", "eligible"]
    assert "short side" in by_id.loc["aaaaaaaaaaaaaaa2", "reason"]
    assert "not reproducible" in by_id.loc["aaaaaaaaaaaaaaa3", "reason"]


def test_build_pool_refuses_a_non_cc_by_licence(tmp_path):
    """`docs/02` §1: CC BY 2.0 is the only licence audited to clear both
    commercial use and redistribution. A row under anything else means the
    harvest changed, and the run should stop rather than quietly shrink."""
    d = tmp_path / "portrait"
    d.mkdir()
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (640, 416, 3), dtype=np.uint8)).save(
        d / "aaaaaaaaaaaaaaa1.jpg", "JPEG", quality=90, subsampling=0)
    attr = tmp_path / "attribution.csv"
    pd.DataFrame({"ImageID": ["aaaaaaaaaaaaaaa1"], "Author": ["a"],
                  "OriginalURL": ["u"],
                  "License": ["https://creativecommons.org/licenses/by-nc/2.0/"]
                  }).to_csv(attr, index=False)
    with pytest.raises(ValueError, match="CC BY 2.0"):
        build_pool(d, attr)
