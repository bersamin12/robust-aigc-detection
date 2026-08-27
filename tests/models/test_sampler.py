import numpy as np
import pytest
import torch

from aigcdet.features.bank import N_VIEWS, BankWriter, FeatureBank
from aigcdet.models.sampler import PairedSampler


def _bank(tmp_path, n=40, dim=6):
    w = BankWriter(str(tmp_path / "b"), n, N_VIEWS, dim, "t", 0)
    rng = np.random.default_rng(0)
    for i in range(n):
        pres = np.zeros((N_VIEWS, 6), np.float32); pres[1:, 0] = 1.0
        sev = np.zeros((N_VIEWS, 6), np.float32); sev[1:, 0] = 0.4
        w.write_image(i, {"path": f"/p{i}", "label": i % 2,
                          "generator": f"g{i % 3}", "source": "s", "split": "train"},
                      feats=rng.normal(size=(N_VIEWS, dim)).astype(np.float32),
                      presence=pres, severity=sev,
                      proxies=np.zeros((N_VIEWS, 3), np.float32),
                      recipes=["[]"] * N_VIEWS)
    w.close()
    return FeatureBank.open(str(tmp_path / "b"))


# --- brief's own tests (Step 1), unmodified in intent ----------------------

def test_batch_shapes_and_pairing(tmp_path):
    b = _bank(tmp_path)
    s = PairedSampler(b, np.arange(40), n_src=4, m_deg=2, rng=np.random.default_rng(0))
    batch = next(iter(s))
    assert batch["f_clean"].shape == (8, 6) and batch["f_deg"].shape == (8, 6)
    assert batch["y_clean"].shape == (8,) and batch["presence_deg"].shape == (8, 6)
    # Each source image contributes m_deg rows sharing one clean embedding.
    assert torch.allclose(batch["f_clean"][0], batch["f_clean"][1])
    assert not torch.allclose(batch["f_deg"][0], batch["f_deg"][1])


def test_labels_match_between_clean_and_degraded_rows(tmp_path):
    b = _bank(tmp_path)
    s = PairedSampler(b, np.arange(40), n_src=8, m_deg=2, rng=np.random.default_rng(1))
    for batch in s:
        assert torch.equal(batch["y_clean"], batch["y_deg"])


def test_degraded_rows_always_have_nonzero_degradation(tmp_path):
    b = _bank(tmp_path)
    s = PairedSampler(b, np.arange(40), n_src=8, m_deg=2, rng=np.random.default_rng(2))
    for batch in s:
        assert (batch["presence_deg"].sum(dim=1) > 0).all()


def test_batches_are_class_balanced(tmp_path):
    b = _bank(tmp_path)
    s = PairedSampler(b, np.arange(40), n_src=8, m_deg=1, rng=np.random.default_rng(3))
    for batch in s:
        assert batch["y_clean"].sum().item() == 4      # half positives


def test_recon_is_returned_only_when_requested(tmp_path):
    b = _bank(tmp_path)
    b.attach_recon(np.zeros((40, N_VIEWS, 12), np.float32))
    s = PairedSampler(b, np.arange(40), n_src=4, m_deg=1,
                      rng=np.random.default_rng(4), use_recon=True)
    batch = next(iter(s))
    assert batch["r_deg"].shape == (4, 12)
    s2 = PairedSampler(b, np.arange(40), n_src=4, m_deg=1, rng=np.random.default_rng(4))
    assert next(iter(s2))["r_deg"] is None


def test_epoch_length_matches_the_index_pool(tmp_path):
    b = _bank(tmp_path)
    s = PairedSampler(b, np.arange(40), n_src=8, m_deg=2, rng=np.random.default_rng(5))
    assert len(s) == len(list(s))


# --- guarantees called out explicitly for this task -------------------------

def _bank_with_clean_marker(tmp_path, n=40, dim=4, marker=999.0):
    """View 0's feature vector is a sentinel unreachable by the augmented
    views' random draws, so any leak of view 0 into f_deg is directly
    observable rather than inferred from presence being a fixture artefact."""
    w = BankWriter(str(tmp_path / "bm"), n, N_VIEWS, dim, "t", 0)
    rng = np.random.default_rng(0)
    for i in range(n):
        pres = np.zeros((N_VIEWS, 6), np.float32); pres[1:, 0] = 1.0
        sev = np.zeros((N_VIEWS, 6), np.float32); sev[1:, 0] = 0.4
        feats = rng.normal(size=(N_VIEWS, dim)).astype(np.float32)
        feats[0] = marker
        w.write_image(i, {"path": f"/p{i}", "label": i % 2,
                          "generator": f"g{i % 3}", "source": "s", "split": "train"},
                      feats=feats, presence=pres, severity=sev,
                      proxies=np.zeros((N_VIEWS, 3), np.float32),
                      recipes=["[]"] * N_VIEWS)
    w.close()
    return FeatureBank.open(str(tmp_path / "bm"))


def test_view_zero_is_never_drawn_as_the_degraded_partner(tmp_path):
    """If view 0 (clean) were ever drawn as the degraded half, the
    consistency loss would compare the clean view against itself: KL and MSE
    both collapse to zero, the loss looks healthy, and the mechanism
    silently contributes nothing for those pairs. A single-batch check could
    miss a rare leak (e.g. an off-by-one that admits view 0 with low
    probability), so draw thousands of rows across many batches, many
    epochs, and several rng seeds."""
    b = _bank_with_clean_marker(tmp_path)
    marker = torch.full((4,), 999.0)
    total_rows = 0
    for seed in range(5):
        s = PairedSampler(b, np.arange(40), n_src=8, m_deg=3,
                          rng=np.random.default_rng(seed))
        for _epoch in range(20):
            for batch in s:
                is_marker = torch.isclose(batch["f_deg"], marker).all(dim=1)
                assert not is_marker.any()
                total_rows += batch["f_deg"].shape[0]
    assert total_rows > 5000, "test did not draw enough rows to be decisive"


def _bank_identity_encoded(tmp_path, n=40):
    """Every one of an image's 11 views stores the exact same value (its own
    index, exactly representable in float16 for n < 2048) in its single
    feature dim. f_clean and f_deg are therefore bit-identical if and only
    if they were drawn from the same source image."""
    w = BankWriter(str(tmp_path / "bi"), n, N_VIEWS, 1, "t", 0)
    for i in range(n):
        pres = np.zeros((N_VIEWS, 6), np.float32); pres[1:, 0] = 1.0
        sev = np.zeros((N_VIEWS, 6), np.float32); sev[1:, 0] = 0.4
        feats = np.full((N_VIEWS, 1), float(i), dtype=np.float32)
        w.write_image(i, {"path": f"/p{i}", "label": i % 2,
                          "generator": f"g{i % 3}", "source": "s", "split": "train"},
                      feats=feats, presence=pres, severity=sev,
                      proxies=np.zeros((N_VIEWS, 3), np.float32),
                      recipes=["[]"] * N_VIEWS)
    w.close()
    return FeatureBank.open(str(tmp_path / "bi"))


def test_pairs_never_cross_images(tmp_path):
    """The clean and degraded halves of every row must come from the SAME
    source image. A sampler that paired across images would make the
    consistency loss punish the model for distinguishing two genuinely
    different pictures -- actively harmful, and silent like the view-0 bug."""
    b = _bank_identity_encoded(tmp_path)
    s = PairedSampler(b, np.arange(40), n_src=8, m_deg=3, rng=np.random.default_rng(3))
    rows = 0
    for batch in s:
        assert torch.equal(batch["f_clean"], batch["f_deg"])
        rows += batch["f_clean"].shape[0]
    assert rows > 0


def test_generator_balance_is_stratified_not_proportional_to_pool_size(tmp_path):
    """Real AIGC datasets are imbalanced across generators. A sampler that
    draws uniformly from the raw pool reproduces that skew in every batch,
    which then shows up downstream as a generalisation result that is
    really a sampling artefact. Build a positive pool where one generator
    holds 94% of the images and confirm draws are close to balanced across
    all 5 families over many draws, not dominated by the majority one.
    (The positive pool is the 50 odd-indexed rows; gA covers 44 of them.)"""
    n = 100  # i % 2 == 1 gives 50 positive rows
    w = BankWriter(str(tmp_path / "bg"), n, N_VIEWS, 3, "t", 0)
    rng = np.random.default_rng(0)
    pos_gens = iter(["gA"] * 44 + ["gB"] * 2 + ["gC"] * 2 + ["gD"] + ["gE"])
    for i in range(n):
        pres = np.zeros((N_VIEWS, 6), np.float32); pres[1:, 0] = 1.0
        sev = np.zeros((N_VIEWS, 6), np.float32); sev[1:, 0] = 0.4
        label = i % 2
        gen = next(pos_gens) if label == 1 else "real"
        w.write_image(i, {"path": f"/p{i}", "label": label, "generator": gen,
                          "source": "s", "split": "train"},
                      feats=rng.normal(size=(N_VIEWS, 3)).astype(np.float32),
                      presence=pres, severity=sev,
                      proxies=np.zeros((N_VIEWS, 3), np.float32),
                      recipes=["[]"] * N_VIEWS)
    w.close()
    b = FeatureBank.open(str(tmp_path / "bg"))
    s = PairedSampler(b, np.arange(n), n_src=10, m_deg=1, rng=np.random.default_rng(7))

    n_draws = 5000
    drawn = s._draw_stratified(s.pos, n_draws)
    gens_drawn = s.generators[drawn]
    counts = {g: int((gens_drawn == g).sum()) for g in np.unique(gens_drawn)}

    assert set(counts) == {"gA", "gB", "gC", "gD", "gE"}
    for g, c in counts.items():
        frac = c / n_draws
        assert 0.10 < frac < 0.30, f"{g} drawn {frac:.3f} of the time, expected ~0.2"


def test_determinism_with_equivalently_seeded_generators(tmp_path):
    """Project rule: no global seeding. Two samplers built from
    equivalently-seeded generators must produce identical batch sequences."""
    b = _bank(tmp_path)
    s1 = PairedSampler(b, np.arange(40), n_src=8, m_deg=2, rng=np.random.default_rng(42))
    s2 = PairedSampler(b, np.arange(40), n_src=8, m_deg=2, rng=np.random.default_rng(42))
    n_batches = 0
    for batch1, batch2 in zip(s1, s2):
        for key in ("f_clean", "f_deg", "y_clean", "y_deg",
                    "presence_deg", "severity_deg"):
            assert torch.equal(batch1[key], batch2[key])
        n_batches += 1
    assert n_batches == len(s1)


def test_rng_is_required_no_implicit_seeding(tmp_path):
    b = _bank(tmp_path)
    with pytest.raises(TypeError):
        PairedSampler(b, np.arange(40), n_src=4, m_deg=2)


def test_augmented_only_draws_distinct_degraded_views(tmp_path):
    b = _bank(tmp_path)
    s = PairedSampler(b, np.arange(40), n_src=4, m_deg=5,
                      rng=np.random.default_rng(11), augmented_only=True)
    deg_views = s._draw_degraded_views(20)
    assert deg_views.min() >= 1 and deg_views.max() < N_VIEWS
    for row in deg_views:
        assert len(set(row.tolist())) == len(row)


def test_augmented_only_rejects_m_deg_larger_than_available_views(tmp_path):
    b = _bank(tmp_path)
    with pytest.raises(ValueError):
        PairedSampler(b, np.arange(40), n_src=4, m_deg=N_VIEWS,
                      rng=np.random.default_rng(0), augmented_only=True)
