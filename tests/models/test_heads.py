import torch

from aigcdet.models.heads import ClassifierHead, DegradationHead, Detector


def test_degradation_head_shapes_and_severity_range():
    h = DegradationHead(dim_in=32)
    out = h(torch.randn(4, 32))
    assert out["presence"].shape == (4, 6)
    assert out["severity"].shape == (4, 6)
    assert out["embedding"].shape == (4, 256)
    assert (out["severity"] >= 0).all() and (out["severity"] <= 1).all()


def test_classifier_head_shapes():
    c = ClassifierHead(dim_in=32)
    out = c(torch.randn(5, 32))
    assert out["logit"].shape == (5,) and out["hidden"].shape == (5, 512)


def test_degradation_head_output_depends_on_input():
    """Guards against a stub that returns input-independent output: the shape
    and severity-range checks above would not catch that, but two distinct
    inputs must produce distinct presence/severity/embedding here."""
    h = DegradationHead(dim_in=32)
    a = h(torch.zeros(4, 32))
    b = h(torch.ones(4, 32))
    assert not torch.allclose(a["presence"], b["presence"])
    assert not torch.allclose(a["severity"], b["severity"])
    assert not torch.allclose(a["embedding"], b["embedding"])


def test_film_changes_the_hidden_state_and_plain_head_ignores_cond():
    """FiLM is WIRED to `cond` -- asserted on a TRAINED projection.

    The projection is zero-initialised (see `test_film_is_the_identity_at_
    initialisation`), so at step 0 it is a pass-through and `cond` provably
    cannot matter. Filling the weights here is what keeps this test about the
    wiring rather than about the initialisation; without it the assertion
    below would be testing that the zero-init is absent.
    """
    # fork_rng(devices=[]) forks only the CPU RNG (never touches CUDA, so no
    # GPU process starts) and restores it on exit, so the manual_seed below
    # cannot leak into later tests' unseeded randomness.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        f = torch.randn(3, 32)
        cond = torch.randn(3, 256)
        film = ClassifierHead(dim_in=32, use_film=True)
        with torch.no_grad():          # stand in for a trained projection
            film.film.weight.normal_(0.0, 0.1)
            film.film.bias.normal_(0.0, 0.1)
        a = film(f, cond)["hidden"]
        b = film(f, torch.zeros_like(cond))["hidden"]
        assert not torch.allclose(a, b)
        plain = ClassifierHead(dim_in=32, use_film=False)
        assert torch.allclose(plain(f, cond)["hidden"], plain(f)["hidden"])


def test_film_is_the_identity_at_initialisation():
    """A freshly built FiLM head must return EXACTLY the un-conditioned hidden
    state, for any `cond`.

    Default `nn.Linear` init emits random gamma/beta, and FiLM's output is not
    renormalised, so an untrained block applies an arbitrary affine to a
    LayerNorm-ed `h`. Rung a7_norecon measured the cost on 2026-08-30: the
    consistency term opened at con=44.6 against a3's 0.032 and ran away to
    1.5e8, collapsing the classifier to constant output (val_auc 0.5031). With
    the projection zeroed, A7 starts from its base rung exactly, so the rung
    measures FiLM instead of measuring a random perturbation.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        f = torch.randn(4, 32)
        film = ClassifierHead(dim_in=32, use_film=True)
        plain_h = film(f, None)["hidden"]
        for scale in (1.0, 100.0):     # no cond magnitude may perturb it
            cond = torch.randn(4, 256) * scale
            assert torch.equal(film(f, cond)["hidden"], plain_h)


def test_film_renormalises_so_hidden_cannot_run_away():
    """`hidden` must stay on the LayerNorm scale for ANY gamma/beta.

    A3's consistency term is an MSE over `hidden` (train_head.py:168-170).
    Un-normalised FiLM makes that term unbounded: inflating gamma is a
    cheaper way to move the loss than matching clean to degraded, which is
    how a7_norecon reached con=1.5e8 and a constant classifier. The zero-init
    alone did not fix it (0.0296 at val_auc 0.5601 on 2026-08-30) because it
    only sets the starting point. This pins the trajectory instead: even a
    projection scaled to emit huge gamma leaves `hidden` at unit RMS.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        f = torch.randn(8, 32)
        cond = torch.randn(8, 256)
        plain = ClassifierHead(dim_in=32, use_film=False)
        baseline = plain(f)["hidden"].pow(2).mean().sqrt()
        film = ClassifierHead(dim_in=32, use_film=True)
        for scale in (1.0, 10.0, 1000.0):
            with torch.no_grad():
                film.film.weight.normal_(0.0, scale)
                film.film.bias.normal_(0.0, scale)
            rms = film(f, cond)["hidden"].pow(2).mean().sqrt()
            # LayerNorm pins RMS to ~1 regardless of the modulation size.
            assert torch.isclose(rms, baseline, rtol=0.2), (scale, rms, baseline)


def test_film_leaves_the_identity_once_it_is_trained():
    """The zero-init must be a STARTING point, not a dead branch: the film
    projection has to receive gradient, or A7 is A3 with extra parameters and
    a negative result would be unfalsifiable."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        head = ClassifierHead(dim_in=32, use_film=True)
        out = head(torch.randn(4, 32), torch.randn(4, 256))
        out["logit"].sum().backward()
        assert head.film.weight.grad is not None
        assert head.film.weight.grad.abs().sum() > 0


def test_detector_without_recon_matches_feature_width():
    d = Detector(dim_feat=16, use_recon=False)
    out = d(torch.randn(2, 16))
    assert out["logit"].shape == (2,) and out["presence"].shape == (2, 6)


def test_detector_with_recon_consumes_the_concatenated_width():
    d = Detector(dim_feat=16, use_recon=True, recon_dim=12)
    out = d(torch.randn(2, 16), torch.randn(2, 12))
    assert out["logit"].shape == (2,)


def test_detector_raises_when_recon_expected_but_missing():
    import pytest
    d = Detector(dim_feat=16, use_recon=True, recon_dim=12)
    with pytest.raises(ValueError, match="recon"):
        d(torch.randn(2, 16))


def test_stop_gradient_isolates_the_degradation_head_when_film_is_on():
    """With FiLM enabled, no classifier gradient may reach the degradation head
    (spec §3.4): otherwise `d` stops meaning 'degradation'."""
    d = Detector(dim_feat=16, use_recon=False, use_film=True)
    out = d(torch.randn(4, 16))
    out["logit"].sum().backward()
    grads = [p.grad for p in d.degradation.parameters() if p.grad is not None]
    assert all(g.abs().sum() == 0 for g in grads) or not grads


def test_trainable_parameter_count_is_small():
    d = Detector(dim_feat=1024, use_recon=True)
    n = sum(p.numel() for p in d.parameters() if p.requires_grad)
    assert n < 3_000_000, f"heads should stay ~2M parameters, got {n}"
