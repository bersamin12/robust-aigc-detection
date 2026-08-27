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


def test_film_changes_the_hidden_state_and_plain_head_ignores_cond():
    torch.manual_seed(0)
    f = torch.randn(3, 32)
    cond = torch.randn(3, 256)
    film = ClassifierHead(dim_in=32, use_film=True)
    a = film(f, cond)["hidden"]
    b = film(f, torch.zeros_like(cond))["hidden"]
    assert not torch.allclose(a, b)
    plain = ClassifierHead(dim_in=32, use_film=False)
    assert torch.allclose(plain(f, cond)["hidden"], plain(f)["hidden"])


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
