"""`aigcdet.models.pooling`: the readout variants from the NTIRE 2026 report."""
import pytest
import torch

from aigcdet.models.heads import Detector
from aigcdet.models.pooling import (
    TOKEN_POOL_ATTN, TOKEN_POOL_CLS, TOKEN_POOL_CLS_MEAN, TOKEN_POOL_CLS_REG_MEAN,
    TOKEN_POOL_MEAN, TOKEN_POOLS, TokenPooler, build_pooler, pooled_dim,
    token_pool_dim, tower_poolers,
)


class _Spec:
    def __init__(self, dim, num_prefix_tokens):
        self.dim, self.num_prefix_tokens = dim, num_prefix_tokens


D, P, N = 16, 5, 7          # dinov2regl's layout: CLS + 4 registers, then patches


def _tokens(b=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, P + N, D, generator=g)


@pytest.mark.parametrize("mode,width", [
    (TOKEN_POOL_MEAN, D), (TOKEN_POOL_CLS, D), (TOKEN_POOL_CLS_MEAN, 2 * D),
    (TOKEN_POOL_CLS_REG_MEAN, (P + 1) * D), (TOKEN_POOL_ATTN, D)])
def test_each_mode_pools_to_its_declared_width(mode, width):
    pl = TokenPooler(mode, D, P)
    assert pl.out_dim == token_pool_dim(mode, D, P) == width
    out = pl(_tokens())
    assert out.shape == (3, width) and out.dtype == torch.float32


def test_mean_is_the_bank_readout_patch_mean_without_prefix_tokens():
    h = _tokens()
    assert torch.equal(TokenPooler(TOKEN_POOL_MEAN, D, P)(h), h[:, P:].mean(1))


def test_cls_is_token_zero():
    h = _tokens()
    assert torch.equal(TokenPooler(TOKEN_POOL_CLS, D, P)(h), h[:, 0])


def test_cls_reg_mean_concatenates_in_token_order():
    h = _tokens()
    out = TokenPooler(TOKEN_POOL_CLS_REG_MEAN, D, P)(h)
    assert torch.equal(out[:, :D], h[:, 0])
    for r in range(1, P):
        assert torch.equal(out[:, r * D:(r + 1) * D], h[:, r])
    assert torch.equal(out[:, P * D:], h[:, P:].mean(1))


def test_cls_mean_is_cls_then_patch_mean():
    h = _tokens()
    out = TokenPooler(TOKEN_POOL_CLS_MEAN, D, P)(h)
    assert torch.equal(out[:, :D], h[:, 0]) and torch.equal(out[:, D:], h[:, P:].mean(1))


def test_cls_reg_mean_needs_registers():
    with pytest.raises(ValueError, match="register"):
        token_pool_dim(TOKEN_POOL_CLS_REG_MEAN, D, 1)


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        TokenPooler("gem", D, P)
    with pytest.raises(ValueError):
        build_pooler("gem", [_Spec(D, P)])


def test_only_attention_pooling_has_parameters():
    for mode in TOKEN_POOLS:
        n = sum(p.numel() for p in TokenPooler(mode, D, P).parameters())
        assert (n > 0) == (mode == TOKEN_POOL_ATTN), mode


def test_attention_pooling_is_deterministic_and_trainable():
    torch.manual_seed(0)
    a = TokenPooler(TOKEN_POOL_ATTN, D, P)
    torch.manual_seed(0)
    b = TokenPooler(TOKEN_POOL_ATTN, D, P)
    h = _tokens()
    assert torch.equal(a(h), b(h))
    # A random projection, not .sum(): a LayerNorm output sums to zero
    # identically, so .sum() would carry no gradient to anything.
    (a(h) * torch.randn(3, D, generator=torch.Generator().manual_seed(1))).sum().backward()
    assert a.query.grad is not None and a.query.grad.abs().sum() > 0


def test_attention_pooling_reads_only_the_patches():
    """Registers absorb global junk; the learned query must not see them."""
    torch.manual_seed(0)
    pl = TokenPooler(TOKEN_POOL_ATTN, D, P)
    h = _tokens()
    h2 = h.clone()
    h2[:, :P] += 100.0
    assert torch.allclose(pl(h), pl(h2))


def test_attention_pooling_accepts_a_half_precision_tower():
    torch.manual_seed(0)
    pl = TokenPooler(TOKEN_POOL_ATTN, D, P)
    out = pl(_tokens().to(torch.bfloat16))
    assert out.dtype == torch.float32 and torch.isfinite(out).all()


def test_build_pooler_is_none_for_the_mean_and_a_list_for_two_towers():
    specs = [_Spec(D, P), _Spec(D, P)]
    assert build_pooler(TOKEN_POOL_MEAN, specs) is None
    one = build_pooler(TOKEN_POOL_CLS, specs[:1])
    assert isinstance(one, TokenPooler)
    two = build_pooler(TOKEN_POOL_CLS, specs)
    assert isinstance(two, torch.nn.ModuleList) and len(two) == 2
    assert pooled_dim(TOKEN_POOL_CLS_MEAN, specs) == 4 * D


def test_tower_poolers_matches_the_head_to_the_tower_count():
    specs = [_Spec(D, P), _Spec(D, P)]
    head = Detector(dim_feat=2 * D, pooler=build_pooler(TOKEN_POOL_CLS, specs))
    assert len(tower_poolers(head, 2)) == 2
    with pytest.raises(ValueError):
        tower_poolers(head, 1)
    plain = Detector(dim_feat=D)
    assert tower_poolers(plain, 3) == [None, None, None]
    single = Detector(dim_feat=D, pooler=build_pooler(TOKEN_POOL_CLS, specs[:1]))
    assert tower_poolers(single, 1)[0] is single.pooler
    with pytest.raises(ValueError):
        tower_poolers(single, 2)


def test_the_pooler_lives_in_the_heads_state_dict():
    """Trained, averaged and saved with the head -- so it must be in there."""
    torch.manual_seed(0)
    head = Detector(dim_feat=D, pooler=build_pooler(TOKEN_POOL_ATTN, [_Spec(D, P)]))
    keys = [k for k in head.state_dict() if k.startswith("pooler.")]
    assert "pooler.query" in keys and any("attn" in k for k in keys)
    # And a legacy head (no pooler) loads a legacy state dict unchanged.
    legacy = Detector(dim_feat=D)
    legacy.load_state_dict(Detector(dim_feat=D).state_dict())
    assert not any(k.startswith("pooler.") for k in legacy.state_dict())
