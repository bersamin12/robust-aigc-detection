"""How a ViT's token sequence becomes the one vector the head reads.

Every fine-tuned arm to date pools the same way the frozen banks do: the mean
of the final-layer patch tokens, CLS and registers dropped
(`features.backbones._pool`). The NTIRE 2026 report (arXiv 2604.11487) has
the strongest entries reading the tower differently -- Ant International's
two experts use attention pooling and first-token (CLS) pooling, Shallow Real
concatenates CLS, the four register tokens and the patch mean -- so the choice
is a measurable variable here rather than a fixed convention.

The pooler is part of the HEAD, not the tower: it is trained at the head's
learning rate, averaged by SWA with the head, and saved under the head's
state dict, so a checkpoint's `dim_feat` and `config["token_pool"]` together
describe the readout with nothing implicit. `TOKEN_POOL_MEAN` is the
convention every earlier checkpoint used; for it `build_pooler` returns
`None`, and `_forward_tower` takes the historical `_pool` path, bit for bit.
"""
from __future__ import annotations

import torch
import torch.nn as nn

#: The historical readout: mean over final-layer patch tokens.
TOKEN_POOL_MEAN = "mean"
#: The CLS token alone (DINOv2's `pooler_output`).
TOKEN_POOL_CLS = "cls"
#: CLS concatenated with the patch mean.
TOKEN_POOL_CLS_MEAN = "cls_mean"
#: CLS, every register token, and the patch mean, concatenated.
TOKEN_POOL_CLS_REG_MEAN = "cls_reg_mean"
#: One learned query attending over the patch tokens (multi-head), then a
#: LayerNorm. The only mode with parameters of its own.
TOKEN_POOL_ATTN = "attn"
TOKEN_POOLS = (TOKEN_POOL_MEAN, TOKEN_POOL_CLS, TOKEN_POOL_CLS_MEAN,
               TOKEN_POOL_CLS_REG_MEAN, TOKEN_POOL_ATTN)


def token_pool_dim(mode: str, dim: int, num_prefix_tokens: int) -> int:
    """Width of the pooled vector for one tower of hidden size `dim`."""
    if mode not in TOKEN_POOLS:
        raise ValueError(f"token_pool must be one of {TOKEN_POOLS}, got {mode!r}")
    if mode == TOKEN_POOL_CLS_MEAN:
        return 2 * dim
    if mode == TOKEN_POOL_CLS_REG_MEAN:
        if num_prefix_tokens < 2:
            raise ValueError(
                f"{mode} reads the register tokens, and this tower has "
                f"num_prefix_tokens={num_prefix_tokens} (CLS only). Use "
                f"{TOKEN_POOL_CLS_MEAN}, which is the same readout without them.")
        return (num_prefix_tokens + 1) * dim
    return dim


class TokenPooler(nn.Module):
    """(B, T, D) tokens -> (B, out_dim) float32, per `mode`.

    Token 0 is CLS and tokens 1..num_prefix_tokens-1 are registers, which is
    DINOv2-with-registers' layout and what `BackboneSpec.num_prefix_tokens`
    already encodes for every ViT in the registry.
    """

    def __init__(self, mode: str, dim: int, num_prefix_tokens: int,
                 heads: int = 8):
        super().__init__()
        self.out_dim = token_pool_dim(mode, dim, num_prefix_tokens)
        self.mode, self.dim = mode, int(dim)
        self.num_prefix_tokens = int(num_prefix_tokens)
        if mode == TOKEN_POOL_ATTN:
            if dim % heads:
                raise ValueError(f"dim {dim} is not divisible by {heads} heads")
            self.query = nn.Parameter(torch.empty(1, 1, dim))
            nn.init.trunc_normal_(self.query, std=0.02)
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3:
            raise ValueError(f"expected (B, T, D) tokens, got {tuple(h.shape)}")
        p = self.num_prefix_tokens
        patches = h[:, p:, :]
        if self.mode == TOKEN_POOL_MEAN:
            # Identical to `features.backbones._pool`: mean in the tower's
            # dtype, then float32.
            return patches.mean(dim=1).float()
        if self.mode == TOKEN_POOL_CLS:
            return h[:, 0, :].float()
        if self.mode == TOKEN_POOL_CLS_MEAN:
            return torch.cat([h[:, 0, :], patches.mean(dim=1)], dim=-1).float()
        if self.mode == TOKEN_POOL_CLS_REG_MEAN:
            return torch.cat([h[:, 0, :], h[:, 1:p, :].flatten(1),
                              patches.mean(dim=1)], dim=-1).float()
        # attn. The pooler's parameters are float32 (it lives in the head) and
        # the tokens arrive in whatever the tower runs in, so cast once here;
        # under autocast the matmuls still run in the autocast dtype.
        x = patches.to(self.query.dtype)
        q = self.query.expand(x.shape[0], -1, -1)
        out, _ = self.attn(q, x, x, need_weights=False)
        return self.norm(out[:, 0, :]).float()

    def extra_repr(self) -> str:
        return (f"mode={self.mode}, dim={self.dim}, "
                f"num_prefix_tokens={self.num_prefix_tokens}, out_dim={self.out_dim}")


def build_pooler(mode: str, specs) -> nn.Module | None:
    """One `TokenPooler` per tower spec, or `None` for the historical mean.

    A single spec gives a bare pooler; several give an `nn.ModuleList` in the
    towers' order, which is the order the head's input concatenates them in.
    """
    if mode not in TOKEN_POOLS:
        raise ValueError(f"token_pool must be one of {TOKEN_POOLS}, got {mode!r}")
    specs = list(specs)
    if not specs:
        raise ValueError("build_pooler needs at least one tower spec")
    if mode == TOKEN_POOL_MEAN:
        return None
    poolers = [TokenPooler(mode, sp.dim, sp.num_prefix_tokens) for sp in specs]
    return poolers[0] if len(poolers) == 1 else nn.ModuleList(poolers)


def pooled_dim(mode: str, specs) -> int:
    """The head's input width: the towers' pooled widths, summed."""
    return sum(token_pool_dim(mode, sp.dim, sp.num_prefix_tokens) for sp in specs)


def tower_poolers(head: nn.Module, n_towers: int) -> list:
    """The per-tower pooler list `_forward_tower` wants, from a head built by
    `Detector(..., pooler=build_pooler(...))`. `None` entries mean `_pool`."""
    p = getattr(head, "pooler", None)
    if p is None:
        return [None] * n_towers
    if isinstance(p, nn.ModuleList):
        if len(p) != n_towers:
            raise ValueError(
                f"the head holds {len(p)} poolers and this run has {n_towers} towers")
        return list(p)
    if n_towers != 1:
        raise ValueError(f"the head holds one pooler and this run has {n_towers} towers")
    return [p]
