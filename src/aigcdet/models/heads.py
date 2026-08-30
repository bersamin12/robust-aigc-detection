"""Trainable heads (spec §3.4). Everything here runs on cached embeddings.

The headline model does NOT use FiLM: the degradation head feeds calibration,
EQI, and the dashboard, not the classifier. Conditioning is rung A7, a
hypothesis under test, because DCPT reports that architectural additions
overfit on limited training data.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from aigcdet.augment.recipes import N_FAMILIES


class DegradationHead(nn.Module):
    def __init__(self, dim_in: int, hidden: int = 256, n_families: int = N_FAMILIES):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dim_in, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.presence = nn.Linear(hidden, n_families)
        self.severity = nn.Linear(hidden, n_families)

    def forward(self, f: torch.Tensor) -> dict[str, torch.Tensor]:
        e = self.trunk(f)
        return {"presence": self.presence(e),
                "severity": torch.sigmoid(self.severity(e)),
                "embedding": e}


class ClassifierHead(nn.Module):
    def __init__(self, dim_in: int, hidden: int = 512,
                 use_film: bool = False, cond_dim: int = 256):
        super().__init__()
        self.use_film = use_film
        self.trunk = nn.Sequential(
            nn.Linear(dim_in, hidden), nn.GELU(), nn.LayerNorm(hidden))
        if use_film:
            self.film = nn.Linear(cond_dim, hidden * 2)
            # Identity at initialisation. With default `nn.Linear` init this
            # layer emits random gamma/beta, so `(1 + gamma) * h + beta`
            # applies an arbitrary affine to a freshly LayerNorm-ed `h` before
            # a single gradient step -- and FiLM's output is NOT renormalised.
            # Measured cost of not doing this (rung a7_norecon, 2026-08-30):
            # the consistency term starts at con=44.6 instead of a3's 0.032,
            # 1400x larger, and runs away to 1.5e8 over 30 epochs while
            # `cls` pins to ln(2) -- a classifier collapsed to constant output
            # at val_auc 0.5031. Zeroing the projection makes the block an
            # exact pass-through at step 0, so A7 starts from its own base
            # rung rather than from a random perturbation of it, which is what
            # makes "does FiLM help?" a question about FiLM.
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
            # Renormalise AFTER the modulation. `hidden` is what A3's
            # consistency term takes an MSE over (train_head.py:168-170), and
            # `(1 + gamma) * h + beta` is unbounded: nothing constrains how
            # far a conditioned `h` can drift from the LayerNorm-ed scale the
            # loss weight was tuned against. Zeroing the projection fixes the
            # STARTING point but not the trajectory -- rung a7_norecon still
            # diverged after the zero-init (2026-08-30: 0.0296 at val_auc
            # 0.5601), because once gamma leaves zero the consistency MSE
            # grows quadratically in |h| and minimising it by inflating
            # gamma is cheaper than matching clean to degraded. This puts
            # `hidden` back on the same scale as every non-FiLM rung, so the
            # loss weights transfer and A7 is comparable to A3 rather than
            # being a different optimisation problem wearing the same name.
            #
            # Applied whether or not `cond` is supplied, so the block's
            # architecture does not depend on a call-site argument -- that is
            # what keeps the zero-init an EXACT identity (LayerNorm is
            # idempotent on an already-normalised `h` at unit weight and zero
            # bias) rather than an approximate one.
            self.film_norm = nn.LayerNorm(hidden)
        self.out = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(),
                                 nn.Linear(hidden // 2, 1))

    def forward(self, f: torch.Tensor, cond: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        h = self.trunk(f)
        if self.use_film:
            if cond is not None:
                gamma, beta = self.film(cond).chunk(2, dim=-1)
                h = (1.0 + gamma) * h + beta
            h = self.film_norm(h)
        return {"logit": self.out(h).squeeze(-1), "hidden": h}


class Detector(nn.Module):
    """Degradation head + classifier head over a cached embedding."""

    def __init__(self, dim_feat: int, use_recon: bool = False,
                 recon_dim: int = 12, use_film: bool = False):
        super().__init__()
        self.use_recon = use_recon
        self.use_film = use_film
        dim_in = dim_feat + (recon_dim if use_recon else 0)
        self.degradation = DegradationHead(dim_in)
        self.classifier = ClassifierHead(dim_in, use_film=use_film)

    def forward(self, f: torch.Tensor, r: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if self.use_recon:
            if r is None:
                raise ValueError("this Detector expects recon features `r`")
            f = torch.cat([f, r], dim=-1)
        deg = self.degradation(f)
        # Stop-gradient: the classifier must not reshape `d` into a general
        # purpose feature, or the degradation readout stops being meaningful.
        cond = deg["embedding"].detach() if self.use_film else None
        cls = self.classifier(f, cond)
        return {"logit": cls["logit"], "hidden": cls["hidden"],
                "presence": deg["presence"], "severity": deg["severity"],
                "deg_embedding": deg["embedding"]}
