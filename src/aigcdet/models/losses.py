"""Losses (spec §3.5).

    L = L_cls                                  # BCE, class-balanced
      + lambda_deg * L_deg                     # degradation multi-task
      + alpha * KL(p_clean || p_degraded)      # prediction consistency
      + beta  * MSE(h_c_clean, h_c_degraded)   # feature consistency

The feature-consistency term acts on `hidden`, the classifier head's live,
trainable ~512-d state h_c (see aigcdet.models.heads.Detector), never the
frozen cached backbone embedding. An earlier draft wrote it on the cached
embedding, where clean and degraded values are both constants with no
gradient path at all: the term would appear in the loss, the config and the
ablation table while contributing exactly nothing.

The prediction-consistency term is the ASYMMETRIC KL(p_clean || p_degraded):
the clean prediction is the reference distribution the degraded prediction is
pulled towards. KL(p || q) != KL(q || p) in general, so callers must not
treat the argument order as interchangeable.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

_EPS = 1e-6


@dataclass
class LossWeights:
    lambda_deg: float = 0.3
    alpha: float = 1.0
    beta: float = 1.0


def classification_loss(logit: torch.Tensor, y: torch.Tensor,
                         pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Class-balanced BCE (spec §3.5, L_cls)."""
    return F.binary_cross_entropy_with_logits(logit, y.float(), pos_weight=pos_weight)


def degradation_loss(pred_presence: torch.Tensor, pred_severity: torch.Tensor,
                      tgt_presence: torch.Tensor, tgt_severity: torch.Tensor) -> torch.Tensor:
    """Multi-task degradation loss (spec §3.5, L_deg).

    Presence is BCE over all families. Severity is smooth-L1, masked to
    families that are actually present: an absent family's severity target
    is meaningless (the transform was never applied), so a wrong prediction
    there must not be penalised.
    """
    pres = F.binary_cross_entropy_with_logits(pred_presence, tgt_presence.float())
    mask = tgt_presence.float()
    denom = mask.sum().clamp(min=1.0)
    sev = (F.smooth_l1_loss(pred_severity, tgt_severity, reduction="none") * mask).sum() / denom
    return pres + sev


def _kl_bernoulli(logit_p: torch.Tensor, logit_q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) for Bernoulli distributions given as logits.

    Asymmetric: KL(p || q) != KL(q || p) in general. `p` (from `logit_p`) is
    the reference distribution.
    """
    p = torch.sigmoid(logit_p).clamp(_EPS, 1 - _EPS)
    q = torch.sigmoid(logit_q).clamp(_EPS, 1 - _EPS)
    kl = p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()
    return kl.mean()


def consistency_loss(logit_clean: torch.Tensor, logit_deg: torch.Tensor,
                      hidden_clean: torch.Tensor, hidden_deg: torch.Tensor,
                      alpha: float, beta: float) -> torch.Tensor:
    """Clean/degraded consistency (spec §3.5): prediction KL + feature MSE.

    `hidden_clean`/`hidden_deg` must be the classifier's trainable hidden
    state h_c, never a frozen/cached embedding (see module docstring).

    This function does not detach either side itself: callers that want the
    clean branch to act as a fixed target (rather than both branches meeting
    in the middle by collapsing the representation, as `total_loss` does via
    `.detach()` on `logit_clean`/`hidden_clean`) must detach it before calling.
    """
    pred = _kl_bernoulli(logit_clean, logit_deg)
    feat = F.mse_loss(hidden_deg, hidden_clean)
    return alpha * pred + beta * feat


def total_loss(out_clean: dict, out_deg: dict, batch: dict,
               weights: LossWeights, pos_weight: torch.Tensor | None = None
               ) -> tuple[torch.Tensor, dict[str, float]]:
    """Composite loss (spec §3.5).

    `out_clean`/`out_deg` are `Detector` forward outputs (logit, hidden,
    presence, severity). `batch` supplies y_clean, y_deg, presence_deg,
    severity_deg.

    The clean branch is detached before entering the consistency term: it is
    treated as the fixed target the degraded branch is pulled towards,
    rather than letting both drift to meet in the middle, which a symmetric
    loss could satisfy by collapsing the representation instead of becoming
    robust to degradation.
    """
    l_cls = 0.5 * (classification_loss(out_clean["logit"], batch["y_clean"], pos_weight)
                   + classification_loss(out_deg["logit"], batch["y_deg"], pos_weight))
    l_deg = degradation_loss(out_deg["presence"], out_deg["severity"],
                              batch["presence_deg"], batch["severity_deg"])
    l_con = consistency_loss(out_clean["logit"].detach(), out_deg["logit"],
                              out_clean["hidden"].detach(), out_deg["hidden"],
                              weights.alpha, weights.beta)
    total = l_cls + weights.lambda_deg * l_deg + l_con
    return total, {"cls": l_cls.detach().item(), "deg": l_deg.detach().item(),
                    "con": l_con.detach().item(), "total": total.detach().item()}
