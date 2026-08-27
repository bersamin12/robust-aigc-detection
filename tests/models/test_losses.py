import pytest
import torch

from aigcdet.models.heads import ClassifierHead, DegradationHead, Detector
from aigcdet.models.losses import (
    LossWeights, classification_loss, consistency_loss, degradation_loss, total_loss,
)


def test_classification_loss_is_near_zero_for_confident_correct_logits():
    logit = torch.tensor([10.0, -10.0])
    y = torch.tensor([1.0, 0.0])
    assert classification_loss(logit, y).item() < 1e-3


def test_classification_loss_is_large_for_confident_wrong_logits():
    """Sibling of the near-zero test above, so a constant-0.0 stub cannot
    pass both: the near-zero test alone is satisfied by such a stub."""
    logit = torch.tensor([10.0, -10.0])
    y = torch.tensor([0.0, 1.0])  # exactly backwards
    assert classification_loss(logit, y).item() > 5.0


def test_degradation_loss_is_zero_at_a_perfect_prediction():
    tgt_p = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    tgt_s = torch.tensor([[0.7, 0.0, 0.0, 0.0, 0.0, 0.0]])
    pred_p = torch.tensor([[20.0, -20.0, -20.0, -20.0, -20.0, -20.0]])
    loss = degradation_loss(pred_p, tgt_s.clone(), tgt_p, tgt_s)
    assert loss.item() < 1e-3


def test_severity_error_is_masked_to_present_families():
    """A wrong severity on an absent family must not be penalised: its target
    is meaningless when the transform was never applied."""
    tgt_p = torch.tensor([[1.0, 0.0] + [0.0] * 4])
    tgt_s = torch.tensor([[0.5, 0.0] + [0.0] * 4])
    pred_p = torch.tensor([[20.0, -20.0] + [-20.0] * 4])
    good = degradation_loss(pred_p, tgt_s.clone(), tgt_p, tgt_s)
    noisy = tgt_s.clone()
    noisy[0, 1] = 0.9  # absent family, wrong severity
    same = degradation_loss(pred_p, noisy, tgt_p, tgt_s)
    assert torch.isclose(good, same, atol=1e-6)


def test_degradation_loss_is_large_for_a_confidently_wrong_prediction():
    """Neither of the two tests above would fail against a constant-0.0
    stub on its own (a perfect prediction and a masked-out disagreement both
    legitimately score near zero); this one requires an actual wrong,
    present-family prediction to score high."""
    tgt_p = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    tgt_s = torch.tensor([[0.9, 0.0, 0.0, 0.0, 0.0, 0.0]])
    pred_p = torch.tensor([[-20.0, 20.0, 20.0, 20.0, 20.0, 20.0]])  # presence exactly backwards
    pred_s = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # severity wrong on the present family
    assert degradation_loss(pred_p, pred_s, tgt_p, tgt_s).item() > 5.0


def test_consistency_loss_is_zero_when_clean_and_degraded_agree():
    lg = torch.tensor([1.5, -0.3])
    hd = torch.randn(2, 8, generator=torch.Generator().manual_seed(0))
    assert consistency_loss(lg, lg.clone(), hd, hd.clone(), 1.0, 1.0).item() == pytest.approx(0.0, abs=1e-6)


def test_consistency_loss_grows_with_disagreement():
    lg_a = torch.tensor([2.0, 2.0])
    h = torch.randn(2, 8, generator=torch.Generator().manual_seed(1))
    near = consistency_loss(lg_a, torch.tensor([1.9, 1.9]), h, h.clone(), 1.0, 1.0)
    far = consistency_loss(lg_a, torch.tensor([-4.0, -4.0]), h, h.clone(), 1.0, 1.0)
    assert far > near


def test_consistency_gradient_reaches_the_hidden_state():
    """Guards the v1 bug: the feature term must act on a trainable tensor."""
    gen = torch.Generator().manual_seed(4)
    h_clean = torch.randn(3, 8, generator=gen, requires_grad=True)
    h_deg = torch.randn(3, 8, generator=gen, requires_grad=True)
    lg = torch.zeros(3, requires_grad=True)
    consistency_loss(lg, lg.clone(), h_clean, h_deg, 0.0, 1.0).backward()
    assert h_deg.grad is not None and h_deg.grad.abs().sum() > 0


def test_loss_weights_defaults_are_explicit():
    w = LossWeights()
    assert w.lambda_deg > 0 and w.alpha > 0 and w.beta > 0


# --- Additional tests: the KL direction, and that switching a coefficient to
# zero genuinely removes that term's gradient contribution, not just its
# value. These are the assertions that keep the ablation ladder and the
# project's headline claim honest. ---


def test_kl_direction_is_clean_to_degraded_not_symmetric():
    """spec §3.5: alpha * KL(p_clean || p_deg). KL is asymmetric, so this is
    NOT the same as KL(p_deg || p_clean) or a symmetrised average of the two.
    A future swap of the argument order inside the KL term must fail this
    test rather than pass silently (both directions are non-negative, so a
    weaker test would not catch a swap)."""
    logit_clean = torch.tensor([4.0])   # p_clean ~ 0.982
    logit_deg = torch.tensor([0.0])     # p_deg = 0.5
    h = torch.zeros(1, 4)

    loss = consistency_loss(logit_clean, logit_deg, h, h.clone(), alpha=1.0, beta=0.0)

    eps = 1e-6
    p = torch.sigmoid(logit_clean).clamp(eps, 1 - eps)
    q = torch.sigmoid(logit_deg).clamp(eps, 1 - eps)
    kl_clean_deg = (p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()).mean()
    kl_deg_clean = (q * (q / p).log() + (1 - q) * ((1 - q) / (1 - p)).log()).mean()

    assert not torch.isclose(kl_clean_deg, kl_deg_clean, atol=1e-4), \
        "test is degenerate: the two directions coincide for this input"
    assert torch.isclose(loss, kl_clean_deg, atol=1e-5)
    assert not torch.isclose(loss, kl_deg_clean, atol=1e-5)


def test_alpha_zero_removes_prediction_consistency_gradient_contribution():
    """Isolate the KL(prediction) term from the MSE(feature) term: hidden
    states are identical constants here, so all gradient on `w` can only flow
    through the prediction term. alpha=0 must zero that gradient exactly."""
    w = torch.tensor(2.0, requires_grad=True)
    logit_clean = torch.zeros(1)
    h = torch.zeros(1, 3)

    consistency_loss(logit_clean, w * torch.ones(1), h, h.clone(), alpha=1.0, beta=1.0).backward()
    grad_on = w.grad.clone()
    w.grad = None

    consistency_loss(logit_clean, w * torch.ones(1), h, h.clone(), alpha=0.0, beta=1.0).backward()
    grad_off = w.grad

    assert grad_on.abs().item() > 0
    assert grad_off.abs().item() == pytest.approx(0.0)


def test_beta_zero_removes_feature_consistency_gradient_contribution():
    """Mirror of the alpha test: logits are identical constants, so gradient
    on `w` can only flow through the MSE(feature) term. beta=0 must zero it."""
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    hidden_clean = torch.zeros(1, 2)
    logit = torch.zeros(1)

    consistency_loss(logit, logit.clone(), hidden_clean, w.unsqueeze(0), alpha=1.0, beta=1.0).backward()
    grad_on = w.grad.clone()
    w.grad = None

    consistency_loss(logit, logit.clone(), hidden_clean, w.unsqueeze(0), alpha=1.0, beta=0.0).backward()
    grad_off = w.grad

    assert grad_on.abs().sum().item() > 0
    assert grad_off.abs().sum().item() == pytest.approx(0.0)


def test_consistency_gradient_reaches_real_classifier_parameters():
    """The bug this task exists to avoid: an earlier design applied the
    consistency term to the frozen, cached backbone embedding. With a frozen
    backbone both the clean and degraded embeddings are constants, so that
    term would have zero gradient path despite appearing in the loss, the
    config and the ablation table.

    Here it must act on `hidden`, the live output of Detector.classifier
    (h_c), which carries a real grad_fn back to the classifier's own
    trainable weights. This is the single most important assertion in this
    module: it fails loudly if a future change reroutes the term back onto a
    detached or cached tensor.
    """
    gen = torch.Generator().manual_seed(0)
    detector = Detector(dim_feat=16)
    f_clean = torch.randn(4, 16, generator=gen)
    f_deg = torch.randn(4, 16, generator=gen)

    out_clean = detector(f_clean)
    out_deg = detector(f_deg)

    loss = consistency_loss(out_clean["logit"].detach(), out_deg["logit"],
                             out_clean["hidden"].detach(), out_deg["hidden"],
                             alpha=1.0, beta=1.0)
    loss.backward()

    grads = [p.grad for p in detector.classifier.parameters() if p is not None]
    assert any(g is not None for g in grads), "no gradient reached the classifier at all"
    total_abs_grad = sum(g.abs().sum().item() for g in grads if g is not None)
    assert total_abs_grad > 0, "classifier parameters received exactly zero gradient"


def test_total_loss_detaches_clean_branch_from_the_consistency_term():
    """Regression guard for the two `.detach()` calls in `total_loss` itself
    (not `consistency_loss`'s, which is deliberately undetached — see its
    docstring). If either is removed, gradient from the consistency term
    leaks back through the CLEAN forward pass, and the optimiser could
    minimise L by collapsing the clean prediction/hidden state onto the
    degraded one instead of pulling the degraded one towards the clean one —
    the "meet in the middle" failure mode `total_loss`'s docstring warns
    about.

    Isolated with two independent `ClassifierHead` instances (clean and
    degraded branches do NOT share parameters here, unlike the real
    `Detector`), so a leak can't hide inside gradient shared with the
    degraded branch. `y_clean` is set to the clean logit's own prediction
    (detached), which makes L_cls's gradient into the clean branch exactly
    zero by construction (BCE-with-logits' gradient is `sigmoid(logit) - y`,
    zero when `y == sigmoid(logit)`), leaving the consistency term as the
    only thing that could put gradient on `clf_clean`'s parameters.

    This single assertion (`clean_grad == 0`) covers BOTH detach calls: removing
    logit_clean's detach opens a nonzero-gradient path via the KL term,
    removing hidden_clean's detach opens one via the MSE term, and either
    alone makes `clean_grad` nonzero. Verified by hand (not left in the
    suite) that deleting either one flips this test to fail.
    """
    gen = torch.Generator().manual_seed(5)
    dim = 16
    clf_clean = ClassifierHead(dim_in=dim)
    clf_deg = ClassifierHead(dim_in=dim)
    deg_head = DegradationHead(dim_in=dim)

    f_clean = torch.randn(4, dim, generator=gen)
    f_deg = torch.randn(4, dim, generator=gen)

    out_clean = clf_clean(f_clean)
    out_deg = clf_deg(f_deg)
    out_deg = {**out_deg, **{k: v for k, v in deg_head(f_deg).items() if k in ("presence", "severity")}}

    y_clean_matching = torch.sigmoid(out_clean["logit"]).detach()
    batch = {
        "y_clean": y_clean_matching,
        "y_deg": torch.ones(4),
        "presence_deg": torch.zeros(4, 6),
        "severity_deg": torch.zeros(4, 6),
    }
    loss, _ = total_loss(out_clean, out_deg, batch, LossWeights(lambda_deg=1.0, alpha=1.0, beta=1.0))
    loss.backward()

    clean_grad = sum(p.grad.abs().sum().item() for p in clf_clean.parameters() if p.grad is not None)
    deg_branch_grad = sum(p.grad.abs().sum().item() for p in clf_deg.parameters() if p.grad is not None)

    assert clean_grad == pytest.approx(0.0), \
        "gradient leaked into the clean branch's classifier — a total_loss detach is missing"
    # Sanity: the pipeline is actually live and the degraded branch does get
    # gradient (from L_cls and the consistency term), so a zero on the clean
    # side is meaningful and not an artefact of a dead graph.
    assert deg_branch_grad > 0


def test_lambda_deg_zero_removes_degradation_head_gradient_via_total_loss():
    """Ablation switch check for L_deg: with lambda_deg=0 the degradation
    head's parameters (which, with use_film=False, feed only L_deg — never
    the classifier) must receive exactly zero gradient, while the classifier
    still receives gradient from L_cls and the consistency term."""
    gen = torch.Generator().manual_seed(0)
    detector = Detector(dim_feat=16, use_film=False)
    f_clean = torch.randn(4, 16, generator=gen)
    f_deg = torch.randn(4, 16, generator=gen)
    batch = {
        "y_clean": torch.zeros(4),
        "y_deg": torch.ones(4),
        "presence_deg": torch.zeros(4, 6),
        "severity_deg": torch.zeros(4, 6),
    }

    out_clean = detector(f_clean)
    out_deg = detector(f_deg)
    loss_on, _ = total_loss(out_clean, out_deg, batch,
                             LossWeights(lambda_deg=1.0, alpha=1.0, beta=1.0))
    loss_on.backward()
    deg_grad_on = sum(p.grad.abs().sum().item() for p in detector.degradation.parameters()
                       if p.grad is not None)
    cls_grad_on = sum(p.grad.abs().sum().item() for p in detector.classifier.parameters()
                       if p.grad is not None)
    detector.zero_grad()

    out_clean = detector(f_clean)
    out_deg = detector(f_deg)
    loss_off, parts_off = total_loss(out_clean, out_deg, batch,
                                      LossWeights(lambda_deg=0.0, alpha=1.0, beta=1.0))
    loss_off.backward()
    deg_grad_off = sum(p.grad.abs().sum().item() for p in detector.degradation.parameters()
                        if p.grad is not None)
    cls_grad_off = sum(p.grad.abs().sum().item() for p in detector.classifier.parameters()
                        if p.grad is not None)

    assert deg_grad_on > 0
    assert deg_grad_off == pytest.approx(0.0)
    assert cls_grad_on > 0 and cls_grad_off > 0
    assert parts_off["deg"] > 0  # the *value* is still reported, only its gradient is gated


def test_total_loss_returns_finite_scalar_and_component_dict():
    gen = torch.Generator().manual_seed(2)
    detector = Detector(dim_feat=16)
    f_clean = torch.randn(4, 16, generator=gen)
    f_deg = torch.randn(4, 16, generator=gen)
    batch = {
        "y_clean": torch.zeros(4),
        "y_deg": torch.ones(4),
        "presence_deg": torch.zeros(4, 6),
        "severity_deg": torch.zeros(4, 6),
    }
    out_clean = detector(f_clean)
    out_deg = detector(f_deg)
    loss, parts = total_loss(out_clean, out_deg, batch, LossWeights())

    assert torch.isfinite(loss)
    assert set(parts) == {"cls", "deg", "con", "total"}
    assert all(torch.isfinite(torch.tensor(v)) for v in parts.values())
