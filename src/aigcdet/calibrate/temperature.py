"""Calibration (spec §3.7).

Global temperature scaling is the standard baseline. The conditional variant
lets the temperature depend on the estimated degradation, which is the point:
a detector that stays accurate but becomes wildly overconfident at JPEG-30 is
dangerous in a moderation pipeline, and one scalar cannot fix both regimes.

Fitted on internal validation rows only -- `fit` requires a per-row `split=`
column and checks it (see the package docstring) -- with the classifier frozen.

Both fits are optimised with L-BFGS and both check that they actually
converged: the final gradient must be at the tolerance and the resulting
temperature must lie inside [T_MIN, T_MAX]. Neither check is cosmetic. A
validation set the frozen classifier separates perfectly drives T towards 0,
and one it gets backwards drives T towards infinity; in both cases the
optimiser converges happily and returns a temperature that would destroy every
probability it is later applied to.

Inference is guarded separately, because the fit-time range check says nothing
about a `cond` outside the validation range: see `ConditionalTemperature`.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit

from aigcdet.calibrate import (
    CalibrationError,
    check_fit_split,
    fit_standardiser,
)

#: A useful temperature lives well inside these bounds. Outside them the fit
#: has collapsed (T -> 0, infinite confidence) or dissolved (T -> inf, every
#: prediction 0.5), and is reported as a failure rather than returned.
T_MIN = 1e-2
T_MAX = 1e2

#: Max-abs gradient accepted as converged. Chosen against float64 L-BFGS with
#: a strong-Wolfe line search, which reaches ~1e-5 on the fits here.
GRAD_TOL = 1e-4

#: Initial trial step handed to L-BFGS. The strong-Wolfe line search chooses
#: the step it actually takes, so this is not a learning rate and sweeping it
#: over six orders of magnitude moves the fitted temperature in the fourth
#: decimal. It is not exposed as a parameter for exactly that reason.
_LBFGS_INIT_STEP = 1.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # scipy's expit, not 1/(1+exp(-x)): the naive form overflows on a
    # saturating logit (a small T multiplying a large logit is precisely what
    # this module produces) and emits a RuntimeWarning on the way.
    return expit(x)


def _check_logits(logits: np.ndarray) -> np.ndarray:
    lg = np.asarray(logits, dtype=np.float64)
    if lg.ndim != 1:
        raise ValueError(f"logits must be 1-D, got shape {lg.shape}")
    if lg.size == 0:
        raise ValueError("logits is empty")
    if not np.isfinite(lg).all():
        raise ValueError("logits must be finite; got NaN or inf")
    return lg


def _check_labels(y: np.ndarray, n: int) -> np.ndarray:
    yt = np.asarray(y)
    if yt.shape != (n,):
        raise ValueError(f"y must have shape ({n},), got {yt.shape}")
    uniq = np.unique(yt)
    if not np.isin(uniq, (0, 1)).all():
        raise ValueError(f"y must be 0/1, got values {uniq.tolist()}")
    if uniq.size < 2:
        raise ValueError(
            "temperature scaling needs both classes in the validation set; "
            f"got only {uniq.tolist()}")
    return yt.astype(np.float64)


def _check_converged(grad_norm: float, loss: float, temperatures: np.ndarray,
                     what: str, n_iter: int) -> None:
    if not math.isfinite(loss):
        raise CalibrationError(f"{what}: loss is not finite ({loss})")
    if not np.isfinite(temperatures).all():
        raise CalibrationError(f"{what}: fitted temperature is not finite")
    if grad_norm > GRAD_TOL:
        raise CalibrationError(
            f"{what}: L-BFGS did not converge in {n_iter} iterations "
            f"(max|grad| = {grad_norm:.3g} > {GRAD_TOL:g}); raise the iteration "
            f"budget or check the validation logits")
    lo, hi = float(temperatures.min()), float(temperatures.max())
    if lo < T_MIN or hi > T_MAX:
        raise CalibrationError(
            f"{what}: fitted temperature outside [{T_MIN:g}, {T_MAX:g}] "
            f"(min {lo:.3g}, max {hi:.3g}). A temperature at the bottom means "
            f"the validation set is separated perfectly and calibration is "
            f"degenerate; one at the top means the scores are anti-correlated "
            f"with the labels. Neither is a calibration to ship")


class GlobalTemperature:
    """p = sigmoid(z / T) with one scalar T, the standard baseline.

    Fit on internal validation rows only, with the classifier frozen.
    """

    def __init__(self) -> None:
        self.temperature = 1.0
        self._fitted = False

    def fit(self, logits: np.ndarray, y: np.ndarray, *, split,
            max_iter: int = 100) -> "GlobalTemperature":
        """Fit T on `logits`/`y`.

        `split` is required and holds one split label per row; every row must
        be the internal validation split (spec §6.7).

        Raises CalibrationError if the optimiser does not converge or lands on
        a degenerate temperature.
        """
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        lg_np = _check_logits(logits)
        check_fit_split(split, lg_np.size)
        y_np = _check_labels(y, lg_np.size)

        lg = torch.from_numpy(lg_np)
        yt = torch.from_numpy(y_np)
        log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=_LBFGS_INIT_STEP, max_iter=max_iter,
                                line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(lg / log_t.exp(), yt)
            loss.backward()
            return loss

        opt.step(closure)
        # Re-evaluate at the final iterate: this is the loss and gradient the
        # convergence check must see, not the ones from some earlier step.
        loss = float(closure().item())
        grad_norm = float(log_t.grad.abs().max().item())
        t = float(log_t.detach().exp().item())
        _check_converged(grad_norm, loss, np.array([t]), "global temperature", max_iter)

        self.temperature = t
        self._fitted = True
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("GlobalTemperature is not fitted; call fit() first")
        p = _sigmoid(_check_logits(logits) / self.temperature)
        # Tripwire, not a live safeguard: expit cannot leave [0, 1] and the
        # logits are checked finite, so nothing here can currently trip it. It
        # exists to catch a future change to the link function.
        if not ((p >= 0.0) & (p <= 1.0)).all():
            raise CalibrationError("calibrated probabilities left [0, 1]")
        return p


class ConditionalTemperature:
    """T(cond) = softplus(Linear(cond)) + eps, so temperature is always positive.

    `cond` is the degradation evidence -- the degradation head's estimate and
    the handcrafted proxies (spec §3.7, `T(d, h)`). Fit on internal validation
    rows only, with the classifier frozen.

    Conditioning columns are standardised on the fit rows, and a column that is
    constant there (relative to its own scale) is neutralised: it is centred to
    exactly zero and its weight is initialised at zero, so it never receives
    gradient and can never move the temperature, whatever value it takes at
    inference. That is the honest reading of a degradation family validation
    happened never to exercise.

    **Temperatures are clamped at inference to the range the fit produced.**
    A linear model extrapolates without limit, and downwards extrapolation here
    is not a small error: `T = eps = 0.01` MULTIPLIES the logit by 100. On a
    real two-regime fit (w = +1.55, b = +2.39) a cond just below the validation
    range gives T ~ 0.4, two units below it gives T = 0.0145, and p goes to
    1.000000 on an image the model has *less* reason to be sure about than the
    ones it was fitted on. Since `eps` equals `T_MIN`, the fit-time range check
    cannot see that path either. Clamping to the observed fit range is the
    honest statement: outside the validation range there is no evidence for a
    more extreme temperature than validation ever justified.
    """

    def __init__(self, cond_dim: int, eps: float = 1e-2) -> None:
        if not isinstance(cond_dim, (int, np.integer)) or cond_dim < 1:
            raise ValueError(f"cond_dim must be a positive int, got {cond_dim!r}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps!r}")
        self.cond_dim = int(cond_dim)
        self.eps = float(eps)
        # Deterministic init, so no RNG is drawn and no global seed is touched:
        # zero weights and a bias with softplus(bias) ~= 1 start the fit at the
        # identity, i.e. at the uncalibrated model. `skip_init` is what keeps
        # the promise -- a plain nn.Linear would draw (and discard) a Kaiming
        # init from the global torch RNG, advancing a stream this package has
        # no business touching.
        self.net = torch.nn.utils.skip_init(
            nn.Linear, self.cond_dim, 1, dtype=torch.float64)
        nn.init.zeros_(self.net.weight)
        nn.init.constant_(self.net.bias, 0.5414)   # softplus(0.5414) ~= 1.0
        self.constant_columns: tuple[int, ...] = ()
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self._t_lo = 0.0
        self._t_hi = 0.0
        self._fitted = False

    def _check_cond(self, cond: np.ndarray) -> np.ndarray:
        c = np.asarray(cond, dtype=np.float64)
        if c.ndim != 2:
            raise ValueError(f"cond must be 2-D (n, cond_dim), got shape {c.shape}")
        if c.shape[1] != self.cond_dim:
            raise ValueError(
                f"cond has {c.shape[1]} columns but this ConditionalTemperature "
                f"was built with cond_dim={self.cond_dim}")
        if not np.isfinite(c).all():
            raise ValueError("cond must be finite; got NaN or inf")
        return c

    def _z(self, cond: np.ndarray) -> np.ndarray:
        return (cond - self._mu) / self._sd

    def fit(self, logits: np.ndarray, y: np.ndarray, cond: np.ndarray,
            epochs: int = 300, *, split) -> "ConditionalTemperature":
        """Fit T(cond) on internal validation rows.

        `split` is required and holds one split label per row; every row must
        be the internal validation split (spec §6.7). `epochs` is the L-BFGS
        iteration budget. Raises CalibrationError if the fit does not converge
        inside that budget or lands on a degenerate temperature.
        """
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        lg_np = _check_logits(logits)
        n = lg_np.size
        check_fit_split(split, n)
        y_np = _check_labels(y, n)
        c_np = self._check_cond(cond)
        if c_np.shape[0] != n:
            raise ValueError(
                f"cond has {c_np.shape[0]} rows but logits has {n}")
        # cond_dim + 1 parameters; fewer than twice that many rows cannot
        # identify them, and the fit would be noise dressed as a temperature.
        min_rows = 2 * (self.cond_dim + 1)
        if n < min_rows:
            raise ValueError(
                f"need at least {min_rows} validation rows to fit "
                f"cond_dim={self.cond_dim} (got {n})")

        self._fitted = False
        self._mu, self._sd, self.constant_columns = fit_standardiser(c_np)

        lg = torch.from_numpy(lg_np)
        yt = torch.from_numpy(y_np)
        c = torch.from_numpy(self._z(c_np))
        opt = torch.optim.LBFGS(self.net.parameters(), lr=_LBFGS_INIT_STEP,
                                max_iter=epochs, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            opt.zero_grad()
            t = nn.functional.softplus(self.net(c)).squeeze(-1) + self.eps
            loss = nn.functional.binary_cross_entropy_with_logits(lg / t, yt)
            loss.backward()
            return loss

        opt.step(closure)
        loss = float(closure().item())
        grad_norm = max(float(p.grad.abs().max().item()) for p in self.net.parameters())
        fitted = self._raw_temperatures(c_np)
        _check_converged(grad_norm, loss, fitted, "conditional temperature", epochs)
        self._t_lo = float(fitted.min())
        self._t_hi = float(fitted.max())
        self._fitted = True
        return self

    def _raw_temperatures(self, cond: np.ndarray) -> np.ndarray:
        """T before clamping. Only the fit itself may see these."""
        with torch.no_grad():
            t = nn.functional.softplus(
                self.net(torch.from_numpy(self._z(cond)))).squeeze(-1)
        return t.numpy() + self.eps

    def temperatures(self, cond: np.ndarray) -> np.ndarray:
        """T for each row of `cond`, clamped to the range the fit produced.

        Strictly positive by construction, and never more extreme in either
        direction than validation itself justified -- see the class docstring
        for why extrapolating downwards is the dangerous direction.
        """
        if not self._fitted:
            raise RuntimeError("ConditionalTemperature is not fitted; call fit() first")
        raw = self._raw_temperatures(self._check_cond(cond))
        return np.clip(raw, self._t_lo, self._t_hi)

    def transform(self, logits: np.ndarray, cond: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ConditionalTemperature is not fitted; call fit() first")
        lg = _check_logits(logits)
        t = self.temperatures(cond)
        if t.shape != lg.shape:
            raise ValueError(f"cond has {t.size} rows but logits has {lg.size}")
        p = _sigmoid(lg / t)
        # Tripwire, not a live safeguard -- see GlobalTemperature.transform.
        if not ((p >= 0.0) & (p <= 1.0)).all():
            raise CalibrationError("calibrated probabilities left [0, 1]")
        return p
