"""Evidence Quality Index (spec §3.6).

EQI is fitted, not hand-defined: it is the model's probability of being correct
given the degradation evidence, estimated on validation data. That makes it
interpretable ("this image retains ~40% usable evidence") and directly usable
for abstention, rather than a hand-tuned severity score.

Fitted on internal validation rows only -- `fit` requires a per-row `split=`
column and checks it (see the package docstring). Fitting EQI on the rows it is
later scored on would make the abstention curve report the training fit rather
than the evidence.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from aigcdet.calibrate import CalibrationError, check_fit_split, fit_standardiser


class EQI:
    """P(the detector is correct | degradation evidence), in [0, 1].

    `cond` is the degradation evidence (spec §3.6: the degradation head's `d`,
    optionally with the handcrafted proxies `h`); `correct` is 0/1, whether the
    frozen detector got that row right.
    """

    def __init__(self, seed: int = 20260827, max_iter: int = 2000) -> None:
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        self.seed = seed
        self.model = LogisticRegression(max_iter=max_iter, random_state=seed)
        self.constant_columns: tuple[int, ...] = ()
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self._fitted = False

    def _check_cond(self, cond: np.ndarray) -> np.ndarray:
        c = np.asarray(cond, dtype=np.float64)
        if c.ndim != 2:
            raise ValueError(f"cond must be 2-D (n, n_features), got shape {c.shape}")
        if c.shape[0] == 0:
            raise ValueError("cond is empty")
        if not np.isfinite(c).all():
            raise ValueError("cond must be finite; got NaN or inf")
        if self._mu is not None and c.shape[1] != self._mu.shape[1]:
            raise ValueError(
                f"EQI was fitted on {self._mu.shape[1]} columns, got {c.shape[1]}")
        return c

    def _z(self, cond: np.ndarray) -> np.ndarray:
        # Constant columns centre to exactly zero and are divided by 1, so they
        # contribute nothing instead of exploding through a near-zero scale.
        return (cond - self._mu) / self._sd

    def fit(self, cond: np.ndarray, correct: np.ndarray, *, split) -> "EQI":
        """Fit on internal validation rows: `cond` evidence, `correct` 0/1.

        `split` is required and holds one split label per row; every row must
        be the internal validation split (spec §6.7).
        """
        # Drop any previous fit before validating, so a refit at a different
        # conditioning width is not rejected by the stale one it replaces.
        self._fitted = False
        self._mu = None
        self._sd = None
        self.constant_columns = ()
        c = self._check_cond(cond)
        check_fit_split(split, c.shape[0])
        yc = np.asarray(correct)
        if yc.shape != (c.shape[0],):
            raise ValueError(
                f"correct must have shape ({c.shape[0]},), got {yc.shape}")
        uniq = np.unique(yc)
        if not np.isin(uniq, (0, 1)).all():
            raise ValueError(f"correct must be 0/1, got values {uniq.tolist()}")
        if uniq.size < 2:
            raise ValueError(
                "EQI needs both outcomes (some correct, some wrong) in the "
                f"validation set; got only {uniq.tolist()}")
        min_rows = 2 * (c.shape[1] + 1)
        if c.shape[0] < min_rows:
            raise ValueError(
                f"need at least {min_rows} validation rows to fit "
                f"{c.shape[1]} conditioning columns (got {c.shape[0]})")

        self._mu, self._sd, self.constant_columns = fit_standardiser(c)

        self.model.fit(self._z(c), yc.astype(int))
        n_iter = int(np.max(self.model.n_iter_))
        if n_iter >= self.model.max_iter:
            raise CalibrationError(
                f"EQI's logistic regression hit its {self.model.max_iter}-iteration "
                f"limit without converging")
        self._fitted = True
        return self

    def predict(self, cond: np.ndarray) -> np.ndarray:
        """P(correct) per row, in [0, 1]. The range is enforced, not assumed.

        **Deliberately NOT clamped to the range the fit produced**, unlike
        `ConditionalTemperature.temperatures`. The two look symmetric and are
        not, in either direction:

        * A FLOOR at the fitted minimum is actively harmful. `fit_policy` sets
          `eqi_threshold` to `quantile(EQI_val, 1 - target_coverage)`, so at
          `target_coverage = 1.0` the gate IS the fitted minimum, and `decide`
          admits it (`ev >= threshold`). A row whose degradation evidence is
          far worse than anything validation saw scores ~1e-10 and is correctly
          sent to a human; floored to the fitted minimum it lands exactly on
          the gate and is auto-decided instead. The clamp would manufacture the
          failure it was added to prevent -- a confident automatic decision on
          evidence the fit never covered. This is pinned by a test.
        * A CAP at the fitted maximum is inert where it would matter and
          destructive where it would not. The cap is always at or above the
          gate (the gate is a quantile of the very values the cap is the
          maximum of), so it can never move a row from auto-decided to review:
          routing is bit-identical with and without it. What it does change is
          the ORDER -- it ties every above-range row at one value, and
          `eval.metrics.risk_coverage` ranks abstentions by EQI, so the capped
          curve resolves the ranking worse for no operational gain.

        The hazard `ConditionalTemperature` clamps against has no analogue
        here. There `T` is a *multiplier*: extrapolated down to `eps` it is an
        unbounded 100x amplification of the logit and drives `p` to 1.000000,
        a number nothing downstream can tell from a well-founded one. EQI is
        not a multiplier; it is the reported quantity itself, a logistic
        probability bounded in [0, 1] by construction and checked below.

        The residual risk is real but is not clamp-shaped: an out-of-range
        `cond` in a direction validation never spanned can project HIGH and
        auto-decide a badly degraded image. No output clamp reaches that -- the
        clamp's own ceiling sits above the gate. Closing it needs an
        input-space out-of-distribution test on `cond` and a decision about how
        much extra review load to spend, which is a policy question, not a
        bound on this function's output.
        """
        if not self._fitted:
            raise RuntimeError("EQI is not fitted; call fit() first")
        p = self.model.predict_proba(self._z(self._check_cond(cond)))[:, 1]
        if not (np.isfinite(p).all() and ((p >= 0.0) & (p <= 1.0)).all()):
            raise CalibrationError(
                f"EQI must return values in [0, 1], got range "
                f"[{np.min(p)!r}, {np.max(p)!r}]")
        return p
