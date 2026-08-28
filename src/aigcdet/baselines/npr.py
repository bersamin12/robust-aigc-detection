"""NPR-style baseline (spec §6.3).

Captures up-sampling artifacts through neighbouring-pixel relationships.
Near-free to implement and expected to COLLAPSE under resize and blur, which
is why it is included: that failure is the most informative row in the
robustness table.

A baseline that is accidentally crippled flatters the headline model, so the
feature below was measured against three alternatives before being kept, on a
fixture whose two classes are the SAME pixels differing only in whether the
stride-2 cell grid lines up with the decoder's up-sampling grid -- a control of
the same design as the one `tests/baselines/test_npr.py` runs, swept over
increasing post-up-sample blur so the artifact gets progressively fainter.
Ranked by how faint an artifact they could still separate, the within/across
neighbour contrast used here was the best of the four: at the blur where it
still scored AUC 1.00, the literal "NPR = x - up(down(x))" residual normalised
by the mean neighbour difference scored 0.94, and the same residual normalised
by the image standard deviation scored 0.83. Normalising the residual by a
quantity that is itself grid-blind throws away the contrast that carries the
signal.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def npr_feature(img: np.ndarray, stride: int = 2) -> np.ndarray:
    """Compare within-cell and across-cell neighbour differences.

    A generator's transposed-convolution up-sampling makes pixels inside a
    stride-sized cell more similar to each other than the cell grid would
    otherwise predict.

    The two ratio entries are what actually carries the artifact: on an
    up-sampled image they sit well below 1, and on the same pixels with the
    grid shifted by one they sit above it. The two raw magnitudes are kept
    because they let a downstream head normalise the ratios against how much
    neighbour variation the image has at all.

    Raises rather than returning NaN when the image or the stride cannot
    support the measurement: with fewer than `2 * stride` rows or columns, or
    with `stride < 2`, one of the two masks selects nothing and its mean is a
    silent NaN that would poison a whole feature bank.
    """
    if stride < 2:
        raise ValueError(
            f"stride must be at least 2 to have both within-cell and "
            f"across-cell neighbour pairs; got {stride}")
    g = img.astype(np.float32).mean(axis=2)
    h, w = g.shape
    h, w = h - h % stride, w - w % stride
    if h < 2 * stride or w < 2 * stride:
        raise ValueError(
            f"image is too small for stride {stride}: needs at least "
            f"{2 * stride} rows and columns after truncation to a whole "
            f"number of cells, got {h}x{w}")
    g = g[:h, :w]
    dh = np.abs(np.diff(g, axis=1))
    dv = np.abs(np.diff(g, axis=0))
    cols, rows = np.arange(dh.shape[1]), np.arange(dv.shape[0])
    within_h = dh[:, cols % stride != (stride - 1)].mean()
    across_h = dh[:, cols % stride == (stride - 1)].mean()
    within_v = dv[rows % stride != (stride - 1), :].mean()
    across_v = dv[rows % stride == (stride - 1), :].mean()
    eps = 1e-6
    return np.array([within_h, across_h,
                     within_h / (across_h + eps),
                     within_v / (across_v + eps)], dtype=np.float32)


class NPRDetector:
    """A logistic head on the four NPR statistics.

    Deliberately tiny: the point of the baseline is what the FEATURE can see,
    so a bigger head would report the head's capacity instead.
    """

    def __init__(self, seed: int = 20260827):
        self.clf = make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=2000, random_state=seed))

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "NPRDetector":
        """Fit on TRAIN or internal-validation rows only -- never on the rows
        the reported number is computed from.

        This head has no `split=` guard of its own (unlike everything in
        `aigcdet.calibrate`, which refuses a fit whose per-row split labels are
        not all internal validation). Nothing here can see which rows it was
        handed, so the caller carries the whole obligation: select the fit rows
        by `bank.meta["split"]` before calling, and score the held-out rows
        separately. Fitting and scoring the same matrix reports the head's
        memory, not the baseline, and it silently flatters whatever the
        baseline is being compared against.
        """
        self.clf.fit(features, labels)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """P(AI-generated) per row -- higher means more likely AI-generated,
        matching `aigcdet.eval.metrics`' score convention."""
        return self.clf.predict_proba(features)[:, 1]
