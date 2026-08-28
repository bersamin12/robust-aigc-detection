"""NPR-style baseline (spec §6.3).

Captures up-sampling artifacts through neighbouring-pixel relationships.
Near-free to implement and expected to COLLAPSE under resize and blur, which
is why it is included: that failure is the most informative row in the
robustness table.

**Name this row "NPR-style neighbouring-pixel summary + linear probe" in the
results table, never bare "NPR".** Published NPR trains a ResNet-50 over the
full residual map; this is four scalars into a logistic regression. The head
is not the bottleneck -- on the control below, an RBF SVC and a 400-tree random
forest both reach the same 1.000 on clean data and are mostly WORSE under
degradation (at jpeg50: 0.800 for this head against 0.698 and 0.649; the one
cell where a swap helps is resize0.25, 0.615 for the forest against 0.554) --
so the implementation is internally fair. But an unfootnoted "NPR" row in a
§6.3 comparison understates the published method, exactly as UnivFD carries
its "rung A0" footnote.

The feature was measured against three alternatives on the control that
`tests/baselines/test_npr.py` builds -- two members that are a cyclic
permutation of one image, identical pixel multiset, only the stride-2 phase
different -- swept over increasing post-up-sample blur so the artifact gets
progressively fainter. Held-out AUC:

    post-up-sample sigma             1.1    1.3    1.5
    within/across contrast (kept)  1.000  0.993  0.632
    four within/across ratios      1.000  0.995  0.694
    NPR residual / mean neighbour  1.000  0.970  0.557
    NPR residual / image sd        0.778  0.559  0.510

The kept form and the four-ratio variant are TIED within the noise of a 24-row
held-out split, the four-ratio one very slightly ahead at the two faintest
settings; the brief specifies this one, so it stands, and nobody should re-run
this expecting a clear win. What the sweep does rule out is normalising the
literal "NPR = x - up(down(x))" residual by a grid-BLIND quantity: dividing by
the image standard deviation loses the signal a full blur step earlier than
anything else here, because that denominator carries none of the contrast the
artifact lives in. (An earlier version of this note claimed the kept form was
the outright best of the four. Those numbers came from a crop-based control
that was later found to leak -- see `_grid_control` in the test file.)
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#: Layout of `npr_feature`'s return value. A contract, not a convenience: a
#: bank stores this vector positionally, and swapping two entries is invisible
#: to every aggregate statistic.
NPR_FEATURE_NAMES: tuple[str, ...] = (
    "within_h",       # mean |neighbour difference| INSIDE a cell, horizontally
    "across_h",       # ... and STRADDLING a cell boundary, horizontally
    "contrast_h",     # (within_h - across_h) / (within_h + across_h), in [-1, 1]
    "contrast_v",     # the same contrast, vertically
)

#: Guards the contrast denominator only. Deliberately not used as a ratio
#: denominator: see `npr_feature`.
_EPS = 1e-6


def npr_feature(img: np.ndarray, stride: int = 2) -> np.ndarray:
    """Compare within-cell and across-cell neighbour differences.

    **Input contract: `img` is uint8, HWC, 3-channel RGB** -- what every op in
    `aigcdet.augment.ops` produces and what `data.normalize` writes. Violations
    raise rather than degrade quietly: a float [0, 1] image would divide the two
    magnitude entries by 255 while leaving the contrasts invariant, so a bank
    built from mixed dtype conventions would silently carry two scales in the
    same column; an RGBA image would fold alpha into the luma; a CHW array
    would measure the channel axis as image rows.

    A generator's transposed-convolution up-sampling makes pixels inside a
    stride-sized cell more similar to each other than the cell grid would
    otherwise predict, so `contrast_h`/`contrast_v` go NEGATIVE on an
    up-sampled image and positive on the same pixels with the grid shifted by
    one. The two raw magnitudes are kept so a downstream head can tell a small
    contrast on a flat image from a small contrast on a busy one.

    The contrast is `(within - across) / (within + across + eps)`, bounded in
    [-1, 1], NOT the unbounded ratio `within / (across + eps)`. The ratio is a
    strictly increasing function of this, so the two rank identically -- every
    AUC measured under both agreed to three decimals -- but the ratio explodes
    when `across` is ~0, and that is real content, not a corner case: a
    nearest-neighbour upscale whose cell grid is ANTI-aligned with the
    measurement grid has `across_h == 0` exactly, and the ratio form returns
    ~4.8e7. Pixel art, blown-up thumbnails and nearest-upscaled web images all
    sit in the REAL class of a scraped corpus, and `ops.center_crop(0.8)` can
    supply the odd offset. Such a row makes `StandardScaler` fit a column whose
    mean and variance are set by that one outlier, collapsing every other row
    towards 0. Measured on a faint-artifact bank (post-up-sample blur 1.3) whose
    clean held-out AUC is 0.993, ONE injected anti-aligned row takes it to 0.557
    under the ratio form and to 0.708 under the bounded form -- the bound does
    not make the outlier harmless, it stops it from dominating the column.
    `np.isfinite` cannot see any of this, because 4.8e7 is finite.

    Raises rather than returning NaN when the image or the stride cannot
    support the measurement: with fewer than `2 * stride` rows or columns, or
    with `stride < 2`, one of the two masks selects nothing and its mean is a
    silent NaN that would poison a whole feature bank.
    """
    if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
        got = "not an array" if not isinstance(img, np.ndarray) else f"shape {img.shape}"
        raise ValueError(
            f"npr_feature expects an HWC 3-channel RGB image; got {got}. A 2-D "
            f"greyscale array, an RGBA array and a CHW tensor all land here")
    if img.dtype != np.uint8:
        raise ValueError(
            f"npr_feature expects uint8 on the 0-255 scale (what "
            f"aigcdet.augment.ops produces); got dtype {img.dtype}. Float input "
            f"would scale the two magnitude entries by 1/255 while leaving the "
            f"contrasts invariant, mixing two scales into one bank column")
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
    return np.array([within_h, across_h,
                     (within_h - across_h) / (within_h + across_h + _EPS),
                     (within_v - across_v) / (within_v + across_v + _EPS)],
                    dtype=np.float32)


class NPRDetector:
    """A logistic head on the four NPR statistics.

    Deliberately tiny: the point of the baseline is what the FEATURE can see,
    so a bigger head would report the head's capacity instead. Verified rather
    than assumed -- an RBF SVC and a 400-tree random forest match this head on
    clean data and are mostly worse under degradation, so the head is not what
    limits the baseline.
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
