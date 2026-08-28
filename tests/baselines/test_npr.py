import cv2
import numpy as np
import pytest

from aigcdet.augment import ops
from aigcdet.baselines.npr import NPRDetector, npr_feature


def _upsampled(seed):
    """Mimics a generator's up-sampling: build small, then scale up."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    return ops.resize_roundtrip(np.repeat(np.repeat(small, 4, 0), 4, 1), 1.0)


def _natural(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)


def test_npr_feature_shape_and_finiteness():
    f = npr_feature(_natural(0))
    assert f.shape == (4,) and np.isfinite(f).all()


def test_npr_separates_upsampled_from_natural_images():
    up = np.stack([npr_feature(_upsampled(i)) for i in range(20)])
    nat = np.stack([npr_feature(_natural(100 + i)) for i in range(20)])
    X = np.concatenate([nat, up]); y = np.array([0] * 20 + [1] * 20)
    from aigcdet.eval.metrics import roc_auc
    d = NPRDetector().fit(X, y)
    assert roc_auc(y, d.score(X)) > 0.9


def test_npr_signal_degrades_under_resize_as_expected():
    """The documented failure mode: resampling destroys up-sampling artifacts.
    This test asserts the failure, because the plot of it is a deliverable."""
    from aigcdet.eval.metrics import roc_auc
    up = [_upsampled(i) for i in range(20)]
    nat = [_natural(100 + i) for i in range(20)]
    y = np.array([0] * 20 + [1] * 20)

    X_clean = np.stack([npr_feature(i) for i in nat + up])
    X_deg = np.stack([npr_feature(ops.resize_roundtrip(i, 0.25)) for i in nat + up])
    d = NPRDetector().fit(X_clean, y)
    assert roc_auc(y, d.score(X_clean)) > roc_auc(y, d.score(X_deg))


# --------------------------------------------------------------------------
# The fixtures above are block-replicated noise: the artifact is enormous and
# the two classes differ in content as well as in provenance, so a feature
# that keyed purely on content would pass them. Everything below runs on a
# harder fixture and a control that removes content from the comparison
# entirely.
# --------------------------------------------------------------------------

#: A decoder's post-up-sample convolution, in pixels. 0.8 leaves an artifact
#: that is real but faint -- large enough that a working feature separates the
#: alignment control perfectly, small enough that the transforms below can
#: destroy it, which is the behaviour the robustness table reports.
_POST_CONV_SIGMA = 0.8
_N = 48


def _decoder_output(seed, size=192, sigma=_POST_CONV_SIGMA):
    """A stride-2 decoder up-sample: build at half resolution, replicate each
    pixel into a 2x2 cell (what a transposed convolution's up-sample does),
    then apply the convolution that follows it in every real decoder."""
    rng = np.random.default_rng(seed)
    half = rng.integers(0, 256, (size // 2, size // 2, 3)).astype(np.float32)
    half = cv2.GaussianBlur(half, (7, 7), 1.5, borderType=cv2.BORDER_REFLECT)
    x = np.repeat(np.repeat(half, 2, 0), 2, 1)
    k = 2 * int(4 * sigma + 0.5) + 1
    x = cv2.GaussianBlur(x, (k, k), sigma, borderType=cv2.BORDER_REFLECT)
    return np.clip(x, 0, 255).astype(np.uint8)


def _grid_control(img, size=160):
    """Two overlapping crops of the SAME image: one whose stride-2 cells line
    up with the decoder's up-sampling grid, one shifted by a single pixel so
    they do not.

    They share all but one row and one column, so content, spectrum and
    sharpness are the same to within that edge. The one thing that genuinely
    differs is grid alignment -- so separation here IS the up-sampling
    artifact, and a feature that reported content could not produce it.
    """
    return img[0:size, 0:size], img[1:size + 1, 1:size + 1]


@pytest.fixture(scope="module")
def grid_pairs():
    return [_grid_control(_decoder_output(s)) for s in range(_N)]


def _control_auc(pairs, transform=None):
    """Held-out AUC for aligned (label 1) vs shifted (label 0), fitting the
    head on half the rows and scoring the other half -- never the rows it was
    fitted on."""
    from aigcdet.eval.metrics import roc_auc
    aligned = [p[0] for p in pairs]
    shifted = [p[1] for p in pairs]
    if transform is not None:
        aligned = [transform(i) for i in aligned]
        shifted = [transform(i) for i in shifted]
    n = len(pairs)
    X = np.stack([npr_feature(i) for i in shifted + aligned])
    y = np.array([0] * n + [1] * n)
    fit_rows = np.zeros(2 * n, bool)
    fit_rows[: n // 2] = True
    fit_rows[n: n + n // 2] = True
    d = NPRDetector().fit(X[fit_rows], y[fit_rows])
    return roc_auc(y[~fit_rows], d.score(X[~fit_rows]))


def test_npr_feature_measures_the_upsampling_grid_not_the_image_content(grid_pairs):
    """Overlapping crops of one image, differing by a one-pixel grid shift. A
    feature that reports image content cannot tell these apart; one that
    reports the up-sampling grid must."""
    aligned = np.stack([npr_feature(a) for a, _ in grid_pairs])
    shifted = np.stack([npr_feature(s) for _, s in grid_pairs])
    # Both ratio entries, on EVERY image, not just on average.
    assert (aligned[:, 2] < shifted[:, 2]).all()
    assert (aligned[:, 3] < shifted[:, 3]).all()
    # And the gap is a real one, not a rounding artefact: within-cell
    # neighbours are markedly more similar when the grid lines up. The bounds
    # are tight on purpose -- a mutant that pools the within-cell and
    # across-cell columns together still lands at 0.90/1.13, inside a lazier
    # band, while the real feature sits at 0.79/1.25.
    assert aligned[:, 2].mean() < 0.85 and shifted[:, 2].mean() > 1.15
    assert aligned[:, 3].mean() < 0.85 and shifted[:, 3].mean() > 1.15


def _anisotropic(seed, axis, size=192, sigma=_POST_CONV_SIGMA):
    """Up-sampled along ONE axis only, so the artifact exists in that
    direction and not in the other."""
    rng = np.random.default_rng(seed)
    shape = (size, size // 2, 3) if axis == 1 else (size // 2, size, 3)
    base = rng.integers(0, 256, shape).astype(np.float32)
    base = cv2.GaussianBlur(base, (7, 7), 1.5, borderType=cv2.BORDER_REFLECT)
    x = np.repeat(base, 2, axis=axis)
    k = 2 * int(4 * sigma + 0.5) + 1
    x = cv2.GaussianBlur(x, (k, k), sigma, borderType=cv2.BORDER_REFLECT)
    return np.clip(x, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("axis,artifact_entry,clean_entry", [(1, 2, 3), (0, 3, 2)])
def test_each_ratio_entry_reports_its_own_axis(axis, artifact_entry, clean_entry):
    """Entry 2 is the horizontal ratio and entry 3 the vertical one. Every
    other fixture here is isotropic, so entry 3 could be silently computed from
    the horizontal differences and nothing would notice -- this is the only
    test that separates the two axes."""
    f = np.stack([npr_feature(_anisotropic(s, axis)) for s in range(8)])
    assert (f[:, artifact_entry] < 0.85).all()
    assert f[:, clean_entry] == pytest.approx(1.0, abs=0.05)


def test_stride_selects_which_upsampling_grid_is_measured():
    """A stride-2 up-sampled image is invisible to a stride-3 measurement:
    stride is a real parameter, not decoration."""
    imgs = [_decoder_output(s) for s in range(8)]
    at2 = np.mean([npr_feature(i, stride=2)[2] for i in imgs])
    at3 = np.mean([npr_feature(i, stride=3)[2] for i in imgs])
    assert at2 < 0.85
    assert at3 == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("name,transform,lo,hi", [
    # The artifact survives: mild recompression, mild blur, colour jitter.
    ("clean", None, 0.95, 1.0),
    ("jpeg90", lambda i: ops.jpeg(i, 90), 0.95, 1.0),
    ("blur0.5", lambda i: ops.blur(i, 0.5), 0.95, 1.0),
    ("blur1.0", lambda i: ops.blur(i, 1.0), 0.95, 1.0),
    ("jitter", lambda i: ops.jitter(i, 0.2, 0.2, 0.2), 0.95, 1.0),
    # The artifact is destroyed: anything that resamples or heavily low-passes.
    ("blur2.0", lambda i: ops.blur(i, 2.0), 0.0, 0.80),
    ("resize0.25", lambda i: ops.resize_roundtrip(i, 0.25), 0.0, 0.80),
    ("crop0.8", lambda i: ops.center_crop(i, 0.8), 0.0, 0.80),
])
def test_npr_robustness_profile_under_the_projects_own_transforms(
        grid_pairs, name, transform, lo, hi):
    """The robustness table, asserted. Colour jitter is a pixel-value change
    and leaves the grid alone; resize and centre crop resample and wipe it out.
    That split is the deliverable, so it is pinned here rather than described.
    """
    assert lo <= _control_auc(grid_pairs, transform) <= hi


def test_npr_signal_falls_monotonically_with_jpeg_quality(grid_pairs):
    """JPEG quantises the high frequencies the artifact lives in, so the
    baseline should decay across the eval grid's quality ladder rather than
    fall off a cliff at one point."""
    aucs = [_control_auc(grid_pairs, lambda i, q=q: ops.jpeg(i, q))
            for q in (90, 70, 50, 30)]
    assert aucs == sorted(aucs, reverse=True), aucs
    assert aucs[-1] < aucs[0] - 0.1, aucs


def test_npr_feature_ignores_the_ragged_edge_that_does_not_fill_a_cell():
    """Rows and columns past the last whole cell must be truncated, not folded
    in: including them shifts every column's cell membership by one and swaps
    the within/across masks over half the image."""
    rng = np.random.default_rng(7)
    ragged = rng.integers(0, 256, (163, 199, 3), dtype=np.uint8)
    assert npr_feature(ragged) == pytest.approx(npr_feature(ragged[:162, :198]))


@pytest.mark.parametrize("bad_stride", [0, 1])
def test_npr_feature_refuses_a_stride_with_no_across_cell_pairs(bad_stride):
    """At stride 1 every neighbour pair is 'within' and the across-cell mean is
    the mean of an empty slice -- a silent NaN that would poison a bank."""
    with pytest.raises(ValueError, match="stride must be at least 2"):
        npr_feature(_natural(0), stride=bad_stride)


def test_npr_feature_refuses_an_image_too_small_to_hold_two_cells():
    rng = np.random.default_rng(3)
    tiny = rng.integers(0, 256, (3, 40, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="too small for stride 2"):
        npr_feature(tiny)


def test_npr_detector_generalises_to_rows_it_was_not_fitted_on(grid_pairs):
    """Fit on half, score the other half. The brief's own tests fit and score
    the same matrix, which cannot distinguish a working baseline from a
    memorising one."""
    assert _control_auc(grid_pairs) > 0.9


def test_npr_detector_fit_returns_self_for_chaining():
    X = np.stack([npr_feature(_natural(i)) for i in range(6)]
                 + [npr_feature(_upsampled(i)) for i in range(6)])
    y = np.array([0] * 6 + [1] * 6)
    d = NPRDetector()
    assert d.fit(X, y) is d


def test_npr_detector_scores_are_probabilities_of_the_ai_class():
    """`score` must return P(label 1) per row, on [0, 1], one per row -- the
    convention `aigcdet.eval.metrics` and the calibration package assume."""
    X = np.stack([npr_feature(_natural(i)) for i in range(6)]
                 + [npr_feature(_upsampled(i)) for i in range(6)])
    y = np.array([0] * 6 + [1] * 6)
    s = NPRDetector().fit(X, y).score(X)
    assert s.shape == (12,)
    assert ((s >= 0.0) & (s <= 1.0)).all()
    # Direction: the up-sampled rows -- label 1 -- must score higher.
    assert s[6:].mean() > s[:6].mean()
