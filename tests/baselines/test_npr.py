import cv2
import numpy as np
import pytest

from aigcdet.augment import ops
from aigcdet.baselines.npr import NPR_FEATURE_NAMES, NPRDetector, npr_feature


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


# ==========================================================================
# The three tests above are the brief's. Their fixtures are block-replicated
# noise, where the two classes differ in CONTENT as well as in provenance:
# a feature with no grid awareness at all -- [std, mean|dh|, Laplacian
# variance, mean] -- passes all three of them. They are kept because the brief
# specifies them, but they are not the coverage. Everything below is.
# ==========================================================================

#: A decoder's post-up-sample convolution, in pixels. 0.8 leaves an artifact
#: that is real but faint -- large enough that a working feature separates the
#: grid control perfectly, small enough that the transforms below can destroy
#: it, which is the behaviour the robustness table reports.
_POST_CONV_SIGMA = 0.8
_SIZE = 160
_N = 48


def _wrap_blur(x, sigma):
    """Circular gaussian blur: pad by the kernel radius with `mode="wrap"`,
    filter, then crop. Keeps the image exactly periodic, which is what makes
    the control below a cyclic permutation rather than two different crops."""
    k = 2 * int(4 * sigma + 0.5) + 1
    r = k // 2
    padded = np.pad(x, ((r, r), (r, r), (0, 0)), mode="wrap")
    return cv2.GaussianBlur(padded, (k, k), sigma,
                            borderType=cv2.BORDER_CONSTANT)[r:-r, r:-r]


def _torus_decoder(seed, size=_SIZE, sigma=_POST_CONV_SIGMA, upsample=True):
    """A stride-2 decoder up-sample, built to wrap exactly.

    With `upsample=True`: build at half resolution, replicate each pixel into a
    2x2 cell (what a transposed convolution's up-sample does), then apply the
    convolution that follows it in every real decoder. With `upsample=False`:
    the identical pipeline at native resolution, so the image has the same
    smoothness and the same spectrum shape but NO stride-2 grid. That is the
    negative arm, and it is what makes the robustness profile self-validating.
    """
    rng = np.random.default_rng(seed)
    n = size // 2 if upsample else size
    base = _wrap_blur(rng.integers(0, 256, (n, n, 3)).astype(np.float32), 1.5)
    x = np.repeat(np.repeat(base, 2, 0), 2, 1) if upsample else base
    return np.clip(_wrap_blur(x, sigma), 0, 255).astype(np.uint8)


def _grid_control(img):
    """One image and the SAME image rolled by one pixel on the torus.

    Because `_torus_decoder` wraps exactly, `np.roll` is a cyclic permutation:
    the two members hold an IDENTICAL PIXEL MULTISET (asserted in
    `test_the_control_members_are_a_cyclic_permutation_of_one_image`), so every
    statistic that is a function of the multiset alone -- mean, standard
    deviation, histogram, brightness, contrast -- is bit-identical between
    them and cannot contribute anything. What differs is the stride-2 phase.

    This is NOT a proof that only the artifact can separate them. A finite
    window's edge treatment is not shift-invariant, so a spatial statistic can
    still pick up a border term; on the earlier version of this control, built
    from two overlapping CROPS instead of a roll, a deliberately grid-blind
    feature scored 0.976 through `_control_auc`. The roll shrinks that leak but
    does not prove it away. What rules it out is the negative arm: the same
    pipeline with `upsample=False` must not separate. See
    `test_the_control_has_nothing_to_find_when_there_is_no_upsampling`.
    """
    return img, np.roll(img, (-1, -1), axis=(0, 1))


@pytest.fixture(scope="module")
def grid_pairs():
    return [_grid_control(_torus_decoder(s)) for s in range(_N)]


@pytest.fixture(scope="module")
def no_upsample_pairs():
    return [_grid_control(_torus_decoder(s, upsample=False)) for s in range(_N)]


def _control_auc(pairs, transform=None):
    """Held-out AUC for aligned (label 1) vs rolled (label 0), fitting the head
    on half the rows and scoring the other half -- never the rows it was
    fitted on."""
    from aigcdet.eval.metrics import roc_auc
    aligned = [p[0] for p in pairs]
    rolled = [p[1] for p in pairs]
    if transform is not None:
        aligned = [transform(i) for i in aligned]
        rolled = [transform(i) for i in rolled]
    n = len(pairs)
    X = np.stack([npr_feature(i) for i in rolled + aligned])
    y = np.array([0] * n + [1] * n)
    fit_rows = np.zeros(2 * n, bool)
    fit_rows[: n // 2] = True
    fit_rows[n: n + n // 2] = True
    d = NPRDetector().fit(X[fit_rows], y[fit_rows])
    return roc_auc(y[~fit_rows], d.score(X[~fit_rows]))


def test_the_control_members_are_a_cyclic_permutation_of_one_image(grid_pairs,
                                                                   no_upsample_pairs):
    """The property the whole control rests on, asserted rather than assumed.

    If `_torus_decoder` stopped wrapping -- a `BORDER_REFLECT` blur, say -- the
    roll would stop being a permutation and the two members would differ in
    content, which is exactly the contamination this control exists to avoid.
    """
    for pairs in (grid_pairs, no_upsample_pairs):
        for a, b in pairs:
            assert a.shape == b.shape
            assert np.array_equal(np.sort(a.ravel()), np.sort(b.ravel()))
            assert not np.array_equal(a, b)          # a real shift, not a no-op


def test_npr_feature_measures_the_upsampling_grid_not_the_pixel_multiset(grid_pairs):
    """Same pixel multiset, one-pixel grid roll. The contrast entries must flip
    sign: within-cell neighbours are more similar than across-cell ones when
    the grid lines up, and less similar when it does not."""
    aligned = np.stack([npr_feature(a) for a, _ in grid_pairs])
    rolled = np.stack([npr_feature(b) for _, b in grid_pairs])
    # Both contrast entries, on EVERY image, not just on average.
    assert (aligned[:, 2] < rolled[:, 2]).all()
    assert (aligned[:, 3] < rolled[:, 3]).all()
    # The bounds are tight on purpose. A mutant that widens the within-cell
    # mask to cover every column still flips sign, landing at -0.053/+0.059;
    # the real feature sits at -0.111/+0.111, so 0.08 separates them.
    assert aligned[:, 2].mean() < -0.08 and rolled[:, 2].mean() > 0.08
    assert aligned[:, 3].mean() < -0.08 and rolled[:, 3].mean() > 0.08


def test_the_control_has_nothing_to_find_when_there_is_no_upsampling(
        no_upsample_pairs, grid_pairs):
    """The negative arm. Same pipeline, same smoothness, same roll -- only the
    `np.repeat` up-sample removed. If this separated, every number in the
    robustness profile would be measuring the fixture instead of the artifact.

    Three ways of saying it, in increasing order of how much they prove:
    the magnitude collapses (16x, deterministic), the direction stops being
    consistent, and the held-out AUC loses its edge. The AUC arm is the noisy
    one -- across nine seed bases it wanders over 0.25-0.86 with no consistent
    direction, which is what "no signal" looks like at 24 held-out rows -- so
    the magnitude bound is the load-bearing assertion here.
    """
    off = np.abs(np.stack([npr_feature(a) for a, _ in no_upsample_pairs]
                          + [npr_feature(b) for _, b in no_upsample_pairs])[:, 2])
    on = np.abs(np.stack([npr_feature(a) for a, _ in grid_pairs]
                         + [npr_feature(b) for _, b in grid_pairs])[:, 2])
    assert off.max() < 0.02 < 0.08 < on.min()

    a = np.stack([npr_feature(x) for x, _ in no_upsample_pairs])
    b = np.stack([npr_feature(x) for _, x in no_upsample_pairs])
    consistent = float((a[:, 2] < b[:, 2]).mean())
    assert 0.2 < consistent < 0.8, consistent

    off_auc, on_auc = _control_auc(no_upsample_pairs), _control_auc(grid_pairs)
    assert off_auc < 0.9, off_auc
    assert on_auc > 0.95, on_auc


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
def test_each_contrast_entry_reports_its_own_axis(axis, artifact_entry, clean_entry):
    """Entry 2 is the horizontal contrast and entry 3 the vertical one. Every
    other fixture here is isotropic, so entry 3 could be silently computed from
    the horizontal differences and nothing would notice -- this is the only
    test that separates the two axes."""
    f = np.stack([npr_feature(_anisotropic(s, axis)) for s in range(8)])
    assert (f[:, artifact_entry] < -0.08).all()
    assert f[:, clean_entry] == pytest.approx(0.0, abs=0.03)


def test_stride_selects_which_upsampling_grid_is_measured():
    """A stride-2 up-sampled image is invisible to a stride-3 measurement:
    stride is a real parameter, not decoration."""
    imgs = [_torus_decoder(s) for s in range(8)]
    at2 = np.mean([npr_feature(i, stride=2)[2] for i in imgs])
    at3 = np.mean([npr_feature(i, stride=3)[2] for i in imgs])
    assert at2 < -0.08
    assert at3 == pytest.approx(0.0, abs=0.03)


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


# --------------------------------------------------------------------------
# The feature's own contract: exact values, layout, and what it accepts.
# --------------------------------------------------------------------------

def _striped(period_values, axis):
    """uint8 HWC RGB, 8x8, constant along `axis`, with `period_values` repeated
    along the other. Put entirely in the RED channel so that a mutant reading
    one channel instead of the luma reads 3x the value."""
    row = np.array(period_values * 2, dtype=np.float32)
    plane = np.tile(row, (8, 1)) if axis == 1 else np.tile(row[:, None], (1, 8))
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[..., 0] = plane.astype(np.uint8)
    return img


@pytest.mark.parametrize("axis,expected", [
    # Stripes across the columns: within-cell pairs are flat (0), across-cell
    # pairs jump by 120/3 = 40 in luma. Vertical neighbours never differ, so
    # both vertical means are 0 and the vertical contrast is exactly 0.
    (1, [0.0, 40.0, -1.0, 0.0]),
    # Stripes down the rows: now the HORIZONTAL magnitudes are the flat pair
    # and the artifact shows up in the vertical contrast instead.
    (0, [0.0, 0.0, 0.0, -1.0]),
])
def test_the_four_entries_are_exactly_what_their_names_say(axis, expected):
    """Hand-computable values, pinning the layout `NPR_FEATURE_NAMES` documents.

    Kills three mutations that every aggregate test above survives: swapping
    entries 0 and 1 (within <-> across), zeroing the two magnitude entries, and
    reading a single channel instead of the luma (which would report 120, not
    40, because the stripes live only in RED).
    """
    assert NPR_FEATURE_NAMES == ("within_h", "across_h", "contrast_h", "contrast_v")
    img = _striped([0, 0, 120, 120], axis)
    assert npr_feature(img) == pytest.approx(np.array(expected, np.float32), abs=1e-5)


def test_contrast_entries_stay_bounded_on_an_anti_aligned_nearest_upscale():
    """A nearest-neighbour upscale whose cell grid is ANTI-aligned with the
    measurement grid has `across_h == 0` EXACTLY, so an unbounded
    `within / (across + 1e-6)` ratio returns ~4.8e7 -- finite, so
    `np.isfinite` waves it through, and large enough that one such row in a fit
    set makes `StandardScaler` normalise the column to that single outlier.

    This is ordinary content, not a corner case: pixel art, blown-up
    thumbnails and nearest-upscaled web images all sit in the REAL class of a
    scraped corpus, and `ops.center_crop(0.8)` can supply the odd offset.
    """
    rng = np.random.default_rng(0)
    small = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
    aligned = np.repeat(np.repeat(small, 2, 0), 2, 1)
    anti = np.ascontiguousarray(aligned[1:, 1:])

    f_aligned, f_anti = npr_feature(aligned), npr_feature(anti)
    # center_crop(0.8) is how an ordinary eval-grid transform reaches the same
    # odd offset, so it is checked on the same path.
    f_cropped = npr_feature(ops.center_crop(aligned, 0.8))
    for f in (f_aligned, f_anti, f_cropped):
        assert np.isfinite(f).all()
        assert (np.abs(f[2:]) <= 1.0).all(), f
        assert (f[:2] <= 255.0).all(), f
    # The artifact is still detected -- the phase is simply inverted.
    assert f_aligned[2] < -0.99 and f_anti[2] > 0.99


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


@pytest.mark.parametrize("name,make", [
    ("2-D greyscale", lambda r: r.integers(0, 256, (64, 64), dtype=np.uint8)),
    ("RGBA", lambda r: r.integers(0, 256, (64, 64, 4), dtype=np.uint8)),
    ("CHW", lambda r: r.integers(0, 256, (3, 64, 64), dtype=np.uint8)),
])
def test_npr_feature_refuses_anything_that_is_not_hwc_rgb(name, make):
    """Left unguarded, each of these fails differently and misleadingly: 2-D
    greyscale raises `AxisError` from the channel mean, RGBA folds alpha into
    the luma and returns a plausible-looking vector, and a CHW tensor measures
    the 3-channel axis as image rows and complains that the image is too
    small."""
    with pytest.raises(ValueError, match="HWC 3-channel RGB"):
        npr_feature(make(np.random.default_rng(1)))


@pytest.mark.parametrize("scale", [1.0, 255.0])
def test_npr_feature_refuses_float_input_which_would_rescale_the_magnitudes(scale):
    """Float [0, 1] input leaves the two contrasts invariant while dividing the
    two magnitudes by 255, so a bank built from mixed dtype conventions carries
    two scales in one column and nothing downstream can tell."""
    img = _natural(0).astype(np.float32) / (255.0 / scale)
    with pytest.raises(ValueError, match="expects uint8"):
        npr_feature(img)


# --------------------------------------------------------------------------
# The head.
# --------------------------------------------------------------------------

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
