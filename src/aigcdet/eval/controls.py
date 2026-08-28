"""Content-blind control (spec §4.2 defence 2, §6.5).

Train a classifier that CANNOT see content -- 16x16 thumbnails, or file
metadata alone -- and report its AUC. A high score means the dataset is
separable without looking at the image, so every headline number is suspect.
A near-chance score is positive evidence that the real model's signal is
content. This is run on our own splits AND on the official demo set.

Reading this module's output honestly
-------------------------------------
The control is only worth publishing if it can be believed when it says
"clean", so two ways it could lie are handled explicitly here.

**The estimator-branch confound.** `metadata_features`' fourth column comes
from `features.proxies.estimate_jpeg_quality`, which has two branches: an
exact quantisation-table read for real JPEG files, and a pixel-statistics
fallback (leave-one-family-out MAE of 14 to 31 quality points, per its own
docstring) for everything else. On the organisers' benchmark the real class
(COCO val2017) is JPEG and the AI class (DALL-E) is PNG, so the two classes
take *different branches* and land on different scales. The column is then
partly a format label rather than a property of the images, and a "broken"
verdict resting on it would be an artefact of the estimator, not evidence
about the dataset. So every row's branch is recorded (`quality_estimator_branches`),
`metadata_features` warns when a call spans both branches, and
`content_blind_auc` returns `verdict == "confounded"` -- refusing the
broken/suspect/clean answer -- when the branch assignment predicts the label
(`BRANCH_CONFOUND_THRESHOLD`). The raw result is still returned alongside,
but the *number* moves with the verdict -- `auc_ignoring_branch_confound`,
`auc_ci_ignoring_branch_confound`, `verdict_ignoring_branch_confound` -- so
`result["auc"]` raises `KeyError` instead of yielding a figure that reads as
clean once it is copied into a table. Nothing is hidden; nothing quotable is
left without its caveat attached to its name.

The same defence covers the case where provenance was never supplied at all.
A quality column whose branches nobody recorded is in exactly the same
position as one whose branches predict the label -- the number cannot be
vouched for -- so it gets the same treatment rather than a caveat demoted to a
string somebody has to read: `quality_branches` is REQUIRED, a caller with no
provenance passes `BRANCH_PROVENANCE_NOT_VERIFIED`, and the result comes back
as `auc_unverified_branch_provenance`. A caller whose features have no quality
column at all (thumbnails, geometry) passes `NO_QUALITY_COLUMN` and keeps the
plain `auc`, because there is nothing there to caveat -- and a false caveat on
the thumbnail control would be its own harm.

The confound runs the other way too. Matching the file formats makes the
branch split vanish and the control go quiet -- but with every image now on
the *pixel* branch, column 3 is no longer file metadata at all: it is a
blockiness statistic read off the pixels, whose error (14 to 31 points)
exceeds most of the differences it would need to resolve. A near-chance
result from it is therefore weak evidence, not proof of a clean dataset.
`metadata_control` reports that state as `quality_branch_check["status"] ==
"pixel_derived"` and says so in `caveats`, and it always reports the
geometry-only control (width/height/aspect, which no estimator touches)
beside the combined one.

**Leakage.** A scaler, imputer or selector fitted on the full array before
splitting inflates the cross-validated AUC, which for this function means a
*false alarm* that the dataset is broken. Everything is fitted strictly
inside folds, via a `Pipeline` handed to `cross_val_predict`; do not add a
preprocessing step outside it.
"""
from __future__ import annotations

import warnings

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aigcdet.eval.metrics import bootstrap_ci, roc_auc
from aigcdet.features.proxies import estimate_jpeg_quality

VERDICT_THRESHOLDS = {"broken": 0.85, "suspect": 0.70}

# `estimate_jpeg_quality` reads the quantisation table when it can and falls
# back to a pixel statistic when it cannot. These name the two branches.
BRANCH_EXACT = "exact"
BRANCH_ESTIMATED = "estimated"

# The branch flag alone, treated as a classifier, is scored by AUC and
# direction-normalised into [0.5, 1]. For a binary flag that AUC is
# 0.5 + |exact-share(class 1) - exact-share(class 0)| / 2, so 0.55 is a
# ten-percentage-point difference in exact-branch share between the classes.
# Above that, the quality column carries enough format information for a
# broken/suspect verdict to be an estimator artefact, and the verdict is
# withheld rather than reported.
BRANCH_CONFOUND_THRESHOLD = 0.55

#: The two explicit opt-outs for `content_blind_auc`'s REQUIRED
#: `quality_branches` argument, on the model of `report.BANKS_NOT_VERIFIED`.
#: There is no default, because a default of "no provenance supplied" made the
#: number this function cannot vouch for the path of least resistance.
#:
#: They are two sentinels and not one because the caller knows something this
#: function cannot: whether `features` contains the estimated-quality column at
#: all. `content_blind_auc` is handed a bare `(N, d)` float array with no
#: column names -- a thumbnail matrix and a metadata matrix have the same type
#: -- so inferring "has a quality column" from its shape would be a guess, and
#: a guess here fails towards reassurance on exactly the input that needs the
#: caveat. The caller declares which case it is in; nothing is inferred.
#:
#: `NO_QUALITY_COLUMN`: these features carry no quality estimate at all
#: (thumbnails, the geometry-only columns). There is no branch to check and
#: the plain `auc` is fully vouched for. It must NOT be renamed: a false
#: caveat on the thumbnail control -- the §4.2 headline -- is its own harm.
NO_QUALITY_COLUMN = "no-quality-column"

#: `BRANCH_PROVENANCE_NOT_VERIFIED`: these features DO carry the quality column
#: and the caller cannot say which branch produced each row. Nothing is hidden
#: -- the AUC, its CI and the verdict are all returned -- but under
#: `..._unverified_branch_provenance` names, so `result["auc"]` raises
#: `KeyError` rather than yielding a bare figure. That is the same defence the
#: confounded path already gets, for the same failure: a number copied into a
#: results table without the caveat that qualifies it.
BRANCH_PROVENANCE_NOT_VERIFIED = "branch-provenance-not-verified"

#: Sentinel default marking `quality_branches` as required while still allowing
#: an error that names the two opt-outs, rather than a bare TypeError that
#: mentions neither.
_REQUIRED = object()


def thumbnail_features(paths: list[str], size: int = 16) -> np.ndarray:
    """`(N, size*size*3)` float32 in [0, 1]: each image as a tiny RGB thumbnail.

    A 16x16 thumbnail plainly contains content, so "content-blind" here is a
    claim about *which* content, not about none. Downsampling by an order of
    magnitude destroys every band a generator's fingerprint lives in --
    high-frequency residuals, resampling and quantisation traces, local texture
    statistics -- while keeping colour distribution and gross composition. A
    classifier limited to this cannot be reading generation artefacts, so a
    high AUC means the two classes differ in overall appearance: palette,
    brightness, subject framing, or the scene distribution the sources happen
    to contain. That is a dataset property, not a detector capability, which is
    why it invalidates the headline numbers. A near-chance AUC is the useful
    negative result: the classes are not separable by gross appearance, so the
    detector's signal has to come from somewhere finer.

    Two properties are load-bearing and are pinned by tests, because losing
    either turns this into an average-colour detector that returns near-chance
    on a dataset whose classes differ in layout -- a false "clean" on exactly
    the evidence this control exists to provide:

    * The `size * size` grid is kept, not collapsed. Composition is half of
      what is being measured.
    * The resampler averages over the pixels it discards. A point-sampling
      filter (`Image.NEAREST`) would *alias* the fine structure into the
      thumbnail instead of destroying it, so the result would no longer be
      blind to the high-frequency band the detector actually uses.
    """
    out = []
    for p in paths:
        with Image.open(p) as im:
            t = im.convert("RGB").resize((size, size), Image.BILINEAR)
        out.append(np.asarray(t, dtype=np.float32).reshape(-1) / 255.0)
    return np.stack(out).astype(np.float32)


def _has_quantization_table(path: str) -> bool:
    """Whether `estimate_jpeg_quality(_, path)` will take its exact branch.

    Mirrors the guard in `proxies.estimate_jpeg_quality` exactly, including its
    exception set, so the two agree on every file (pinned by a test that checks
    the branch label against the estimator's observable behaviour).
    """
    try:
        with Image.open(path) as im:
            tables = getattr(im, "quantization", None)
            if tables:
                np.asarray(tables[0], dtype=np.float64).reshape(8, 8)
                return True
    except (OSError, ValueError, KeyError):
        return False
    return False


def _metadata_rows(paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """`((N, 4) float32, (N,) branch labels)` -- one pass over the files."""
    rows, branches = [], []
    for p in paths:
        with Image.open(p) as im:
            w, h = im.size
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        q = estimate_jpeg_quality(arr, p)
        rows.append([float(w), float(h), float(w) / max(1.0, h), float(q)])
        branches.append(BRANCH_EXACT if _has_quantization_table(p) else BRANCH_ESTIMATED)
    return np.asarray(rows, dtype=np.float32), np.asarray(branches)


def metadata_features(paths: list[str]) -> np.ndarray:
    """`(N, 4)`: width, height, aspect ratio, estimated JPEG quality.

    Warns when the input spans both branches of `estimate_jpeg_quality`, which
    puts column 3 on two different scales and makes it partly a format label.
    Use `metadata_control` (or pass `quality_estimator_branches(paths)` to
    `content_blind_auc`) to have that checked against the labels rather than
    merely announced.
    """
    features, branches = _metadata_rows(paths)
    _warn_if_branches_are_mixed(branches)
    return features


def quality_estimator_branches(paths: list[str]) -> np.ndarray:
    """`(N,)` of `"exact"` / `"estimated"`: which branch produced each quality.

    `"exact"` is a genuine file-metadata read of the JPEG quantisation table.
    `"estimated"` is the pixel-statistics fallback -- content, not metadata.
    """
    return _metadata_rows(paths)[1]


def _warn_if_branches_are_mixed(branches: np.ndarray) -> None:
    if len(set(branches.tolist())) > 1:
        warnings.warn(
            "estimated JPEG quality came from two different branches of "
            "estimate_jpeg_quality (exact quantisation table for some files, "
            "the pixel-statistics fallback for others). The two are on "
            "different scales, so that column is partly a file-format label. "
            "If the format split follows the class label, a content-blind "
            "verdict resting on it is an estimator artefact -- run "
            "metadata_control() to have that checked.",
            UserWarning,
            stacklevel=3,
        )


def _verdict(auc: float) -> str:
    if auc > VERDICT_THRESHOLDS["broken"]:
        return "broken"
    if auc > VERDICT_THRESHOLDS["suspect"]:
        return "suspect"
    return "clean"


def _branch_check(branches: np.ndarray, labels: np.ndarray) -> dict:
    """How strongly the estimator branch alone predicts the label.

    Unknown branch labels are rejected rather than coerced. `== BRANCH_EXACT`
    maps anything it does not recognise -- a typo, a third branch added to
    `estimate_jpeg_quality` later -- to "not exact", which reads out as
    `pixel_derived`, `confounded: False` and a "no file carried a quantisation
    table" caveat: three false statements and no error. A defence layer must
    not fail towards reassurance.
    """
    branches = np.asarray(branches)
    unknown = sorted(set(branches.tolist()) - {BRANCH_EXACT, BRANCH_ESTIMATED})
    if unknown:
        raise ValueError(
            f"unknown estimator branch label(s) {unknown}; expected only "
            f"{BRANCH_EXACT!r} and {BRANCH_ESTIMATED!r}. Silently treating an "
            "unrecognised label as 'not exact' would report a confound-free "
            "dataset that has not been checked."
        )
    exact = (branches == BRANCH_EXACT).astype(float)
    classes = np.unique(labels)
    fractions = {int(c): float(exact[labels == c].mean()) for c in classes}
    if exact.min() == exact.max() or len(classes) < 2:
        branch_auc = 0.5
    else:
        raw = roc_auc(labels, exact)
        branch_auc = max(raw, 1.0 - raw)
    if exact.max() == 0.0:
        status = "pixel_derived"
    elif exact.min() == 1.0:
        status = "file_metadata"
    else:
        status = "mixed"
    return {
        "status": status,
        "branch_label_auc": branch_auc,
        "exact_fraction_by_class": fractions,
        "counts": {
            BRANCH_EXACT: int(exact.sum()),
            BRANCH_ESTIMATED: int(len(exact) - exact.sum()),
        },
        "confounded": bool(branch_auc >= BRANCH_CONFOUND_THRESHOLD),
    }


def content_blind_auc(features: np.ndarray, labels: np.ndarray,
                      seed: int = 20260827, n_splits: int = 5,
                      quality_branches: np.ndarray | str = _REQUIRED) -> dict:
    """Cross-validated AUC of a classifier restricted to `features`.

    Returns `auc`, `auc_ci`, `verdict` (`broken` / `suspect` / `clean` per
    `VERDICT_THRESHOLDS`), `n_splits`, and `quality_branch_check`.

    Every step -- scaling included -- is fitted inside the fold: the whole
    classifier is a `Pipeline` and `features` is handed to `cross_val_predict`
    untouched, so no test fold informs its own prediction. Fitting *anything*
    on the full array first -- a scaler, an imputer, a feature selector --
    inflates the AUC, and an inflated AUC here reads as "your dataset is
    broken": the leak's failure mode is a false alarm. Do not preprocess
    `features` in this function; put the step in the pipeline.

    `quality_branches` is REQUIRED and has three legitimate values. Pass the
    real branch labels (`quality_estimator_branches(paths)`) whenever the
    features include an estimated-JPEG-quality column -- `metadata_control`
    does this for you and is the cheap path, since it reads each file once for
    both the features and their provenance. Pass `NO_QUALITY_COLUMN` when the
    features carry no quality estimate at all (thumbnails, geometry only), and
    `BRANCH_PROVENANCE_NOT_VERIFIED` when they do carry one and you cannot say
    which branch produced each row.

    Two of those three make the number un-quotable-bare, and by the same
    mechanism. If the branch assignment predicts the label
    (`BRANCH_CONFOUND_THRESHOLD`), `verdict` becomes `"confounded"` and `auc`,
    `auc_ci` and the plain verdict move to `auc_ignoring_branch_confound`,
    `auc_ci_ignoring_branch_confound` and `verdict_ignoring_branch_confound`.
    If provenance was never supplied for a quality column, `verdict` becomes
    `"unverified_branch_provenance"` and the three move to
    `auc_unverified_branch_provenance`,
    `auc_ci_unverified_branch_provenance` and
    `verdict_unverified_branch_provenance`. The keys are long and
    self-incriminating on purpose: neither number can be lifted into a results
    table without its caveat coming along, and code that reads `result["auc"]`
    raises `KeyError` rather than quoting one. The failure mode being defended
    is identical in the two cases, which is why the defence is.

    Only `NO_QUALITY_COLUMN` -- and a real branch array that comes back
    unconfounded -- leaves a plain `auc` in the result, because only there is
    there something to vouch for it.
    """
    if quality_branches is _REQUIRED:
        raise ValueError(
            "content_blind_auc requires `quality_branches`: pass "
            "quality_estimator_branches(paths) so the estimator-branch "
            "confound is checked against the labels, or declare the "
            "alternative -- quality_branches=NO_QUALITY_COLUMN if these "
            "features carry no quality estimate at all (thumbnails, geometry "
            "only), or quality_branches=BRANCH_PROVENANCE_NOT_VERIFIED if they "
            "do and you cannot say which branch produced each row. The last is "
            "not a way to silence the check: it renames the AUC to "
            "auc_unverified_branch_provenance so it cannot be quoted bare. "
            "metadata_control() supplies the real thing and is the cheap path.")
    if quality_branches is None:
        raise ValueError(
            "quality_branches=None does not say which case you are in. Pass "
            "NO_QUALITY_COLUMN if these features carry no quality column, or "
            "BRANCH_PROVENANCE_NOT_VERIFIED if they do and its provenance is "
            "unknown; the two produce different results and only you know "
            "which is true of `features`.")
    if isinstance(quality_branches, str) and quality_branches not in (
            NO_QUALITY_COLUMN, BRANCH_PROVENANCE_NOT_VERIFIED):
        # Checked BEFORE the cross-validated fit and the 500-resample
        # bootstrap, so a typo costs a millisecond rather than a full run.
        raise ValueError(
            f"quality_branches must be one branch label per row, or the exact "
            f"sentinel NO_QUALITY_COLUMN or BRANCH_PROVENANCE_NOT_VERIFIED, "
            f"got {quality_branches!r}")

    labels = np.asarray(labels)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, random_state=seed))
    proba = cross_val_predict(clf, features, labels, cv=n_splits,
                              method="predict_proba")[:, 1]
    auc = roc_auc(labels, proba)
    ci = bootstrap_ci(roc_auc, labels, proba, n=500, seed=seed)
    verdict = _verdict(auc)

    result = {"auc": auc, "auc_ci": ci, "verdict": verdict, "n_splits": n_splits}
    if isinstance(quality_branches, str):
        # Validated above, so this is one of the two sentinels.
        if quality_branches == NO_QUALITY_COLUMN:
            result["quality_branch_check"] = (
                "not applicable: caller declared no quality column")
            return result
        result["quality_branch_check"] = "not performed: no branch provenance supplied"
        result["auc_unverified_branch_provenance"] = result.pop("auc")
        result["auc_ci_unverified_branch_provenance"] = result.pop("auc_ci")
        result["verdict_unverified_branch_provenance"] = result.pop("verdict")
        result["verdict"] = "unverified_branch_provenance"
        return result

    check = _branch_check(quality_branches, labels)
    result["quality_branch_check"] = check
    if check["confounded"]:
        result["auc_ignoring_branch_confound"] = result.pop("auc")
        result["auc_ci_ignoring_branch_confound"] = result.pop("auc_ci")
        result["verdict_ignoring_branch_confound"] = result.pop("verdict")
        result["verdict"] = "confounded"
    return result


def metadata_control(paths: list[str], labels: np.ndarray,
                     seed: int = 20260827, n_splits: int = 5) -> dict:
    """The metadata control, run with its estimator provenance attached.

    Prefer this over calling `content_blind_auc(metadata_features(paths), ...)`
    by hand: it passes the branch labels through, so the estimator-branch
    confound is checked rather than silently folded into one number.

    Beyond `content_blind_auc`'s keys it returns `geometry` (the same control
    on width/height/aspect alone -- the columns no estimator touches, so its
    verdict is trustworthy whatever the formats are) and `caveats`: an empty
    list only when the quality column is a real quantisation-table read for
    every file. The branch status and per-branch counts are not repeated at the
    top level; they live in `quality_branch_check["status"]` and
    `quality_branch_check["counts"]`, one place so the two cannot disagree.
    """
    labels = np.asarray(labels)
    features, branches = _metadata_rows(paths)
    _warn_if_branches_are_mixed(branches)

    result = content_blind_auc(features, labels, seed=seed, n_splits=n_splits,
                               quality_branches=branches)
    geometry = content_blind_auc(features[:, :3], labels, seed=seed,
                                 n_splits=n_splits,
                                 quality_branches=NO_QUALITY_COLUMN)
    # Refines the generic declaration with the reason it is true here.
    geometry["quality_branch_check"] = "not applicable: geometry columns only"
    check = result["quality_branch_check"]

    caveats = []
    if check["confounded"]:
        caveats.append(
            "the estimator branch predicts the label (branch-only AUC "
            f"{check['branch_label_auc']:.3f}); the quality column is partly a "
            "file-format label, so the combined verdict is withheld. The "
            "geometry-only verdict below is unaffected."
        )
    elif check["status"] == "mixed":
        caveats.append(
            "the quality column mixes exact and pixel-estimated values; the "
            "branch split does not follow the label, but the column spans two "
            "scales and is not a like-for-like comparison."
        )
    if check["status"] == "pixel_derived":
        caveats.append(
            "no file carried a quantisation table, so the quality column is a "
            "pixel statistic (blockiness), not metadata -- its leave-one-family-out "
            "error is 14 to 31 quality points, which exceeds most real "
            "differences between sources. A near-chance result from it is weak "
            "evidence of a clean dataset, not proof; re-check with the geometry-"
            "only verdict and with thumbnail_features."
        )

    result["geometry"] = geometry
    result["caveats"] = caveats
    return result
