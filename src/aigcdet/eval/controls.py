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
(`BRANCH_CONFOUND_THRESHOLD`). The raw verdict is still returned alongside,
under `verdict_ignoring_branch_confound`; nothing is hidden, but the headline
field cannot be quoted without the confound being visible.

The confound runs the other way too. Matching the file formats makes the
branch split vanish and the control go quiet -- but with every image now on
the *pixel* branch, column 3 is no longer file metadata at all: it is a
blockiness statistic read off the pixels, whose error (14 to 31 points)
exceeds most of the differences it would need to resolve. A near-chance
result from it is therefore weak evidence, not proof of a clean dataset.
`metadata_control` reports that state as `quality_feature_status ==
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
    """How strongly the estimator branch alone predicts the label."""
    exact = (np.asarray(branches) == BRANCH_EXACT).astype(float)
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
                      quality_branches: np.ndarray | None = None) -> dict:
    """Cross-validated AUC of a classifier restricted to `features`.

    Returns `auc`, `auc_ci`, `verdict` (`broken` / `suspect` / `clean` per
    `VERDICT_THRESHOLDS`), `n_splits`, and `quality_branch_check`.

    Every step -- scaling included -- is fitted inside the fold via a
    `Pipeline`, so no test fold informs its own prediction. Standardising on
    the full array first would inflate the AUC, and an inflated AUC here reads
    as "your dataset is broken": the leak's failure mode is a false alarm.

    Pass `quality_branches` (from `quality_estimator_branches`) whenever the
    features include an estimated-JPEG-quality column. If the branch
    assignment predicts the label, `verdict` becomes `"confounded"` and the
    unqualified verdict moves to `verdict_ignoring_branch_confound`. Without
    it, `quality_branch_check` records that no check was performed rather than
    leaving the result looking checked.
    """
    labels = np.asarray(labels)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, random_state=seed))
    proba = cross_val_predict(clf, features, labels, cv=n_splits,
                              method="predict_proba")[:, 1]
    auc = roc_auc(labels, proba)
    ci = bootstrap_ci(roc_auc, labels, proba, n=500, seed=seed)
    verdict = _verdict(auc)

    result = {"auc": auc, "auc_ci": ci, "verdict": verdict, "n_splits": n_splits}
    if quality_branches is None:
        result["quality_branch_check"] = "not performed: no branch provenance supplied"
        return result

    check = _branch_check(np.asarray(quality_branches), labels)
    result["quality_branch_check"] = check
    if check["confounded"]:
        result["verdict"] = "confounded"
        result["verdict_ignoring_branch_confound"] = verdict
    return result


def metadata_control(paths: list[str], labels: np.ndarray,
                     seed: int = 20260827, n_splits: int = 5) -> dict:
    """The metadata control, run with its estimator provenance attached.

    Prefer this over calling `content_blind_auc(metadata_features(paths), ...)`
    by hand: it passes the branch labels through, so the estimator-branch
    confound is checked rather than silently folded into one number.

    Beyond `content_blind_auc`'s keys it returns `geometry` (the same control
    on width/height/aspect alone -- the columns no estimator touches, so its
    verdict is trustworthy whatever the formats are), `quality_feature_status`,
    `quality_branch_counts`, and `caveats`: an empty list only when the quality
    column is a real quantisation-table read for every file.
    """
    labels = np.asarray(labels)
    features, branches = _metadata_rows(paths)
    _warn_if_branches_are_mixed(branches)

    result = content_blind_auc(features, labels, seed=seed, n_splits=n_splits,
                               quality_branches=branches)
    geometry = content_blind_auc(features[:, :3], labels, seed=seed, n_splits=n_splits)
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
    result["quality_feature_status"] = check["status"]
    result["quality_branch_counts"] = check["counts"]
    result["caveats"] = caveats
    return result
