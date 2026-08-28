"""Resolution-only control: the floor every headline number must be read against.

This is not a detector. It is a deliberately trivial classifier that sees
**image dimensions and nothing else** -- no pixel ever reaches it -- and whose
only job is to answer one question: *how much of our reported score is
available without looking at the image at all?*

Why this exists
---------------
The frozen manifest's resolution column predicts the label. Measured on
`data/manifest.parquet` (138,116 rows), the WildFake short side splits like
this:

    short side    real     fake    % fake
      128            0    1,262      100%
      200       40,000   11,516       22%
      224            0    7,002      100%
      256            0   25,728      100%
      450            0    6,158      100%
      512       15,000   10,316       41%

The four 100%-fake WildFake sizes hold 40,150 images, 29.1% of the whole
138,116-row manifest. Counting every short side that is 100% fake across both
sources, not just WildFake's, it is 40,905 rows -- 29.6%. A single threshold,
`short side > 200`, scores 0.7258 accuracy against a 0.5290 majority baseline.

That is a property of how the sources were assembled, not of how the images
were made.

On the organisers' scored benchmark it is worse, and it is not a tendency but
a clean separation. Over ALL 13,841 demo images (not a sample): every one of
the 4,998 COCO real images has a short side of exactly 200, and all 8,843
DALL-E 3 images have a short side between 346 and 1,746. The two ranges do not
touch. `short_side > 200 -> AI-generated` is a perfect classifier on the
benchmark, AUC 1.000, and it never decodes a pixel.

So a headline TPR on that benchmark is uninterpretable on its own. This module
produces the number it has to be read against.

How to read the output
----------------------
`resolution_leak_report` returns the TPR at the project operating point
(`aigcdet.operating_point.TARGET_FPR`) achievable from dimensions alone, fitted
on one split and scored on a disjoint one. `describe` renders it as prose that
carries its own interpretation, because the number is only meaningful with the
sentence attached: **a model that does not substantially beat this has not been
shown to detect generation artefacts.** It has been shown to detect the
dataset's assembly.

Relationship to `eval.controls`
-------------------------------
`eval.controls.metadata_control(...)["geometry"]` is a neighbouring measurement
and is NOT this. Three differences, each of which matters here:

* It takes file **paths** and opens every image with PIL, so pixels are read
  (`_metadata_rows` decodes to RGB for the quality column) and the geometry
  columns are a slice of a larger matrix. This module takes a **DataFrame** and
  can only ever see `width` and `height`; there is no path to a pixel.
* It reports a **cross-validated AUC** over one pool. This reports **TPR at the
  project operating point**, fitted on one split and scored on a disjoint one
  -- the same quantity the ablation table reports for a real model, so the two
  are directly comparable. An AUC is not.
* Its purpose is a verdict on the dataset (`broken`/`suspect`/`clean`). This
  one's purpose is a **floor for the results table**.

Both should be run. Only this one belongs in a row next to a model's score.

Nothing here loads weights, opens a file, or starts a GPU process.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from aigcdet.eval.metrics import roc_auc, tpr_at_fpr
from aigcdet.operating_point import TARGET_FPR, fpr_label, tpr_column_name

#: The label column, 0 = authentic and 1 = AI-generated, as
#: `aigcdet.data.manifest.MANIFEST_COLUMNS` defines it. Named rather than
#: inlined at each use so that "which column was this scored against?" has one
#: answer. A metric computed against any other column in the frame -- a
#: generator name, a source, a prediction someone joined on -- is not the
#: quantity this module claims to report.
LABEL_COLUMN = "label"

#: The ONLY columns any feature here may be derived from. This tuple is the
#: whole integrity claim of the module: `resolution_features` selects these two
#: columns out of the frame and never touches another, so a frame carrying a
#: perfectly-separating pixel statistic produces exactly the same feature
#: matrix as one carrying none. That is checked by a test that hands it both.
DIMENSION_COLUMNS: tuple[str, ...] = ("width", "height")

#: The split column, read per row. Same name and same meaning as
#: `bank.meta["split"]` and `MANIFEST_COLUMNS`.
SPLIT_COLUMN = "split"

#: Short sides that image-generation pipelines emit *by construction* --
#: powers of two and the ViT/latent-diffusion sizes built around them.
#:
#: Declared A PRIORI and deliberately NOT fitted to this dataset's leak. Both
#: of the manifest's most lopsided sizes are informative about that choice:
#: 224 and 256 are in this tuple and are 100% fake, while 450 is 100% fake and
#: is NOT in it, and 200 -- the whole real half of the benchmark -- is not in
#: it either. Tuning the tuple to the observed split would make this feature a
#: memorised label rather than a stated prior about generators, and would leave
#: the control quietly measuring itself.
CANONICAL_SHORT_SIDES: tuple[int, ...] = (
    64, 128, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280,
    1536, 2048,
)

#: Layout of `resolution_features`' columns. A positional contract, like
#: `npr.NPR_FEATURE_NAMES`: swapping two entries is invisible to every
#: aggregate statistic and to the fitted tree's accuracy.
#:
#: Logs, not raw pixels counts, because resolution is multiplicative -- 256 is
#: to 512 what 512 is to 1024 -- and a depth-limited tree splitting on a raw
#: scale spends its depth budget in the crowded low end.
RESOLUTION_FEATURE_NAMES: tuple[str, ...] = (
    "log2_width",
    "log2_height",
    "log2_short_side",
    "log2_long_side",
    "aspect_ratio",            # long / short, so it is orientation-blind
    "log2_area",
    "short_side_is_canonical", # 0/1 against CANONICAL_SHORT_SIDES
)

#: Row label a results table must use for this row, and the sentence that has
#: to travel with it. Exported here rather than written into
#: `aigcdet.baselines.__init__`'s registry by this module, so that wiring it in
#: stays one visible edit in that file (ruling R38/I3's mechanism: the wording
#: is data, not prose).
BASELINE_ROW_LABEL = (
    "Resolution-only control (image dimensions, no pixels read)")
BASELINE_ROW_FOOTNOTE = (
    "not a detector: a depth-limited decision tree on width and height alone, "
    "fitted on one split and scored on a disjoint one. It is the floor every "
    "other row must be read against -- a row that does not substantially "
    "exceed it has not been shown to detect generation artefacts.")

#: How much a real detector has to beat this control by before its score is
#: evidence about generation artefacts rather than about resolution. A margin,
#: not a threshold on the model's own score: at a control TPR of 0.99 there is
#: no headroom left and NO model score is interpretable on that population,
#: which is the situation on the organisers' benchmark and is exactly what
#: `describe` has to say out loud.
SUBSTANTIAL_MARGIN = 0.10

#: Pinned so the tree, and therefore the reported floor, is reproducible.
#: Matches the seed the other baselines and controls in this project use.
DEFAULT_SEED = 20260827

#: Depth-limited on purpose. The point of this baseline is that it is
#: *obviously* trivial and human-readable: `format_tree` prints the whole
#: model, and at this depth that is a page of `short_side <= x` tests a
#: reviewer can check by eye. It is not depth-limited to handicap it -- the
#: leak it measures is a handful of discrete resolutions, which is what a
#: shallow tree represents best.
DEFAULT_MAX_DEPTH = 6

#: Leaves must be big enough for their class proportion to BE a probability.
#:
#: This was 5, and 5 is wrong in a way that matters and is not obvious. A tree
#: scores by leaf purity, so every pure leaf ties at exactly 1.0 and TPR at 1%
#: FPR is read off that whole tied block at once. One five-row leaf that is
#: pure by chance on the fit rows -- and at 50/50 class odds that is one leaf
#: in 32 -- carries real negatives at score time, spends the entire 1% FPR
#: budget, and drops the reported TPR to 0.0000. Measured on twelve synthetic
#: continuum draws, `min_samples_leaf=5` produced a usable 0 < TPR@1% < TPR@5%
#: in 1 of 12; at 50 it was 10 of 12.
#:
#: The direction of that failure is why the default moved rather than the test
#: being reseeded: a collapsed floor UNDERSTATES the leak, and an understated
#: floor flatters every model reported above it. This baseline exists to be a
#: lower bound, so its failure mode must not be silent optimism.
#:
#: 50 is not a tuned number -- it is roughly where a leaf proportion stops
#: being noise (+/- 0.07 at worst) -- and it was checked to change nothing on
#: the real data: on the frozen manifest, train -> val_internal gives TPR@1%
#: 0.5654 and AUC 0.9220 at both 5 and 50. It costs resolution only on frames
#: of a few hundred rows, where a trivial control should be a stump anyway.
DEFAULT_MIN_SAMPLES_LEAF = 50


def _require_columns(frame: pd.DataFrame, columns, what: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"{what} takes a DataFrame of manifest rows, not "
            f"{type(frame).__name__}. The type is the guarantee: this baseline "
            f"is a control whose whole claim is that it cannot see a pixel, so "
            f"it accepts no image, no path and no feature matrix a caller "
            f"could have put pixel statistics into")
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{what} needs column(s) {missing}; the frame has "
            f"{list(frame.columns)}")


def resolution_features(frame: pd.DataFrame) -> np.ndarray:
    """`(n, len(RESOLUTION_FEATURE_NAMES))` float64, from dimensions ALONE.

    Reads `DIMENSION_COLUMNS` out of `frame` and nothing else. Every other
    column -- pixel statistics, paths, generator names, the label itself -- is
    ignored by construction, not by convention: the two columns are selected by
    name, so there is no code path by which a third could reach the output.
    Hand this function two frames that differ only outside `DIMENSION_COLUMNS`
    and it returns bit-identical matrices.

    Refuses non-positive dimensions rather than returning `-inf`. A width of 0
    is a corrupt manifest row, and `log2(0)` would put a single `-inf` into a
    column that `DecisionTreeClassifier` accepts without complaint and then
    splits on, silently making that one row its own leaf.
    """
    _require_columns(frame, DIMENSION_COLUMNS, "resolution_features")
    w = frame[DIMENSION_COLUMNS[0]].to_numpy(dtype=np.float64)
    h = frame[DIMENSION_COLUMNS[1]].to_numpy(dtype=np.float64)
    if w.size and (np.min(w) <= 0 or np.min(h) <= 0):
        bad = int(((w <= 0) | (h <= 0)).sum())
        raise ValueError(
            f"{bad} of {w.size} rows have a non-positive width or height; a "
            f"manifest row cannot describe a zero-pixel image, and log2(0) "
            f"would become a -inf the tree happily splits on")
    short = np.minimum(w, h)
    long = np.maximum(w, h)
    canonical = np.isin(short.astype(np.int64), np.asarray(CANONICAL_SHORT_SIDES))
    # Only exact integer sizes can be canonical; a fractional short side is not
    # a size any pipeline emits, and `astype` above would truncate it into one.
    canonical &= short == np.floor(short)
    return np.column_stack([
        np.log2(w),
        np.log2(h),
        np.log2(short),
        np.log2(long),
        long / short,
        np.log2(w * h),
        canonical.astype(np.float64),
    ])


def _check_split_column(frame: pd.DataFrame, what: str) -> np.ndarray:
    _require_columns(frame, (SPLIT_COLUMN,), what)
    return frame[SPLIT_COLUMN].to_numpy()


class ResolutionBaseline:
    """A shallow decision tree on the seven dimension features.

    Two departures from `npr.NPRDetector`, both deliberate and both load-bearing
    for what this class is *for*:

    **It takes a DataFrame, never a feature matrix.** `NPRDetector.fit` takes an
    `(n, d)` array, which is right for a detector -- the caller owns the
    features. Here that signature would defeat the entire control: anyone could
    hand it a matrix with a pixel statistic in it and the resulting number would
    still be published as "achievable from dimensions alone". The frame goes
    through `resolution_features`, inside this class, on every call.

    **It refuses to be scored on the rows it was fitted on.** `NPRDetector`
    documents that obligation and leaves it with the caller. That is defensible
    for a detector whose number is compared against other detectors' numbers
    under the same discipline; it is not defensible here, because this number's
    job is to be a FLOOR. A floor inflated by memorising the fit rows makes
    every model above it look better than it is -- the error points the wrong
    way, towards reassurance, so the guard is structural: `fit` records which
    split it consumed and `score` rejects any row carrying it.
    """

    def __init__(self, seed: int = DEFAULT_SEED,
                 max_depth: int = DEFAULT_MAX_DEPTH,
                 min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF):
        self.clf = DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            random_state=seed)
        #: The split label every fit row carried. `None` until `fit` runs.
        self.fit_split_: str | None = None
        self.n_fit_: int = 0

    def fit(self, frame: pd.DataFrame) -> "ResolutionBaseline":
        """Fit on `frame`, whose rows must ALL carry one split label.

        The single-split requirement is what makes `score`'s guard checkable.
        A fit spanning two splits has no one split to refuse later, so the
        train-on-test check would silently become a no-op -- and this class
        exists to produce a number that cannot be quietly inflated.

        Labels come from `frame[LABEL_COLUMN]`, not from a separate argument,
        so the frame is the whole input and there is no second place for the
        answer to enter from.
        """
        _require_columns(frame, (LABEL_COLUMN,), "ResolutionBaseline.fit")
        splits = _check_split_column(frame, "ResolutionBaseline.fit")
        present = sorted({str(s) for s in splits.tolist()})
        if len(present) != 1:
            raise ValueError(
                f"ResolutionBaseline.fit needs every row to carry ONE split "
                f"label so that score() can refuse those rows later; got "
                f"{present}. Select the fit split before calling")
        y = frame[LABEL_COLUMN].to_numpy()
        if len(np.unique(y)) < 2:
            raise ValueError(
                f"the fit rows carry a single class ({np.unique(y).tolist()}); "
                f"a control fitted on one class reports a constant score and "
                f"an AUC of exactly 0.5, which is indistinguishable in a "
                f"results table from 'there is no resolution leak'")
        self.clf.fit(resolution_features(frame), y)
        self.fit_split_ = present[0]
        self.n_fit_ = len(frame)
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """P(AI-generated) per row -- higher means more likely AI-generated,
        matching `aigcdet.eval.metrics`' score convention and `npr`'s.

        Refuses any row carrying the split `fit` consumed. See the class
        docstring for why this guard is here and not in the caller.
        """
        if self.fit_split_ is None:
            raise RuntimeError("ResolutionBaseline.score before fit")
        splits = _check_split_column(frame, "ResolutionBaseline.score")
        overlap = int((splits.astype(str) == self.fit_split_).sum())
        if overlap:
            raise ValueError(
                f"{overlap} of {len(frame)} rows to be scored carry split "
                f"{self.fit_split_!r}, which is the split this baseline was "
                f"fitted on. Scoring them would report the tree's memory as "
                f"the resolution floor, and an inflated floor makes every "
                f"model above it look better than it is")
        return self.clf.predict_proba(resolution_features(frame))[:, 1]

    def format_tree(self) -> str:
        """The whole fitted model as text -- every threshold, readable by eye.

        The baseline's credibility rests on a reader being able to confirm it
        is trivial, so the model is printable rather than merely described.
        """
        if self.fit_split_ is None:
            raise RuntimeError("ResolutionBaseline.format_tree before fit")
        return export_text(self.clf, feature_names=list(RESOLUTION_FEATURE_NAMES))


def resolution_leak_report(frame: pd.DataFrame, *, fit_split: str,
                           score_split: str,
                           target_fpr: float = TARGET_FPR,
                           seed: int = DEFAULT_SEED,
                           max_depth: int = DEFAULT_MAX_DEPTH,
                           min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF) -> dict:
    """Fit on `fit_split`, score on `score_split`, report the floor.

    Returns the TPR at `target_fpr` under `operating_point.tpr_column_name`'s
    DERIVED key -- `tpr_at_1pct` at the project default -- alongside `auc`, the
    two split names, their row counts, and the declarations
    (`target_fpr`, `operating_point`) that let a reader of the artefact check
    which operating point produced the number. The number and its provenance
    travel together, on the model of `run_ablation._selection_summary`: a bare
    float in a results file is a figure nobody can audit.

    `fit_split` and `score_split` must name different splits, and the two row
    sets are checked to be disjoint after selection as well as before -- the
    second check is not redundant, it is what catches the masks being built
    from the wrong column or swapped.
    """
    _require_columns(frame, (LABEL_COLUMN, SPLIT_COLUMN) + DIMENSION_COLUMNS,
                     "resolution_leak_report")
    if fit_split == score_split:
        raise ValueError(
            f"fit_split and score_split are both {fit_split!r}; fitting and "
            f"scoring the same rows reports the tree's memory, not the "
            f"resolution floor")
    splits = frame[SPLIT_COLUMN].to_numpy().astype(str)
    fit_mask = splits == str(fit_split)
    score_mask = splits == str(score_split)
    for name, value, mask in (("fit_split", fit_split, fit_mask),
                              ("score_split", score_split, score_mask)):
        if not mask.any():
            raise ValueError(
                f"{name}={value!r} matches no rows; the frame carries splits "
                f"{sorted(set(splits.tolist()))}")
    if (fit_mask & score_mask).any():
        raise ValueError(
            "the fit and score row sets overlap, which cannot happen for two "
            "different split labels -- the masks were built from the wrong "
            "column")

    model = ResolutionBaseline(seed=seed, max_depth=max_depth,
                               min_samples_leaf=min_samples_leaf)
    model.fit(frame[fit_mask])
    scored = frame[score_mask]
    s = model.score(scored)
    y = scored[LABEL_COLUMN].to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError(
            f"the score split {score_split!r} carries a single class "
            f"({np.unique(y).tolist()}); TPR at a false-POSITIVE rate is "
            f"undefined without negatives, and an AUC over one class is not a "
            f"number. In the frozen manifest this is true of "
            f"'heldout_generator', which is 100% generated")
    if len(np.unique(s)) < 2:
        # The same defence `aeroblade_scores` puts on an all-zero LPIPS column,
        # for the same reason. A constant score is an AUC of exactly 0.5 and a
        # TPR of exactly 0.0 at any FPR -- which in a results table is
        # indistinguishable from the honest finding "resolution carries no
        # signal here", and that reading is the dangerous one: it publishes a
        # floor of zero and makes every model above it look unimpeachable.
        raise ValueError(
            f"the fitted tree assigns the same score ({s[0]}) to all "
            f"{len(s)} scored rows, so it found no usable split at all. That "
            f"reports as AUC 0.5 and TPR 0.0, which cannot be told apart from "
            f"a genuine absence of resolution leak. Usual causes: every fit "
            f"row shares one resolution, or min_samples_leaf "
            f"({min_samples_leaf}) exceeds what the {int(fit_mask.sum())} fit "
            f"rows can support")
    return {
        tpr_column_name(target_fpr): tpr_at_fpr(y, s, target_fpr),
        "auc": roc_auc(y, s),
        # Declared, not implied. `tpr_column_name` renames the key when the
        # operating point moves, and these two say which point that was in a
        # form a reader does not have to parse out of a column name.
        "target_fpr": float(target_fpr),
        "operating_point": fpr_label(target_fpr),
        # Read back off the OBJECTS, not off the masks that were meant to
        # build them: `model.fit_split_` is the label the tree actually
        # consumed and `n_fit_` is how many rows it actually saw. Reporting
        # `fit_mask.sum()` here instead describes the intent, so a swap of the
        # two frames leaves the counts looking correct while the number they
        # annotate came from the other split -- a report that is wrong and
        # says it is right.
        "fit_split": model.fit_split_,
        "score_split": str(score_split),
        "n_fit": model.n_fit_,
        "n_score": int(len(scored)),
        "positive_rate_score_split": float(np.mean(y == 1)),
        "features": list(RESOLUTION_FEATURE_NAMES),
        "tree": model.format_tree(),
    }


def describe(result: dict) -> str:
    """The report as prose that carries its own interpretation.

    The number is useless bare and actively misleading in a table without this
    sentence, so `describe` is the intended way to surface it: a reader who
    sees `0.97` next to a model's `0.98` has to be told, in the same breath,
    that the 0.97 was obtained without decoding a pixel.
    """
    key = tpr_column_name(result["target_fpr"])
    tpr = result[key]
    point = result["operating_point"]
    headroom = 1.0 - tpr
    lines = [
        f"Resolution-only control: TPR {tpr:.4f} at {point} FPR "
        f"(AUC {result['auc']:.4f}).",
        f"Fitted on {result['n_fit']} {result['fit_split']!r} rows, scored on "
        f"{result['n_score']} disjoint {result['score_split']!r} rows "
        f"({result['positive_rate_score_split']:.1%} AI-generated).",
        "",
        "What this number is: the score achievable from IMAGE DIMENSIONS "
        "ALONE. The classifier is a shallow decision tree over "
        f"{len(result['features'])} features derived only from width and "
        "height. It never decoded a pixel, opened a file, or loaded a weight.",
        "",
        f"How to read any model against it: a detector scoring at or below "
        f"{tpr:.4f} TPR at {point} FPR on this population has demonstrated "
        f"nothing about generation artefacts, and a detector that does not "
        f"exceed it by a substantial margin (>= {SUBSTANTIAL_MARGIN:.2f}) has "
        f"not been shown to be reading the image rather than its shape.",
    ]
    if headroom < SUBSTANTIAL_MARGIN:
        lines += [
            "",
            f"WARNING: this control leaves only {headroom:.4f} of headroom "
            f"below a perfect score, which is less than the "
            f"{SUBSTANTIAL_MARGIN:.2f} margin above. On this population NO "
            f"model score is interpretable as evidence of artefact detection, "
            f"however high it is: the population is separable by resolution, "
            f"so a perfect detector and a ruler are indistinguishable here. "
            f"Report the number, report this control beside it, and do not "
            f"claim artefact detection from this population alone.",
        ]
    return "\n".join(lines)
