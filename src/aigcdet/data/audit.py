"""Profile the classes before normalising them (spec §4.2 defence 1).

The output table goes in docs/data_audit.md and in the README; it is the
figure that motivates the whole normalisation step.

`jpeg_q` uses `aigcdet.features.proxies.estimate_jpeg_quality` with the file
path passed in, so a real JPEG's quantisation table is read exactly. Only
non-JPEG inputs fall back to the pixel-based estimate; see that function's
docstring for the fallback's known miscalibration.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from PIL import Image

from aigcdet.features.proxies import estimate_jpeg_quality

# Thresholds shared by every comparison audit_flags makes (class-level,
# per-source-vs-opposite-class, and within-class): a >1.5x median-width
# ratio, or a >10-point median JPEG-quality gap.
_WIDTH_RATIO_THRESHOLD = 1.5
_JPEG_Q_DIFF_THRESHOLD = 10.0


def audit_table(paths: list[str], labels: list[int], sources: list[str]) -> pd.DataFrame:
    """One row per (source, label): n, dominant format, median width/height,
    median estimated JPEG quality."""
    rows = []
    for p, lab, src in zip(paths, labels, sources):
        with Image.open(p) as im:
            fmt, (w, h) = im.format, im.size
            q = estimate_jpeg_quality(np.asarray(im.convert("RGB")), p)
        rows.append({"source": src, "label": lab, "fmt": fmt,
                     "width": w, "height": h, "jpeg_q": q})
    df = pd.DataFrame(rows)
    return (df.groupby(["source", "label"], as_index=False)
              .agg(n=("fmt", "size"),
                   fmt_top=("fmt", lambda s: s.mode().iloc[0]),
                   width_median=("width", "median"),
                   height_median=("height", "median"),
                   jpeg_q_median=("jpeg_q", "median")))


def _width_ratio_flagged(a: float, b: float) -> bool:
    return max(a, b) / max(1.0, min(a, b)) > _WIDTH_RATIO_THRESHOLD


def _jpeg_q_flagged(a: float, b: float) -> bool:
    return abs(a - b) > _JPEG_Q_DIFF_THRESHOLD


def audit_flags(df: pd.DataFrame) -> list[str]:
    """Warn where authentic and generated images differ in ways a detector
    could exploit without looking at content at all.

    Three comparisons, because the confound this audit exists to catch
    usually enters through a *source*, not through the label, and a
    comparison that pools every source of a class together before comparing
    can dilute it to nothing:

    1. Class-level (pooled): each class's median-of-per-source-medians. The
       original, coarsest check -- cheap to read, but blind to a class that
       mixes a low-resolution source with a high-resolution one if their
       medians happen to average out close to the other class's.
    2. Per-source vs. the opposite class, pooled: catches an individual
       source that differs from the opposite class even when other sources
       of its own label would otherwise average it out. Reuses the same
       pooled opposite-class figures (1) already computed, so a source is
       judged against the same yardstick, just without hiding inside its
       own class's pool first.
    3. Within-class heterogeneity: two sources sharing a label that differ
       materially from *each other*. Invisible to both (1) and (2) -- a
       source can sit close to the opposite class's pooled figure while
       still being wildly unlike a same-label source, e.g. one real source
       at ~500px and another at ~2000px, whose pooled median lands near the
       fake class's ~1024px and clears both of the checks above even though
       the resolution split within the real class is itself exploitable.
    """
    flags: list[str] = []
    real, fake = df[df["label"] == 0], df[df["label"] == 1]
    if real.empty or fake.empty:
        return flags

    # 1. Class-level, pooled.
    if set(real["fmt_top"]) != set(fake["fmt_top"]):
        flags.append(
            f"Format confound: authentic {sorted(set(real['fmt_top']))} vs "
            f"generated {sorted(set(fake['fmt_top']))}")
    rw, fw = real["width_median"].median(), fake["width_median"].median()
    if _width_ratio_flagged(rw, fw):
        flags.append(f"Resolution confound: median width {rw:.0f} vs {fw:.0f}")
    rq, fq = real["jpeg_q_median"].median(), fake["jpeg_q_median"].median()
    if _jpeg_q_flagged(rq, fq):
        flags.append(f"JPEG-quality confound: median q {rq:.0f} vs {fq:.0f}")

    # 2. Per-source vs. the opposite class, pooled.
    real_fmts, fake_fmts = set(real["fmt_top"]), set(fake["fmt_top"])
    for _, row in df.iterrows():
        is_real = row["label"] == 0
        own_label = "authentic" if is_real else "generated"
        opp_label = "generated" if is_real else "authentic"
        opp_fmts = fake_fmts if is_real else real_fmts
        opp_w = fw if is_real else rw
        opp_q = fq if is_real else rq
        src = row["source"]
        if row["fmt_top"] not in opp_fmts:
            flags.append(
                f"Format confound: source '{src}' ({own_label}) is "
                f"{row['fmt_top']} but no {opp_label} source is")
        if _width_ratio_flagged(row["width_median"], opp_w):
            flags.append(
                f"Resolution confound: source '{src}' ({own_label}) median "
                f"width {row['width_median']:.0f} vs {opp_label} class {opp_w:.0f}")
        if _jpeg_q_flagged(row["jpeg_q_median"], opp_q):
            flags.append(
                f"JPEG-quality confound: source '{src}' ({own_label}) median "
                f"q {row['jpeg_q_median']:.0f} vs {opp_label} class {opp_q:.0f}")

    # 3. Within-class heterogeneity.
    for label, name in ((0, "authentic"), (1, "generated")):
        rows = list(df[df["label"] == label].iterrows())
        for (_, a), (_, b) in combinations(rows, 2):
            if a["fmt_top"] != b["fmt_top"]:
                flags.append(
                    f"Format heterogeneity within {name}: source "
                    f"'{a['source']}' is {a['fmt_top']} but '{b['source']}' "
                    f"is {b['fmt_top']}")
            if _width_ratio_flagged(a["width_median"], b["width_median"]):
                flags.append(
                    f"Resolution heterogeneity within {name}: source "
                    f"'{a['source']}' median width {a['width_median']:.0f} vs "
                    f"'{b['source']}' {b['width_median']:.0f}")
            if _jpeg_q_flagged(a["jpeg_q_median"], b["jpeg_q_median"]):
                flags.append(
                    f"JPEG-quality heterogeneity within {name}: source "
                    f"'{a['source']}' median q {a['jpeg_q_median']:.0f} vs "
                    f"'{b['source']}' {b['jpeg_q_median']:.0f}")

    return flags
