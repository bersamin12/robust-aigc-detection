"""Profile the classes before normalising them (spec §4.2 defence 1).

The output table goes in docs/data_audit.md and in the README; it is the
figure that motivates the whole normalisation step.

`jpeg_q` uses `aigcdet.features.proxies.estimate_jpeg_quality` with the file
path passed in, so a real JPEG's quantisation table is read exactly. Only
non-JPEG inputs fall back to the pixel-based estimate; see that function's
docstring for the fallback's known miscalibration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from aigcdet.features.proxies import estimate_jpeg_quality


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


def audit_flags(df: pd.DataFrame) -> list[str]:
    """Warn where authentic and generated images differ in ways a detector
    could exploit without looking at content at all."""
    flags: list[str] = []
    real, fake = df[df["label"] == 0], df[df["label"] == 1]
    if real.empty or fake.empty:
        return flags
    if set(real["fmt_top"]) != set(fake["fmt_top"]):
        flags.append(
            f"Format confound: authentic {sorted(set(real['fmt_top']))} vs "
            f"generated {sorted(set(fake['fmt_top']))}")
    rw, fw = real["width_median"].median(), fake["width_median"].median()
    if max(rw, fw) / max(1.0, min(rw, fw)) > 1.5:
        flags.append(f"Resolution confound: median width {rw:.0f} vs {fw:.0f}")
    rq, fq = real["jpeg_q_median"].median(), fake["jpeg_q_median"].median()
    if abs(rq - fq) > 10:
        flags.append(f"JPEG-quality confound: median q {rq:.0f} vs {fq:.0f}")
    return flags
