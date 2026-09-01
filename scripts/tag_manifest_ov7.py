#!/usr/bin/env python
"""Add decoder-lineage and model columns to an AI-OV7 manifest.

WHY THIS EXISTS
---------------
`manifest_ov7.parquet` carries `generator` (the family, e.g. `sdxl_t2i`) and
`label`, but NOT `lineage` -- and lineage is the axis the whole corpus is built
around: `heldout_groups()` holds out a whole decoder lineage, not a family, so
"train on everything except the held-out decoder" is not expressible against
the manifest alone. It also has one identical `licence` paragraph on every row,
which asserts that "per-row `licence_tag` is recorded so an Apache-only subset
can be cut without regenerating" -- true of `pairs.parquet`, not of the
manifest. This closes both gaps.

`MANIFEST_COLUMNS` is deliberately NOT widened: that schema is shared with the
union stream, and every feature bank on disk fingerprints its manifest. This
writes a separate tagged file instead, so nothing already built is invalidated.

NO JOIN IS NEEDED, WHICH IS WHY THIS IS SAFE. Every column added here is a pure
function of `generator` via `generate.registry`, so there is no dependence on
the ImageID -> sequential-number renumbering that `build_dataset.py` performs
and does not record. Authentic rows (`generator == ""`) get the real's own
licence and empty model fields rather than being dropped or guessed at.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from aigcdet.generate import registry

REAL_LICENCE = "CC BY 2.0 - https://creativecommons.org/licenses/by/2.0/"

#: Added columns. `lineage` is the one that matters; the rest are what make an
#: licence- or architecture-limited subset cuttable without regenerating.
ADDED = ["lineage", "model", "hf_id", "licence_tag", "arch", "method",
         "is_heldout_lineage"]


def family_tags() -> dict[str, dict[str, str]]:
    """family -> its registry tags, over every suite this corpus can contain."""
    out: dict[str, dict[str, str]] = {}
    for suite_name in registry.SUITES:
        suite = registry.SUITES[suite_name]
        for fam, spec in suite.items():
            model = registry.MODELS[spec.model]
            out[fam] = {
                "lineage": model.lineage,
                "model": spec.model,
                "hf_id": model.hf_id,
                "licence_tag": model.licence_tag,
                "arch": model.arch,
                "method": spec.method,
                "is_heldout_lineage": model.lineage == registry.HELDOUT_LINEAGE,
            }
    return out


def tag(df: pd.DataFrame) -> pd.DataFrame:
    tags = family_tags()
    unknown = sorted(set(df["generator"].unique()) - set(tags) - {""})
    if unknown:
        raise SystemExit(
            f"generator(s) {unknown} are not in registry.SUITES. This manifest "
            f"was not produced by generate_ov7.py, or the registry lost a "
            f"family. Refusing to guess a lineage -- the held-out rung reads it.")

    out = df.copy()
    for col in ADDED:
        default = False if col == "is_heldout_lineage" else ""
        out[col] = [tags[g][col] if g else default for g in out["generator"]]
    # A real is not "of" a lineage, but it does carry a licence obligation.
    out.loc[out["generator"] == "", "licence_tag"] = "CC BY 2.0"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True,
                    help="written separately; the input is never modified, so "
                         "feature banks fingerprinting it stay valid")
    a = ap.parse_args(argv)

    df = pd.read_parquet(a.manifest)
    if "generator" not in df.columns:
        raise SystemExit(f"{a.manifest} has no `generator` column")
    out = tag(df)
    out.to_parquet(a.out, index=False)

    print(f"{len(out)} rows -> {a.out}")
    print(f"added: {', '.join(ADDED)}\n")
    print(out.groupby(["is_heldout_lineage", "lineage", "generator"])
             .size().to_string())
    print("\nby label and lineage:")
    print(out.pivot_table(index="lineage", columns="label", values="path",
                          aggfunc="count", fill_value=0).to_string())
    print("\nlicence_tag:")
    print(out["licence_tag"].value_counts().to_string())
    held = int(out["is_heldout_lineage"].sum())
    print(f"\nheld-out lineage {registry.HELDOUT_LINEAGE!r}: {held} rows; "
          f"trainable fakes: {int(((out.label == 1) & ~out.is_heldout_lineage).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
