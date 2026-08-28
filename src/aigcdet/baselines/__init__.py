"""Published baselines the headline model is compared against (spec §6.3).

Only NPR needs code. The other two are already paid for:

- **UnivFD** is rung A0 evaluated on the `clipl` bank. It is a linear probe on
  frozen CLIP features, which is exactly what A0 is, so it needs no module, no
  second CLIP path and no extra weights. Report it in the results table under
  its published name with a footnote saying which rung produced the number.
- **AEROBLADE** is the `r` branch alone, unthresholded, read out of the
  already-cached `features.recon.recon_features` vector -- see `aeroblade.py`.
  It never touches a VAE here; the round-trip happened once, upstream. It
  defaults to the paper's LPIPS distance, with L1 selectable, so the row we
  report is the published method rather than a weaker variant of it.

Results-table naming, because two of the three rows are not what their
published names would imply on their own:

| row | wording to use |
|---|---|
| UnivFD | "UnivFD (rung A0 on the `clipl` bank)" |
| AEROBLADE | "AEROBLADE (LPIPS round-trip distance, training-free)" |
| NPR | "NPR-style neighbouring-pixel summary + linear probe" |

The NPR footnote is not optional: published NPR trains a ResNet-50 over the
full residual map, and this is four scalars into a logistic regression.

Nothing in this package loads model weights or starts a GPU process.
"""

#: Row label every results table must use for each baseline, keyed by the short
#: name a row is likely to carry. Data rather than prose alone, so a script that
#: emits a §6.3 comparison cannot drift from the wording above (ruling R38/I3).
BASELINE_ROW_LABELS: dict[str, str] = {
    "univfd": "UnivFD (rung A0 on the `clipl` bank)",
    "aeroblade": "AEROBLADE (LPIPS round-trip distance, training-free)",
    "npr": "NPR-style neighbouring-pixel summary + linear probe",
}

#: Why each short name understates its published method. Emitted alongside the
#: label so a reader of the table is told, not left to look it up.
BASELINE_ROW_FOOTNOTES: dict[str, str] = {
    "univfd": ("this row is rung A0 of our own ladder, evaluated on the `clipl` "
               "bank -- a linear probe on frozen CLIP features, which is what "
               "UnivFD is."),
    "aeroblade": ("the published LPIPS round-trip distance, read from the cached "
                  "recon vector and unthresholded. An L1 variant is selectable "
                  "and must be labelled as such if it is the number reported."),
    "npr": ("published NPR trains a ResNet-50 over the full residual map; this "
            "row is four neighbouring-pixel scalars into a logistic regression, "
            "so an unfootnoted `NPR` row would understate the published method."),
}

#: Rungs of our own ladder that ARE a published baseline when run on the right
#: bank, and must be relabelled if the number is quoted as that baseline.
RUNG_IS_A_BASELINE: dict[str, str] = {"a0": "univfd"}
