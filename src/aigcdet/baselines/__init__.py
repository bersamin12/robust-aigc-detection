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
