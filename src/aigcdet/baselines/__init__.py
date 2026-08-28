"""Published baselines the headline model is compared against (spec §6.3).

Only NPR needs code. The other two are already paid for:

- **UnivFD** is rung A0 evaluated on the `clipl` bank. It is a linear probe on
  frozen CLIP features, which is exactly what A0 is, so it needs no module, no
  second CLIP path and no extra weights. Report it in the results table under
  its published name with a footnote saying which rung produced the number.
- **AEROBLADE** is the `r` branch alone, unthresholded, read out of the
  already-cached `features.recon.recon_features` vector -- see `aeroblade.py`.
  It never touches a VAE here; the round-trip happened once, upstream.

Nothing in this package loads model weights or starts a GPU process.
"""
