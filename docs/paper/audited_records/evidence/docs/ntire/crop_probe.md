# The 200px crop window, costed

`scripts/ntire_crop_probe.py`, run 2026-10-07 on the two frozen
round-two checkpoints. The gate in that script's header was registered
before any arm had a number. `ntire_test` is not touched: it was read
once on 2026-10-04 and is sealed.

## Finding

`CanonPolicy(mode="crop", crop_side=200)` equalises the number of native
pixels every image contributes and leaves FIELD OF VIEW free. Field of
view is `200 / short side`, and train and eval do not share its
distribution:

| split | n | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|
| `train` | 407,223 | 0.391 | 0.391 | 0.391 | 0.481 | 0.521 |
| `ntire_val` | 10,000 | 0.100 | 0.231 | 0.312 | 0.431 | 0.893 |
| `ntire_val_hard` | 2,500 | 0.112 | 0.223 | 0.298 | 0.417 | 0.784 |

Train is a near point mass: every train image sits at 384-512 short side
(0% above 512, median 512), so its field of view is 0.391-0.521 for
the central 90% of rows. On `ntire_val_hard`, 86.4% of rows fall outside that band and
70.8% fall below it, down to 0.050. The policy
trains at one scale and tests across a nine-fold range of a covariate it
was written to be invariant to. Under `squish` the field of view is
1.000 for every row in every split.

## Arms on ntire_val_hard

### Branch A -- dinov2regl @512, depth 24, final

| arm | views | robust | clean | 80-336 | 352-512 | 528-736 | 752-4016 |
|---|---|---|---|---|---|---|---|
| `center` | 1.00 | 0.8561 | 0.9912 | 0.8777 | 0.8738 | 0.8088 | 0.8618 |
| `grid9` | 8.86 | 0.8495 | 0.9978 | 0.8819 | 0.8733 | 0.8094 | 0.8434 |
| `fov3` | 8.86 | 0.8703 | 0.9977 | 0.8819 | 0.8733 | 0.8199 | 0.9123 |
| `squish` | 1.00 | 0.9482 | 0.9906 | 0.9386 | 0.9632 | 0.9372 | 0.9688 |
| `band_squish` | 1.00 | 0.9129 | 0.9848 | 0.9221 | 0.9293 | 0.9052 | 0.9088 |

| arm | robust delta | clean delta | PRIMARY | small-bucket gain | large-bucket gain | MECHANISM |
|---|---|---|---|---|---|---|
| `grid9` | -0.0066 | +0.0066 | **fail** | +0.0041 | -0.0089 | **fail** |
| `fov3` | +0.0142 | +0.0065 | **PASS** | +0.0041 | +0.0309 | **PASS** |
| `squish` | +0.0920 | -0.0006 | **PASS** | +0.0609 | +0.1178 | **PASS** |
| `band_squish` | +0.0567 | -0.0065 | **fail** | +0.0444 | +0.0718 | **PASS** |

### Branch B -- dual dinov2regl224 @224, depth 24, swa

| arm | views | robust | clean | 80-336 | 352-512 | 528-736 | 752-4016 |
|---|---|---|---|---|---|---|---|
| `center` | 1.00 | 0.8872 | 0.9801 | 0.9195 | 0.9110 | 0.8805 | 0.8303 |
| `grid9` | 8.86 | 0.9049 | 0.9931 | 0.9395 | 0.9240 | 0.8945 | 0.8549 |
| `fov3` | 8.86 | 0.9215 | 0.9942 | 0.9395 | 0.9240 | 0.8979 | 0.9361 |
| `squish` | 1.00 | 0.9599 | 0.9886 | 0.9642 | 0.9724 | 0.9602 | 0.9524 |
| `band_squish` | 1.00 | 0.9594 | 0.9878 | 0.9601 | 0.9740 | 0.9589 | 0.9504 |

| arm | robust delta | clean delta | PRIMARY | small-bucket gain | large-bucket gain | MECHANISM |
|---|---|---|---|---|---|---|
| `grid9` | +0.0177 | +0.0130 | **PASS** | +0.0200 | +0.0193 | **fail** |
| `fov3` | +0.0343 | +0.0141 | **PASS** | +0.0200 | +0.0616 | **PASS** |
| `squish` | +0.0727 | +0.0085 | **PASS** | +0.0447 | +0.1009 | **PASS** |
| `band_squish` | +0.0722 | +0.0077 | **PASS** | +0.0406 | +0.0993 | **PASS** |

`center` reproduces each branch's frozen round-two number exactly
(A 0.8561 against 0.8561,
B 0.8872 against 0.8873), which is the
probe's own sanity check.

## Fusion, fitted on ntire_val alone

Both z-scores and the weight come from `ntire_val`; `ntire_val_hard` is a
pure readout, nothing is fitted on it.

| policy | split | robust | clean | AUC | TPR@1%FPR |
|---|---|---|---|---|---|
| centre window | `ntire_val` | 0.9718 | 0.9967 | 0.9855 | 0.8678 |
| centre window | `ntire_val_hard` | 0.8856 | 0.9904 | 0.9505 | 0.5160 |
| squish | `ntire_val` | 0.9894 | 0.9964 | 0.9930 | 0.9162 |
| squish | `ntire_val_hard` | 0.9631 | 0.9930 | 0.9807 | 0.7520 |

## Was it the resolution cue?

`squish` does not band-limit, so the resampling signature
`augment/canonical.py` exists to destroy is back in the pixels. Three
controls say the gain is not resolution reading.

1. **The model could not have learned it.** Native short side scores AUC
   0.5275 against the label on `train`, where no image exceeds 512. The
   cue only appears on the eval sets (0.5749 pooled on `ntire_val`, 0.7450
   in its largest bucket) -- at resolutions the model never saw.
2. **It survives holding resolution fixed.** Within equal-count deciles of
   native short side, where the resample factor is near-constant, `squish`
   beats `center` in 10 of 10 strata; the n-weighted within-stratum means
   are 0.8883 -> 0.9637 on branch B, a +0.0754 gain against +0.0727
   pooled.
3. **It survives removing the cue outright.** `band_squish` forces a
   bandwidth ceiling of 160 on the squished square, so native resolution
   is no longer recoverable from the pixels. Branch B keeps
   +0.0722 of its +0.0727,
   i.e. all of it. Branch A keeps
   +0.0567 of +0.0920;
   the shortfall is the real high-frequency detail a 3.2x band cut
   destroys at nominal_side 512, which is the cost canonical.py already
   documents, not a lost shortcut.

## What this is not

A test-time measurement on two checkpoints trained under the crop policy.
It establishes that the window, not the backbone or the recipe, is what
caps robust AUC here. It does NOT report a trained whole-image model, and
the new policy has never been read out on `ntire_test`.

`grid9` is the arm that separates the two candidate explanations. Nine
windows give nine times the pixels at the SAME field of view: branch B
gains only +0.0177 and branch A LOSES -0.0066.
Pixel coverage is not the lever; field of view is. `fov3` interpolates
between them exactly as that reading predicts, and its gain is largest in
the bucket where the centre window saw least
(+0.0506 on A's 752-4016px rows).

