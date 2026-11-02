# Clean rows above 1100px on the fit splits

Crop-trained (round two) against squish-trained (picks.json), each as trained. Paired 95% CI, 2000 resamples stratified by label.

| split | branch | n (real/fake) | crop -> squish | diff [95% CI] |
|---|---|---|---|---|
| ntire_val | A | 871 (688/183) | 0.9890 -> 0.9999 | +0.0110 [+0.0060, +0.0169] |
| ntire_val | B | 871 (688/183) | 0.9886 -> 0.9989 | +0.0104 [+0.0045, +0.0177] |
| ntire_val_hard | A | 295 (168/127) | 0.9814 -> 0.9782 | -0.0032 [-0.0224, +0.0162] |
| ntire_val_hard | B | 295 (168/127) | 0.9820 -> 0.9824 | +0.0004 [-0.0161, +0.0171] |

Commit `5ecff8fc3c3070ed0c9b1534666aaf39a5ff1c41`.
