# Squish gain by distortion group (ntire_val_hard, distorted rows)

Frozen round-two crop checkpoints (A2 final, B2 SWA); robust AUC of the probe's `squish` arm minus its `center` arm within each group, 2000 paired resamples stratified by label.

| group | n | short p50 | A centre -> squish | A gain [95% CI] | B centre -> squish | B gain [95% CI] |
|---|---|---|---|---|---|---|
| photometric only | 241 | 784 | 0.8638 -> 0.9542 | +0.0903 [+0.0420, +0.1386] | 0.8853 -> 0.9679 | +0.0826 [+0.0431, +0.1257] |
| downscale, no crop | 740 | 432 | 0.8613 -> 0.9566 | +0.0953 [+0.0731, +0.1199] | 0.8962 -> 0.9608 | +0.0646 [+0.0438, +0.0869] |
| crop or aspect crop | 191 | 320 | 0.8654 -> 0.9245 | +0.0591 [+0.0197, +0.1051] | 0.8901 -> 0.9534 | +0.0633 [+0.0242, +0.1092] |
| perspective | 78 | 536 | 0.7672 -> 0.9127 | +0.1455 [+0.0522, +0.2480] | 0.8082 -> 0.9484 | +0.1402 [+0.0469, +0.2414] |

Commit `5ecff8fc3c3070ed0c9b1534666aaf39a5ff1c41`.
