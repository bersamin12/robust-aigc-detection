# G1 -- squish vs crop, proxy protocol, three seeds

Pre-registered in `docs/ntire/paper_prereg.md`. Proxy: one clipl tower, depth 24, 1 epoch over 25% of train, bank `ntire_meta`, mean pool, cosine + SWA, final weights. Each seed pair shares its training subset, head init and batch order; only `--canon-mode` differs.

| seed | crop robust | squish robust | diff [95% CI] | crop clean | squish clean | diff [95% CI] |
|---|---|---|---|---|---|---|
| s1 | 0.7491 | 0.8927 | +0.1435 [+0.1198, +0.1691] | 0.9651 | 0.9651 | +0.0000 [-0.0114, +0.0110] |
| s2 | 0.7522 | 0.9017 | +0.1495 [+0.1262, +0.1738] | 0.9623 | 0.9650 | +0.0027 [-0.0089, +0.0143] |
| s3 | 0.7434 | 0.8828 | +0.1394 [+0.1148, +0.1646] | 0.9692 | 0.9635 | -0.0057 [-0.0177, +0.0060] |

Mean robust diff +0.1442 (gate > +0.0100); seeds with CI above zero 3/3 (gate >= 2); mean clean diff -0.0010 (gate >= -0.0050).

**G1 PASSED** -- full training proceeds.

Commit `9b8b08ebea1988f64f9ae50f5421ab3c602a8e7d`.

## ntire_val (secondary)

### seed 1, ntire_val
n=10,000 (robust 5,000, clean 5,000), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `crop` | 0.9499 [0.9461, 0.9537] | 0.9040 [0.8954, 0.9121] | 0.9862 [0.9836, 0.9887] |
| `squish` | 0.9707 [0.9677, 0.9734] | 0.9562 [0.9508, 0.9610] | 0.9823 [0.9790, 0.9855] |

Paired difference against `crop`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `squish` | +0.0207 [+0.0167, +0.0249] * | +0.0522 [+0.0447, +0.0601] * | -0.0039 [-0.0077, -0.0000] * |

`*` = the 95% interval on the paired difference excludes zero.

### seed 2, ntire_val
n=10,000 (robust 5,000, clean 5,000), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `crop` | 0.9519 [0.9483, 0.9555] | 0.9074 [0.8990, 0.9153] | 0.9858 [0.9833, 0.9882] |
| `squish` | 0.9717 [0.9688, 0.9745] | 0.9592 [0.9540, 0.9638] | 0.9822 [0.9791, 0.9852] |

Paired difference against `crop`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `squish` | +0.0197 [+0.0156, +0.0239] * | +0.0518 [+0.0445, +0.0593] * | -0.0037 [-0.0073, -0.0002] * |

`*` = the 95% interval on the paired difference excludes zero.

### seed 3, ntire_val
n=10,000 (robust 5,000, clean 5,000), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `crop` | 0.9540 [0.9503, 0.9576] | 0.9049 [0.8966, 0.9130] | 0.9882 [0.9859, 0.9904] |
| `squish` | 0.9739 [0.9713, 0.9765] | 0.9606 [0.9557, 0.9651] | 0.9842 [0.9813, 0.9871] |

Paired difference against `crop`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `squish` | +0.0199 [+0.0162, +0.0237] * | +0.0557 [+0.0481, +0.0635] * | -0.0040 [-0.0074, -0.0007] * |

`*` = the 95% interval on the paired difference excludes zero.

