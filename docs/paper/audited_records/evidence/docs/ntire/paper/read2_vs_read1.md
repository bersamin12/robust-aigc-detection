### ntire_test: read 2 (squish) vs read 1 (crop), saved fused scores only, no model run
n=2,500 (robust 1,250, clean 1,250), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `read1` | 0.9108 [0.9007, 0.9210] | 0.7931 [0.7693, 0.8168] | 0.9856 [0.9803, 0.9901] |
| `read2` | 0.9211 [0.9108, 0.9303] | 0.8547 [0.8333, 0.8746] | 0.9715 [0.9626, 0.9794] |

Paired difference against `read1`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `read2` | +0.0103 [-0.0000, +0.0204] | +0.0615 [+0.0404, +0.0833] * | -0.0141 [-0.0237, -0.0056] * |

`*` = the 95% interval on the paired difference excludes zero.
