### dinov2 s3: squish-trained vs crop-trained under test-time squish (exploratory, not registered)
n=2,500 (robust 1,250, clean 1,250), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `testtime` | 0.8671 [0.8531, 0.8805] | 0.8299 [0.8076, 0.8516] | 0.9020 [0.8851, 0.9180] |
| `trained` | 0.9073 [0.8958, 0.9188] | 0.8776 [0.8578, 0.8957] | 0.9342 [0.9207, 0.9466] |

Paired difference against `testtime`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `trained` | +0.0401 [+0.0307, +0.0495] * | +0.0478 [+0.0329, +0.0638] * | +0.0322 [+0.0214, +0.0439] * |

`*` = the 95% interval on the paired difference excludes zero.
