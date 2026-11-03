### A2, ntire_val_hard, against center
n=2,500 (robust 1,250, clean 1,250), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `center` | 0.9408 [0.9323, 0.9488] | 0.8561 [0.8351, 0.8754] | 0.9912 [0.9871, 0.9944] |
| `squish` | 0.9730 [0.9680, 0.9777] | 0.9482 [0.9366, 0.9591] | 0.9906 [0.9870, 0.9937] |
| `short_center` | 0.9700 [0.9644, 0.9753] | 0.9415 [0.9294, 0.9534] | 0.9903 [0.9867, 0.9935] |
| `letterbox` | 0.9693 [0.9636, 0.9742] | 0.9482 [0.9368, 0.9590] | 0.9866 [0.9816, 0.9909] |

Paired difference against `center`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `squish` | +0.0322 [+0.0244, +0.0403] * | +0.0920 [+0.0727, +0.1121] * | -0.0006 [-0.0049, +0.0039] |
| `short_center` | +0.0293 [+0.0216, +0.0373] * | +0.0853 [+0.0671, +0.1038] * | -0.0009 [-0.0054, +0.0039] |
| `letterbox` | +0.0285 [+0.0203, +0.0369] * | +0.0920 [+0.0725, +0.1128] * | -0.0046 [-0.0097, +0.0005] |

`*` = the 95% interval on the paired difference excludes zero.
