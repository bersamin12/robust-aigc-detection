### B2swa, ntire_val_hard, against center
n=2,500 (robust 1,250, clean 1,250), 2000 paired resamples stratified by label x is_distorted, 95% percentile CI

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `center` | 0.9419 [0.9332, 0.9498] | 0.8872 [0.8694, 0.9036] | 0.9801 [0.9737, 0.9855] |
| `squish` | 0.9765 [0.9717, 0.9812] | 0.9599 [0.9497, 0.9692] | 0.9886 [0.9845, 0.9924] |
| `short_center` | 0.9757 [0.9707, 0.9805] | 0.9575 [0.9469, 0.9674] | 0.9891 [0.9850, 0.9928] |
| `letterbox` | 0.9720 [0.9665, 0.9772] | 0.9558 [0.9441, 0.9659] | 0.9851 [0.9798, 0.9899] |

Paired difference against `center`:

| model | auc | robust_auc | clean_auc |
|---|---|---|---|
| `squish` | +0.0346 [+0.0269, +0.0429] * | +0.0727 [+0.0567, +0.0901] * | +0.0085 [+0.0028, +0.0151] * |
| `short_center` | +0.0337 [+0.0260, +0.0419] * | +0.0703 [+0.0544, +0.0865] * | +0.0090 [+0.0031, +0.0158] * |
| `letterbox` | +0.0301 [+0.0220, +0.0387] * | +0.0685 [+0.0522, +0.0862] * | +0.0050 [-0.0013, +0.0121] |

`*` = the 95% interval on the paired difference excludes zero.
