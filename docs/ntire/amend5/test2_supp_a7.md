Amendment-7 supplement (R1) to Test 2. It changes no verdict: Test 2's registered PASS and its write-up `test2_regime-grid.md` stand as they are.

# Test 2, R1: intervals on the level-averaged D

## What was added

Test 2 judged its curve on point estimates. It gave a paired interval for
each single condition, but not for the level average
L(level) = mean over JPEG, blur, noise of D(op, level), not for the steps of
the curve, and not for the sign change. Amendment 7 (R1) registers these.
`scripts/amend5_test2_supp_a7.py` computes them from Test 2's per-image
parquets, with one bootstrap draw set per (model, bucket) shared by all ten
conditions: `paired_diff`'s index scheme, seed 20260827, 2,000 draws, label
strata. The script asserts that every single-condition D and interval equals
Test 2's json, and that pooled D(jpeg_heavy) equals a fresh `paired_diff`
call. Both held. No model was loaded; `ntire_test` is refused.

D = AUC(squish) - AUC(native). The 80-336 bucket is empty on clean NTIRE, as
in Test 2.

## Level-averaged D, with 95% intervals

| model | bucket (n) | none | light | medium | heavy |
|---|---|---|---|---|---|
| A | pooled (6,250) | -0.0011 [-0.0025, +0.0003] | +0.0011 [-0.0004, +0.0025] | +0.0075 [+0.0056, +0.0096] | +0.0285 [+0.0252, +0.0319] |
| A | 352-512 (176) | +0.0003 [+0.0000, +0.0012] | +0.0013 [-0.0003, +0.0041] | +0.0042 [+0.0004, +0.0097] | +0.0192 [+0.0068, +0.0349] |
| A | 528-736 (2,694) | **-0.0013 [-0.0027, -0.0003]** | -0.0008 [-0.0023, +0.0003] | +0.0010 [-0.0007, +0.0024] | **+0.0082 [+0.0058, +0.0107]** |
| A | 752+ (3,380) | +0.0016 [-0.0010, +0.0043] | +0.0054 [+0.0026, +0.0083] | +0.0161 [+0.0125, +0.0199] | +0.0491 [+0.0432, +0.0552] |
| B | pooled (6,250) | +0.0043 [+0.0025, +0.0063] | +0.0076 [+0.0055, +0.0099] | +0.0141 [+0.0115, +0.0170] | +0.0271 [+0.0235, +0.0307] |
| B | 352-512 (176) | +0.0011 [-0.0012, +0.0047] | +0.0016 [-0.0025, +0.0074] | +0.0047 [-0.0011, +0.0131] | +0.0104 [+0.0017, +0.0218] |
| B | 528-736 (2,694) | **-0.0011 [-0.0023, +0.0001]** | -0.0003 [-0.0017, +0.0010] | +0.0014 [-0.0004, +0.0030] | **+0.0069 [+0.0045, +0.0094]** |
| B | 752+ (3,380) | +0.0134 [+0.0097, +0.0173] | +0.0184 [+0.0143, +0.0227] | +0.0293 [+0.0244, +0.0345] | +0.0486 [+0.0422, +0.0550] |
| B-Free | pooled (6,250) | -0.1139 [-0.1228, -0.1055] | -0.1080 [-0.1166, -0.0996] | -0.0884 [-0.0971, -0.0800] | -0.0346 [-0.0437, -0.0257] |
| B-Free | 352-512 (176) | -0.0433 [-0.0734, -0.0193] | -0.0442 [-0.0725, -0.0208] | -0.0474 [-0.0717, -0.0257] | -0.0357 [-0.0551, -0.0172] |
| B-Free | 528-736 (2,694) | -0.0535 [-0.0613, -0.0457] | -0.0528 [-0.0605, -0.0451] | -0.0468 [-0.0541, -0.0399] | -0.0292 [-0.0362, -0.0220] |
| B-Free | 752+ (3,380) | -0.1462 [-0.1606, -0.1317] | -0.1324 [-0.1467, -0.1177] | -0.0945 [-0.1092, -0.0797] | -0.0019 [-0.0168, +0.0133] |

## Steps of the curve, with 95% intervals

| model | bucket | none -> light | light -> medium | medium -> heavy | none -> heavy |
|---|---|---|---|---|---|
| A | pooled | +0.0022 [+0.0017, +0.0027] | +0.0065 [+0.0056, +0.0075] | +0.0210 [+0.0188, +0.0232] | +0.0296 [+0.0266, +0.0328] |
| A | 352-512 | +0.0010 [-0.0006, +0.0039] | +0.0029 [+0.0004, +0.0064] | +0.0150 [+0.0059, +0.0260] | +0.0189 [+0.0068, +0.0345] |
| A | 528-736 | +0.0005 [+0.0002, +0.0009] | +0.0017 [+0.0012, +0.0024] | +0.0073 [+0.0057, +0.0089] | +0.0095 [+0.0074, +0.0118] |
| A | 752+ | +0.0038 [+0.0028, +0.0048] | +0.0107 [+0.0091, +0.0127] | +0.0330 [+0.0291, +0.0370] | +0.0475 [+0.0424, +0.0528] |
| B | pooled | +0.0033 [+0.0026, +0.0040] | +0.0065 [+0.0055, +0.0075] | +0.0130 [+0.0113, +0.0146] | +0.0227 [+0.0201, +0.0255] |
| B | 352-512 | +0.0005 [-0.0019, +0.0032] | +0.0031 [+0.0003, +0.0070] | +0.0057 [+0.0006, +0.0123] | +0.0093 [+0.0020, +0.0187] |
| B | 528-736 | +0.0008 [+0.0004, +0.0012] | +0.0017 [+0.0010, +0.0024] | +0.0055 [+0.0042, +0.0070] | +0.0080 [+0.0061, +0.0101] |
| B | 752+ | +0.0050 [+0.0038, +0.0062] | +0.0109 [+0.0091, +0.0129] | +0.0193 [+0.0165, +0.0224] | +0.0352 [+0.0305, +0.0401] |
| B-Free | pooled | +0.0060 [+0.0047, +0.0073] | +0.0196 [+0.0172, +0.0219] | +0.0538 [+0.0498, +0.0578] | +0.0793 [+0.0735, +0.0855] |
| B-Free | 352-512 | -0.0009 [-0.0079, +0.0065] | -0.0032 [-0.0141, +0.0093] | +0.0116 [-0.0072, +0.0333] | +0.0075 [-0.0179, +0.0391] |
| B-Free | 528-736 | +0.0007 [-0.0008, +0.0021] | +0.0059 [+0.0035, +0.0086] | +0.0176 [+0.0132, +0.0222] | +0.0243 [+0.0180, +0.0308] |
| B-Free | 752+ | +0.0138 [+0.0116, +0.0162] | +0.0379 [+0.0339, +0.0423] | +0.0926 [+0.0863, +0.0991] | +0.1443 [+0.1346, +0.1546] |

## What the numbers show

- **"D rises with level" survives with intervals on both branches.** The
  none -> heavy rise lies above zero in every populated bucket and pooled.
  No step has an interval below zero anywhere, so no decrease is resolved.
  Every step is resolved above zero in 528-736, 752+ and pooled. In 352-512
  (n = 176) the none -> light step includes zero on both branches; there the
  curve is flat at the start, not shown to rise.
- **"Sign change in one bucket" survives for Branch A only.** The crossing is
  in 528-736 on both branches, between none (lowest) and heavy (highest).
  A: D(none) -0.0013 [-0.0027, -0.0003] and L(heavy) +0.0082 [+0.0058,
  +0.0107], so the crossing is resolved. B: D(none) -0.0011 [-0.0023,
  +0.0001] includes zero, so the negative side is not resolved and the sign
  change is unresolved. Light is also negative on both branches, but its
  interval includes zero. No other bucket crosses zero on either branch.
  Pooled over size, A crosses on point estimates (D(none) -0.0011), but
  its interval [-0.0025, +0.0003] includes zero.
- B-Free: the pooled rise and the 528-736 and 752+ rises are above zero;
  the 352-512 rise includes zero (+0.0075 [-0.0179, +0.0391]). B-Free's
  level average never crosses zero.

## L5 prediction: untestable

Test 2's L5 prediction ("JPEG's D rises more steeply than noise's") is
reported as untestable, as amendment 7 registers. The three op types' levels
are not matched in damage strength, so a steeper rise for one op type cannot
be told apart from that op simply being stronger at "heavy". Pixel RMSE
(0-255, native resolution) of each condition against the clean image,
recomputed here on 200 primary images (seed 20261019) with Test 2's own
`damage` function:

| op | light | medium | heavy |
|---|---|---|---|
| JPEG (q90/70/30) | 1.6 | 3.1 | 5.7 |
| blur (sigma 0.5/1/2) | 2.5 | 6.9 | 11.1 |
| noise (sigma 0.02/0.05/0.10) | 5.0 | 12.2 | 23.6 |

(Means over images; medians and 10th/90th percentiles are in the json.)
Heavy noise moves pixels about 4x as much as heavy JPEG. For description
only: the rise none -> heavy, pooled, is JPEG +0.0246 [+0.0214, +0.0281],
blur +0.0262 [+0.0228, +0.0300], noise +0.0380 [+0.0335, +0.0428] on A, and
JPEG +0.0203 [+0.0173, +0.0233], blur +0.0154 [+0.0130, +0.0179], noise
+0.0325 [+0.0284, +0.0367] on B. Noise minus JPEG: A +0.0134 [+0.0098,
+0.0173], B +0.0122 [+0.0090, +0.0156]. This ordering follows the RMSE
ordering and says nothing about op type at matched strength.

## Limits

- RMSE is one measure of damage strength, at native scale. It is not what
  the model sees after the resize, and it does not weigh frequency content.
  It shows that the levels are unmatched; it does not supply a matching.
- The intervals are within-sample (resampling images of one source). The
  352-512 bucket has 176 images.
- "Resolved" here reads each interval on its own; no correction for the
  number of cells.

All numbers: `test2_supp_a7.json`. Script: `scripts/amend5_test2_supp_a7.py`.

Base commit: `bbb1306` (Test 2's files and this supplement staged, not committed).
