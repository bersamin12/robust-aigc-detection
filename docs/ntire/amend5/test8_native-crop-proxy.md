# Test 8: fair native crop baseline (`native-crop-proxy`)

## 1. Summary & results

**Verdict: PASS.** Registered condition (amendment 5, Test 8): "**Pass:**
mean over seeds of robust AUC (squish proxy - native-224 proxy) on
`ntire_val_hard` > 0, with a CI above zero in at least 2 of 3 seeds. The 200
px crop proxies are reported alongside." Observed: mean +0.0765, CI above
zero in 3 of 3 seeds.

| seed | squish - native-224, robust AUC on `ntire_val_hard` (95% CI) |
|---|---|
| 1 | +0.0773 [+0.0572, +0.0991] |
| 2 | +0.0798 [+0.0596, +0.1015] |
| 3 | +0.0724 [+0.0529, +0.0941] |
| mean | **+0.0765** |

All `ntire_val_hard` numbers are fit-split numbers: that split was used to
choose squish (pre-registration, "Rules that apply to every test").

Main reading:
- Removing the 200 -> 224 enlargement does not rescue the crop policy. The
  native-224 proxy scores almost the same as the 200 px crop proxy. On
  `ntire_val_hard` crop200 - native-224 is -0.0055, -0.0131 and -0.0059 in
  robust AUC; only seed 2's CI excludes zero.
- Squish's advantage over a crop baseline is therefore not an artefact of
  the crop policy's 1.12x upscale. A native-resolution window of the input
  size trails squish by as much as the 200 px window does.

## 2. Current state

- Registered in `docs/ntire/paper_prereg.md`, "Amendment 5", Test 8, at
  `991dc37` (amendment 6 does not touch Test 8).
- Comparators are the existing G1 proxies `abl_squish_s{1,2,3}` and
  `abl_crop_s{1,2,3}`, with their pipeline scores from
  `outputs/scores/paper/`. They are the same comparator files Test 7 uses.
- Before this test, a reviewer could argue that the 200 px crop baseline is
  handicapped by its upscale to 224. Test 8 is the registered answer to that.

## 3. Opportunities

- The paper can state that the crop baseline is fair: a 224 px native window
  at a 224 input, with no resampling for images of short side >= 224,
  performs like the 200 px crop.
- Together with Test 7: multi-scale crops close most of the gap and native
  crops close none of it. This is consistent with the field of view seen in
  training mattering, not the crop's upscale. It does not separate field of
  view from training-time downscaling, because rrc views are also downscaled
  to 224.

## 4. Testing methodology

### 4.1 Arms

| arm | training canonicalisation | test-time presentation | scored by |
|---|---|---|---|
| native-224 (`abl_nat224_s<i>`) | `--canon-mode crop --crop-side 224 --nominal-side 224 --crop-clamp` | one 224 px native centre window | `scripts/score_plan_splits.py` |
| squish (`abl_squish_s<i>`) | whole frame at 224 | squish | `scripts/score_plan_splits.py` |
| crop200 (`abl_crop_s<i>`) | 200 px native window, resized to 224 | 200 px centre window | `scripts/score_plan_splits.py` |

Images with short side >= 224 are windowed with no resampling. Smaller images
are taken whole and resized (crop_clamp), as registered. Everything else is
the G1 proxy protocol, copied flag for flag from stage P of
`scripts/paper_pipeline.sh` (same flags as Test 7, §4.1 there, except the
canonicalisation). Seeds 20260827, 20260828 and 20260829 pair with
comparator seed indices 1, 2 and 3. All three arms use the same scorer with
the same flags (`--crop-clamp --degrade-seed 20261008 --batch 32`).
`score_plan_splits.py` rebuilds each checkpoint's own policy, so the
native-224 arm is tested with its own 224 px centre window.

### 4.2 Timing

100 timed steps with 8 workers: mean GPU utilisation 92.1%, 1.64 s/step,
peak RAM 7 GB. That is above 70%, so the worker count stayed at 8. Seeds took
50-52 min each.

### 4.3 Statistics

`scripts/paired_bootstrap.py`: 2,000 resamples stratified by label x
is_distorted, seed 20260827, the same indices for the three arms of a seed.
The baseline is native-224, so every difference reads (arm - native-224).

### 4.4 Choices the registration does not fix

1. **No fail branch is registered.** The analysis reports a mean <= 0 as
   "FAIL (not a registered branch)" and a positive mean with fewer than 2
   seeds' CI above zero as IN BETWEEN. Neither occurred.
2. **`ntire_val` is reported as well.** It is not judged (§5.2).
3. **Seed pairing.** Native-224 seed i is compared with comparator seed
   index i.

### 4.5 Code provenance

The new code was staged, not committed, when the queue ran it, so the
`commit` field in the json reads `bbb1306`. The queue logged the md5 of the
staged diff at start (`logs/queue.log`, 05:07:36):
`8657da3d3fd2d70b24bf3e3eb75c9602`. At write-up time the staged diff of the
two run files (`scripts/amend5_test8.sh` and
`scripts/amend5_test8_analysis.py`) still has that md5. Test 8 needs no
source change: `crop` mode with `crop_side` 224 already existed.

## 5. Findings

### 5.1 `ntire_val_hard` (registered split, fit-split numbers)

| seed | arm | AUC | robust AUC (95% CI) | clean AUC |
|---|---|---|---|---|
| 1 | native-224 | 0.8753 | 0.8093 [0.7852, 0.8315] | 0.9288 |
| 1 | squish | 0.9115 | 0.8866 [0.8678, 0.9046] | 0.9346 |
| 1 | crop200 | 0.8728 | 0.8037 [0.7792, 0.8268] | 0.9264 |
| 2 | native-224 | 0.8730 | 0.8083 [0.7835, 0.8315] | 0.9254 |
| 2 | squish | 0.9139 | 0.8881 [0.8702, 0.9052] | 0.9384 |
| 2 | crop200 | 0.8693 | 0.7952 [0.7704, 0.8184] | 0.9266 |
| 3 | native-224 | 0.8710 | 0.8052 [0.7810, 0.8287] | 0.9245 |
| 3 | squish | 0.9073 | 0.8776 [0.8578, 0.8957] | 0.9342 |
| 3 | crop200 | 0.8681 | 0.7993 [0.7746, 0.8224] | 0.9234 |

Paired differences (arm - native-224), 95% CI:

| seed | squish: robust | squish: clean | crop200: robust | crop200: clean |
|---|---|---|---|---|
| 1 | +0.0773 [+0.0572, +0.0991] | +0.0058 [-0.0108, +0.0228] | -0.0055 [-0.0155, +0.0044] | -0.0024 [-0.0094, +0.0047] |
| 2 | +0.0798 [+0.0596, +0.1015] | +0.0131 [-0.0034, +0.0300] | -0.0131 [-0.0233, -0.0033] | +0.0013 [-0.0071, +0.0091] |
| 3 | +0.0724 [+0.0529, +0.0941] | +0.0098 [-0.0071, +0.0266] | -0.0059 [-0.0159, +0.0038] | -0.0011 [-0.0094, +0.0062] |

![Figure](test8_native-crop-proxy_fig.pdf)

*Figure 1. Robust AUC difference on `ntire_val_hard`, squish - native-224
(blue) and crop200 - native-224 (red), per seed, 95% CI.*

### 5.2 `ntire_val` (reported, not judged)

| seed | squish - native-224: robust | squish - native-224: clean | crop200 - native-224: robust |
|---|---|---|---|
| 1 | +0.0297 [+0.0237, +0.0360] | +0.0089 [+0.0049, +0.0129] | -0.0071 [-0.0105, -0.0036] |
| 2 | +0.0318 [+0.0255, +0.0382] | +0.0102 [+0.0063, +0.0140] | -0.0028 [-0.0062, +0.0005] |
| 3 | +0.0335 [+0.0273, +0.0398] | +0.0105 [+0.0065, +0.0144] | -0.0033 [-0.0067, +0.0001] |

On `ntire_val` the squish advantage is smaller (+0.030 to +0.034 robust)
but every CI is above zero, and here squish is also ahead on clean images
(+0.009 to +0.011). Native-224 is ahead of crop200 by 0.003-0.007, with one
CI of three excluding zero.

## 6. Results

| question | answer |
|---|---|
| Registered verdict | **PASS**: mean +0.0765, CI above zero 3/3 |
| Does a native 224 px window (no upscale) close the gap to squish? | No: it trails squish by 0.072-0.080 robust AUC on `ntire_val_hard` |
| Native-224 vs crop200 | Within 0.013. Native-224 is ahead in every seed and split, with the CI excluding zero in 1 of 3 seeds on each split |

## 7. Future work

- Test 8 was not registered on the external sets, so they were not scored.
  Synthbuster is where crop beats squish (Tests 7 and 11). Scoring native-224
  there would show whether the native window keeps the crop policy's clean
  advantage on large images. It is about 15 GPU-minutes.

## 8. What didn't work

- Nothing failed. All three seeds trained and scored (exit 0). The timing
  check needed no worker increase.
- The queue's timing step logged "grep: .../test8_timing_w8.log: No such file
  or directory" twice. The script polled the log before the training process
  had created it. The timing row was recorded correctly
  (`outputs/test8/timing.tsv`), so this did not affect the run.

---

Numbers from `bbb1306` plus the staged diff with md5
`8657da3d3fd2d70b24bf3e3eb75c9602` (§4.5). Source:
`docs/ntire/amend5/test8_native-crop-proxy.json`.
