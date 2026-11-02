# Test 6: two-view routed fusion, validation part

Amendment 5, Test 6. Branch `dion/test6-routed-fusion`, from `bbb1306`.
This file covers everything before read 3. **Read 3 has not run.**

## 1. Summary & results

**Registered verdict: FAIL. Condition 1 fails, so Test 6 cannot pass
whatever read 3 shows.** The registration requires both conditions:

> 1. Robust AUC on `ntire_val_hard` is within the paired CI of squish alone
>    (the CI of routed - squish includes zero or is above it).

Routed - squish, robust AUC on `ntire_val_hard`: **-0.0054 [-0.0099,
-0.0009]**. The interval lies below zero.

- Condition 2 is judged on `ntire_test` (read 3) only. Its "also reported"
  analogues: clean rows > 1100 px, routed - crop fusion = **+0.0078** on
  `ntire_val` (871 rows) and **+0.0112** on `ntire_val_hard` (295 rows).
  The routed fusion is *above* the crop fusion on both. The fit script's
  `within_0.010_of_crop` flag reads the condition as |diff| <= 0.010 and so
  marks `ntire_val_hard` false; read as "the cost recovers" (routed not more
  than 0.010 below crop), both pass. §5.3 sets out the two readings for Dion
  to choose before read 3.
- The routing is effectively off on Branch B: its fit chose b = 0, so B's
  weight is a constant w = sigmoid(0.25) = 0.562, whatever the damage.
- The squish logits re-scored for this test drift from the frozen probe
  parquets on 3 of 4 branch x split pairs (max |logit| 0.35). AUC moves by at
  most 0.00017. No reported number changes at 4 dp except the fourth decimal
  of A2 `ntire_val_hard` robust AUC (0.9482 -> 0.9480), and condition 1 does
  not change (§5.4).

## 2. Current state

The paper's squish view wins under damage and costs AUC on clean images
larger than 1100 px (Sec 8: 0.974 crop vs 0.880 squish, Branch A, on
`ntire_test`). Test 6 asks whether a per-image mix of the whole-image view
(`squish`) and the window view (`grid9`), weighted by the branch's own
predicted damage, keeps the robust gain and removes the clean cost.

## 3. Opportunities

- The routed fusion is better than squish alone on clean rows of
  `ntire_val_hard` (+0.0049 [+0.0031, +0.0069]) and on all rows (+0.0023
  [+0.0002, +0.0045]). It loses only on robust AUC. A paper sentence such as
  "routing trades 0.005 robust AUC for 0.005 clean AUC" is supported by
  these numbers; the registered condition is not met.
- On Branch B, damage severity carried no signal for the mixing weight
  (b = 0). A constant 56/44 squish/grid9 mix was the best B could do.

## 4. Testing methodology

- **Models:** A2 (final weights) and B2swa (SWA weights), frozen.
- **Views:** `squish` logit and the degradation head's per-family severity
  on the squish view, scored by `scripts/amend5_test6_score.py` (GPU, bf16,
  batch 16), 05:10-05:16 on 2026-10-20. `grid9` logits are the per-window
  logits Test 9 saved, averaged as in `ntire_crop_probe.py`.
- **Rule (registered):** s = w * z_squish + (1 - w) * z_grid9, w =
  sigmoid(a + b * d), per branch; z-scores fitted on `ntire_val`; branches
  fused as in the paper (`eval.late_fusion`).
- **Fit:** grid search on `ntire_val` only, a in [-4, 4], b in [0, 8], step
  0.25, objective mean(robust AUC, clean AUC). Written to
  `test6_frozen.json` and staged.
- **Choices the registration leaves open** (recorded in
  `test6_frozen.json`):
  - d = the maximum of the six per-family severities (jpeg, blur, resize,
    noise, jitter, crop). The registration says "overall damage severity";
    the head has no single overall output.
  - (a, b) fitted per branch.
  - Grid ties: smallest b, then a closest to 0. Each branch had exactly one
    optimum (no ties).
  - Branch fusion weight: swept 0 to 1 in 0.05 on `ntire_val`; best 0.5
    (objective 0.99443).
- **Statistics:** `paired_bootstrap`, 2,000 resamples, label x is_distorted
  strata, seed 20260827.
- `ntire_test` is refused by name in both scripts.

## 5. Findings

### 5.1 Fitted parameters

| branch | a | b | ntire_val objective | grid ties | mean w on ntire_val | mean w on ntire_val_hard |
|---|---|---|---|---|---|---|
| A (A2) | -0.75 | 1.75 | 0.99306 | 1 | 0.599 | 0.611 |
| B (B2swa) | +0.25 | 0.00 | 0.99315 | 1 | 0.562 | 0.562 |

### 5.2 Condition 1, `ntire_val_hard`

| arm | AUC | robust AUC | clean AUC |
|---|---|---|---|
| routed fusion | 0.9830 | 0.9578 | 0.9979 |
| squish alone (fused) | 0.9807 | 0.9632 | 0.9930 |
| crop fusion (read 1 recipe) | 0.9505 | 0.8868 | 0.9900 |

Routed - squish: AUC +0.0023 [+0.0002, +0.0045]; robust **-0.0054
[-0.0099, -0.0009]**; clean +0.0049 [+0.0031, +0.0069].

`ntire_val_hard` was the split used to choose squish in the paper, so these
are fit-split numbers in that sense; the registration names this split for
condition 1 nonetheless.

### 5.3 Clean > 1100 px (also reported, not judged)

| split | n (fake) | routed | crop fusion | squish alone | routed - crop |
|---|---|---|---|---|---|
| ntire_val | 871 (183) | 0.9992 | 0.9914 | 0.9986 | +0.0078 |
| ntire_val_hard | 295 (127) | 0.9979 | 0.9866 | 0.9849 | +0.0112 |

Two readings of "within 0.010 of read 1's crop fusion":
- two-sided, |routed - crop| <= 0.010: `ntire_val` yes, `ntire_val_hard` no
  (the fit script's flag);
- one-sided, routed >= crop - 0.010 ("clean cost recovers"): yes on both.

The condition's title is "Clean cost recovers", and Sec 8's cost is crop
above squish, which supports the one-sided reading. This choice decides how
read 3 is judged and must be fixed before read 3 runs. **Dion's call.**
Note that on these rows squish alone is already within 0.010 of crop
(-0.0017 on `ntire_val_hard`, +0.0072 on `ntire_val`), so on the validation
splits there is little cost to recover.

### 5.4 Squish reproduction drift

`amend5_test6_score.py` re-runs the `squish` presentation to get the
severity output, and compares its logits with the frozen probe parquets
(`outputs/scores/probe_<tag>_squish_<split>.parquet`, written by Justin on
2026-10-07).

| pair | max abs logit diff | p99 | median | rows > 0.01 | AUC change (all / robust / clean) |
|---|---|---|---|---|---|
| A2 ntire_val | 0.205 | 0.095 | 0.0053 | 36% | +0.00001 / +0.00003 / -0.00001 |
| A2 ntire_val_hard | 0.234 | 0.135 | 0.0129 | 57% | -0.00008 / -0.00017 / -0.00003 |
| B2swa ntire_val | 4.3e-6 | 2.9e-6 | 4.8e-7 | 0% | 0 / 0 / 0 |
| B2swa ntire_val_hard | 0.352 | 0.182 | 0.0108 | 52% | +0.00009 / +0.00013 / +0.00004 |

Logit standard deviation is 5.5-6.1, so the largest drift is about 0.06 SD.

**Cause: not determined.** Checked:
- Precision is the same: both `ntire_crop_probe.score_arm` and the Test 6
  scorer call `_forward_tower(..., torch.bfloat16, ...)`. The drift is not a
  bf16-vs-fp32 difference.
- Batch size differs: the Test 6 scorer uses `--batch 16`, the probe's
  default is 8. The batch Justin used on 2026-10-07 is not recorded. bf16
  kernels can give different results for different batch shapes.
- Batch size alone does not explain the pattern: B2swa is bit-exact on
  `ntire_val` but drifts on `ntire_val_hard`, with the same script and batch
  in both runs. The two frozen files may have come from different runs or
  settings.

A GPU rerun of the probe at `--batch 8` would settle it; not done, because
Test 7 holds the GPU.

**Effect:** the largest AUC change is 0.00017. Condition 1's interval
(-0.0099, -0.0009) is more than 5x that from zero, so the verdict does not
depend on the drift. The fit used the reproduced logits throughout, so the
routed and squish-alone arms are internally consistent.

### 5.5 Not run (deviation)

The registration also evaluates on "the three external sets as distributed,
`_jpeg95` and `_deg`", and reports Synthbuster's clean > 1100 comparison.
These need `grid9` and squish-with-severity scored on about 84k images, more
than 10 GPU-hours, and are not queued. They do not affect the verdict:
condition 1 is judged on `ntire_val_hard`, condition 2 on `ntire_test`.

## 6. Results

- Condition 1: **FAIL** (-0.0054 [-0.0099, -0.0009]).
- Condition 2: not yet judged (read 3). On the validation splits the routed
  fusion is 0.008-0.011 above the crop fusion on clean large rows.
- Test 6 overall: **FAIL**, fixed by condition 1.
- Frozen parameters: `docs/ntire/amend5/test6_frozen.json` (staged, awaiting
  Dion's commit). Numbers: `test6_val.json`; drift check:
  `outputs/test6/drift_check.json`.

## 7. Future work

**Whether read 3 still runs.** The registration does not make read 3
conditional on condition 1. What it says:

> **Read 3 conditions.** (i) The routing rule, (a, b), z-scores and the branch
> fusion weight are fitted on `ntire_val` only and frozen. They are written to
> `docs/ntire/amend5/test6_frozen.json`, staged, and committed before the read.
> (ii) Read 3 runs only after every other test in this amendment has its
> results written. (iii) It runs exactly once, by one script that scores the
> frozen routed fusion and nothing else. (iv) The paper reports all three reads
> with their dates. Nothing read 3 shows can change a choice. Robust and pooled
> AUC on `ntire_test` are reported as well, but only condition 2 is judged
> there.

So:
- Running read 3 cannot turn Test 6 into a PASS. It still answers condition
  2 (does routing remove Sec 8's clean cost on `ntire_test`), which is the
  part the paper's cost section needs.
- Not running it keeps `ntire_test` at two reads. The registration says
  read 3 is "at most once more", so skipping is allowed, and the paper would
  then report Test 6 on validation only.
- Before read 3, fix the reading of "within 0.010" (§5.3), commit
  `test6_frozen.json`, and wait for Tests 7 and 8 to have results written
  (condition ii).

Other: the external-set evaluation (§5.5) if GPU time allows; a `--batch 8`
probe rerun to locate the drift (§5.4).

## 8. What didn't work

- Routing by predicted damage did not keep squish's robust AUC on
  `ntire_val_hard`.
- On Branch B the damage head's severity gave no useful routing signal
  (b = 0).

Numbers from commit `bbb1306` with the Test 6 scripts staged (not committed)
at run time; staged-code md5 `ff5c79b14d9b7e7de3160e71cee35ed6` (logged by
the queue at 05:07).
