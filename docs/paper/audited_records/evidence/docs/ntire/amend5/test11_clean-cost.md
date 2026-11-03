# Test 11: clean cost on large images (`clean-cost`)

Amendment 5, Test 11. Analysis of existing per-image scores only. No model
was loaded, no image was read, and `ntire_test` was not touched.

## 1. Summary & results

**Registered verdict: NOT MET (cost unexplained), on both branches.** The
registered condition reads: "at least one generator or size bucket where
squish's clean AUC is below crop's by more than 0.020 with a CI below zero,
and no such drop outside it. That pattern is reported as the cost's
location." The registered alternative applies: "Otherwise: the cost is
reported as unexplained, and Test 6 is checked for whether it removes it."

- **Branch A** (the branch that carries the `ntire_test` clean cost in the
  paper) shows no qualifying drop on Synthbuster. Every generator's change is
  within 0.007 of zero. Synthbuster does not reproduce branch A's large-image
  cost (0.974 -> 0.880 on `ntire_test`, short side > 1100 px).
- **Branch B** shows qualifying drops in two places that are not one
  location: `dalle2` (1024 px fakes, in the <= 1100 bucket, -0.110) and
  `firefly`, which is the whole > 1100 bucket (-0.040). The location condition
  fails because `dalle2` lies outside the > 1100 bucket.
- **Size and generator are the same variable on Synthbuster above 1100 px.**
  The > 1100 bucket holds exactly the 997 Firefly fakes and nothing else.
  Synthbuster therefore cannot separate a large-image effect from a Firefly
  effect.
- The NTIRE check (clean rows > 1100 px on the fit splits) reproduces
  `docs/ntire/paper/analysis_valsize.md` to 4 dp. Neither fit split shows a
  cost on either branch.

| branch | flagged generators | flagged size buckets | registered reading |
|---|---|---|---|
| A | none | none | cost unexplained |
| B | `dalle2`, `firefly` | > 1100 (= `firefly`) | cost unexplained (two locations) |

## 2. Current state

The paper (Sec. "Where whole-image input costs") places the clean loss of
test read 2 on `ntire_test` images above 1100 px for branch A. It notes that
the validation splits do not show it, and that `ntire_test` has no generator
labels. The hypothesis in the paper is that large, clean generated images
keep fine traces at native resolution, and that squishing them to 512 px
discards those traces. Test 11 looks for the same cost on Synthbuster, which
has generator labels, and recomputes the fit-split check.

## 3. Opportunities

- If the cost had appeared on Synthbuster for branch A in one generator or
  one size bucket, the paper could name its location. It did not, so the
  paper's "we cannot say what" stands for branch A.
- Branch B's Synthbuster drops are new information. They do not match the
  `ntire_test` pattern, where branch B changes by under 1.3 points at every
  size. They are worth one sentence in the paper's held-out results.

## 4. Testing methodology

### 4.1 Data

| item | value |
|---|---|
| split | `ext_synthbuster` (clean copy only; `_deg` and `_jpeg95` not read) |
| rows | 9,990: 994 RAISE reals, 8,996 fakes over 9 generators |
| real short side | 833-835 px (never split by size, as registered) |
| sizes | `data/manifest_external.parquet` `width`/`height`, joined on `path` (all rows matched) |
| crop-trained | `outputs/scores/paper/A2_*`, `B2swa_*` |
| squish-trained | `outputs/scores/paper/A3sqswa_*`, `B3sqswa_*` (`picks.json`) |

Each model is scored as trained, under its own input policy. "crop vs
squish" in this test is therefore crop-trained against squish-trained, as in
`paper_analyses.valsize` and in the paper's test-read comparison.

### 4.2 Analyses

1. **Per generator:** clean AUC of that generator's fakes against all 994
   reals. `diff` = AUC(squish-trained) - AUC(crop-trained).
2. **By fake short side** (<= 1100, > 1100): fakes only. Reports the mean
   logit and TPR at the threshold giving 1% FPR on the reals (the 0.99
   quantile of each model's real logits; realised FPR 1.01% = 10 of 994).
   The bucket's AUC against all reals is added so the pass condition can be
   judged on buckets as well (see §8.2).
3. **NTIRE check:** clean `ntire_val` / `ntire_val_hard` rows with short
   side > 1100 px, recomputed as Open question 4 of the amendment asks.

### 4.3 Statistics

`paper_analyses.paired_diff`: 2,000 paired resamples, seed 20260827, the
same indices for both models, 95% percentile intervals. Strata are by label.
Every row is clean, so label x is_distorted reduces to label. A **drop** is
`diff < -0.020` and the upper CI bound `< 0`.

### 4.4 How the location rule was applied

The registered "no such drop outside it" was read as follows. All flagged
cells must lie inside one location: either a single generator, or a single
size bucket that contains every flagged generator's fakes. Both flagged lists
are in the JSON, so a different reading can be checked against them.

## 5. Findings

### 5.1 Per generator (analysis 1)

Clean AUC, crop-trained -> squish-trained, with the paired difference.

| generator | fake short side | n fake | A crop -> squish | A diff [95% CI] | B crop -> squish | B diff [95% CI] |
|---|---|---|---|---|---|---|
| dalle2 | 1024 | 1000 | 0.9980 -> 0.9992 | +0.0012 [+0.0003, +0.0023] | 0.9941 -> 0.8841 | **-0.1100 [-0.1255, -0.0949]** |
| dalle3 | 1024 | 1000 | 1.0000 -> 0.9995 | -0.0005 [-0.0013, +0.0000] | 0.9998 -> 0.9991 | -0.0007 [-0.0023, +0.0002] |
| firefly | 1536-2048 | 997 | 0.9996 -> 0.9929 | -0.0067 [-0.0103, -0.0036] | 0.9990 -> 0.9591 | **-0.0399 [-0.0492, -0.0314]** |
| glide | 256 | 1000 | 0.9984 -> 0.9987 | +0.0003 [-0.0005, +0.0012] | 0.9946 -> 0.9808 | -0.0138 [-0.0204, -0.0083] |
| midjourney-v5 | 896-1024 | 999 | 1.0000 -> 0.9981 | -0.0019 [-0.0032, -0.0009] | 0.9998 -> 0.9828 | -0.0170 [-0.0227, -0.0113] |
| stable-diffusion-1-3 | 512 | 1000 | 1.0000 -> 1.0000 | +0.0000 [+0.0000, +0.0000] | 1.0000 -> 1.0000 | -0.0000 [-0.0000, +0.0000] |
| stable-diffusion-1-4 | 512 | 1000 | 1.0000 -> 1.0000 | +0.0000 [+0.0000, +0.0000] | 1.0000 -> 0.9997 | -0.0003 [-0.0007, -0.0000] |
| stable-diffusion-2 | 576-1024 | 1000 | 1.0000 -> 0.9994 | -0.0005 [-0.0011, -0.0001] | 0.9998 -> 0.9882 | -0.0116 [-0.0160, -0.0077] |
| stable-diffusion-xl | 576-1024 | 1000 | 1.0000 -> 0.9999 | -0.0001 [-0.0002, -0.0000] | 0.9996 -> 0.9944 | -0.0052 [-0.0089, -0.0023] |

Bold marks a registered drop. Branch A's crop-trained AUC is at or above
0.998 for every generator, so this split leaves branch A little room to show
a cost. Branch B's squish-trained model is below its crop-trained model on 8
of 9 generators. Only two of those clear the 0.020 size.

![per-generator change](test11_clean-cost_fig.pdf)

### 5.2 By fake short side (analysis 2)

| bucket | n fake | generators | branch | AUC crop -> squish | AUC diff [95% CI] | mean fake logit crop -> squish | TPR@1%FPR crop -> squish |
|---|---|---|---|---|---|---|---|
| <= 1100 | 7999 | all but firefly | A | 0.9995 -> 0.9994 | -0.0002 [-0.0005, +0.0001] | 6.33 -> 6.75 | 0.987 -> 0.990 |
| <= 1100 | 7999 | all but firefly | B | 0.9985 -> 0.9786 | -0.0198 [-0.0224, -0.0172] | 5.99 -> 3.72 | 0.962 -> 0.883 |
| > 1100 | 997 | firefly only | A | 0.9996 -> 0.9929 | -0.0067 [-0.0103, -0.0036] | 5.28 -> 3.94 | 0.984 -> 0.943 |
| > 1100 | 997 | firefly only | B | 0.9990 -> 0.9591 | **-0.0399 [-0.0492, -0.0314]** | 4.86 -> -1.16 | 0.968 -> 0.721 |

- On branch A, the direction matches the paper's mechanism. Under squish the
  mean score of the large fakes falls (5.28 -> 3.94), while the small fakes'
  mean score rises (6.33 -> 6.75), and TPR on the large fakes drops 4.1
  points. The AUC change (-0.0067) is a third of the registered 0.020.
- Branch B's <= 1100 bucket misses the 0.020 size by 0.0002 (-0.0198). Its
  CI lies wholly below zero. It is not flagged, but the margin is too small
  to call it a clean "no".

### 5.3 NTIRE fit splits above 1100 px (analysis 3)

| split | branch | n (real/fake) | crop -> squish | diff [95% CI] |
|---|---|---|---|---|
| ntire_val | A | 871 (688/183) | 0.9890 -> 0.9999 | +0.0110 [+0.0060, +0.0169] |
| ntire_val | B | 871 (688/183) | 0.9886 -> 0.9989 | +0.0104 [+0.0045, +0.0177] |
| ntire_val_hard | A | 295 (168/127) | 0.9814 -> 0.9782 | -0.0032 [-0.0224, +0.0162] |
| ntire_val_hard | B | 295 (168/127) | 0.9820 -> 0.9824 | +0.0004 [-0.0161, +0.0171] |

These are identical, to 4 dp, to `docs/ntire/paper/analysis_valsize.md`
(commit `5ecff8f`). The paper's 871 / 295 figures stand.

## 6. Results

1. Registered verdict: **cost unexplained** on both branches. Under the
   amendment, Test 6 is now checked for whether routing removes the cost.
2. For branch A, the branch with the `ntire_test` cost, Synthbuster shows
   the expected direction on its only large generator (Firefly). The size is
   small: -0.0067 AUC, -4.1 points TPR@1%FPR.
3. On Synthbuster, branch B, not branch A, carries the squish-trained clean
   cost. Its worst generator, dalle2 at 1024 px, sits below the 1100 px cut.
4. Synthbuster cannot separate "large image" from "Firefly". This holds for
   any size cut between 1025 and 1535 px, because no fake has a short side in
   that range.

## 7. Future work

- **Test 6 check** (registered consequence): once Test 6's routed fusion is
  frozen, report its clean AUC on the same Synthbuster generators and buckets.
- **No new scoring is needed for Test 11.** All four models already have
  `ext_synthbuster` scores, so the registration's "otherwise ... scored"
  branch never applied. No GPU command is pending.
- Not registered, so not run: the same tables on SocialRF / CommunityAI.
  Their size spread may separate size from generator better than Synthbuster
  does.

## 8. What didn't work

### 8.1 Synthbuster as a size probe

The registered size split is degenerate on this set. The > 1100 bucket is
one generator, and the reals cannot be split by size (833-835 px). A
size-only reading of Synthbuster is not available.

### 8.2 Interpretation points

- The registration lists the bucket's mean logit and TPR@1%FPR, but its
  pass condition is about "clean AUC" in a "size bucket". The bucket AUC
  used for the condition (bucket fakes against all reals) is therefore this
  write-up's construction. It follows the per-generator definition.
- The location rule (§4.4) is a reading of "no such drop outside it". Under
  any reading, branch A has nothing flagged. Branch B has a 1024 px
  generator flagged and the > 1100 bucket flagged. Only a reading that
  treats "generator dalle2 or bucket > 1100" as one location would pass, and
  the condition's wording does not support that.
- TPR and mean logit are point estimates. The registration asks for paired
  CIs only on AUC, and the plan forbids a new bootstrap.

---

Numbers from commit `bbb1306` (amendment 6) plus the staged
`scripts/amend5_test11.py`. Seed 20260827, 2,000 resamples. Per-image join:
`/scratch/anonymous/dion_extensions/outputs/test11/test11_synthbuster_rows.parquet`.
