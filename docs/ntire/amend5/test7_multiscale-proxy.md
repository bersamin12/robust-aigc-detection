# Test 7: multi-scale training baseline (`multiscale-proxy`)

## 1. Summary & results

**Verdict: PASS.** Registered condition (amendment 5, Test 7): "**Pass:** mean
over seeds of robust AUC (squish proxy - multi-scale proxy) on
`ntire_val_hard` > 0, with a CI above zero in at least 2 of 3 seeds."
Observed: mean +0.0187, CI above zero in 3 of 3 seeds. The fail branch
("multi-scale matches or beats squish (mean difference <= 0)") is not met.

| seed | squish - multi-scale, robust AUC on `ntire_val_hard` (95% CI) |
|---|---|
| 1 | +0.0225 [+0.0129, +0.0325] |
| 2 | +0.0191 [+0.0094, +0.0290] |
| 3 | +0.0146 [+0.0048, +0.0250] |
| mean | **+0.0187** |

All `ntire_val_hard` numbers are fit-split numbers: that split was used to
choose squish (pre-registration, "Rules that apply to every test").

Main reading:
- Training on random resized crops (area 0.20-1.00 of the frame) does not
  match a squish-trained proxy on damaged images. It does close most of the
  crop-to-squish gap: 73-81% on `ntire_val_hard`, 87-92% on `ntire_val`
  (§5.3, unregistered).
- The difference is in robust AUC only. Clean AUC is the same within
  ±0.002 and every clean CI includes zero.
- Every multi-scale seed chose the `squish` presentation on `ntire_val`. So
  the arm being compared is "multi-scale training, whole frame at test
  time". The remaining gap therefore comes from training on the whole
  frame, not from presenting it at test time.

## 2. Current state

- Registered in `docs/ntire/paper_prereg.md`, "Amendment 5", Test 7, at
  `991dc37` (amendment 6, `bbb1306`, does not touch Test 7).
- Comparators are the existing G1 proxies `abl_squish_s{1,2,3}` and
  `abl_crop_s{1,2,3}`. Their `ntire_val` / `ntire_val_hard` scores are the
  pipeline's own (`outputs/scores/paper/`). Their external-set scores did not
  exist, so they were scored here, as the registration requires.
- Before this test, the paper's argument that a crop-trained detector's
  deficit comes from the training size band had no multi-scale control.

## 3. Opportunities

- The multi-scale proxy is the strongest non-squish training policy measured
  so far. It recovers about 80% of the robust gap with a random view per
  sample and the same backbone, and on SocialRF it beats the squish proxy
  (§5.4).
- The registered fail branch would have restated the contribution as
  "training on one size band breaks detectors". With a PASS, the stronger
  claim stands: squish-trained beats multi-scale-trained on damaged images.
  The margin is modest (+0.019 on the fit split, +0.003 to +0.005 on
  `ntire_val`), so the paper should state it with its interval.

## 4. Testing methodology

### 4.1 Arms

| arm | training canonicalisation | test-time presentation | scored by |
|---|---|---|---|
| multi-scale (`abl_rrc_s<i>`) | random resized crop, area U(0.20, 1.00), aspect log-uniform [3/4, 4/3], resized to 224 | chosen per seed on `ntire_val` (all chose `squish`) | `scripts/ntire_crop_probe.py` |
| squish (`abl_squish_s<i>`) | whole frame at 224 | squish | `scripts/score_plan_splits.py` |
| crop (`abl_crop_s<i>`) | 200 px native window, resized to 224 | 200 px centre window | `scripts/score_plan_splits.py` |

Everything else is the G1 proxy protocol, copied flag for flag from stage P
of `scripts/paper_pipeline.sh`: dinov2regl224, depth 24, 1 epoch over 25% of
train, bank `ntire_meta`, mean token pool, cosine LR with SWA from 75%,
final weights, proportional family weighting, n-src 64, m-deg 2. Seeds
20260827, 20260828 and 20260829 pair with comparator seed indices 1, 2 and 3.

### 4.2 Order of operations (registered)

1. 100 timed steps: 8 workers, mean GPU utilisation 89.1%, 1.73 s/step, peak
   RAM 7 GB. That is above 70%, so the worker count stayed at 8.
2. For each seed: train, score `ntire_val` under `center`, `short_center` and
   `squish`, and choose by mean(robust AUC, clean AUC). Ties would go to the
   earlier presentation in that order. Only then score `ntire_val_hard` and
   the three external sets.
3. Comparators on the external sets, then `scripts/amend5_test7_analysis.py
   report`.

Presentation choice on `ntire_val`, mean(robust, clean):

| seed | center | short_center | squish | chosen |
|---|---|---|---|---|
| 1 | 0.9317 | 0.9686 | 0.9703 | squish |
| 2 | 0.9342 | 0.9705 | 0.9716 | squish |
| 3 | 0.9334 | 0.9694 | 0.9707 | squish |

### 4.3 Statistics

`scripts/paired_bootstrap.py`: 2,000 resamples stratified by label x
is_distorted, seed 20260827, the same indices for the three arms of a seed.
The baseline is multi-scale, so every difference reads (arm - multi-scale).

### 4.4 Choices the registration does not fix

1. **rrc sampling.** Follows torchvision's `RandomResizedCrop.get_params`:
   up to 10 attempts to fit a window of the drawn area and aspect, then a
   fallback to the whole image with its aspect clamped into [3/4, 4/3],
   centred. The draws are numpy draws from `canonical_rng(seed, row_id,
   view)`, so each of the 11 views of an image gets its own window.
2. **"resized to 224".** Implemented as squish's own two-step INTER_AREA
   resize (short side to 224 if larger, then the long side). rrc and squish
   therefore resample identically, and rrc with area fraction 1 is squish.
3. **The `center` presentation for an rrc checkpoint.** An rrc policy has no
   deterministic window, so `center` is the crop proxies' eval convention:
   one 200 px native centre window at nominal side 224, clamp on
   (`RRC_CENTER_CROP_SIDE` in `scripts/ntire_crop_probe.py`). It was not
   chosen for any seed, so this choice does not reach the verdict.
4. **Scorer.** The multi-scale arm is scored by the probe's `squish` arm.
   The comparators are scored by `score_plan_splits.py`. The two
   presentations are pixel-identical (a unit test pins `canonicalise` squish
   to `views_squish`). Batch size and other runtime settings differ. Test 6
   measured the effect of the same probe-vs-pipeline difference on frozen
   checkpoints at 0.00017 AUC or less.
5. **Seed pairing.** Multi-scale seed i is compared with comparator seed
   index i.

### 4.5 Code provenance

The new code was staged, not committed, when the queue ran it, so every
`commit` field in the json reads `bbb1306`. The queue logged the md5 of the
staged diff at start (`logs/queue.log`, 05:07:36):
`d9f0da811f2de9d81d5fa901cbc8804b`. At write-up time the staged diff of the
eight run files still has that md5, so the code that produced the numbers is
exactly what is staged. Files: `src/aigcdet/augment/canonical.py` (new
`rrc` mode), `src/aigcdet/features/{extract,replay}.py` (rrc draws a window
per view, like crop), `scripts/train_unfreeze.py` (help text),
`scripts/ntire_crop_probe.py` (rrc `center`), `scripts/amend5_test7.sh`,
`scripts/amend5_test7_analysis.py` and `tests/augment/test_canonical_rrc.py`.
`scripts/amend5_test7_supp.py` (§5.3) was added after the run.

## 5. Findings

### 5.1 `ntire_val_hard` (registered split, fit-split numbers)

| seed | arm | AUC | robust AUC (95% CI) | clean AUC |
|---|---|---|---|---|
| 1 | multi-scale | 0.8990 | 0.8641 [0.8435, 0.8834] | 0.9340 |
| 1 | squish | 0.9115 | 0.8866 [0.8678, 0.9046] | 0.9346 |
| 1 | crop | 0.8728 | 0.8037 [0.7792, 0.8268] | 0.9264 |
| 2 | multi-scale | 0.9024 | 0.8690 [0.8485, 0.8888] | 0.9371 |
| 2 | squish | 0.9139 | 0.8881 [0.8702, 0.9052] | 0.9384 |
| 2 | crop | 0.8693 | 0.7952 [0.7704, 0.8184] | 0.9266 |
| 3 | multi-scale | 0.8969 | 0.8631 [0.8423, 0.8823] | 0.9323 |
| 3 | squish | 0.9073 | 0.8776 [0.8578, 0.8957] | 0.9342 |
| 3 | crop | 0.8681 | 0.7993 [0.7746, 0.8224] | 0.9234 |

Paired differences (arm - multi-scale), 95% CI:

| seed | squish: robust | squish: clean | crop: robust | crop: clean |
|---|---|---|---|---|
| 1 | +0.0225 [+0.0129, +0.0325] | +0.0006 [-0.0057, +0.0073] | -0.0603 [-0.0805, -0.0406] | -0.0075 [-0.0241, +0.0075] |
| 2 | +0.0191 [+0.0094, +0.0290] | +0.0013 [-0.0045, +0.0075] | -0.0738 [-0.0948, -0.0532] | -0.0105 [-0.0265, +0.0044] |
| 3 | +0.0146 [+0.0048, +0.0250] | +0.0019 [-0.0041, +0.0084] | -0.0637 [-0.0851, -0.0435] | -0.0090 [-0.0257, +0.0065] |

Multi-scale sits between the two comparators on robust AUC in every seed.
On clean AUC no difference is resolved.

![Figure](test7_multiscale-proxy_fig.pdf)

*Figure 1. Robust AUC difference on `ntire_val_hard`, squish - multi-scale
(blue) and crop - multi-scale (red), per seed, 95% CI.*

### 5.2 The "All-NaN slice" warnings

The analysis printed 9 `RuntimeWarning: All-NaN slice encountered` lines.
They come from the 9 external-set comparisons (3 sets x 3 seeds). The
external sets as distributed have no distorted rows (`n_robust` = 0), so
`robust_auc` is NaN in every resample and its quantiles are NaN. Every NaN
in the json is a `robust_auc` cell under `external_as_distributed`. The
verdict reads only `ntire_val_hard` (`verdict(per_seed)` with `per_seed`
built from `EVAL_SPLIT`), where `n_robust` = 1,250 and no value is NaN. AUC
and clean AUC on the external sets are finite and are reported in §5.4.

### 5.3 Supplementary (unregistered): `ntire_val` and the share of the gap closed

Squish was chosen with `ntire_val_hard` in view, so the same comparison was
run on `ntire_val` (`scripts/amend5_test7_supp.py`, same bootstrap; json
`test7_multiscale-proxy_supp.json`). The multi-scale presentation was chosen
on `ntire_val`, so this split is not held out for that arm either.

| seed | squish - multi-scale, robust, `ntire_val` | share of crop->squish robust gap closed, `ntire_val_hard` | same, `ntire_val` |
|---|---|---|---|
| 1 | +0.0029 [+0.0005, +0.0054] | 0.73 | 0.92 |
| 2 | +0.0035 [+0.0010, +0.0060] | 0.79 | 0.90 |
| 3 | +0.0047 [+0.0022, +0.0074] | 0.81 | 0.87 |

Share closed = (multi-scale - crop) / (squish - crop), robust AUC. On
`ntire_val` the squish advantage is about one fifth of its `ntire_val_hard`
size, but its CI is still above zero in every seed.

### 5.4 External sets as distributed (reported, not judged)

AUC (all rows are clean, so AUC = clean AUC). Differences are (arm -
multi-scale) with 95% CI.

| set | seed | multi-scale | squish | crop | squish - multi-scale | crop - multi-scale |
|---|---|---|---|---|---|---|
| SocialRF | 1 | 0.9466 | 0.9397 | 0.9284 | -0.0070 [-0.0099, -0.0042] | -0.0182 [-0.0248, -0.0116] |
| SocialRF | 2 | 0.9465 | 0.9369 | 0.9239 | -0.0095 [-0.0123, -0.0070] | -0.0225 [-0.0290, -0.0161] |
| SocialRF | 3 | 0.9475 | 0.9432 | 0.9263 | -0.0044 [-0.0069, -0.0018] | -0.0212 [-0.0279, -0.0146] |
| CommunityAI | 1 | 0.9479 | 0.9554 | 0.9296 | +0.0075 [+0.0053, +0.0096] | -0.0183 [-0.0230, -0.0137] |
| CommunityAI | 2 | 0.9485 | 0.9548 | 0.9284 | +0.0063 [+0.0041, +0.0083] | -0.0201 [-0.0247, -0.0157] |
| CommunityAI | 3 | 0.9454 | 0.9571 | 0.9344 | +0.0117 [+0.0095, +0.0138] | -0.0110 [-0.0156, -0.0065] |
| Synthbuster | 1 | 0.9474 | 0.9516 | 0.9917 | +0.0043 [+0.0012, +0.0072] | +0.0443 [+0.0392, +0.0492] |
| Synthbuster | 2 | 0.9565 | 0.9538 | 0.9927 | -0.0027 [-0.0055, +0.0001] | +0.0362 [+0.0316, +0.0408] |
| Synthbuster | 3 | 0.9631 | 0.9590 | 0.9941 | -0.0041 [-0.0068, -0.0014] | +0.0310 [+0.0269, +0.0353] |

- **SocialRF:** multi-scale beats squish in all 3 seeds, with every CI
  below zero. Both beat crop.
- **CommunityAI:** squish beats multi-scale in all 3 seeds. Both beat crop.
- **Synthbuster:** crop is best by 0.03-0.04, in line with the Test 11
  finding that squish costs on Synthbuster. Multi-scale and squish are within
  0.005 of each other, with the sign varying by seed.

## 6. Results

| question | answer |
|---|---|
| Registered verdict | **PASS**: mean +0.0187, CI above zero 3/3 |
| Does multi-scale training match squish on damaged NTIRE? | No. It closes 73-81% of the crop->squish robust gap on `ntire_val_hard` and 87-92% on `ntire_val` |
| Clean cost? | None resolved on NTIRE (all clean CIs include zero) |
| External sets | Mixed: multi-scale > squish on SocialRF, squish > multi-scale on CommunityAI, crop best on Synthbuster |

## 7. Future work

- A wider area range (e.g. U(0.08, 1.00), torchvision's default) or a
  larger share of whole-frame views would test whether the remaining gap is
  the share of training views that see the whole frame.
- The SocialRF reversal deserves a size breakdown. It is the set where Test
  12 found squish's ROC crossing the crop proxy's.
- Squish and multi-scale arms on the external sets were scored by different
  scripts (§4.4.4). Rescoring the multi-scale arm with `score_plan_splits.py`
  would remove that difference. It is about 30 GPU-minutes.

## 8. What didn't work

- Nothing failed in the run. All three seeds trained and scored (exit 0,
  about 63 min per seed). The timing check needed no worker increase.
- The registration's "Under 70% ... workers are then raised" step did not
  trigger, so the 12- and 14-worker paths in `scripts/amend5_test7.sh` are
  untested on the GPU.

---

Numbers from `bbb1306` plus the staged diff with md5
`d9f0da811f2de9d81d5fa901cbc8804b` (§4.5). Source:
`docs/ntire/amend5/test7_multiscale-proxy.json` and
`test7_multiscale-proxy_supp.json`.
