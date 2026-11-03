# The NTIRE 2026 stream

Two `dinov2regl` branches, fine-tuned separately on one RTX 4090, fused at
score level, judged on the NTIRE 2026 "Robust AI-Generated Image Detection
in the Wild" organisers' own labelled sets (arXiv 2604.11487). Leaderboard
reference: winner 0.9723 robust / 0.9974 clean ROC AUC.

## Populations (`data/manifest_ntire.parquet`, `splits.json`)

| split | rows | what |
|---|---|---|
| `train` | 407,223 | NTIRE train shards 0-5, COCO train2017, Open Images reals, OV7 11,978 pairs, OV7 42,646 pairs (11 families), commercial-API pairs (490 fakes from 5 APIs + 120 reals) |
| `val` | 45,199 | seeded 10% per-source slice of train |
| `ntire_val` | 10,000 | organisers' validation set (half distorted) |
| `ntire_val_hard` | 2,500 | organisers' hard validation set |
| `ntire_test` | 2,500 | organisers' public test set -- **read once** |

`sid_set` and `wildfake` were deleted from disk (`data_removed.md`). The
42k-pair OV7 build comes from Kaggle `fiendfrye/ai-ov7-42k-pairs-11-families`
(symlinked at `$EXP/raw_ov7_42k`; download with the new-style token in
`KAGGLE_API_TOKEN`, not as a username/key pair). The commercial-API pairs
come from Kaggle `selennnnn/commercial-api-dataset` (symlinked at
`$EXP/raw_commercial_api`; the `parity_resample/` fakes, as `manifest.csv`
lists them).

## Recipe

* Views: 1 clean + 10 degraded, recipe pool `ntire` (24 ops grouped under
  the six family bits, up to 5 chained), `data/banks/ntire_meta`. Round
  two uses `ntire2` (26 ops; see below).
* Loss: the a3 recipe (class-balanced BCE + 0.3 degradation multitask +
  clean/degraded consistency).
* Sampler: class-balanced, fake families drawn **proportional** to size.
* Branch B: 2x `dinov2regl224`, crop 200 -> 224, depth 24, chunk 8,
  cosine + 3% warmup, SWA over the last 25%, 3 epochs. ~15.6 GiB.
* Branch A: `dinov2regl` @518, crop 200 -> 512, depth 24, chunk 2, same
  schedule, 2 epochs. ~9.9 GiB.
* Fusion: z-scored logits, weight swept on `ntire_val` + `ntire_val_hard`
  against mean(robust AUC, clean AUC), ties to 0.5 (`eval/late_fusion.py`).

## Running

```
bash scripts/ntire_pipeline.sh          # B -> A -> score + fit fusion (unattended)
bash scripts/epoch_watch.sh outputs/dual/ntire_dual224:16 outputs/unfreeze/ntire_d24_518:4
                                        # scores each epoch snapshot on ntire_val_hard as it lands
STAGE=test bash scripts/ntire_one_gpu.sh # the ONE read of ntire_test
```

Logs: `logs/pipeline.log`, `logs/ntire_dual224.log`, `logs/ntire_d24_518.log`,
`logs/score_*.log`, `logs/slim_watch.log`, `logs/epoch_watch*.log`. Results:
`fusion_ntire.json` (selection splits), `fusion_ntire_final.json` (with the
test read), `outputs/scores/epochs/summary.md` (the per-epoch curve, from
`scripts/metrics_table.py`).

## Round two: `ntire2` (`scripts/ntire_v2.sh`)

The round-one epoch curve on `ntire_val_hard` (below) ranked two distortions
weakest in every epoch: NTIRE's **downscale** (in 931 of the 1,250 distorted
rows; factor 0.3-0.8 of the full image, left small) and **JPEG AI** (248
rows; a learned codec, no block grid). Neither was in the `ntire` pool as
the organisers apply it: `downscale` there is a round trip on the crop, and
no op re-synthesised textures. Pool `ntire2` appends:

* `downscale_keep` -- a PRE-canonical op (`recipes.PRE_CANONICAL_OPS`): the
  decoded image is shrunk by 0.3-0.8 and the crop window taken from the
  small image, as at scoring time. Needs `--crop-clamp`, which the trainer
  now checks, so an image that falls below the window is taken whole.
* `jpeg_ai` -- compressai's `bmshj2018_factorized` at quality 1-4 (optional
  extra `ntire`; ~0.3 s per 224px view on a busy CPU thread).

Within the resize and jpeg families the draw is weighted towards them
(`OP_WEIGHTS`: 1 in 2 resize draws, 2 in 5 jpeg draws). Bank
`data/banks/ntire2_meta`; runs `ntire2_dual224` / `ntire2_d24_518`; the
script scores final and SWA weights, picks the better of each on the
selection splits, and writes `fusion_ntire2.json` and `round2_scores.md`.

## Round one, per-epoch (`outputs/scores/epochs/summary.md`)

| dual224 | robust | clean | AUC |
|---|---|---|---|
| epoch 1 | 0.8601 | 0.9703 | 0.9231 |
| epoch 2 | 0.8606 | 0.9819 | 0.9319 |
| epoch 3 | 0.8647 | 0.9856 | 0.9355 |
| epoch 3, SWA | 0.8666 | 0.9855 | 0.9364 |

Weakest distortions at epoch 3 (AUC of that token's rows against every
clean row): downscale 0.945, jpeg_ai 0.952, quantization 0.967; all others
0.974-0.983.

## Readout ablation (`scripts/ntire_pool_ablation.sh`, `models/pooling.py`)

Both branches read the tower the way every earlier arm did: the mean of the
final-layer patch tokens, CLS and registers dropped. The challenge report has
the strongest entries reading it differently -- Ant International (2nd)
runs one expert on attention pooling and one on first-token (CLS) pooling;
Shallow Real concatenates CLS, the four register tokens and the patch mean.
`--token-pool` on both trainers makes that a variable:

| mode | head input | what |
|---|---|---|
| `mean` | D | patch mean (the default; the historical path, bit for bit) |
| `cls` | D | the CLS token |
| `cls_mean` | 2D | CLS ++ patch mean |
| `cls_reg_mean` | 6D | CLS ++ 4 registers ++ patch mean (Shallow Real) |
| `attn` | D | one learned query, 8-head attention over the patches, LayerNorm (Ant) |

The pooler is part of the head (trained at the head's LR, averaged by SWA,
saved under `state_dict`), and a checkpoint's `dim_feat` follows it. The
ablation runs after the pipeline, on the cheap proxy: one `dinov2regl224`
tower, depth 24, one epoch over a quarter of train, scored on `ntire_val_hard`
and `ntire_val`; table in `pool_ablation.md`. A winner there is a candidate
for a re-run of the real branches, not a result in itself.

## Readout ablation result (2026-09-07, `pool_ablation.md`)

Mean pooling won on the proxy (objective 0.9067; cls 0.9056, cls_mean
0.9057, cls_reg_mean 0.9054, attn 0.8995), by too little to move a branch
off it. The proxy baseline for anything else run in that setting is
therefore `abl_pool_mean`: ntire_val_hard robust 0.8040 / clean 0.9264,
ntire_val robust 0.9280 / clean 0.9683.

## Round one result and fusion (2026-09-07, `fusion_ntire.json`)

Fitted on ntire_val + ntire_val_hard (12,500 rows), w_a = 0.5. On
ntire_val_hard: A (518, epoch 2) robust 0.8450 / clean 0.9941, B (dual 224,
epoch 3 SWA) 0.8647 / 0.9856, fused 0.8655 / 0.9940. Fusion buys clean AUC
and almost no robust AUC: both branches fail on the same rows (downscale,
JPEG AI, quantization), so a second opinion from the same failure mode
adds nothing there. That is the case for changing the training signal
rather than adding models.

## Degradation-invariant distillation (`--distil`, `scripts/ntire_did_ablation.sh`)

The method contribution. The a3 recipe already has a clean/degraded
consistency term, but it acts on the HEAD's 512-d hidden state, which the
head can satisfy by projecting the distortion away while the tower's own
feature keeps it. `--distil` adds a term on the tower's pooled output:
each degraded view's feature is pulled, by cosine, towards the clean
view's, with the clean side a stop-gradient target
(`models/losses.py: distillation_loss`; design memo in the session's
artifact page). Two targets:

| `--distil` | target | cost |
|---|---|---|
| `self` | the same tower's clean feature, detached | none (it is already computed) |
| `ema` | an EMA of tower and head (`_EmaTeacher`, decay `--distil-ema-decay`, 0.999) | one more forward per source, a second fp32 tower on the card, +1.2 GB per checkpoint |

`--distil` does not switch the a3 terms off; `--alpha 0 --beta 0` does, and
the ablation runs both ways so the effect is attributed to the new term and
not to the missing old ones. `dis` in the progress line is the raw cosine
distance (0-2). A `--distil ema` checkpoint keeps its teacher as a whole
model: `score_plan_splits.py --ema` scores it; `slim_ckpt.py --no-ema`
drops it. Resume restores the teacher and refuses a checkpoint without one.

Proxy arms (same setting as the readout ablation, so `abl_pool_mean` is the
baseline row): `c1self` (replacing the a3 terms), `c1self_a3` (beside
them), `c1ema` (replacing). Go/no-go, fixed before the numbers: c1self
gains more than one robust point on ntire_val_hard over 0.8040 at no more
than half a clean point lost, or components 2 (invariant/aware degradation
readout) and 3 (loss-driven op sampling) are not built on it. Table:
`did_ablation.md`.
## Distillation ablation result (2026-09-14, `did_ablation.md`)

Checkpoints on `/scratch/anonymous/aigcdet/outputs/ablation` (the repo disk was
full; `OUT_DIR=` selects it). ntire_val_hard robust / clean, delta to
`abl_pool_mean` (0.8040 / 0.9264):

| arm | robust | clean | ntire_val robust / clean |
|---|---|---|---|
| `c1self` (distil, a3 off) | 0.7931 (-1.1) | 0.9389 (+1.3) | 0.9251 / 0.9749 |
| `c1self_a3` (distil + a3) | 0.7957 (-0.8) | 0.9234 (-0.3) | 0.9222 / 0.9673 |
| `c1ema` (EMA target, a3 off) | 0.7930 (-1.1) | 0.9325 (+0.6) | 0.9304 / 0.9760 |
| `c1ema` teacher (`--ema`) | 0.7699 | 0.8933 | 0.9105 / 0.9531 |

**The go/no-go fails.** All three arms lose about one robust point on the
hard split, with the same sign and size whichever target is used, while
`c1self` is the largest clean gain of any proxy arm. The readout arms sat
within 0.0002 robust of each other, so the drop is not noise. `dis` fell
0.33 -> 0.07 over the epoch: the tower did collapse the clean/degraded
feature gap, and that is what cost degraded-row separability -- pulling
f(degraded) onto f(clean) asks for artefact cues the degradation has
removed. The EMA teacher scores below its student because, at decay 0.999
over 1,544 steps, it is still a blend with early weights.

Confound known before the numbers: under crop canonicalisation the clean
and degraded views are different 200 px windows, so the term also asks for
crop (content) invariance. A same-window control is the one follow-up that
tests the hypothesis rather than this test of it; a `distil_weight` sweep
would be rescuing the method after seeing the data and is not run.
Components 2 and 3 are not built on this result.

### Same-window control (2026-09-17)

`--share-window` builds every degraded view on its clean partner's crop
window (`build_view(window_view=0)`), so the pair differs in degradation
only. Two arms, both on shared windows, gate written before the numbers:
`sw_c1self` beats `sw_base` by more than one robust point at no more than
half a clean point lost, or DID is recorded as refuted rather than
confounded. ntire_val_hard robust / clean:

| arm | robust | clean | ntire_val robust / clean | `dis` at step 1500 |
|---|---|---|---|---|
| `sw_base` (shared window, a3 off, no distil) | 0.7900 | 0.9364 | 0.9264 / 0.9760 | -- |
| `sw_c1self` (shared window, a3 off, distil) | 0.7884 | 0.9333 | 0.9232 / 0.9742 | 0.062 |
| `c1self` (own windows, for reference) | 0.7931 | 0.9389 | 0.9251 / 0.9749 | 0.075 |

**The control fails too: -0.16 robust, -0.3 clean for the term, against a
pair that differs in nothing else.** With the crop confound removed the
distillation term does nothing a readout swap would not (the readout arms
spanned 0.0002). DID is refuted on this proxy, not confounded; the paper
records it as a negative result and the contribution is coverage (round
two). Secondary reading: `sw_base` sits 1.4 robust points below
`abl_pool_mean` and 1.0 clean point above it, which is the same trade every
arm without the a3 consistency terms shows (`c1self`, `c1ema`); the
window sharing and the missing a3 terms are not separated here, and no
further arm is run to separate them. Component-1 code stays on the branch
as the record of what was tested.

## Smoke run (2026-09-05, 2% of train, 1 epoch each, 500 hard-val rows)

| model | AUC | robust | clean |
|---|---|---|---|
| A (518) | 0.7317 | 0.7106 | 0.7554 |
| B (dual 224) | 0.7010 | 0.6973 | 0.7060 |
| fused (w_a=1.0 on 500 rows) | 0.7317 | 0.7106 | 0.7554 |

Numbers from ~80 optimiser steps; they prove the chain, not the model.
