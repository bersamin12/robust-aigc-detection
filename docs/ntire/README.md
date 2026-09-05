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
  the six family bits, up to 5 chained), `data/banks/ntire_meta`.
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

## Smoke run (2026-09-05, 2% of train, 1 epoch each, 500 hard-val rows)

| model | AUC | robust | clean |
|---|---|---|---|
| A (518) | 0.7317 | 0.7106 | 0.7554 |
| B (dual 224) | 0.7010 | 0.6973 | 0.7060 |
| fused (w_a=1.0 on 500 rows) | 0.7317 | 0.7106 | 0.7554 |

Numbers from ~80 optimiser steps; they prove the chain, not the model.
