# Robustness table: dual@224 ep1 vs prev_d24, plan test_transfer

Scored 2026-09-01 by `scripts/score_robustness_plan.py`. Protocol matches
`eval/grid.py`: canonicalise FIRST (centre crop-200, no rng), then apply the
condition to the canonicalised image, rng keyed (seed=0, row, condition).
Same 2,490 rows for every cell (test_transfer, stratified per (source,label),
random_state=0, min short side 200 — the per-group quota exhausts small
groups, hence < 4,000). Stratification reweights toward rare hard sources, so
absolute AUC here is NOT comparable with the pooled full-split AUC (0.9963);
the Δ-vs-clean column is the measurement. AUC SE ≈ ±0.005 at this n: treat
|Δ| < 0.01 as a tie.

Baseline = `outputs/unfreeze/d24/checkpoint.pt` (single dinov2regl, crop
200→512). Ours = dual@224 ep1 (2× dinov2regl, crop 200→224, norway,
slimmed). Raw per-row probabilities: `outputs/scores_norway/robust_*.parquet`;
full metrics: `docs/robustness_plan_{dual224_ep1,prev_d24}.json`.

| Augmentation | Baseline AUC | Our AUC | Δ vs clean (ours) |
| --- | --- | --- | --- |
| clean (reference) | 0.8891 | 0.9551 | — |
| JPEG q=90 | 0.8947 | 0.9645 | +0.0094 |
| JPEG q=70 | 0.8908 | 0.9645 | +0.0094 |
| JPEG q=50 | 0.8865 | 0.9550 | −0.0001 |
| JPEG q=30 (brief's grid) | 0.8764 | 0.9414 | −0.0137 |
| JPEG q=10 | 0.8478 | 0.8949 | −0.0601 |
| Blur σ=0.5 | 0.8831 | 0.9673 | +0.0122 |
| Blur σ=1.0 (held-out severity) | 0.8719 | 0.9585 | +0.0035 |
| Blur σ=2.0 | 0.8810 | 0.9500 | −0.0051 |
| Resize 0.5× | 0.8910 | 0.9600 | +0.0049 |
| Resize 0.25× | 0.8858 | 0.9353 | −0.0198 |
| Noise σ=0.02 | 0.8977 | 0.9584 | +0.0033 |
| Noise σ=0.05 | 0.8880 | 0.9565 | +0.0015 |
| Noise σ=0.10 | 0.8732 | 0.9411 | −0.0140 |
| Jitter brightness ±20% | 0.8809 | 0.9488 | −0.0062 |
| Jitter contrast ±20% | 0.8842 | 0.9617 | +0.0066 |
| Jitter saturation ±20% | 0.8918 | 0.9572 | +0.0021 |
| Jitter all three ±20% (brief's grid) | 0.8766 | 0.9557 | +0.0006 |
| Center crop 80% | 0.8869 | 0.9595 | +0.0045 |
| **Mean (16 template transforms, q30 and combined jitter excluded)** | **0.8835** | **0.9521** | **−0.0031** |

Baseline Δ vs its own clean, same 16 transforms: −0.0056 mean. Ours degrades
less AND starts higher on every single row.
