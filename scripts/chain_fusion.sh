#!/bin/bash
# Adds the A5 fusion rows that the first convnextt launch omitted.
#
# Pass 2: convnextt base + dinov3l partner  -> completes the convnextt ladder.
# Pass 3: dinov3l  base + convnextt partner -> the question the siglip2l fusion
#         could not ask, because siglip2l and dinov3l are both ViTs. convnextt
#         is the CNN paradigm, so this is the only genuinely cross-family fuse.
set -u
BASE_ARGS="--tier ablation --device cuda"
RUNGS="configs/rungs/a0.yaml configs/rungs/a1.yaml configs/rungs/a2.yaml configs/rungs/a3.yaml configs/rungs/a7_norecon.yaml"

while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "=== pass 2: convnextt + dinov3l ==="
python -u scripts/run_ablation.py \
  --bank data/banks/convnextt --eval-bank data/banks/eval_convnextt \
  --rungs $RUNGS \
  --fuse-bank data/banks/dinov3l --fuse-eval-bank data/banks/eval_dinov3l \
  $BASE_ARGS \
  --out docs/robustness_table_convnextt.md \
  --selection docs/selection_convnextt.json \
  --heatmap docs/robustness_heatmap_convnextt.png \
  --out-dir outputs/rungs_convnextt

echo "=== pass 3: dinov3l + convnextt ==="
# Copy rather than reuse outputs/rungs: pass 3 must NOT clobber the existing
# a5 (dinov3l+siglip2l, 0.8773) that is already reported in docs/.
rm -rf outputs/rungs_dino_cnn
cp -r outputs/rungs outputs/rungs_dino_cnn
# a5_partner is siglip2l-trained (dim 1024); left in place run_ablation would
# silently reuse it as the convnextt partner. a7_norecon predates today's
# LayerNorm fix, so drop it and let the fix be measured on dinov3l too.
rm -rf outputs/rungs_dino_cnn/a5_partner outputs/rungs_dino_cnn/a7_norecon
python -u scripts/run_ablation.py \
  --bank data/banks/dinov3l --eval-bank data/banks/eval_dinov3l \
  --rungs $RUNGS \
  --fuse-bank data/banks/convnextt --fuse-eval-bank data/banks/eval_convnextt \
  $BASE_ARGS \
  --out docs/robustness_table_a5_cnn.md \
  --selection docs/selection_a5_cnn.json \
  --heatmap docs/robustness_heatmap_a5_cnn.png \
  --out-dir outputs/rungs_dino_cnn
echo "=== chain done ==="
