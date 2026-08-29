#!/usr/bin/env bash
# Start the ablation-tier eval bank the moment the training bank finishes.
#
# The eval bank serves EVERY rung; recon serves only A4 and has a kill
# criterion. With a Tuesday deadline, eval-then-recon is the difference
# between a robustness table tonight and one on Sunday afternoon.
#
# It refuses to start if the training extraction did not actually complete:
# a crashed run needs resuming, and burning 1.8 h of GPU on the next job
# while the first one sits half-done is how a night gets lost.
set -u
# The repo root, derived from this script's own location rather than
# pinned to an absolute path: the tree has moved once already, and a
# stale `cd` sends a multi-hour job at the wrong data.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRAIN_PID=${1:?usage: chain_eval_bank.sh <train-pid>}
EXPECTED_ROWS=131116
LOG=logs/dinov3l_eval_bank.log

echo "[chain] waiting for training extraction (pid $TRAIN_PID)"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
echo "[chain] pid $TRAIN_PID exited at $(date '+%F %T')"

ROWS=$(python -c "
import pyarrow.parquet as pq
print(pq.ParquetFile('data/banks/dinov3l/meta.parquet').metadata.num_rows)
" 2>/dev/null || echo 0)
echo "[chain] training bank holds $ROWS / $EXPECTED_ROWS rows"

if [ "$ROWS" != "$EXPECTED_ROWS" ]; then
  echo "[chain] REFUSING to start the eval bank: the training extraction is"
  echo "[chain] incomplete. Resume it first:"
  echo "[chain]   python scripts/extract_features.py --manifest data/manifest.parquet \\"
  echo "[chain]       --backbone dinov3l --out data/banks/dinov3l \\"
  echo "[chain]       --split train,val_internal --resume --workers 6 --batch-size 64"
  exit 1
fi

echo "[chain] training bank COMPLETE. Starting the ablation-tier eval bank."
echo "[chain] 25,332 rows x 20 conditions = 506,640 forwards, ~1.8 h at 79 views/s."
exec python scripts/extract_eval_bank.py \
  --manifest data/eval_manifest.parquet \
  --backbone dinov3l \
  --out data/banks/eval_dinov3l \
  --tier ablation \
  --batch-size 64 \
  --resume >> "$LOG" 2>&1
