#!/bin/bash
# The readout ablation from the NTIRE 2026 report (arXiv 2604.11487): the
# same tower, the same recipe, five ways of turning its tokens into the
# head's input (models/pooling.py). Run on the cheap proxy -- ONE
# dinov2regl224 tower, depth 24, one epoch over a quarter of train -- so
# five arms cost ~4 h of the shared card, not five days. The winner is a
# candidate for a re-run of the real branches, not a result in itself.
#
#   bash scripts/ntire_pool_ablation.sh                # waits for PIPELINE_DONE
#   WAIT_FOR=none bash scripts/ntire_pool_ablation.sh  # right now
#
# Each arm: train -> score ntire_val_hard + ntire_val -> slim the checkpoint.
# Idempotent: an arm with a metrics file is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date "+%m-%d %H:%M:%S")] $*"; }

PY="${PY:-.venv/bin/python}"
EXP="${EXP:-/mnt/r13justinssd/techjam/experiments/data}"
MANIFEST="${MANIFEST:-data/manifest_ntire.parquet}"
BANK="${BANK:-data/banks/ntire_meta}"
MODES="${MODES:-mean cls cls_mean cls_reg_mean attn}"
BACKBONE="${BACKBONE:-dinov2regl224}"
SUBSAMPLE="${SUBSAMPLE:-0.25}"
EPOCHS="${EPOCHS:-1}"
CHUNK="${CHUNK:-8}"
WORKERS="${WORKERS:-14}"
SEED="${SEED:-20260827}"
OUT_DIR="${OUT_DIR:-outputs/ablation}"
SCORES="${SCORES:-outputs/scores/ablation}"
WAIT_FOR="${WAIT_FOR:-logs/pipeline.log}"
NEED_MIB="${NEED_MIB:-12000}"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$OUT_DIR" "$SCORES" logs

if [ "$WAIT_FOR" != none ]; then
  log "waiting for PIPELINE_DONE in $WAIT_FOR"
  while ! grep -q PIPELINE_DONE "$WAIT_FOR" 2>/dev/null; do sleep 300; done
  log "pipeline done"
fi

wait_gpu() {
  local seen=0 free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge "$NEED_MIB" ]; then
      seen=$((seen + 1)); [ "$seen" -ge 2 ] && { log "GPU has ${free} MiB free"; return; }
    else
      seen=0; log "waiting for GPU: ${free} MiB free, need $NEED_MIB"
    fi
    sleep 60
  done
}

for mode in $MODES; do
  name="abl_pool_${mode}"
  if [ -f "$SCORES/${name}_metrics.json" ]; then log "$name done; skipping"; continue; fi
  ck="$OUT_DIR/$name/checkpoint.pt"
  if [ ! -f "$ck" ] && [ -f "$OUT_DIR/$name/checkpoint_slim.pt" ]; then
    ck="$OUT_DIR/$name/checkpoint_slim.pt"
  fi
  if [ ! -f "$ck" ] || ! "$PY" -c "import torch,sys; ck=torch.load('$ck',map_location='cpu',weights_only=False,mmap=True); sys.exit(0 if ck['epoch']>=$EPOCHS else 1)"; then
    wait_gpu
    log "$name: $BACKBONE depth 24, $EPOCHS epoch(s) over $SUBSAMPLE of train, token_pool=$mode"
    "$PY" -u scripts/train_unfreeze.py --bank "$BANK" --root "$EXP" \
      --backbone "$BACKBONE" --depth 24 --name "$name" --out-dir "$OUT_DIR" \
      --epochs "$EPOCHS" --n-src 64 --m-deg 2 --src-chunk "$CHUNK" --workers "$WORKERS" \
      --seed "$SEED" --device cuda --canon-mode crop --crop-side 200 --nominal-side 224 \
      --lr-schedule cosine --warmup-frac 0.03 --min-lr-frac 0.01 --swa --swa-start-frac 0.75 \
      --family-weighting proportional --train-subsample-frac "$SUBSAMPLE" \
      --token-pool "$mode" --resume 2>&1 | tee -a "logs/$name.log"
    ck="$OUT_DIR/$name/checkpoint.pt"
  fi
  "$PY" scripts/score_plan_splits.py --ckpt "$ck" --manifest "$MANIFEST" \
    --splits ntire_val_hard,ntire_val --crop-clamp --batch 32 --chunk 16 --workers 12 \
    --out-prefix "$SCORES/$name" 2>&1 | tee -a "logs/score_$name.log"
  # Weights only from here on; the fat file and its epoch link go.
  if [ "$ck" = "$OUT_DIR/$name/checkpoint.pt" ]; then
    "$PY" scripts/slim_ckpt.py "$ck" && rm -f "$ck" "$OUT_DIR/$name"/checkpoint_ep*.pt
  fi
  "$PY" scripts/metrics_table.py "$SCORES" --pattern 'abl_pool_*_metrics.json' \
    --out docs/ntire/pool_ablation.md > /dev/null
done
log "POOL_ABLATION_DONE"
"$PY" scripts/metrics_table.py "$SCORES" --pattern 'abl_pool_*_metrics.json'
