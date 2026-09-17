#!/bin/bash
# Degradation-invariant distillation, component 1, on the proxy: ONE
# dinov2regl224 tower, depth 24, one epoch over a quarter of train, mean
# readout -- the exact setting of `ntire_pool_ablation.sh`, so its
# `abl_pool_mean` arm IS the baseline row and is not run again. Each arm
# adds the distillation term in one form (docs/ntire/did_design notes):
#
#   c1self      the same tower's clean feature as a stop-gradient target,
#               REPLACING the a3 consistency terms (--alpha 0 --beta 0)
#   c1self_a3   the same, BESIDE the a3 terms (the confound control: is it
#               the new term or the missing old ones?)
#   c1ema       an EMA tower as the target, replacing the a3 terms
#
# Go/no-go, written before the numbers: c1self beats abl_pool_mean by more
# than one robust point on ntire_val_hard at no more than half a clean
# point lost, or components 2 and 3 are not built on it.
#
#   bash scripts/ntire_did_ablation.sh                # right now (GPU permitting)
#   ARMS="c1self" bash scripts/ntire_did_ablation.sh  # one arm
#   OUT_DIR=/scratch/$USER/aigcdet/outputs/ablation bash scripts/ntire_did_ablation.sh  # checkpoints on a disk with room
#
# Idempotent: an arm with a metrics file is skipped. Each arm: train ->
# score ntire_val_hard + ntire_val (final weights; the EMA arm also scores
# its teacher) -> slim the checkpoint -> regenerate the table.
set -euo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date "+%m-%d %H:%M:%S")] $*"; }

PY="${PY:-.venv/bin/python}"
EXP="${EXP:-/mnt/r13justinssd/techjam/experiments/data}"
MANIFEST="${MANIFEST:-data/manifest_ntire.parquet}"
BANK="${BANK:-data/banks/ntire_meta}"
ARMS="${ARMS:-c1self c1self_a3 c1ema}"
BACKBONE="${BACKBONE:-dinov2regl224}"
SUBSAMPLE="${SUBSAMPLE:-0.25}"
EPOCHS="${EPOCHS:-1}"
CHUNK="${CHUNK:-8}"
WORKERS="${WORKERS:-14}"
SEED="${SEED:-20260827}"
DISTIL_WEIGHT="${DISTIL_WEIGHT:-1.0}"
EMA_DECAY="${EMA_DECAY:-0.999}"
OUT_DIR="${OUT_DIR:-outputs/ablation}"
SCORES="${SCORES:-outputs/scores/ablation}"
TABLE="${TABLE:-docs/ntire/did_ablation.md}"
BASELINE="${BASELINE:-abl_pool_mean}"
NEED_MIB="${NEED_MIB:-12000}"
MIN_FREE_GB="${MIN_FREE_GB:-12}"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$OUT_DIR" "$SCORES" logs

# The extra flags per arm. Everything else is the proxy's fixed setting.
flags_for() {
  case "$1" in
    c1self)    echo "--distil self --distil-weight $DISTIL_WEIGHT --alpha 0 --beta 0" ;;
    c1self_a3) echo "--distil self --distil-weight $DISTIL_WEIGHT" ;;
    c1ema)     echo "--distil ema --distil-weight $DISTIL_WEIGHT --distil-ema-decay $EMA_DECAY --alpha 0 --beta 0" ;;
    *) log "unknown arm $1"; return 1 ;;
  esac
}

wait_gpu() {
  local seen=0 free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || true)
    if [ -z "$free" ]; then
      log "nvidia-smi gave no reading (driver/library mismatch?); retrying in 5 min"; sleep 300; continue
    fi
    if [ "$free" -ge "$NEED_MIB" ]; then
      seen=$((seen + 1)); [ "$seen" -ge 2 ] && { log "GPU has ${free} MiB free"; return; }
    else
      seen=0; log "waiting for GPU: ${free} MiB free, need $NEED_MIB"
    fi
    sleep 60
  done
}

wait_disk() {
  local free_gb
  while true; do
    free_gb=$(df --output=avail -BG "$OUT_DIR" | tail -1 | tr -dc 0-9)
    [ "$free_gb" -ge "$MIN_FREE_GB" ] && { log "${free_gb} GB free"; return; }
    log "only ${free_gb} GB free, need $MIN_FREE_GB; waiting"; sleep 600
  done
}

table() {
  "$PY" scripts/metrics_table.py "$SCORES" --pattern 'abl_did_*_metrics.json' \
    --pattern "${BASELINE}_metrics.json" --out "$TABLE" > /dev/null
}

for arm in $ARMS; do
  name="abl_did_${arm}"
  extra=$(flags_for "$arm")
  if [ -f "$SCORES/${name}_metrics.json" ]; then log "$name done; skipping"; continue; fi
  ck="$OUT_DIR/$name/checkpoint.pt"
  if [ ! -f "$ck" ] && [ -f "$OUT_DIR/$name/checkpoint_slim.pt" ]; then
    ck="$OUT_DIR/$name/checkpoint_slim.pt"
  fi
  if [ ! -f "$ck" ] || ! "$PY" -c "import torch,sys; ck=torch.load('$ck',map_location='cpu',weights_only=False,mmap=True); sys.exit(0 if ck['epoch']>=$EPOCHS else 1)"; then
    wait_disk; wait_gpu
    log "$name: $BACKBONE depth 24, $EPOCHS epoch(s) over $SUBSAMPLE of train, $extra"
    # shellcheck disable=SC2086
    "$PY" -u scripts/train_unfreeze.py --bank "$BANK" --root "$EXP" \
      --backbone "$BACKBONE" --depth 24 --name "$name" --out-dir "$OUT_DIR" \
      --epochs "$EPOCHS" --n-src 64 --m-deg 2 --src-chunk "$CHUNK" --workers "$WORKERS" \
      --seed "$SEED" --device cuda --canon-mode crop --crop-side 200 --nominal-side 224 \
      --lr-schedule cosine --warmup-frac 0.03 --min-lr-frac 0.01 --swa --swa-start-frac 0.75 \
      --family-weighting proportional --train-subsample-frac "$SUBSAMPLE" \
      --token-pool mean $extra --resume 2>&1 | tee -a "logs/$name.log"
    ck="$OUT_DIR/$name/checkpoint.pt"
  fi
  "$PY" scripts/score_plan_splits.py --ckpt "$ck" --manifest "$MANIFEST" \
    --splits ntire_val_hard,ntire_val --crop-clamp --batch 32 --chunk 16 --workers 12 \
    --out-prefix "$SCORES/$name" 2>&1 | tee -a "logs/score_$name.log"
  case "$arm" in
    c1ema)
      # The teacher is a model too; its row is the same arm under `ema`.
      "$PY" scripts/score_plan_splits.py --ckpt "$ck" --manifest "$MANIFEST" --ema \
        --splits ntire_val_hard,ntire_val --crop-clamp --batch 32 --chunk 16 --workers 12 \
        --out-prefix "$SCORES/${name}_teacher" 2>&1 | tee -a "logs/score_${name}_teacher.log" ;;
  esac
  if [ "$ck" = "$OUT_DIR/$name/checkpoint.pt" ]; then
    "$PY" scripts/slim_ckpt.py "$ck" && rm -f "$ck" "$OUT_DIR/$name"/checkpoint_ep*.pt
  fi
  table
done
log "DID_ABLATION_DONE"
table
cat "$TABLE"
