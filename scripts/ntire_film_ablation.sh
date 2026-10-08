#!/bin/bash
# Degradation-CONDITIONED classification on the proxy: ONE dinov2regl224
# tower, depth 24, one epoch over a quarter of train, mean readout -- the
# exact setting of `ntire_pool_ablation.sh` and `ntire_did_ablation.sh`, so
# `abl_pool_mean` IS the baseline row and is not run again.
#
# The hypothesis is the MIRROR of DID's. DID made the features forget the
# degradation (cosine distillation toward the clean view) and lost a robust
# point; its same-window control (2026-09-17) showed that was the term
# itself, not the crop confound. The degradation head already PREDICTS the
# degradation -- six presence bits and six severities, supervised by the a3
# recipe -- and the headline model spends that prediction on calibration
# only. These arms spend it on the classifier instead: FiLM-modulate the
# classifier's hidden state by the degradation trunk's (detached) embedding.
#
#   film        --film: each row conditioned on ITS OWN degradation embedding
#   film_shuf   --film --film-shuffle: the INFORMATION CONTROL. Every FiLM
#               parameter and the extra film_norm are present and trained;
#               only the row assignment changes, so each image is modulated
#               by its neighbour's embedding. Turning FiLM on adds a linear
#               and a LayerNorm to the classifier, so a gain could be
#               capacity rather than conditioning -- this arm is the only
#               thing that tells those apart, and it exists BEFORE the
#               numbers because DID's control did not.
#
# Prior (docs/overnight_2026-08-31.md, the OLD TechJam metric
# `heldout_robust_tpr_at_1pct` on a 20k probe, a3 -> a7_norecon): mixed
# across backbones, +1.6 on crop dinov2regl (0.7858 -> 0.8022, which is this
# backbone and this canon policy), +14.5 on band eva02l, -5.4 on band
# dinov2regl, -4.3 on band dinov3l; 6 of 9 arms improved. It was never
# adopted because a7 sat outside `ELIGIBLE_RUNGS` -- a protocol decision,
# not a measurement. It has never been scored on robust ROC AUC, and never
# against the NTIRE distortion families. That is the experiment here.
#
# Go/no-go, written before the numbers:
#
#   PRIMARY      film beats abl_pool_mean (0.8040 robust / 0.9264 clean on
#                ntire_val_hard) by more than one robust point at no more
#                than half a clean point lost. Otherwise conditioning is
#                recorded as refuted on this proxy, as DID was.
#   ATTRIBUTION  film beats film_shuf by more than one robust point. If
#                PRIMARY passes and this does not, the gain is the head's
#                extra capacity and normalisation, and the result is
#                recorded as that -- NOT as degradation conditioning.
#
# Both must hold for FiLM to go into round three. No distil-weight-style
# sweep, no arm added after seeing a number: a post-hoc rescue is how a
# negative result becomes a false positive, which is the whole reason the
# DID gate was written down first.
#
#   bash scripts/ntire_film_ablation.sh
#   ARMS="film" bash scripts/ntire_film_ablation.sh
#   OUT_DIR=/scratch/$USER/aigcdet/outputs/ablation bash scripts/ntire_film_ablation.sh
#
# Idempotent: an arm with a metrics file is skipped. Each arm: train ->
# score ntire_val_hard + ntire_val (final weights) -> slim -> table.
set -euo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date "+%m-%d %H:%M:%S")] $*"; }

PY="${PY:-.venv/bin/python}"
EXP="${EXP:-/mnt/r13justinssd/techjam/experiments/data}"
MANIFEST="${MANIFEST:-data/manifest_ntire.parquet}"
BANK="${BANK:-data/banks/ntire_meta}"
ARMS="${ARMS:-film film_shuf}"
BACKBONE="${BACKBONE:-dinov2regl224}"
SUBSAMPLE="${SUBSAMPLE:-0.25}"
EPOCHS="${EPOCHS:-1}"
CHUNK="${CHUNK:-8}"
WORKERS="${WORKERS:-14}"
SEED="${SEED:-20260827}"
OUT_DIR="${OUT_DIR:-outputs/ablation}"
SCORES="${SCORES:-outputs/scores/ablation}"
TABLE="${TABLE:-docs/ntire/film_ablation.md}"
BASELINE="${BASELINE:-abl_pool_mean}"
NEED_MIB="${NEED_MIB:-12000}"
MIN_FREE_GB="${MIN_FREE_GB:-12}"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$OUT_DIR" "$SCORES" logs

# The extra flags per arm. Everything else is the proxy's fixed setting, and
# the a3 terms stay ON in both arms -- unlike the DID arms, nothing here
# replaces them, so `abl_pool_mean` is an exact minus-one-change baseline.
flags_for() {
  case "$1" in
    film)      echo "--film" ;;
    film_shuf) echo "--film --film-shuffle" ;;
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
  "$PY" scripts/metrics_table.py "$SCORES" --pattern 'abl_film_*_metrics.json' \
    --pattern "${BASELINE}_metrics.json" --out "$TABLE" > /dev/null
}

for arm in $ARMS; do
  name="abl_film_${arm}"
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
  if [ "$ck" = "$OUT_DIR/$name/checkpoint.pt" ]; then
    "$PY" scripts/slim_ckpt.py "$ck" && rm -f "$ck" "$OUT_DIR/$name"/checkpoint_ep*.pt
  fi
  table
done
log "FILM_ABLATION_DONE"
table
cat "$TABLE"
