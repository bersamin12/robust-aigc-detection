#!/bin/bash
# Round two, unattended: both branches retrained on the `ntire2` recipe pool
# (NTIRE's permanent downscale + a learned codec for JPEG AI, the two
# distortions the round-one epoch curve ranked weakest on ntire_val_hard),
# under --crop-clamp, then scored (final AND SWA weights), the better of
# each picked on the selection splits, and the fusion fitted.
#
#   bash scripts/ntire_v2.sh                 # waits for the ablation + the bank
#   WAIT_FOR=none bash scripts/ntire_v2.sh   # right now
#   OUT_DIR=/scratch/$USER/aigcdet/outputs bash scripts/ntire_v2.sh   # checkpoints elsewhere
#
# Checkpoints go under $OUT_DIR/{dual,unfreeze}/<name> (default outputs/, the
# repo disk); scores, logs and the fusion stay in the repo. The free-space
# check measures $OUT_DIR, which is where the ~22 GB of fat checkpoints land.
#
# Standalone on purpose: ntire_one_gpu.sh / ntire_pipeline.sh are still being
# read by the round-one run, and bash reads a script incrementally, so they
# are not edited while it lives. `ntire_test` is NOT read here.
set -euo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date "+%m-%d %H:%M:%S")] $*"; }

PY="${PY:-.venv/bin/python}"
EXP="${EXP:-/mnt/r13justinssd/techjam/experiments/data}"
MANIFEST="${MANIFEST:-data/manifest_ntire.parquet}"
BANK="${BANK:-data/banks/ntire2_meta}"
BANK_LOG="${BANK_LOG:-logs/ntire2_bank.log}"
WAIT_FOR="${WAIT_FOR:-logs/pool_ablation.log}"     # POOL_ABLATION_DONE
NAME_A="${NAME_A:-ntire2_d24_518}"
NAME_B="${NAME_B:-ntire2_dual224}"
A_EPOCHS="${A_EPOCHS:-2}"
B_EPOCHS="${B_EPOCHS:-3}"
A_CHUNK="${A_CHUNK:-2}"
DUAL_CHUNK="${DUAL_CHUNK:-8}"
TOKEN_POOL="${TOKEN_POOL:-mean}"
WORKERS="${WORKERS:-14}"
SEED="${SEED:-20260827}"
SCORES="${SCORES:-outputs/scores}"
OUT_DIR="${OUT_DIR:-outputs}"
FIT_SPLITS="${FIT_SPLITS:-ntire_val,ntire_val_hard}"
MIN_FREE_GB="${MIN_FREE_GB:-22}"
SCORE_MIB="${SCORE_MIB:-8000}"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p logs "$SCORES" "$OUT_DIR/dual/$NAME_B" "$OUT_DIR/unfreeze/$NAME_A"

if [ "$WAIT_FOR" != none ]; then
  log "waiting for POOL_ABLATION_DONE in $WAIT_FOR and BANK2_DONE in $BANK_LOG"
  while ! grep -q POOL_ABLATION_DONE "$WAIT_FOR" 2>/dev/null || ! grep -q BANK2_DONE "$BANK_LOG" 2>/dev/null; do
    if grep -q Traceback "$BANK_LOG" 2>/dev/null; then log "bank build failed; see $BANK_LOG"; exit 1; fi
    sleep 300
  done
fi
[ -f "$BANK/config.json" ] || { log "no bank at $BANK"; exit 1; }
grep -q '"recipe_pool": "ntire2"' "$BANK/config.json" || { log "$BANK is not an ntire2 bank"; exit 1; }
"$PY" -c "import aigcdet.augment.ops as o, sys; sys.exit(0 if o.jpeg_ai_available() else 1)" \
  || { log "compressai missing: uv pip install compressai"; exit 1; }
free_gb=$(df --output=avail -BG "$OUT_DIR" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge "$MIN_FREE_GB" ] || { log "only ${free_gb} GB free, need $MIN_FREE_GB"; exit 1; }
log "prerequisites met (${free_gb} GB free on $OUT_DIR)"

nohup bash scripts/slim_watch.sh "$OUT_DIR/dual/$NAME_B" "$OUT_DIR/unfreeze/$NAME_A" > logs/slim_watch_v2.log 2>&1 &
SLIM_PID=$!
# PIPELINE_LOG=/dev/null: the watcher would otherwise exit on round one's
# PIPELINE_DONE before the first snapshot lands; it is killed below instead.
PIPELINE_LOG=/dev/null nohup bash scripts/epoch_watch.sh "$OUT_DIR/dual/$NAME_B:16" "$OUT_DIR/unfreeze/$NAME_A:4" > logs/epoch_watch_v2.log 2>&1 &
WATCH_PID=$!
log "slim watcher $SLIM_PID, epoch watcher $WATCH_PID"

wait_gpu() {
  local need_mib="$1" seen=0 free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge "$need_mib" ]; then
      seen=$((seen + 1)); [ "$seen" -ge 2 ] && { log "GPU has ${free} MiB free"; return; }
    else
      seen=0; log "waiting for GPU: ${free} MiB free, need $need_mib"
    fi
    sleep 60
  done
}
complete() {   # <ckpt> <epochs>
  [ -f "$1" ] && "$PY" -c "import torch,sys; ck=torch.load('$1',map_location='cpu',weights_only=False,mmap=True); sys.exit(0 if ck['epoch']>=$2 else 1)"
}
common=(--bank "$BANK" --root "$EXP" --depth 24 --n-src 64 --m-deg 2
        --workers "$WORKERS" --seed "$SEED" --device cuda
        --canon-mode crop --crop-side 200 --crop-clamp
        --lr-schedule cosine --warmup-frac 0.03 --min-lr-frac 0.01
        --swa --swa-start-frac 0.75 --family-weighting proportional
        --token-pool "$TOKEN_POOL" --resume)
done_B() { complete "$OUT_DIR/dual/$NAME_B/checkpoint.pt" "$B_EPOCHS"; }
done_A() { complete "$OUT_DIR/unfreeze/$NAME_A/checkpoint.pt" "$A_EPOCHS"; }
train_B() {
  log "B: dual dinov2regl224 on $BANK, $B_EPOCHS epochs -> logs/$NAME_B.log"
  "$PY" -u scripts/train_dual.py "${common[@]}" --backbone dinov2regl224 --name "$NAME_B" \
    --out-dir "$OUT_DIR/dual" --perturb-tower2 0.0 --epochs "$B_EPOCHS" --src-chunk "$DUAL_CHUNK" \
    --nominal-side 224 2>&1 | tee -a "logs/$NAME_B.log"
}
train_A() {
  log "A: dinov2regl @518 on $BANK, $A_EPOCHS epochs -> logs/$NAME_A.log"
  "$PY" -u scripts/train_unfreeze.py "${common[@]}" --backbone dinov2regl --name "$NAME_A" \
    --out-dir "$OUT_DIR/unfreeze" --epochs "$A_EPOCHS" --src-chunk "$A_CHUNK" 2>&1 | tee -a "logs/$NAME_A.log"
}
run_stage() {   # <fn> <need_mib> <done-fn>
  local attempt
  # Ask first, wait second: a resumed run whose training is already finished
  # would otherwise sit in wait_gpu behind another user for a card it never uses.
  "$3" && { log "$1: already complete"; return 0; }
  for attempt in 1 2 3 4 5 6; do
    wait_gpu "$2"
    if "$1"; then return 0; fi
    log "$1 failed (attempt $attempt); pausing 10 min"; sleep 600
  done
  log "$1 failed 6 times"; exit 1
}
score() {   # <ckpt> <prefix> <splits> <chunk> [extra...]
  local ckpt="$1" prefix="$2" splits="$3" chunk="$4"; shift 4
  for sp in ${splits//,/ }; do
    [ -f "${prefix}_${sp}.parquet" ] && { log "score: ${prefix}_${sp}.parquet exists"; continue; }
    wait_gpu "$SCORE_MIB"
    "$PY" scripts/score_plan_splits.py --ckpt "$ckpt" --manifest "$MANIFEST" --splits "$sp" \
      --crop-clamp --batch 32 --chunk "$chunk" --workers 12 --out-prefix "$prefix" "$@" \
      2>&1 | tee -a "logs/score_$(basename "$prefix").log"
  done
}
pick() {   # <prefix-final> <prefix-swa>: the better objective over the fit splits
  "$PY" - "$1" "$2" "$FIT_SPLITS" <<'PYEOF'
import json, sys
def obj(prefix, splits):
    m = json.load(open(f"{prefix}_metrics.json"))["splits"]
    return sum((m[s]["robust_auc"] + m[s]["clean_auc"]) / 2 for s in splits) / len(splits)
a, b, splits = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
oa, ob = obj(a, splits), obj(b, splits)
print(a if oa >= ob else b)
print(f"  {a}: {oa:.4f}  {b}: {ob:.4f}", file=sys.stderr)
PYEOF
}

run_stage train_B 17000 done_B
run_stage train_A 13000 done_A
CB=$OUT_DIR/dual/$NAME_B/checkpoint.pt; CA=$OUT_DIR/unfreeze/$NAME_A/checkpoint.pt
score "$CB" "$SCORES/B2" "val,$FIT_SPLITS" 16
score "$CB" "$SCORES/B2swa" "$FIT_SPLITS" 16 --swa
score "$CA" "$SCORES/A2" "val,$FIT_SPLITS" 8
score "$CA" "$SCORES/A2swa" "$FIT_SPLITS" 8 --swa
PB=$(pick "$SCORES/B2" "$SCORES/B2swa"); PA=$(pick "$SCORES/A2" "$SCORES/A2swa")
log "fusing $PA + $PB"
"$PY" scripts/fuse_branches.py --a "$PA" --b "$PB" --fit-splits "$FIT_SPLITS" \
  $( [ "$PA" = "$SCORES/A2" ] && [ "$PB" = "$SCORES/B2" ] && echo "--eval-splits val" ) \
  --out docs/ntire/fusion_ntire2.json
"$PY" scripts/metrics_table.py "$SCORES" --pattern '[AB]2*_metrics.json' --out docs/ntire/round2_scores.md > /dev/null
kill "$SLIM_PID" "$WATCH_PID" 2>/dev/null || true
log "NTIRE_V2_DONE"
