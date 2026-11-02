#!/bin/bash
# Exploratory, not registered (decided 2026-10-15, after the amendment 4
# results): the test-time probe of amendment 4, run on the DINOv2 G1 crop
# proxies, so the CLIP trained-vs-test-time comparison has a matched DINOv2
# counterpart. Hard-val only; ntire_test is never touched.
#
#   bash scripts/dinov2_probe.sh    # ~3 x 10 min of GPU
set -uo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date "+%m-%d %H:%M:%S")] $*"; }

PY="${PY:-.venv/bin/python}"
RUNS=/scratch/justin_r13/aigcdet/outputs/paper
SC=outputs/scores/paper
DOCS=docs/ntire/paper
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p logs

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  log "REFUSING: tracked files are modified; commit first"; exit 1
fi
log "dinov2 probe at $(git rev-parse HEAD)"

wait_gpu() {
  local need="$1" seen=0 free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || true)
    if [ -z "$free" ]; then sleep 300; continue; fi
    if [ "$free" -ge "$need" ]; then
      seen=$((seen + 1)); [ "$seen" -ge 2 ] && { log "GPU has ${free} MiB free"; return; }
    else
      [ "$seen" -gt 0 ] || [ $((SECONDS % 1800)) -lt 60 ] && log "waiting for GPU: ${free} MiB free, need $need"
      seen=0
    fi
    sleep 60
  done
}

for i in 1 2 3; do
  pre="$SC/abl_probe_s$i"
  [ -f "${pre}_center_ntire_val_hard.parquet" ] && [ -f "${pre}_squish_ntire_val_hard.parquet" ] && continue
  for attempt in 1 2 3 4 5 6; do
    wait_gpu 9000
    "$PY" -u scripts/ntire_crop_probe.py --ckpt "$RUNS/abl_crop_s$i/checkpoint_slim.pt" \
      --arms center,squish --split ntire_val_hard --out "$DOCS/abl_probe_s$i.json" \
      --parquet-prefix "$pre" --batch 32 --chunk 8 --workers 12 2>&1 \
      | grep -av "it/s\]" | tee -a logs/dinov2_probe_detail.log
    [ "${PIPESTATUS[0]}" -eq 0 ] && break
    log "abl s$i: attempt $attempt failed"
    [ "$attempt" -eq 6 ] && { log "DINOV2_PROBE_FAILED at s$i"; exit 1; }
    sleep 600
  done
done

"$PY" scripts/paper_results.py testtime --prefix abl \
  --label "Exploratory, not registered" --out "$DOCS/abl_testtime" | head -8
for i in 1 2 3; do
  "$PY" scripts/paired_bootstrap.py \
    --model testtime="$SC/abl_probe_s${i}_squish_ntire_val_hard.parquet" \
    --model trained="$SC/abl_squish_s${i}_ntire_val_hard.parquet" --baseline testtime \
    --title "dinov2 s$i: squish-trained vs crop-trained under test-time squish (exploratory, not registered)" \
    --out "$DOCS/abl_trained_vs_testtime_s$i.json" > /dev/null || { log "DINOV2_PROBE_FAILED at bootstrap"; exit 1; }
done
log "DINOV2_PROBE_DONE"
