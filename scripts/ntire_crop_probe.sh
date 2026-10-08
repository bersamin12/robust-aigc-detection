#!/usr/bin/env bash
# Test-time crop probe on the two frozen round-two checkpoints. Reads
# ntire_val_hard only; the gate lives in scripts/ntire_crop_probe.py's header
# and was registered before any arm had a number.
set -u
cd /mnt/r13justinssd/techjam/aigcdet
export PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
log() { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

log "branch B (dual224, swa)"
$PY -u scripts/ntire_crop_probe.py \
  --ckpt /scratch/justin_r13/aigcdet/outputs/dual/ntire2_dual224/checkpoint.pt --swa \
  --out docs/ntire/crop_probe_B2swa.json \
  --parquet-prefix outputs/scores/probe_B2swa \
  --batch 8 --chunk 16 --workers 12 || { log "B FAILED"; exit 1; }

log "branch A (d24_518, final)"
$PY -u scripts/ntire_crop_probe.py \
  --ckpt /scratch/justin_r13/aigcdet/outputs/unfreeze/ntire2_d24_518/checkpoint.pt \
  --out docs/ntire/crop_probe_A2.json \
  --parquet-prefix outputs/scores/probe_A2 \
  --batch 4 --chunk 8 --workers 12 || { log "A FAILED"; exit 1; }

log "CROP_PROBE_DONE"
