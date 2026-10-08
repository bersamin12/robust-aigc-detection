#!/usr/bin/env bash
# Round two of the crop probe: the bandwidth control, and the fusion refit.
#   band_squish  full field of view with the bandwidth ceiling restored at
#                160, so native resolution is no longer in the pixels. Asks
#                whether the gain can be kept while the residual cue is
#                closed -- which decides the canon policy a retrain uses.
#   squish on ntire_val  so the fusion z-scores and weight can be fitted on
#                ntire_val ALONE and read out on ntire_val_hard, leaving
#                hard-val a split nothing was fitted on. ntire_test is NOT
#                touched: it was read once on 2026-10-04 and is sealed.
set -u
cd /mnt/r13justinssd/techjam/aigcdet
export PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
log() { echo "[$(date +%m-%d\ %H:%M:%S)] $*"; }

A=/scratch/justin_r13/aigcdet/outputs/unfreeze/ntire2_d24_518/checkpoint.pt
B=/scratch/justin_r13/aigcdet/outputs/dual/ntire2_dual224/checkpoint.pt

log "band_squish control, branch B"
$PY -u scripts/ntire_crop_probe.py --ckpt "$B" --swa --arms band_squish \
  --band-side 160 --out docs/ntire/crop_probe_B2swa_band.json \
  --parquet-prefix outputs/scores/probe_B2swa \
  --batch 32 --chunk 16 --workers 12 2>&1 | grep -av "it/s\]" || exit 1
log "band_squish control, branch A"
$PY -u scripts/ntire_crop_probe.py --ckpt "$A" --arms band_squish \
  --band-side 160 --out docs/ntire/crop_probe_A2_band.json \
  --parquet-prefix outputs/scores/probe_A2 \
  --batch 32 --chunk 8 --workers 12 2>&1 | grep -av "it/s\]" || exit 1

log "squish on ntire_val, branch B"
$PY -u scripts/ntire_crop_probe.py --ckpt "$B" --swa --arms squish \
  --split ntire_val --out docs/ntire/crop_probe_B2swa_val.json \
  --parquet-prefix outputs/scores/probe_B2swa \
  --batch 32 --chunk 16 --workers 12 2>&1 | grep -av "it/s\]" || exit 1
log "squish on ntire_val, branch A"
$PY -u scripts/ntire_crop_probe.py --ckpt "$A" --arms squish \
  --split ntire_val --out docs/ntire/crop_probe_A2_val.json \
  --parquet-prefix outputs/scores/probe_A2 \
  --batch 32 --chunk 8 --workers 12 2>&1 | grep -av "it/s\]" || exit 1

log "fusion, fitted on ntire_val alone, read out on ntire_val_hard"
log "-- centre window (the frozen policy)"
$PY -u scripts/fuse_branches.py --a outputs/scores/A2 --b outputs/scores/B2swa \
  --fit-splits ntire_val --eval-splits ntire_val_hard \
  --out docs/ntire/fusion_center_valfit.json || exit 1
log "-- squish"
$PY -u scripts/fuse_branches.py \
  --a outputs/scores/probe_A2_squish --b outputs/scores/probe_B2swa_squish \
  --fit-splits ntire_val --eval-splits ntire_val_hard \
  --out docs/ntire/fusion_squish_valfit.json || exit 1

log "PROBE2_DONE"
