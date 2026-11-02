#!/usr/bin/env bash
# Amendment 2 (docs/ntire/paper_prereg.md): aspect-preserving whole-image arms
# on the frozen round-two checkpoints, then paired bootstraps against center
# and squish. Fit splits only; ntire_test is never touched.
set -euo pipefail
cd /mnt/r13justinssd/techjam/aigcdet
export PYTHONPATH=src
PY=.venv/bin/python
OUT=docs/ntire/paper
SC=outputs/scores/paper
A2=/scratch/justin_r13/aigcdet/outputs/unfreeze/ntire2_d24_518/checkpoint.pt
B2=/scratch/justin_r13/aigcdet/outputs/dual/ntire2_dual224/checkpoint.pt
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }
wait_gpu() {  # two clean readings 60 s apart
  local need=$1 a b
  while true; do
    a=$(free_mib || echo 0); [ -n "$a" ] || a=0
    if [ "$a" -ge "$need" ]; then
      sleep 60; b=$(free_mib || echo 0); [ -n "$b" ] || b=0
      [ "$b" -ge "$need" ] && { log "GPU has $b MiB free"; return; }
    fi
    sleep 120
  done
}

[ -z "$(git status --porcelain -- scripts docs/ntire/paper_prereg.md)" ] || { log "REFUSING: dirty tree"; exit 1; }
COMMIT=$(git rev-parse HEAD)
log "aspect probe at $COMMIT"

run() {  # tag ckpt swa batch split
  local tag=$1 ck=$2 swa=$3 batch=$4 split=$5
  local js=$OUT/aspect_probe_${tag}_${split}.json
  if [ -f "$js" ] && grep -q '"letterbox"' "$js"; then log "skip $js"; return; fi
  for try in 1 2 3 4 5 6; do
    wait_gpu 12000
    if $PY scripts/ntire_crop_probe.py --ckpt "$ck" $swa --split "$split" \
        --arms short_center,letterbox --batch "$batch" \
        --out "$js" --parquet-prefix "$SC/aspect_${tag}"; then return; fi
    log "attempt $try failed for $tag $split"
  done
  log "ASPECT_PROBE_FAILED $tag $split"; exit 1
}

run A2 "$A2" "" 4 ntire_val_hard
run B2swa "$B2" --swa 8 ntire_val_hard
run A2 "$A2" "" 4 ntire_val
run B2swa "$B2" --swa 8 ntire_val

for tag in A2 B2swa; do
  for base in center squish; do
    $PY scripts/paired_bootstrap.py \
      --model center=outputs/scores/probe_${tag}_center_ntire_val_hard.parquet \
      --model squish=outputs/scores/probe_${tag}_squish_ntire_val_hard.parquet \
      --model short_center=$SC/aspect_${tag}_short_center_ntire_val_hard.parquet \
      --model letterbox=$SC/aspect_${tag}_letterbox_ntire_val_hard.parquet \
      --baseline $base --title "$tag, ntire_val_hard, against $base" \
      --out $OUT/aspect_boot_${tag}_vs_${base}.json > /dev/null
  done
done
echo "$COMMIT" > $OUT/aspect_probe.done
log "ASPECT_PROBE_DONE"
