#!/bin/bash
# Unattended: for each NTIRE train shard named, download -> unzip -> delete
# zip -> normalise into the union tree -> delete the raw extraction. One shard
# at a time, so the transient never exceeds one zip plus one extraction
# (~40 GB) on top of the ~21 GB each normalised shard keeps.
#
#   SHARDS="4 5" bash scripts/ntire_shards_chain.sh
#
# A shard whose sidecar parquet already exists is skipped; a zip that is
# already (partly) present is resumed by curl -C -.
set -euo pipefail
cd "$(dirname "$0")/.."
EXP="${EXP:-/mnt/r13justinssd/techjam/experiments/data}"
PY="${PY:-.venv/bin/python}"
RAW="${RAW:-$EXP/raw_ntire_ext}"
REPO_HF="deepfakesMSU/NTIRE-RobustAIGenDetection-train"
FLOOR_GB="${FLOOR_GB:-25}"
export PYTHONPATH="$PWD/src"

log() { echo "[$(date +%H:%M:%S)] $*"; }
free_gb() { df -BG --output=avail "$EXP" | tail -1 | tr -dc '0-9'; }

for k in ${SHARDS:-4 5}; do
  if [ -f "$EXP/ntire_shard${k}_rows.parquet" ]; then
    log "shard $k: sidecar exists, skipping"; continue
  fi
  if [ ! -d "$RAW/extracted/shard_$k/images" ]; then
    if [ "$(free_gb)" -lt $((FLOOR_GB + 40)) ]; then
      log "shard $k: only $(free_gb) GB free, need ~40 GB transient above the ${FLOOR_GB} GB floor"; exit 1
    fi
    mkdir -p "$RAW"
    log "shard $k: downloading"
    ( cd "$RAW" && curl -sSL -C - -o "shard_$k.zip" \
        "https://huggingface.co/datasets/$REPO_HF/resolve/main/shard_$k.zip" )
    log "shard $k: extracting"
    unzip -q -o "$RAW/shard_$k.zip" -d "$RAW/extracted"
    rm -f "$RAW/shard_$k.zip"
  fi
  log "shard $k: normalising ($(free_gb) GB free)"
  "$PY" scripts/normalize_ntire_shards.py --src "$RAW/extracted" --shard "$k" \
      --dest "$EXP/normalized_union" --sidecar-dir "$EXP" --skipped-dir docs/ntire \
      --workers "${WORKERS:-12}"
  rm -rf "$RAW/extracted/shard_$k"
  log "shard $k: done ($(free_gb) GB free)"
done
log "all shards done"
