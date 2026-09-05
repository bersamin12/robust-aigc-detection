#!/bin/bash
# One NTIRE shard at a time: download -> extract -> delete the zip.
#
# The full train split is 114.4 GB of zips and extracts to roughly the same
# again (the payload is JPEG, so the zip is not compressing much). /mnt/
# berstorage has ~92 GB free, so pulling all six shards is not an option and
# a partial pull that dies at shard 5 leaves no way to tell what is missing.
# Sequential-with-delete keeps peak usage at ~42 GB (one zip + its extraction)
# and leaves a complete, inspectable shard after every step.
set -euo pipefail
REPO="deepfakesMSU/NTIRE-RobustAIGenDetection-train"
# Overridable: the r13 box keeps the shards under experiments/data/raw_ntire_ext.
DEST="${DEST:-/mnt/berstorage/techjam/ntire}"
SHARD="$1"

mkdir -p "$DEST"
cd "$DEST"
if [ -d "$DEST/extracted/shard_${SHARD}/images" ]; then
  echo "[$(date +%H:%M:%S)] shard ${SHARD} already extracted; nothing to do"
  exit 0
fi
echo "[$(date +%H:%M:%S)] downloading shard_${SHARD}.zip"
# `hf` if the venv has it, else a resumable curl straight at the resolve URL --
# the repo is ungated and the file is a single blob either way.
if command -v hf >/dev/null 2>&1; then
  hf download "$REPO" "shard_${SHARD}.zip" --repo-type dataset --local-dir "$DEST"
else
  curl -sSL -C - -o "shard_${SHARD}.zip" \
    "https://huggingface.co/datasets/${REPO}/resolve/main/shard_${SHARD}.zip"
fi
echo "[$(date +%H:%M:%S)] extracting"
unzip -q -o "shard_${SHARD}.zip" -d "$DEST/extracted"
echo "[$(date +%H:%M:%S)] removing the zip"
rm -f "shard_${SHARD}.zip"
df -h "$DEST" | tail -1
echo "[$(date +%H:%M:%S)] shard ${SHARD} done"
