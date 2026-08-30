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
DEST="/mnt/berstorage/techjam/ntire"
SHARD="$1"

cd "$DEST"
echo "[$(date +%H:%M:%S)] downloading shard_${SHARD}.zip"
hf download "$REPO" "shard_${SHARD}.zip" --repo-type dataset --local-dir "$DEST"
echo "[$(date +%H:%M:%S)] extracting"
unzip -q -o "shard_${SHARD}.zip" -d "$DEST/extracted"
echo "[$(date +%H:%M:%S)] removing the zip"
rm -f "shard_${SHARD}.zip"
df -h "$DEST" | tail -1
echo "[$(date +%H:%M:%S)] shard ${SHARD} done"
