#!/bin/bash
# Kill the NTIRE chain if free space on the corpus volume gets dangerous.
#
# The shard pipeline needs a ~40 GB transient (zip + its extraction) and the
# volume is at 98%. Other writers are live on the same disk: the ablation
# ladder checkpoints, the fusion run, and the Open Images harvest. A full
# volume would fail all of them, not just the download -- and a half-extracted
# shard is worse than no shard, because nothing downstream can tell.
set -u
TARGET_PID="$1"
FLOOR_GB="${2:-12}"
while kill -0 "$TARGET_PID" 2>/dev/null; do
  FREE=$(df -BG --output=avail /mnt/berstorage | tail -1 | tr -dc '0-9')
  if [ "$FREE" -lt "$FLOOR_GB" ]; then
    echo "[$(date +%H:%M:%S)] FREE=${FREE}G below ${FLOOR_GB}G floor -- killing NTIRE chain $TARGET_PID"
    pkill -P "$TARGET_PID" 2>/dev/null
    kill "$TARGET_PID" 2>/dev/null
    pkill -f 'acquire_ntire_shard' 2>/dev/null
    echo "[$(date +%H:%M:%S)] killed. Partial shard may need deleting before a retry."
    exit 1
  fi
  sleep 20
done
echo "[$(date +%H:%M:%S)] NTIRE chain exited on its own; guard standing down. free=${FREE}G"
