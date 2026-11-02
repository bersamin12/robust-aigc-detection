#!/bin/bash
# Keep a long fine-tune's disk footprint bounded: whenever a per-epoch
# snapshot `checkpoint_ep<N>.pt` appears, replace it with its weights-only
# twin (`scripts/slim_ckpt.py`) and delete the fat one. The live
# `checkpoint.pt` is never touched, so `--resume` keeps working.
#
#   bash scripts/slim_watch.sh outputs/dual/ntire_dual224 outputs/unfreeze/ntire_d24_518 &
#
# A snapshot is only slimmed once it is a DIFFERENT inode from checkpoint.pt
# (the trainer hard-links the snapshot, then rewrites checkpoint.pt via a
# temp file at the next epoch), and only once it has stopped growing.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
export PYTHONPATH="$PWD/src"
log() { echo "[$(date +%H:%M:%S)] $*"; }
while true; do
  for dir in "$@"; do
    for f in "$dir"/checkpoint_ep*.pt; do
      [ -e "$f" ] || continue
      case "$f" in *_slim.pt) continue;; esac
      live="$dir/checkpoint.pt"
      if [ -e "$live" ] && [ "$f" -ef "$live" ]; then continue; fi
      s1=$(stat -c %s "$f"); sleep 20; s2=$(stat -c %s "$f")
      [ "$s1" = "$s2" ] || continue
      slim="${f%.pt}_slim.pt"
      if [ -e "$slim" ]; then rm -f "$f"; log "removed fat $f (slim exists)"; continue; fi
      log "slimming $f"
      if "$PY" scripts/slim_ckpt.py "$f"; then rm -f "$f"; log "removed fat $f"; fi
    done
  done
  sleep 120
done
