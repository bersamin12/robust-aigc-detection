#!/usr/bin/env bash
# The two siglipso400m arms, resumed after the eval-root fix of 2026-08-31.
#
# Both have a COMPLETE Stage A bank (368,358 rows, finite) and no eval bank:
# `extract_eval_bank` was resolving the eval manifest against the TRAIN root,
# which is one directory level too deep, so preflight reported all 12,276 shard
# rows missing and the arm died three seconds into a step that follows 6.5 h of
# extraction. `full_scale_arm2.sh` passes `--root $EVAL_ROOT` instead; nothing
# about Stage A changed, so both arms skip straight to the eval bank.
#
# WHY A WAITER AND NOT TWO LAUNCHES. `full_scale.sh` is concurrently building
# eval_full_crop_dinov2regl and will then start band_dinov2regl's Stage A
# (~6.5 h, 4 processes). The arm script's own MAX_CONCURRENT_EXTRACT guard is
# what refused the band arm at 09:22, correctly. Rather than defeat the guard
# with a large cap, wait for capacity and let the guard keep meaning something.
#
# `full_scale_arm.sh siglipso400m crop` (pid 7469) is still finishing Stage A
# under the OLD script text -- deliberately not patched in place, because bash
# re-reads a running script by byte offset. It will merge Stage A (the part
# worth having), then fail its eval step in three seconds exactly as before.
# That failure is expected and is this script's start signal, not an error.
set -uo pipefail
cd /workspace/robust-aigc-detection

log() { echo "[$(date +%H:%M:%S)] $*"; }
n_extract() { pgrep -f "extract_features\.py|extract_eval_bank\.py" | wc -l; }

# One shard per GPU: these arms are now eval-bank-only, and the eval bank is
# 982,020 forwards against Stage A's 368k rows x n_views -- a smaller job that
# does not need the 2-per-GPU packing Stage A was tuned for.
export SHARDS_PER_GPU=1 ALLOW_CONCURRENT=1 MAX_CONCURRENT_EXTRACT=12

wait_for_capacity() {   # $1 = headroom needed, $2 = label
  local need=$1 label=$2 waited=0
  while :; do
    local n; n=$(n_extract)
    if [ "$n" -le $((12 - need)) ]; then
      log "$label: $n extract process(es) running, $need free -- go"; return 0
    fi
    [ $((waited % 600)) -eq 0 ] && log "$label: waiting, $n extract process(es) running"
    sleep 60; waited=$((waited + 60))
    if [ "$waited" -ge 43200 ]; then log "$label: GAVE UP after 12 h"; return 1; fi
  done
}

for mode in crop band; do
  tag="full_${mode}_siglipso400m"
  if [ -f "data/banks/eval_$tag/config.json" ]; then
    log "$tag: eval bank already present, skipping arm"; continue
  fi
  # Stage A must be MERGED before the arm can skip it; for crop that means
  # waiting out pid 7469 rather than racing its merge_banks.
  while [ ! -f "data/banks/$tag/config.json" ]; do
    log "$tag: Stage A not merged yet, waiting"; sleep 120
  done
  wait_for_capacity 4 "$tag" || exit 1
  log "=== $tag: eval bank + ladder ==="
  bash scripts/full_scale_arm2.sh siglipso400m "$mode" \
    > "logs/${tag}_resume.log" 2>&1
  log "  $tag exit $?"
done
log "=== SIGLIP EVAL CHAIN DONE ==="
