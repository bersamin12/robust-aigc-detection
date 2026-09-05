#!/bin/bash
# The whole NTIRE run, unattended, on one 4090:
#   wait for the smoke chain and the full metadata bank -> drop the smoke
#   artefacts -> start the checkpoint slimmer -> B (dual @224) -> A (single
#   @518) -> score both on the selection splits and fit the fusion.
# `ntire_test` is NOT read here; that is STAGE=test of ntire_one_gpu.sh, run
# by hand once, after the numbers on the selection splits are what they are.
set -euo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date "+%m-%d %H:%M:%S")] $*"; }

SMOKE_LOG="${SMOKE_LOG:-logs/smoke_chain.log}"
BANK_LOG="${BANK_LOG:-logs/full_manifest_bank.log}"
export MANIFEST="${MANIFEST:-data/manifest_ntire.parquet}"
export BANK="${BANK:-data/banks/ntire_meta}"

log "waiting for the smoke chain ($SMOKE_LOG) and the full bank ($BANK_LOG)"
while ! grep -q SMOKE_CHAIN_DONE "$SMOKE_LOG" || ! grep -q FULL_BANK_DONE "$BANK_LOG"; do
  if grep -q "Traceback" "$SMOKE_LOG" "$BANK_LOG"; then
    log "a prerequisite failed; see $SMOKE_LOG / $BANK_LOG"; exit 1
  fi
  sleep 60
done
[ -f docs/ntire/fusion_smoke.json ] || { log "smoke fusion json missing; not proceeding"; exit 1; }
log "prerequisites met"

if [ "${KEEP_SMOKE:-0}" != 1 ]; then
  log "removing smoke artefacts"
  rm -rf outputs/dual/smoke_dual224 outputs/unfreeze/smoke_d24_518 data/banks/ntire_meta_smoke
  rm -f outputs/scores/smokeA_* outputs/scores/smokeB_*
fi
df -h "$PWD" | tail -1

mkdir -p outputs/dual/"${NAME_B:-ntire_dual224}" outputs/unfreeze/"${NAME_A:-ntire_d24_518}"
nohup bash scripts/slim_watch.sh outputs/dual/"${NAME_B:-ntire_dual224}" \
      outputs/unfreeze/"${NAME_A:-ntire_d24_518}" > logs/slim_watch.log 2>&1 &
SLIM_PID=$!
log "slim watcher pid $SLIM_PID"

# The card is SHARED. Other users' jobs come and go, and a launch into a
# card with 4 GB free dies at model load. Wait for enough free memory, seen
# twice a minute apart, and retry a stage that still OOMs after a pause.
wait_gpu() {
  local need_mib="$1" seen=0
  while true; do
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge "$need_mib" ]; then
      seen=$((seen + 1)); [ "$seen" -ge 2 ] && { log "GPU has ${free} MiB free; proceeding"; return; }
    else
      seen=0
      log "waiting for GPU: ${free} MiB free, need ${need_mib} (other users' jobs: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';'))"
    fi
    sleep 60
  done
}
run_stage() {
  local stage="$1" need_mib="$2" attempt
  for attempt in 1 2 3 4 5 6; do
    wait_gpu "$need_mib"
    if STAGE="$stage" bash scripts/ntire_one_gpu.sh; then return 0; fi
    log "stage $stage failed (attempt $attempt); pausing 10 min before retrying"
    sleep 600
  done
  log "stage $stage failed 6 times; giving up"; exit 1
}

run_stage B 17000
log "B done; slimming its live checkpoint's optimiser state is left for after A"
run_stage A 13000
log "A done"
run_stage score 8000
kill "$SLIM_PID" 2>/dev/null || true
log "PIPELINE_DONE"
