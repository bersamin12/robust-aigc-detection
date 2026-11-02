#!/usr/bin/env bash
# Amendment 3 (docs/ntire/paper_prereg.md): UnivFD and B-Free under native,
# train_fov and squish presentation. Runs after the amendment-2 aspect probe,
# never touches ntire_test.
set -euo pipefail
cd /mnt/r13justinssd/techjam/aigcdet
export PYTHONPATH=src
PY=.venv/bin/python
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
log "public detector probe at $COMMIT"

while pgrep -f "scripts/aspect_probe.sh" > /dev/null; do sleep 300; done
log "aspect probe no longer running"

wait_gpu 8000
$PY scripts/public_detector_probe.py --selfcheck --device cuda \
  || { log "PUBLIC_PROBE_FAILED selfcheck"; exit 1; }

for det in univfd bfree; do
  ok=0
  for try in 1 2 3 4 5 6; do
    wait_gpu 8000
    if $PY scripts/public_detector_probe.py --detector "$det"; then ok=1; break; fi
    log "attempt $try failed for $det"
  done
  [ "$ok" = 1 ] || { log "PUBLIC_PROBE_FAILED $det"; exit 1; }
done

$PY scripts/public_detector_probe.py --summarise docs/ntire/paper/public_detectors.md > /dev/null
echo "$COMMIT" > docs/ntire/paper/public_probe.done
log "PUBLIC_PROBE_DONE"
