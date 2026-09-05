#!/bin/bash
# The early number. The pipeline scores only when both branches are done,
# ~30 h after the first epoch lands; this scores every per-epoch snapshot on
# the organisers' hard set as soon as it appears, beside the running trainer.
#
#   bash scripts/epoch_watch.sh outputs/dual/ntire_dual224:16 outputs/unfreeze/ntire_d24_518:4
#
# Each argument is <run dir>[:<chunk>]. A snapshot `checkpoint_ep<N>.pt` (or
# its `_slim` twin) is scored once, to $OUT/<run>_ep<N>_<split>.parquet and
# `_metrics.json`, and $OUT/summary.md is regenerated after each. The card is
# shared: a score waits for NEED_MIB free and uses few workers so the trainer
# keeps its CPU. Exits once the pipeline log says PIPELINE_DONE and nothing
# is left to score.
set -u
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
MANIFEST="${MANIFEST:-data/manifest_ntire.parquet}"
SPLITS="${SPLITS:-ntire_val_hard}"
OUT="${OUT:-outputs/scores/epochs}"
NEED_MIB="${NEED_MIB:-4500}"
POLL="${POLL:-120}"
WORKERS="${WORKERS:-6}"
PIPELINE_LOG="${PIPELINE_LOG:-logs/pipeline.log}"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$OUT" logs
log() { echo "[$(date "+%m-%d %H:%M:%S")] $*"; }

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

score_one() {
  local f="$1" name="$2" ep="$3" chunk="$4" prefix="$OUT/${name}_ep${ep}"
  # Settled: same size 20 s apart (the trainer links a finished file, but
  # a slim twin is written beside it and may still be growing).
  local s1 s2; s1=$(stat -c %s "$f"); sleep 20; s2=$(stat -c %s "$f")
  [ "$s1" = "$s2" ] || { log "$f still growing; later"; return 1; }
  local free; free=$(free_mib)
  [ "$free" -ge "$NEED_MIB" ] || { log "GPU has ${free} MiB free, need $NEED_MIB; later"; return 1; }
  log "scoring $f -> $prefix ($SPLITS, chunk $chunk)"
  if "$PY" scripts/score_plan_splits.py --ckpt "$f" --manifest "$MANIFEST" \
       --splits "$SPLITS" --crop-clamp --batch 16 --chunk "$chunk" \
       --workers "$WORKERS" --out-prefix "$prefix" >> "logs/epoch_watch_${name}.log" 2>&1; then
    "$PY" scripts/metrics_table.py "$OUT" --out "$OUT/summary.md" > /dev/null
    log "$(grep -E "^${SPLITS%%,*}:" "logs/epoch_watch_${name}.log" | tail -1)"
  else
    log "scoring $f FAILED; see logs/epoch_watch_${name}.log"
    rm -f "${prefix}_metrics.json"
    return 1
  fi
}

while true; do
  found=0
  for arg in "$@"; do
    dir="${arg%%:*}"; chunk="${arg#*:}"; [ "$chunk" = "$arg" ] && chunk=8
    name=$(basename "$dir")
    for f in "$dir"/checkpoint_ep*.pt; do
      [ -e "$f" ] || continue
      ep=$(basename "$f" | sed -E 's/^checkpoint_ep([0-9]+)(_slim)?\.pt$/\1/')
      [ -n "$ep" ] || continue
      [ -f "$OUT/${name}_ep${ep}_metrics.json" ] && continue
      # Prefer the slim twin when both exist; skip the fat one then.
      case "$f" in
        *_slim.pt) ;;
        *) [ -e "${f%.pt}_slim.pt" ] && continue ;;
      esac
      found=1
      score_one "$f" "$name" "$ep" "$chunk" || true
    done
  done
  if [ "$found" = 0 ] && grep -q PIPELINE_DONE "$PIPELINE_LOG" 2>/dev/null; then
    log "pipeline done and nothing left to score; exiting"; exit 0
  fi
  sleep "$POLL"
done
