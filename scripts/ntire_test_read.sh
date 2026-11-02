#!/bin/bash
# The ONE read of `ntire_test`, the sealed split.
#
# Everything about WHICH models are read comes out of the frozen fusion
# record (`docs/ntire/fusion_ntire2.json`, written 2026-09-23 by
# `ntire_v2.sh` once both branches were trained and the better weights of
# each were picked on the selection splits). The script takes the two
# prefixes, the two checkpoints and each branch's weights tag (final / swa)
# from that file and its metrics files, so it cannot read a model the
# selection did not choose. Nothing here fits anything: `fuse_branches.py`
# re-derives the z-scores and the weight from the FIT-split parquets, which
# already exist, and applies them unchanged to the test rows.
#
#   bash scripts/ntire_test_read.sh
#
# Refuses to run a second time: once `<prefix>_ntire_test.parquet` exists,
# the split has been read and re-reading it after seeing the number is the
# one thing the whole protocol exists to prevent. Deleting those parquets to
# get another read is cheating; if a re-read is ever genuinely needed, the
# honest move is to say so in the write-up.
set -euo pipefail
cd "$(dirname "$0")/.."
log() { echo "[$(date "+%m-%d %H:%M:%S")] $*"; }

PY="${PY:-.venv/bin/python}"
MANIFEST="${MANIFEST:-data/manifest_ntire.parquet}"
FUSION="${FUSION:-docs/ntire/fusion_ntire2.json}"
OUT="${OUT:-docs/ntire/fusion_ntire2_final.json}"
SPLIT="${SPLIT:-ntire_test}"
FIT_SPLITS="${FIT_SPLITS:-ntire_val,ntire_val_hard}"
A_CHUNK="${A_CHUNK:-8}"
B_CHUNK="${B_CHUNK:-16}"
NEED_MIB="${NEED_MIB:-9000}"
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p logs

[ -f "$FUSION" ] || { log "no frozen fusion record at $FUSION; run ntire_v2.sh first"; exit 1; }

# (prefix, ckpt, weights-tag) per branch, straight out of the frozen record.
read -r A_PREFIX A_CKPT A_TAG B_PREFIX B_CKPT B_TAG < <("$PY" - "$FUSION" <<'PYEOF'
import json, sys
rec = json.load(open(sys.argv[1]))
out = []
for side in ("branch_a", "branch_b"):
    b = rec[side]
    tag = json.load(open(f"{b['prefix']}_metrics.json"))["weights"]
    if tag not in ("final", "swa", "ema"):
        sys.exit(f"unexpected weights tag {tag!r} for {side}")
    out += [b["prefix"], b["ckpt"], tag]
print(" ".join(out))
PYEOF
)
log "frozen A: $A_PREFIX ($A_TAG) $A_CKPT"
log "frozen B: $B_PREFIX ($B_TAG) $B_CKPT"
for f in "$A_CKPT" "$B_CKPT"; do
  [ -f "$f" ] || { log "checkpoint missing: $f"; exit 1; }
done
for p in "$A_PREFIX" "$B_PREFIX"; do
  [ -f "${p}_${SPLIT}.parquet" ] && { log "${p}_${SPLIT}.parquet exists: $SPLIT HAS ALREADY BEEN READ"; exit 1; }
  for s in ${FIT_SPLITS//,/ }; do
    [ -f "${p}_${s}.parquet" ] || { log "missing fit-split score ${p}_${s}.parquet; the fusion cannot be re-derived"; exit 1; }
  done
done

wait_gpu() {
  local seen=0 free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || true)
    if [ -n "$free" ] && [ "$free" -ge "$NEED_MIB" ]; then
      seen=$((seen + 1)); [ "$seen" -ge 2 ] && { log "GPU has ${free} MiB free"; return; }
    else
      seen=0; log "waiting for GPU: ${free:-no reading} MiB free, need $NEED_MIB"
    fi
    sleep 60
  done
}

read_one() {   # <prefix> <ckpt> <tag> <chunk>
  local flag=""
  [ "$3" = swa ] && flag="--swa"
  [ "$3" = ema ] && flag="--ema"
  log "reading $SPLIT with $2 ($3)"
  # shellcheck disable=SC2086
  "$PY" scripts/score_plan_splits.py --ckpt "$2" --manifest "$MANIFEST" \
    --splits "$SPLIT" --crop-clamp --batch 32 --chunk "$4" --workers 12 \
    --out-prefix "$1" $flag 2>&1 | tee -a "logs/test_read.log"
}

wait_gpu
read_one "$A_PREFIX" "$A_CKPT" "$A_TAG" "$A_CHUNK"
read_one "$B_PREFIX" "$B_CKPT" "$B_TAG" "$B_CHUNK"

log "applying the frozen fusion (nothing is fitted on $SPLIT)"
"$PY" scripts/fuse_branches.py --a "$A_PREFIX" --b "$B_PREFIX" \
  --fit-splits "$FIT_SPLITS" --test-split "$SPLIT" --write-fused --out "$OUT" \
  2>&1 | tee -a "logs/test_read.log"
log "TEST_READ_DONE -> $OUT"
