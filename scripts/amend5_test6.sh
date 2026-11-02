#!/usr/bin/env bash
# Amendment 5, Test 6, val part: score, fit, freeze (docs/ntire/paper_prereg.md).
#
#   bash scripts/amend5_test6.sh score    # GPU: squish logit + severity, both branches,
#                                         #      ntire_val and ntire_val_hard
#   bash scripts/amend5_test6.sh fit      # CPU: (a, b), z-scores, branch fusion ->
#                                         #      docs/ntire/amend5/test6_frozen.json (staged)
#
# Read 3 of ntire_test is NOT here and must not be: it runs once, last of all,
# after Dion commits test6_frozen.json and every other test is written.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=/scratch/dion_r13/dion_extensions/repo/.venv/bin/python
A2=/scratch/justin_r13/aigcdet/outputs/unfreeze/ntire2_d24_518/checkpoint.pt
B2=/scratch/justin_r13/aigcdet/outputs/dual/ntire2_dual224/checkpoint.pt
OUT=/scratch/dion_r13/dion_extensions/outputs/test6
export PYTHONPATH="$PWD/src:$PWD/scripts" HF_HOME=/scratch/dion_r13/cache/hf
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
log() { echo "[$(date "+%m-%d %H:%M:%S")] test6: $*"; }

cmd_score() {
  local split status=0
  for split in ntire_val ntire_val_hard; do
    if [ ! -f "$OUT/test6_squish_A2_$split.parquet" ]; then
      log "A2 squish+severity on $split"
      "$PY" -u scripts/amend5_test6_score.py --ckpt "$A2" --tag A2 --split "$split" 2>&1 \
        | grep -av "it/s\]"
      [ "${PIPESTATUS[0]}" -eq 0 ] || status=1
    fi
    if [ ! -f "$OUT/test6_squish_B2swa_$split.parquet" ]; then
      log "B2swa squish+severity on $split"
      "$PY" -u scripts/amend5_test6_score.py --ckpt "$B2" --swa --tag B2swa --split "$split" 2>&1 \
        | grep -av "it/s\]"
      [ "${PIPESTATUS[0]}" -eq 0 ] || status=1
    fi
  done
  return $status
}

cmd_fit() {
  "$PY" scripts/amend5_test6_fit.py || return 1
  # Stage only. Dion reviews and commits test6_frozen.json before read 3.
  git add docs/ntire/amend5/test6_frozen.json docs/ntire/amend5/test6_val.json \
    scripts/amend5_test6.sh scripts/amend5_test6_score.py scripts/amend5_test6_fit.py
  log "staged test6_frozen.json; STOP here until Dion commits it"
}

case "${1:-}" in
  score) cmd_score ;;
  fit) cmd_fit ;;
  *) echo "usage: $0 score | fit"; exit 2 ;;
esac
