#!/usr/bin/env bash
# Amendment 7, Test 15: native 512 window for Branch A (docs/ntire/paper_prereg.md).
#
#   bash scripts/amend5_test15.sh timing           # 100 timed steps; writes timing_ok if allowed
#   bash scripts/amend5_test15.sh arm crop200      # train and score the 200 -> 512 arm
#   bash scripts/amend5_test15.sh arm native512    # train and score the native 512 arm
#
# Two proxies on Branch A's backbone (dinov2regl, depth 24, --src-chunk 2 as
# in scripts/ntire_v2.sh), one seed (20260827), input 512:
#   crop200    --crop-side 200 --nominal-side 512   (the window is enlarged 2.56x)
#   native512  --crop-side 512 --nominal-side 512   (no enlargement at short side >= 512)
# Everything else is the G1 proxy protocol, flag for flag as amend5_test8.sh
# (copied from branch dion/test8-native-crop-proxy, staged there). Scoring is
# the pipeline's score_plan_splits.py call, centre window of each
# checkpoint's own policy.
#
# Registered stop rule: if the 100-step timing projects more than 7 h for one
# arm, timing_ok is not written and the arms do not run. The protocol is not
# changed to fit.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=/scratch/dion_r13/dion_extensions/repo/.venv/bin/python
EXP=/mnt/r13justinssd/techjam/experiments/data
OUT=/scratch/dion_r13/dion_extensions/outputs/test15
RUNS=$OUT/runs
LOGS=/scratch/dion_r13/dion_extensions/logs
SEED=20260827
#: Steps in one proxy epoch (1 epoch over 25% of train; Test 8's log shows 1544).
STEPS_PER_EPOCH=1544
MAX_HOURS_PER_ARM=7
WORKER_STEPS=(8 12 14)

export PYTHONPATH="$PWD/src:$PWD/scripts" HF_HOME=/scratch/dion_r13/cache/hf
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
mkdir -p "$RUNS" "$LOGS"
log() { echo "[$(date "+%m-%d %H:%M:%S")] test15: $*"; }

crop_side_of() {
  case "$1" in
    crop200) echo 200 ;;
    native512) echo 512 ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}

# The training command of one arm. $1 = run name, $2 = crop side, $3 = workers,
# $4 = extra flags (log cadence for the timing run).
train_command() {
  echo "$PY" -u scripts/train_unfreeze.py --bank data/banks/ntire_meta --root "$EXP" \
    --backbone dinov2regl --depth 24 --name "$1" --out-dir "$RUNS" \
    --epochs 1 --n-src 64 --m-deg 2 --src-chunk 2 --workers "$3" \
    --seed "$SEED" --device cuda --canon-mode crop --crop-side "$2" --nominal-side 512 --crop-clamp \
    --lr-schedule cosine --warmup-frac 0.03 --min-lr-frac 0.01 --swa --swa-start-frac 0.75 \
    --family-weighting proportional --train-subsample-frac 0.25 \
    --token-pool mean --resume ${4:-}
}

# Mean GPU utilisation over steps 20-100 of a throwaway native512 run with $1
# workers (the larger window is the slower arm to decode). Prints
# "<workers> <mean util %> <s/step at step 100> <peak RAM used GB>".
time_100_steps() {
  local workers=$1 name="timing_w$1" logf="$LOGS/test15_timing_w$1.log" pid
  rm -rf "${RUNS:?}/$name"
  setsid $(train_command "$name" 512 "$workers" "--log-every 10") > "$logf" 2>&1 &
  pid=$!
  local util_samples=() peak_ram=0 ram
  while kill -0 "$pid" 2>/dev/null; do
    if grep -q "step 100/" "$logf"; then break; fi
    if grep -q "step [2-9][0-9]/" "$logf"; then
      util_samples+=("$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)")
      ram=$(free -g | awk '/Mem:/ {print $3}')
      [ "$ram" -gt "$peak_ram" ] && peak_ram=$ram
    fi
    sleep 5
  done
  kill -TERM -- "-$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  rm -rf "${RUNS:?}/$name"
  local per_step mean_util
  per_step=$(sed -n 's/.*step 100\/[0-9]* .* \([0-9.]*\)s\/step.*/\1/p' "$logf" | head -1)
  mean_util=$(printf "%s\n" "${util_samples[@]:-0}" | awk '{s += $1; n++} END {printf "%.1f", (n ? s / n : 0)}')
  echo "$workers $mean_util ${per_step:-nan} $peak_ram"
}

# Writes timing.tsv, timing_chosen_workers and, if the projection allows, timing_ok.
cmd_timing() {
  local chosen="" best_util=0 best_workers=${WORKER_STEPS[0]} best_step=nan line w util per_step ram
  rm -f "$OUT/timing_ok"
  echo -e "workers\tgpu_util_mean_pct\ts_per_step\tpeak_ram_used_gb" > "$OUT/timing.tsv"
  for workers in "${WORKER_STEPS[@]}"; do
    line=$(time_100_steps "$workers")
    log "timing: workers util% s/step peakRAM_GB = $line"
    read -r w util per_step ram <<< "$line"
    echo -e "$w\t$util\t$per_step\t$ram" >> "$OUT/timing.tsv"
    if awk "BEGIN {exit !($util > $best_util)}"; then best_util=$util; best_workers=$w; best_step=$per_step; fi
    if awk "BEGIN {exit !($util >= 70)}"; then chosen=$w; best_step=$per_step; break; fi
  done
  [ -z "$chosen" ] && chosen=$best_workers
  echo "$chosen" > "$OUT/timing_chosen_workers"
  local hours
  hours=$(awk "BEGIN {printf \"%.2f\", $best_step * $STEPS_PER_EPOCH / 3600}" 2>/dev/null || echo nan)
  log "timing: $chosen workers, $best_step s/step, projected $hours h per arm (limit $MAX_HOURS_PER_ARM h)"
  if awk "BEGIN {exit !($hours <= $MAX_HOURS_PER_ARM)}" 2>/dev/null; then
    echo "$hours" > "$OUT/timing_ok"
  else
    log "timing: over the registered limit (or no step time measured); arms will not run"
  fi
}

chosen_workers() {
  cat "$OUT/timing_chosen_workers" 2>/dev/null || echo "${WORKER_STEPS[0]}"
}

finished_epoch() {
  [ -f "$1" ] || { echo -1; return; }
  "$PY" -c "import torch; print(torch.load('$1', map_location='cpu', weights_only=False, mmap=True)['epoch'])" \
    2>/dev/null || echo -1
}

cmd_arm() {
  local arm=$1 side name ckpt prefix
  side=$(crop_side_of "$arm")
  name="abl512_${arm}_s1"
  ckpt="$RUNS/$name/checkpoint.pt"
  prefix="$OUT/$name"
  if [ -f "$RUNS/$name/checkpoint_slim.pt" ]; then
    ckpt="$RUNS/$name/checkpoint_slim.pt"
  elif [ "$(finished_epoch "$ckpt")" -lt 1 ]; then
    log "$name: training, crop side $side, $(chosen_workers) workers"
    $(train_command "$name" "$side" "$(chosen_workers)") 2>&1 | grep -av "it/s\]" \
      | tee -a "$LOGS/test15_$name.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { log "$name: training failed"; return 1; }
  fi

  for split in ntire_val_hard ntire_val; do
    [ -f "${prefix}_${split}.parquet" ] && continue
    "$PY" scripts/score_plan_splits.py --ckpt "$ckpt" --manifest data/manifest_ntire.parquet \
      --splits "$split" --crop-clamp --degrade-seed 20261008 --batch 32 --chunk 8 \
      --workers 8 --out-prefix "$prefix" 2>&1 | grep -av "it/s\]"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { log "$name: scoring $split failed"; return 1; }
  done

  if [ "$ckpt" = "$RUNS/$name/checkpoint.pt" ]; then
    "$PY" scripts/slim_ckpt.py "$ckpt" > /dev/null \
      && rm -f "$ckpt" "$RUNS/$name"/checkpoint_ep*.pt
  fi
  log "$name: done"
}

case "${1:-}" in
  timing) cmd_timing ;;
  arm) cmd_arm "$2" ;;
  *) echo "usage: $0 timing | arm crop200|native512"; exit 2 ;;
esac
