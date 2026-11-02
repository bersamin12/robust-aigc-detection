#!/usr/bin/env bash
# Amendment 11, Test 19: does enlarging the window hurt, at equal view? (`stretch`)
# Registered in docs/ntire/paper_prereg.md, amendment 11.
#
#   bash scripts/amend5_test19.sh smoke     # a few steps + a handful of rows, end to end (not results)
#   bash scripts/amend5_test19.sh timing    # 100 timed steps; writes timing_ok if within budget
#   bash scripts/amend5_test19.sh arm       # train and score the crop196@196 arm
#
# One new proxy, `crop196@196`: Test 15's `crop200@512` arm (scripts/amend5_test15.sh,
# copied flag for flag) with three values changed, all because of the input size:
#   --crop-side 196 --nominal-side 196 --backbone dinov2regl196
# The registered arm is crop200@200. DINOv2's patch is 14 and the existing
# pipeline handles a nominal side that is not a multiple of 14 (512) by squishing
# it to the tower's fixed size (dinov2regl: 518). At 200 that same mechanism
# squishes the window to 518, a 2.59x enlargement, which is what this arm must
# not have. So the registered substitute applies: window 196 at input 196
# (14 x 14 patches), on dinov2regl196, the same dinov2regl tower registered at
# 196, which takes the window with no resampling at all.
#
# The other two arms (crop200@512, native512) are not retrained: their
# per-image logits come from Test 15's runs (outputs/test15).
#
# Registered stop rule: if the 100-step timing projects train + score above
# 6 GPU-h, timing_ok is not written and the arm does not run. The protocol is
# not changed to fit.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=/scratch/dion_r13/dion_extensions/repo/.venv/bin/python
EXP=/mnt/r13justinssd/techjam/experiments/data
OUT=/scratch/dion_r13/dion_extensions/outputs/test19
RUNS=$OUT/runs
LOGS=/scratch/dion_r13/dion_extensions/logs
SEED=20260827
#: Window side and nominal input side of the arm (the registered substitute for 200).
SIDE=196
BACKBONE=dinov2regl196
ARM_NAME="abl196_crop196_s1"
#: Steps in one proxy epoch (1 epoch over 25% of train; Tests 8 and 15 logged 1544).
STEPS_PER_EPOCH=1544
#: Registered budget for train + score, GPU-hours.
MAX_GPU_HOURS=6
#: Scoring allowance added to the projected training time. Test 15's crop200
#: arm spent about 13 min scoring both splits and slimming at 518 px; 0.5 h is
#: a deliberate upper bound, not a measurement.
SCORE_HOURS=0.5
WORKER_STEPS=(8 12 14)

export PYTHONPATH="$PWD/src:$PWD/scripts" HF_HOME=/scratch/dion_r13/cache/hf
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
mkdir -p "$RUNS" "$LOGS"
log() { echo "[$(date "+%m-%d %H:%M:%S")] test19: $*"; }

# The training command. $1 = run name, $2 = output dir, $3 = workers,
# $4 = extra flags (log cadence for timing, subsample for the smoke run).
# Identical to amend5_test15.sh's train_command except --backbone, --crop-side
# and --nominal-side.
train_command() {
  echo "$PY" -u scripts/train_unfreeze.py --bank data/banks/ntire_meta --root "$EXP" \
    --backbone "$BACKBONE" --depth 24 --name "$1" --out-dir "$2" \
    --epochs 1 --n-src 64 --m-deg 2 --src-chunk 2 --workers "$3" \
    --seed "$SEED" --device cuda --canon-mode crop --crop-side "$SIDE" --nominal-side "$SIDE" --crop-clamp \
    --lr-schedule cosine --warmup-frac 0.03 --min-lr-frac 0.01 --swa --swa-start-frac 0.75 \
    --family-weighting proportional --train-subsample-frac 0.25 \
    --token-pool mean --resume ${4:-}
}

# The scoring command, as amend5_test15.sh's. $1 = checkpoint, $2 = manifest,
# $3 = split, $4 = output prefix.
score_command() {
  echo "$PY" scripts/score_plan_splits.py --ckpt "$1" --manifest "$2" \
    --splits "$3" --crop-clamp --degrade-seed 20261008 --batch 32 --chunk 8 \
    --workers 8 --out-prefix "$4"
}

# Mean GPU utilisation over steps 20-100 of a throwaway run of this arm with
# $1 workers. Prints "<workers> <mean util %> <s/step at step 100> <peak RAM used GB>".
time_100_steps() {
  local workers=$1 name="timing_w$1" logf="$LOGS/test19_timing_w$1.log" pid
  rm -rf "${RUNS:?}/$name"
  setsid $(train_command "$name" "$RUNS" "$workers" "--log-every 10") > "$logf" 2>&1 &
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

# Writes timing.tsv, timing_chosen_workers and, if train + score fits
# MAX_GPU_HOURS, timing_ok (holding the projected hours).
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
  local train_hours total_hours
  train_hours=$(awk "BEGIN {printf \"%.2f\", $best_step * $STEPS_PER_EPOCH / 3600}" 2>/dev/null || echo nan)
  total_hours=$(awk "BEGIN {printf \"%.2f\", $train_hours + $SCORE_HOURS}" 2>/dev/null || echo nan)
  log "timing: $chosen workers, $best_step s/step, projected train $train_hours h + score $SCORE_HOURS h = $total_hours GPU-h (limit $MAX_GPU_HOURS)"
  if awk "BEGIN {exit !($total_hours <= $MAX_GPU_HOURS)}" 2>/dev/null; then
    echo "$total_hours" > "$OUT/timing_ok"
  else
    log "timing: over the registered limit (or no step time measured); the arm will not run"
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

# Train (resumable), score both splits, slim the checkpoint. $1 = output dir,
# $2 = manifest, $3 = extra training flags, $4 = workers.
train_and_score() {
  local out=$1 manifest=$2 extra=$3 workers=$4
  local ckpt="$out/runs/$ARM_NAME/checkpoint.pt" prefix="$out/$ARM_NAME"
  if [ -f "$out/runs/$ARM_NAME/checkpoint_slim.pt" ]; then
    ckpt="$out/runs/$ARM_NAME/checkpoint_slim.pt"
  elif [ "$(finished_epoch "$ckpt")" -lt 1 ]; then
    log "$ARM_NAME: training into $out, window $SIDE at input $SIDE, $workers workers"
    $(train_command "$ARM_NAME" "$out/runs" "$workers" "$extra") 2>&1 | grep -av "it/s\]" \
      | tee -a "$LOGS/test19_$(basename "$out")_$ARM_NAME.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { log "$ARM_NAME: training failed"; return 1; }
  fi

  for split in ntire_val_hard ntire_val; do
    [ "$split" != ntire_test ] || { log "refusing ntire_test"; return 1; }
    [ -f "${prefix}_${split}.parquet" ] && continue
    $(score_command "$ckpt" "$manifest" "$split" "$prefix") 2>&1 | grep -av "it/s\]"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { log "$ARM_NAME: scoring $split failed"; return 1; }
  done

  if [ "$ckpt" = "$out/runs/$ARM_NAME/checkpoint.pt" ]; then
    "$PY" scripts/slim_ckpt.py "$ckpt" > /dev/null \
      && rm -f "$ckpt" "$out/runs/$ARM_NAME"/checkpoint_ep*.pt
  fi
  log "$ARM_NAME: done ($out)"
}

cmd_arm() {
  train_and_score "$OUT" data/manifest_ntire.parquet "" "$(chosen_workers)"
}

# A few training steps (0.05% of train instead of 25%, so the epoch, SWA and
# checkpoint all complete), scoring of a handful of rows per split, then the
# analysis in smoke mode. Its numbers are not results.
cmd_smoke() {
  local smoke=$OUT/smoke
  rm -rf "$smoke"; mkdir -p "$smoke/runs"
  "$PY" scripts/amend5_test19_analysis.py --write-smoke-manifest "$smoke/manifest_smoke.parquet" || return 1
  # train_command passes --train-subsample-frac 0.25; argparse keeps the last
  # value given, so this later 0.0005 wins for the smoke run only.
  # 2 loader workers, not 8: the smoke run shares the box with other jobs, and
  # the worker count does not change what is trained.
  train_and_score "$smoke" "$smoke/manifest_smoke.parquet" "--log-every 1 --train-subsample-frac 0.0005" 2 || return 1
  "$PY" scripts/amend5_test19_analysis.py --smoke
}

case "${1:-}" in
  smoke) cmd_smoke ;;
  timing) cmd_timing ;;
  arm) cmd_arm ;;
  *) echo "usage: $0 smoke | timing | arm"; exit 2 ;;
esac
