#!/usr/bin/env bash
# Amendment 5, Test 8: fair native crop baseline (docs/ntire/paper_prereg.md).
#
#   bash scripts/amend5_test8.sh timing        # 100 timed steps, workers chosen
#   bash scripts/amend5_test8.sh seed 1        # train and score one seed
#   bash scripts/amend5_test8.sh analysis      # the registered comparison, json + fig
#
# Arms: three proxy seeds trained with a native 224 px window at the 224
# input size (`--canon-mode crop --crop-side 224 --nominal-side 224
# --crop-clamp`): no enlargement for any image whose short side is at least
# 224; a smaller image gives its largest square, resized, as the current
# policy does below 200 px. Tested with the same 224 px native centre window:
# score_plan_splits rebuilds the checkpoint's own policy, and with no rng
# `canonicalise` takes the centre window. Everything else is the G1 proxy
# protocol, copied flag for flag from stage P of scripts/paper_pipeline.sh
# (the abl_crop / abl_squish arms), and the scoring is the pipeline's own
# score() call, so the comparators' parquets and these are the same code.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=/scratch/dion_r13/dion_extensions/repo/.venv/bin/python
EXP=/mnt/r13justinssd/techjam/experiments/data
OUT=/scratch/dion_r13/dion_extensions/outputs/test8
RUNS=$OUT/runs
LOGS=/scratch/dion_r13/dion_extensions/logs
SEEDS=(20260827 20260828 20260829)
#: Worker counts tried by `timing`, in order. The registration says: under
#: 70% GPU utilisation the CPU augmentation is the bottleneck, so raise the
#: workers and change nothing else. Workers do not change any pixel
#: (features/extract.py: a view depends only on its key and its file).
WORKER_STEPS=(8 12 14)

export PYTHONPATH="$PWD/src:$PWD/scripts" HF_HOME=/scratch/dion_r13/cache/hf
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
mkdir -p "$RUNS" "$LOGS"
log() { echo "[$(date "+%m-%d %H:%M:%S")] test8: $*"; }

# The training command of one arm. $1 = run name, $2 = seed, $3 = workers,
# $4 = extra flags (log cadence for the timing run).
train_command() {
  echo "$PY" -u scripts/train_unfreeze.py --bank data/banks/ntire_meta --root "$EXP" \
    --backbone dinov2regl224 --depth 24 --name "$1" --out-dir "$RUNS" \
    --epochs 1 --n-src 64 --m-deg 2 --src-chunk 8 --workers "$3" \
    --seed "$2" --device cuda --canon-mode crop --crop-side 224 --nominal-side 224 --crop-clamp \
    --lr-schedule cosine --warmup-frac 0.03 --min-lr-frac 0.01 --swa --swa-start-frac 0.75 \
    --family-weighting proportional --train-subsample-frac 0.25 \
    --token-pool mean --resume ${4:-}
}

# Mean GPU utilisation over steps 20-100 of a throwaway run with $1 workers.
# Prints "<workers> <mean util %> <s/step at step 100> <peak RAM used GB>".
# The run is started in its own process group (setsid) so that killing it
# also stops its data-loading worker processes.
time_100_steps() {
  local workers=$1 name="timing_w$1" logf="$LOGS/test8_timing_w$1.log" pid
  rm -rf "${RUNS:?}/$name"
  setsid $(train_command "$name" "${SEEDS[0]}" "$workers" "--log-every 10") > "$logf" 2>&1 &
  pid=$!
  local util_samples=() peak_ram=0 ram
  while kill -0 "$pid" 2>/dev/null; do
    if grep -q "step 100/" "$logf"; then break; fi
    if grep -q "step [2-9][0-9]/" "$logf"; then   # past warm-up (step 20+)
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

# Writes $OUT/timing.tsv (one row per worker count tried) and
# $OUT/timing_chosen_workers (the count every seed then trains with).
cmd_timing() {
  local chosen="" best_util=0 best_workers=${WORKER_STEPS[0]} line w util per_step ram
  echo -e "workers\tgpu_util_mean_pct\ts_per_step\tpeak_ram_used_gb" > "$OUT/timing.tsv"
  for workers in "${WORKER_STEPS[@]}"; do
    line=$(time_100_steps "$workers")
    log "timing: workers util% s/step peakRAM_GB = $line"
    read -r w util per_step ram <<< "$line"
    echo -e "$w\t$util\t$per_step\t$ram" >> "$OUT/timing.tsv"
    if awk "BEGIN {exit !($util > $best_util)}"; then best_util=$util; best_workers=$w; fi
    if awk "BEGIN {exit !($util >= 70)}"; then chosen=$w; break; fi
  done
  # Under 70% at every count: the registration does not abort, it only
  # raises workers. Train at the best count measured; the tsv records it.
  [ -z "$chosen" ] && chosen=$best_workers
  echo "$chosen" > "$OUT/timing_chosen_workers"
  log "timing: training with $chosen workers (see $OUT/timing.tsv)"
}

chosen_workers() {
  cat "$OUT/timing_chosen_workers" 2>/dev/null || echo "${WORKER_STEPS[0]}"
}

# Epoch a checkpoint has finished, or -1 if there is none.
finished_epoch() {
  [ -f "$1" ] || { echo -1; return; }
  "$PY" -c "import torch; print(torch.load('$1', map_location='cpu', weights_only=False, mmap=True)['epoch'])" \
    2>/dev/null || echo -1
}

cmd_seed() {
  local i=$1 seed=${SEEDS[$(( $1 - 1 ))]} name="abl_nat224_s$1"
  local ckpt="$RUNS/$name/checkpoint.pt" prefix="$OUT/$name"
  if [ -f "$RUNS/$name/checkpoint_slim.pt" ]; then
    ckpt="$RUNS/$name/checkpoint_slim.pt"
  elif [ "$(finished_epoch "$ckpt")" -lt 1 ]; then
    log "$name: training, seed $seed, $(chosen_workers) workers"
    $(train_command "$name" "$seed" "$(chosen_workers)") 2>&1 | grep -av "it/s\]" \
      | tee -a "$LOGS/test8_$name.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { log "$name: training failed"; return 1; }
  fi

  # The pipeline's score() call: same scorer, flags and splits as abl_crop / abl_squish.
  for split in ntire_val_hard ntire_val; do
    [ -f "${prefix}_${split}.parquet" ] && continue
    "$PY" scripts/score_plan_splits.py --ckpt "$ckpt" --manifest data/manifest_ntire.parquet \
      --splits "$split" --crop-clamp --degrade-seed 20261008 --batch 32 --chunk 16 \
      --workers 8 --out-prefix "$prefix" 2>&1 | grep -av "it/s\]"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { log "$name: scoring $split failed"; return 1; }
  done

  # Same clean-up as the pipeline: keep a slim copy, drop the fat one.
  if [ "$ckpt" = "$RUNS/$name/checkpoint.pt" ]; then
    "$PY" scripts/slim_ckpt.py "$ckpt" > /dev/null \
      && rm -f "$ckpt" "$RUNS/$name"/checkpoint_ep*.pt
  fi
  log "$name: done"
}

cmd_analysis() {
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" scripts/amend5_test8_analysis.py
}

case "${1:-}" in
  timing) cmd_timing ;;
  seed) cmd_seed "$2" ;;
  analysis) cmd_analysis ;;
  *) echo "usage: $0 timing | seed N | analysis"; exit 2 ;;
esac
