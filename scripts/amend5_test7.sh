#!/usr/bin/env bash
# Amendment 5, Test 7: multi-scale training baseline (docs/ntire/paper_prereg.md).
#
#   bash scripts/amend5_test7.sh timing        # 100 timed steps, workers chosen
#   bash scripts/amend5_test7.sh seed 1        # train, pick presentation, score
#   bash scripts/amend5_test7.sh comparators   # existing crop/squish proxies on the external sets
#   bash scripts/amend5_test7.sh analysis      # the registered comparison, json + fig
#
# Arms: three proxy seeds trained with random resized crop (`--canon-mode rrc`:
# area U(0.20, 1.00) of the native image, aspect log-uniform [3/4, 4/3],
# squished to 224). Everything else is the G1 proxy protocol, copied flag for
# flag from stage P of scripts/paper_pipeline.sh (the abl_crop / abl_squish
# arms): dinov2regl224, depth 24, 1 epoch over 25% of train, bank ntire_meta,
# mean pool, cosine with SWA, final weights, seeds 20260827-29. Within a seed
# the training subset, head initialisation and batch order are the
# comparators' own; only the canonicalisation policy differs.
#
# Order matters for the registration: the test-time presentation (center,
# short_center or squish) is chosen per seed on ntire_val BEFORE that seed is
# scored on ntire_val_hard or any external set.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=/scratch/dion_r13/dion_extensions/repo/.venv/bin/python
EXP=/mnt/r13justinssd/techjam/experiments/data
OUT=/scratch/dion_r13/dion_extensions/outputs/test7
RUNS=$OUT/runs
LOGS=/scratch/dion_r13/dion_extensions/logs
SEEDS=(20260827 20260828 20260829)
EXTERNAL_SPLITS=(ext_aigi_socialrf ext_aigi_community ext_synthbuster)
#: Worker counts tried by `timing`, in order. The registration says: under
#: 70% GPU utilisation the CPU augmentation is the bottleneck, so raise the
#: workers and change nothing else. Workers do not change any pixel
#: (features/extract.py: a view depends only on its key and its file).
WORKER_STEPS=(8 12 14)

export PYTHONPATH="$PWD/src:$PWD/scripts" HF_HOME=/scratch/dion_r13/cache/hf
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
mkdir -p "$RUNS" "$LOGS"
log() { echo "[$(date "+%m-%d %H:%M:%S")] test7: $*"; }

# The training command of one arm. $1 = run name, $2 = seed, $3 = workers,
# $4 = extra flags (log cadence for the timing run).
train_command() {
  echo "$PY" -u scripts/train_unfreeze.py --bank data/banks/ntire_meta --root "$EXP" \
    --backbone dinov2regl224 --depth 24 --name "$1" --out-dir "$RUNS" \
    --epochs 1 --n-src 64 --m-deg 2 --src-chunk 8 --workers "$3" \
    --seed "$2" --device cuda --canon-mode rrc --nominal-side 224 \
    --lr-schedule cosine --warmup-frac 0.03 --min-lr-frac 0.01 --swa --swa-start-frac 0.75 \
    --family-weighting proportional --train-subsample-frac 0.25 \
    --token-pool mean --resume ${4:-}
}

# Mean GPU utilisation over steps 20-100 of a throwaway run with $1 workers.
# Prints "<workers> <mean util %> <s/step at step 100> <peak RAM used GB>".
# The run is started in its own process group (setsid) so that killing it
# also stops its data-loading worker processes.
time_100_steps() {
  local workers=$1 name="timing_w$1" logf="$LOGS/test7_timing_w$1.log" pid
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

# Score one checkpoint with the probe. $1 ckpt, $2 prefix, $3 split, $4 arms.
probe_score() {
  local manifest=data/manifest_ntire.parquet
  case "$3" in ext_*) manifest=data/manifest_external.parquet ;; esac
  [ "$3" = ntire_test ] && { log "REFUSING ntire_test"; return 1; }
  "$PY" -u scripts/ntire_crop_probe.py --ckpt "$1" --manifest "$manifest" --split "$3" \
    --arms "$4" --out "${2}_probe_$3.json" --parquet-prefix "$2" \
    --batch 32 --chunk 16 --workers 8 2>&1 | grep -av "it/s\]"
  return "${PIPESTATUS[0]}"
}

cmd_seed() {
  local i=$1 seed=${SEEDS[$(( $1 - 1 ))]} name="abl_rrc_s$1"
  local ckpt="$RUNS/$name/checkpoint.pt" prefix="$OUT/$name" arm
  if [ -f "$RUNS/$name/checkpoint_slim.pt" ]; then
    ckpt="$RUNS/$name/checkpoint_slim.pt"
  elif [ "$(finished_epoch "$ckpt")" -lt 1 ]; then
    log "$name: training, seed $seed, $(chosen_workers) workers"
    $(train_command "$name" "$seed" "$(chosen_workers)") 2>&1 | grep -av "it/s\]" \
      | tee -a "$LOGS/test7_$name.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { log "$name: training failed"; return 1; }
  fi

  # Presentation choice on ntire_val, before any other split is scored.
  if [ ! -f "${prefix}_choice.json" ]; then
    probe_score "$ckpt" "$prefix" ntire_val center,short_center,squish || return 1
    "$PY" scripts/amend5_test7_analysis.py choose --seed-index "$i" || return 1
  fi
  arm=$("$PY" -c "import json; print(json.load(open('${prefix}_choice.json'))['chosen'])")
  log "$name: presentation $arm (chosen on ntire_val)"

  for split in ntire_val_hard "${EXTERNAL_SPLITS[@]}"; do
    [ -f "${prefix}_${arm}_${split}.parquet" ] && continue
    probe_score "$ckpt" "$prefix" "$split" "$arm" || return 1
  done

  # Same clean-up as the pipeline: keep a slim copy, drop the fat one.
  if [ "$ckpt" = "$RUNS/$name/checkpoint.pt" ]; then
    "$PY" scripts/slim_ckpt.py "$ckpt" > /dev/null \
      && rm -f "$ckpt" "$RUNS/$name"/checkpoint_ep*.pt
  fi
  log "$name: done"
}

# The registration asks for the existing crop / squish proxies on the
# external sets "if they have not been" scored. They have not: scored here
# exactly as the pipeline scored them on NTIRE (score_plan_splits, crop
# clamp on, final weights).
cmd_comparators() {
  local policy i ckpt prefix splits
  splits=$(IFS=,; echo "${EXTERNAL_SPLITS[*]}")
  for i in 1 2 3; do for policy in crop squish; do
    ckpt=/scratch/justin_r13/aigcdet/outputs/paper/abl_${policy}_s$i/checkpoint_slim.pt
    prefix=$OUT/abl_${policy}_s$i
    log "comparator abl_${policy}_s$i on $splits"
    "$PY" scripts/score_plan_splits.py --ckpt "$ckpt" --manifest data/manifest_external.parquet \
      --splits "$splits" --crop-clamp --degrade-seed 20261008 --batch 32 --chunk 16 \
      --workers 8 --out-prefix "$prefix" 2>&1 | grep -av "it/s\]"
    [ "${PIPESTATUS[0]}" -eq 0 ] || log "comparator abl_${policy}_s$i failed"
  done; done
}

cmd_analysis() {
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" scripts/amend5_test7_analysis.py report
}

case "${1:-}" in
  timing) cmd_timing ;;
  seed) cmd_seed "$2" ;;
  comparators) cmd_comparators ;;
  analysis) cmd_analysis ;;
  *) echo "usage: $0 timing | seed N | comparators | analysis"; exit 2 ;;
esac
