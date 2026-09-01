#!/usr/bin/env bash
# DOES MORE DATA HELP, AND THROUGH WHICH CHANNEL?
#
# The 300k corpus changes two things at once -- more unique images AND more
# gradient steps -- so a single "half the data" run cannot say which one buys
# the score. Two runs at depth 8 separate them, both against the existing
# full-data d8 (16,000 rows, 5 epochs, 1,225 steps, OV7 0.7054):
#
#   half5   0.5x rows, 5 epochs  ->  ~610 steps.  Half the data AND half the
#           optimisation. This is the realistic "smaller corpus" point, and
#           the full-vs-half gap here is the total effect of scale.
#   half10  0.5x rows, 10 epochs -> ~1,220 steps. Half the data at the SAME
#           step count as the full run. The only thing that differs from d8 is
#           how many DISTINCT images exist, so this isolates data DIVERSITY.
#
#   d8 - half10  =  the diversity channel
#   half10 - half5 =  the optimisation channel
#
# Why it matters more than the row count suggests: `PairedSampler` draws
# generator families UNIFORMLY, and the probe corpus's smallest fake family
# (BigGAN) has 96 images against ntire's 4,166. At 245 steps/epoch the sampler
# takes ~436 draws per family per epoch, so BigGAN's 96 images are each seen
# ~23 times over the run while ~90% of ntire is never touched. The subsample is
# stratified for exactly that reason (see `_stratified_subsample`): thinning
# the corpus without preserving the mixture would move two variables.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/venv/main/bin/python}
ROOT=${ROOT:-/workspace/data/probe}
TRAIN_BANK=${TRAIN_BANK:-data/banks/probe_crop_dinov2regl_local}
OUT=${OUT:-outputs/unfreeze}
DEPTH=${DEPTH:-8}
WAIT_FOR=${WAIT_FOR:-"=== D8 AUX ABLATION COMPLETE"}
WAIT_LOG=${WAIT_LOG:-logs/d8_aux_heads.log}

WF_MAN=$ROOT/eval_manifest_union_probe.parquet
OV7_MAN=/data/ov7_stage/eval_manifest_ov7_transfer.parquet
OV7_ROOT=/data/ov7_stage/normalized_ov7/open_images_v7

export AIGCDET_DATA_ROOT="$ROOT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
mkdir -p logs "$OUT"
L=logs/data_scaling.log
say () { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L"; }

# name:frac:epochs:gpu
RUNS="half5:0.5:5:0 half10:0.5:10:1"

if [ -n "$WAIT_FOR" ]; then
  say "waiting for '$WAIT_FOR' in $WAIT_LOG before touching the GPUs"
  while ! grep -q "$WAIT_FOR" "$WAIT_LOG" 2>/dev/null; do sleep 30; done
  while pgrep -f "scripts/(extract_features|extract_eval_bank|train_rung|train_head)\.py" >/dev/null; do sleep 20; done
fi

say "=== data scaling at depth $DEPTH: [$RUNS]"
T0=$(date +%s)

# --- stage 1: both subsampled runs, one per GPU -----------------------------
F=$(mktemp)
for r in $RUNS; do
  IFS=: read -r nm frac ep g <<<"$r"
  ( CUDA_VISIBLE_DEVICES=$g "$PY" -u scripts/train_unfreeze.py \
      --bank "$TRAIN_BANK" --root "$ROOT" --depth "$DEPTH" --name "$nm" \
      --out-dir "$OUT" --epochs "$ep" --train-subsample-frac "$frac" \
      --src-chunk 8 --device cuda \
      > "logs/scaling_${nm}.log" 2>&1 || echo "train $nm" >> "$F" ) &
  say "  $nm: frac=$frac epochs=$ep -> GPU $g"
done
wait
[ -s "$F" ] && { say "FAILED:"; cat "$F" | tee -a "$L"; exit 1; }
say "stage 1 done in $(( ($(date +%s)-T0)/60 )) min"

# --- stage 2: each run's own eval banks --------------------------------------
# The tower moved, so the cached features on disk are not the ones this
# checkpoint produces. Not optional, not shareable between runs.
bank_for () {   # name kind
  local nm=$1 kind=$2 man root split bank slots=24
  if [ "$kind" = wf ]; then
    man=$WF_MAN; root=$ROOT; split=""; bank=data/banks/eval_unfreeze_${nm}
  else
    man=$OV7_MAN; root=$OV7_ROOT; split="heldout_generator,val_internal"
    bank=data/banks/eval_ov7_${nm}
  fi
  [ -f "$bank/config.json" ] && { say "  $kind $nm already built, skipping"; return 0; }
  say "  $kind $nm -> $bank ($slots slots)"
  rm -rf "$bank" "$bank".shard*
  local FF; FF=$(mktemp); local s
  for s in $(seq 0 $((slots-1))); do
    ( CUDA_VISIBLE_DEVICES=$((s % 4)) "$PY" -u scripts/extract_eval_bank.py \
        --manifest "$man" --root "$root" --backbone dinov2regl \
        --out "${bank}.shard${s}" --tier ablation --no-subsample \
        ${split:+--split "$split"} --canon-mode crop --crop-side 200 \
        --batch-size 64 --device cuda --tower-checkpoint "$OUT/${nm}/checkpoint.pt" \
        --shard "${s}/${slots}" --resume \
        > "logs/bank_${kind}_${nm}_${s}.log" 2>&1 || echo "$kind $nm shard $s" >> "$FF" ) &
  done
  wait
  [ -s "$FF" ] && { say "FAILED $kind $nm:"; cat "$FF" | tee -a "$L"; return 1; }
  "$PY" -u scripts/merge_banks.py --out "$bank" \
    $(for i in $(seq 0 $((slots-1))); do echo "${bank}.shard${i}"; done) >>"$L" 2>&1
}

for r in $RUNS; do
  nm=${r%%:*}
  bank_for "$nm" wf  || exit 1
  bank_for "$nm" ov7 || exit 1
done
say "stage 2 done"

# --- stage 3: score, each against the bank ITS OWN tower produced ------------
# d8 is carried in as the full-data reference; it is the same depth, the same
# seed and the same recipe, differing only in how much of the corpus it saw.
for kind in ov7 wf; do
  base=data/banks/eval_ov7_; outj=docs/selection_ov7_data_scaling.json
  [ "$kind" = wf ] && { base=data/banks/eval_unfreeze_; outj=docs/selection_wf_data_scaling.json; }
  declare -a R; R=()
  for nm in d8 half5 half10; do
    [ -f "${base}${nm}/config.json" ] || { say "  no ${base}${nm}, skipping"; continue; }
    [ -f "$OUT/${nm}/checkpoint.pt" ] || continue
    R+=(--rung "${nm}=${OUT}/${nm}/checkpoint.pt:${base}${nm}")
  done
  say "scoring $kind over ${#R[@]} rungs"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u scripts/score_unfreeze_ladder.py "${R[@]}" \
    --out "$outj" --device cuda 2>&1 | tee -a "$L"
done

say "=== DATA SCALING COMPLETE in $(( ($(date +%s)-T0)/60 )) min"
