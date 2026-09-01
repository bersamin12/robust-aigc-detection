#!/usr/bin/env bash
# The unfreeze depth ladder, D0..D4, one depth per GPU.
#
# Three stages per depth, and the middle one is the cost people forget:
#   1. train   ~56 min at depth 4 (measured, docs/bench_finetune_dinov2regl.json)
#   2. RE-EXTRACT that depth's eval bank -- a tower whose weights moved does not
#      produce the features in the frozen bank, so there is nothing to reuse
#   3. score every depth on its own bank, checking they differ ONLY in the tower
#
# D0 is trained, not copied from a3. Its tower is frozen and its pixels are the
# cached bank's, but this path runs the tower in float32 where a3's features
# were cached in float16 -- reading D1..D4 against a3 would fold that into the
# depth effect. See FinetuneConfig.tower_dtype.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/venv/main/bin/python}
ROOT=${ROOT:-/workspace/data/probe}
TRAIN_BANK=${TRAIN_BANK:-data/banks/probe_crop_dinov2regl_local}
EVAL_MAN=${EVAL_MAN:-$ROOT/eval_manifest_union_probe.parquet}
OUT=${OUT:-outputs/unfreeze}
DEPTHS=${DEPTHS:-"0 1 2 4"}
EPOCHS=${EPOCHS:-5}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}

export AIGCDET_DATA_ROOT="$ROOT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
mkdir -p logs "$OUT"

echo "depths: $DEPTHS   epochs: $EPOCHS   gpus: $NGPU"
T0=$(date +%s)
FAILED=$(mktemp)

# --- stage 1+2: train each depth and re-extract its bank, one GPU each -------
g=0
for d in $DEPTHS; do
  (
    gpu=$((g % NGPU))
    CUDA_VISIBLE_DEVICES=$gpu "$PY" -u scripts/train_unfreeze.py \
      --bank "$TRAIN_BANK" --root "$ROOT" --depth "$d" --out-dir "$OUT" \
      --epochs "$EPOCHS" --device cuda > "logs/unfreeze_d${d}_train.log" 2>&1 \
      || { echo "d$d train" >> "$FAILED"; exit 1; }

    CUDA_VISIBLE_DEVICES=$gpu "$PY" -u scripts/extract_eval_bank.py \
      --manifest "$EVAL_MAN" --root "$ROOT" --backbone dinov2regl \
      --out "data/banks/eval_unfreeze_d${d}" --tier ablation --no-subsample \
      --canon-mode crop --crop-side 200 --batch-size 32 --device cuda \
      --tower-checkpoint "$OUT/d${d}/checkpoint.pt" \
      > "logs/unfreeze_d${d}_bank.log" 2>&1 \
      || { echo "d$d bank" >> "$FAILED"; exit 1; }
    echo "[$(date +%T)] d$d done"
  ) &
  g=$((g + 1))
done
wait

if [ -s "$FAILED" ]; then
  echo "=== FAILED ==="; cat "$FAILED"
  echo "NOT scoring: a ladder missing a depth is not a ladder." >&2
  exit 1
fi

# --- stage 3: score, with the comparability claim checked first -------------
args=()
for d in $DEPTHS; do
  args+=(--rung "d${d}=${OUT}/d${d}/checkpoint.pt:data/banks/eval_unfreeze_d${d}")
done
"$PY" -u scripts/score_unfreeze_ladder.py "${args[@]}" \
  --out docs/selection_unfreeze_ladder.json --device cuda || exit 1
echo "TOTAL $(( ($(date +%s) - T0) / 60 )) min"
