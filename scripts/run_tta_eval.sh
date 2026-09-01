#!/usr/bin/env bash
# Rung A6's eval bank: 4000 rows x 20 conditions x 8 TTA views = 640,000
# forwards, sharded over every GPU on the box.
#
# The plain eval bank this sits beside was extracted in one process. This one
# is 8x the work, so it is sharded by IMAGE (`--shard i/N`) and recombined with
# scripts/merge_banks.py -- which requires every unrecognised config key to
# agree, and `tta_views` is one, so a shard extracted with a different view
# list cannot be merged in.
#
# Two processes per GPU: the TTA views are CPU work (JPEG encode, gaussian
# blur, two resample round-trips) sitting between GPU batches, and one process
# per GPU leaves the card idle through all of it.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/venv/main/bin/python}
BANK=${BANK:-data/banks/tta_eval_probe_crop_dinov2regl}
PLAIN=${PLAIN:-data/banks/eval_probe_crop_dinov2regl}
MAN=${MAN:-/workspace/data/probe/eval_manifest_union_probe.parquet}
# Where this box mounts the corpus. The manifest stores relative paths, so
# without it every row reads as missing and preflight refuses the run.
ROOT=${ROOT:-/workspace/data/probe}
BACKBONE=${BACKBONE:-dinov2regl}
BATCH=${BATCH:-64}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
PROCS_PER_GPU=${PROCS_PER_GPU:-2}
SLOTS=$((NGPU * PROCS_PER_GPU))
SHARDS=${SHARDS:-$SLOTS}

# One BLAS thread per worker. A fat box hands each process as many threads as
# it has cores, and 16 processes x 224 threads is how a machine runs out of
# them (see the 7b7bc90 fix); the parallelism here is across processes.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

mkdir -p logs
echo "gpus $NGPU  procs/gpu $PROCS_PER_GPU  shards $SHARDS  batch $BATCH"
echo "out  $BANK"
T0=$(date +%s)

FAILED=$(mktemp)
for s in $(seq 0 $((SLOTS - 1))); do
  (
    gpu=$((s % NGPU))
    i=$s
    while [ "$i" -lt "$SHARDS" ]; do
      tag="tta_${i}of${SHARDS}"
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u scripts/extract_eval_bank.py \
        --manifest "$MAN" --root "$ROOT" --backbone "$BACKBONE" \
        --out "${BANK}.shard${i}" \
        --tier ablation --no-subsample --canon-mode crop --crop-side 200 \
        --tta-views all --batch-size "$BATCH" --device cuda \
        --shard "${i}/${SHARDS}" --resume \
        > "logs/${tag}.log" 2>&1 \
        || { echo "shard $i (gpu $gpu) -- logs/${tag}.log" >> "$FAILED"; }
      i=$((i + SLOTS))
    done
  ) &
done
wait                       # reap every worker before going near the merge

if [ -s "$FAILED" ]; then
  echo "=== FAILED SHARDS ==="; cat "$FAILED"; rm -f "$FAILED"
  echo "NOT merging: a merge over a partial cover is a bank with holes." >&2
  exit 1
fi
rm -f "$FAILED"
echo "all $SHARDS shards done in $(( ($(date +%s) - T0) / 60 )) min; merging"

rm -rf "$BANK"
"$PY" -u scripts/merge_banks.py --out "$BANK" \
  $(for i in $(seq 0 $((SHARDS - 1))); do echo "${BANK}.shard${i}"; done) || exit 1

# The post-condition is never just "the file exists". `verify_tta_bank.py`
# proves the identity view reproduces the plain bank bit for bit, which is the
# one check that covers the composition order, the RNG keying and the
# flattening at once.
"$PY" -u scripts/verify_tta_bank.py --tta "$BANK" --plain "$PLAIN" || exit 1
echo "TOTAL $(( ($(date +%s) - T0) / 60 )) min"
