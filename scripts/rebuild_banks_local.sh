#!/usr/bin/env bash
# Re-extract the probe train and eval banks from THIS box's images.
#
# Why this exists. The banks on box #2 were streamed from the machine that
# built them, which saved a Stage A run and was the right call -- the manifest
# fingerprints match and the image bytes are identical (md5-checked across both
# boxes, 2026-08-31). What does NOT match is the compute stack: a fresh
# extraction here disagrees with the imported eval bank on 21 of 24 rows, worst
# |delta| 0.0049 against a feature magnitude of ~0.20, and it disagrees even on
# `clean`, whose recipe is empty and whose crop is the deterministic centre
# window. So the difference is in resampling or the forward, not in the data.
#
# That is harmless as long as everything is read from the same stack, and every
# rung so far was: a3/a4/a4vq/a4both/aF all trained on the imported train bank
# and scored on the imported eval bank. It stops being harmless the moment
# something computed HERE joins that table -- rung A6's TTA bank, or the
# unfreeze ladder's D0, which trains from live pixels and is supposed to
# reproduce cached a3. Those would differ from their controls for a reason that
# has nothing to do with what they are testing.
#
# `assert_tta_bank_matches` did not catch it and could not: it compares
# manifest fingerprints, condition axes and row order, all of which legitimately
# agree. Only the pixels disagree, which is what `verify_tta_bank.py` measures.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/venv/main/bin/python}
ROOT=${ROOT:-/workspace/data/probe}
BB=${BB:-dinov2regl}
TRAIN_MAN=${TRAIN_MAN:-$ROOT/manifest_union_probe.parquet}
EVAL_MAN=${EVAL_MAN:-$ROOT/eval_manifest_union_probe.parquet}
TRAIN_BANK=${TRAIN_BANK:-data/banks/probe_crop_${BB}_local}
EVAL_BANK=${EVAL_BANK:-data/banks/eval_probe_crop_${BB}_local}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
PROCS_PER_GPU=${PROCS_PER_GPU:-3}
SLOTS=$((NGPU * PROCS_PER_GPU))

# `extract_features.py` has no --root flag; it reads this.
export AIGCDET_DATA_ROOT="$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

mkdir -p logs
echo "gpus $NGPU  slots $SLOTS  root $ROOT"
T0=$(date +%s)
FAILED=$(mktemp)

run_shards() {                       # $1=kind  $2=bank  $3=manifest
  local kind=$1 bank=$2 man=$3 i s
  rm -rf "${bank}" "${bank}".shard*
  for s in $(seq 0 $((SLOTS - 1))); do
    (
      gpu=$((s % NGPU)); i=$s
      while [ "$i" -lt "$SLOTS" ]; do
        if [ "$kind" = train ]; then
          CUDA_VISIBLE_DEVICES=$gpu "$PY" -u scripts/extract_features.py \
            --manifest "$man" --backbone "$BB" --out "${bank}.shard${i}" \
            --split train,val_internal --shard "${i}/${SLOTS}" --device cuda \
            --batch-size 32 --canon-mode crop --crop-side 200 --resume \
            > "logs/rebuild_${kind}_${i}.log" 2>&1 \
            || echo "${kind} shard $i" >> "$FAILED"
        else
          CUDA_VISIBLE_DEVICES=$gpu "$PY" -u scripts/extract_eval_bank.py \
            --manifest "$man" --root "$ROOT" --backbone "$BB" \
            --out "${bank}.shard${i}" --tier ablation --no-subsample \
            --canon-mode crop --crop-side 200 --batch-size 32 --device cuda \
            --shard "${i}/${SLOTS}" --resume \
            > "logs/rebuild_${kind}_${i}.log" 2>&1 \
            || echo "${kind} shard $i" >> "$FAILED"
        fi
        i=$((i + SLOTS))
      done
    ) &
  done
  wait
  if [ -s "$FAILED" ]; then
    echo "=== FAILED ==="; cat "$FAILED"
    echo "NOT merging: a merge over a partial cover is a bank with holes." >&2
    exit 1
  fi
  "$PY" -u scripts/merge_banks.py --out "$bank" \
    $(for i in $(seq 0 $((SLOTS - 1))); do echo "${bank}.shard${i}"; done)
}

echo "[$(date +%T)] eval bank: 4000 rows x 20 conditions"
run_shards eval "$EVAL_BANK" "$EVAL_MAN"
echo "[$(date +%T)] train bank: 20000 rows x 11 views"
run_shards train "$TRAIN_BANK" "$TRAIN_MAN"

# The post-condition is never just "the file exists" -- the 2026-08-29 banks
# were 100% NaN and the only check that had been run was the row count.
"$PY" - "$TRAIN_BANK" "$EVAL_BANK" <<'PY'
import json, os, sys
import numpy as np
from aigcdet.features.bank import FeatureBank
for d in sys.argv[1:]:
    b = FeatureBank.open(d); b.check_invariants()
    a = np.asarray(b.feats[:])
    zero = int((np.abs(a).sum(axis=(1, 2)) == 0).sum())
    print(f"{d}\n  {a.shape}  finite={np.isfinite(a).all()}  all-zero rows={zero}"
          f"  |x| median={np.median(np.abs(a)):.4f} max={np.abs(a).max():.3f}"
          f"\n  manifest {b.config['manifest_sha256'][:16]}")
    assert np.isfinite(a).all() and zero == 0, d
PY
echo "TOTAL $(( ($(date +%s) - T0) / 60 )) min"
