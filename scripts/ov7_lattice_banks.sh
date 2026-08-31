#!/usr/bin/env bash
# Phase 1 of the OV7 five-arm lattice: build every arm's eval bank.
#
# Five arms, four shards each = 20 concurrent processes over 4 GPUs. Shard
# count IS thread count here: extract_eval_bank.py takes no --workers, so one
# shard is one decode thread. Four shards per arm keeps ~5 processes per card
# (~12.5 GB of 24) while leaving room for the union stragglers still running.
#
# crop_dinov2regl is REBUILT rather than copied from box 2. The bank exists
# there, but fusion needs every arm's score frame on one box against one
# manifest, and 20 min of idle GPU is cheaper than an 817 MB transfer plus a
# provenance argument about whose config.json is authoritative.
#
# Shard dirs carry the scheme (_s4_) because a 4-shard shard0 and a 6-shard
# shard0 hold different rows under the same name, and --resume would happily
# continue one into the other.
set -uo pipefail
cd /workspace/robust-aigc-detection

MAN=/workspace/data/ov7/eval_manifest_ov7_transfer.parquet
ROOT=/workspace/data/ov7/normalized_ov7/open_images_v7
NSHARD=4
export ALLOW_CONCURRENT=1 MAX_CONCURRENT_EXTRACT=32
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
mkdir -p logs

ARMS="crop:dinov2regl crop:dinov2l crop:siglipso400m band:dinov2regl band:siglipso400m"

echo "[$(date -u +%H:%M)] === phase 1: 5 arms x $NSHARD shards = 20 processes ==="
g=0
for arm in $ARMS; do
  MODE=${arm%%:*}; BB=${arm##*:}
  EXTRA=""; [ "$MODE" = crop ] && EXTRA="--crop-side 200"
  for i in $(seq 0 $((NSHARD-1))); do
    gpu=$((g % 4)); g=$((g+1))
    CUDA_VISIBLE_DEVICES=$gpu python3 -u scripts/extract_eval_bank.py \
      --manifest "$MAN" --root "$ROOT" --backbone "$BB" \
      --out "data/banks/eval_ov7_${MODE}_${BB}_s${NSHARD}_shard${i}" \
      --tier ablation --no-subsample --split heldout_generator,val_internal \
      --canon-mode "$MODE" $EXTRA --batch-size 32 --device cuda \
      --shard "${i}/${NSHARD}" --resume \
      > "logs/ov7lat_${MODE}_${BB}_${i}.log" 2>&1 &
  done
done
wait
echo "[$(date -u +%H:%M)] === all shards exited; merging ==="

FAILED=0
for arm in $ARMS; do
  MODE=${arm%%:*}; BB=${arm##*:}
  OUT="data/banks/eval_ov7_${MODE}_${BB}"
  SH=""
  for i in $(seq 0 $((NSHARD-1))); do SH="$SH data/banks/eval_ov7_${MODE}_${BB}_s${NSHARD}_shard${i}"; done
  python3 -u scripts/merge_banks.py --out "$OUT" $SH \
    > "logs/ov7lat_merge_${MODE}_${BB}.log" 2>&1 \
    && echo "  merged $OUT" \
    || { echo "  MERGE FAILED $OUT"; FAILED=1; }
done

echo "[$(date -u +%H:%M)] === verifying merges against their shards ==="
for arm in $ARMS; do
  MODE=${arm%%:*}; BB=${arm##*:}
  OUT="data/banks/eval_ov7_${MODE}_${BB}"
  SH=""
  for i in $(seq 0 $((NSHARD-1))); do SH="$SH data/banks/eval_ov7_${MODE}_${BB}_s${NSHARD}_shard${i}"; done
  python3 -u scripts/verify_merge.py "$OUT" $SH \
    > "logs/ov7lat_verify_${MODE}_${BB}.log" 2>&1 \
    && echo "  verified $OUT" || { echo "  VERIFY FAILED $OUT"; FAILED=1; }
done

echo "[$(date -u +%H:%M)] === PHASE 1 DONE (failed=$FAILED) ==="
