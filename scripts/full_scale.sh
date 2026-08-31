#!/bin/bash
# Full-scale Stage A + eval bank for the WINNER, dinov2regl under both policies.
#
# WHY THIS ARM. Wave 2 (20k probe): dinov2regl:crop a3 0.7858 is the best single
# shippable arm, and band+crop FUSED reaches 0.8714 -- past the barred dinov3l
# reference's 0.8667 -- for the price of ONE tower run twice (304M against the
# 2B cap, not 608M). So both policies are needed; neither alone is the result.
#
# WHY SHARDED. One arm per GPU is 375,358 / 7.4 img/s = 14 h. Four contiguous
# shards of one arm is 3.5 h, and merge_banks concatenates them into a bank
# bit-identical to a single run: --shard keeps the frozen manifest's INDEX
# LABELS, which are the per-view RNG key, so view pixels do not move.
#
# WHY CROP FIRST. Sequential by policy, not both at once: same total, but crop
# finishes complete at ~5 h instead of both landing at ~10 h. If the box goes
# away we still have the strongest single arm rather than two half-banks.
set -uo pipefail
cd /workspace/robust-aigc-detection

DATA=/workspace/data
MANIFEST=$DATA/manifest_union.parquet
EVAL_MANIFEST=$DATA/eval_manifest_union.parquet
ROOT=$DATA/union/train
# read_manifest() rebases `path` onto $AIGCDET_DATA_ROOT via rel_path. The
# union manifest's own paths are the LOCAL /mnt/berstorage tree, which does
# not exist here, so without this every open() fails on row 0.
export AIGCDET_DATA_ROOT="$ROOT"

# THE EVAL MANIFEST AND THE TRAIN MANIFEST DO NOT SHARE A rel_path ROOT.
#   manifest_union      rel_path = "coco_train2017/real/0000000.png"
#   eval_manifest_union rel_path = "normalized_union/coco_train2017/real/..."
# One AIGCDET_DATA_ROOT therefore cannot serve both: the value that resolves
# Stage A is one level too deep for the eval bank, and read_manifest() rebases
# onto it silently, so the mismatch surfaces only in the eval preflight -- on
# 2026-08-31 that was 6.5 h of Stage A after the mistake was already made.
# $EVAL_ROOT is the eval manifest's own root: `normalized_union` symlinked at
# the corpus, `demo` beside it for the benchmark split.
EVAL_ROOT=${EVAL_ROOT:-$DATA/union/eval_root}


# ONE THREAD PER WORKER. Without this the run dies at launch: 4 shards x 40
# workers = 160 processes, and each numpy import spins up OpenBLAS with one
# thread per core (64), so pthread_create fails with "Resource temporarily
# unavailable" before a single image is read. This is what commit 7b7bc90
# ("one thread per worker, or a fat box runs out of them") fixed inside
# run_pod_arms.sh; a script that spawns its own workers has to repeat it,
# because the caps are environment, not code.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1
BB=dinov2regl
NGPU=4
WORKERS=${WORKERS:-32}
BATCH=${BATCH:-32}
CROP_SIDE=200

log() { echo "[$(date +%H:%M:%S)] $*"; }

finite_check() {   # $1 = bank dir
  python3 - "$1" <<'PYEOF'
import sys, numpy as np, pandas as pd, json, os
d = sys.argv[1]
f = np.load(os.path.join(d, "feats.npy"), mmap_mode="r")
m = pd.read_parquet(os.path.join(d, "meta.parquet"))
cfg = json.load(open(os.path.join(d, "config.json")))
bad = 0
for s in range(0, f.shape[0], 8192):
    bad += int((~np.isfinite(np.asarray(f[s:s+8192], dtype=np.float32))).sum())
peak = float(np.abs(np.asarray(f[:4096], dtype=np.float32)).max())
ok = bad == 0 and len(m) == f.shape[0] == cfg["n_images"]
print(f"FINITE {d}: rows={f.shape[0]} meta={len(m)} cfg={cfg['n_images']} "
      f"nonfinite={bad} max|x|={peak:.2f} -> {'OK' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PYEOF
}

run_policy() {     # $1 = band|crop
  local mode=$1 extra="" tag bank ebank
  [ "$mode" = crop ] && extra="--crop-side $CROP_SIDE"
  tag="full_${mode}_${BB}"; bank="data/banks/$tag"; ebank="data/banks/eval_$tag"

  if [ -f "$bank/config.json" ] && finite_check "$bank" >/dev/null 2>&1; then
    log "$tag: stage A already complete, skipping"
  else
    log "$tag: stage A, $NGPU shards"
    for i in $(seq 0 $((NGPU-1))); do
      CUDA_VISIBLE_DEVICES=$i nohup python3 -u scripts/extract_features.py \
        --manifest "$MANIFEST" --backbone $BB --out "${bank}_shard$i" \
        --split train,val_internal --shard "$i/$NGPU" --device cuda \
        --batch-size $BATCH --workers $WORKERS --resume \
        --canon-mode "$mode" $extra > "logs/${tag}_shard$i.log" 2>&1 &
    done
    wait
    for i in $(seq 0 $((NGPU-1))); do
      finite_check "${bank}_shard$i" || { log "$tag: shard $i FAILED finite check"; return 1; }
    done
    log "$tag: merging $NGPU shards"
    python3 -u scripts/merge_banks.py --out "$bank" \
      $(for i in $(seq 0 $((NGPU-1))); do echo -n "${bank}_shard$i "; done) \
      > "logs/${tag}_merge.log" 2>&1 || { log "$tag: MERGE FAILED"; return 1; }
    finite_check "$bank" || { log "$tag: merged bank FAILED finite check"; return 1; }
  fi

  if [ -f "$ebank/config.json" ]; then
    log "$tag: eval bank already present, skipping"
  else
    log "$tag: eval bank, $NGPU shards"
    for i in $(seq 0 $((NGPU-1))); do
      CUDA_VISIBLE_DEVICES=$i nohup python3 -u scripts/extract_eval_bank.py \
        --manifest "$EVAL_MANIFEST" --root "$EVAL_ROOT" --backbone $BB --out "${ebank}_shard$i" \
        --tier ablation --shard "$i/$NGPU" --device cuda \
        --batch-size $BATCH --resume \
        --canon-mode "$mode" $extra > "logs/e${tag}_shard$i.log" 2>&1 &
    done
    wait
    python3 -u scripts/merge_banks.py --out "$ebank" \
      $(for i in $(seq 0 $((NGPU-1))); do echo -n "${ebank}_shard$i "; done) \
      > "logs/e${tag}_merge.log" 2>&1 || { log "$tag: EVAL MERGE FAILED"; return 1; }
    finite_check "$ebank" || { log "$tag: eval bank FAILED finite check"; return 1; }
  fi

  log "$tag: ladder"
  cfgs=(); for r in a0 a1 a2 a3 a7_norecon; do cfgs+=(configs/rungs/${r}.yaml); done
  python3 -u scripts/run_ablation.py --bank "$bank" --eval-bank "$ebank" \
    --rungs "${cfgs[@]}" --tier ablation --device cuda \
    --out-dir "outputs/rungs_$tag" \
    --out "docs/robustness_table_$tag.md" \
    --selection "docs/selection_$tag.json" > "logs/ladder_$tag.log" 2>&1
  log "  ladder exit $?"
  return 0
}

log "=== SMOKE (LIMIT=64) on the FULL manifest -- validates paths before 10 h ==="
for mode in crop band; do
  extra=""; [ "$mode" = crop ] && extra="--crop-side $CROP_SIDE"
  python3 -u scripts/extract_features.py --manifest "$MANIFEST" --backbone $BB \
    --out "data/banks/smoke_full_${mode}" --split train,val_internal \
    --limit 64 --device cuda --batch-size 16 --workers 8 \
    --canon-mode "$mode" $extra > "logs/smoke_full_${mode}.log" 2>&1
  if finite_check "data/banks/smoke_full_${mode}"; then
    log "  GATE ok: $mode"
  else
    log "  GATE FAILED: $mode -- see logs/smoke_full_${mode}.log"; exit 1
  fi
done
log "smoke passed both policies; committing to the full run"

run_policy crop || { log "CROP FAILED -- stopping before band"; exit 1; }
log "=== CROP COMPLETE ==="
run_policy band || { log "BAND FAILED"; exit 1; }
log "=== BAND COMPLETE ==="

log "=== band+crop fitted fusion at full scale ==="
python3 -u scripts/run_ablation.py \
  --bank "data/banks/full_band_${BB}" --eval-bank "data/banks/eval_full_band_${BB}" \
  --fuse-bank "data/banks/full_crop_${BB}" --fuse-eval-bank "data/banks/eval_full_crop_${BB}" \
  --rungs configs/rungs/a3.yaml --tier ablation --device cuda --fit-fuse-weight \
  --out-dir outputs/rungs_full_bandcrop \
  --out docs/robustness_table_full_bandcrop.md \
  --selection docs/selection_full_bandcrop.json > logs/a5_full_bandcrop.log 2>&1
log "  fusion exit $? ; $(grep -a 'weight fitted' logs/a5_full_bandcrop.log | tail -1)"
log "=== ALL DONE ==="
