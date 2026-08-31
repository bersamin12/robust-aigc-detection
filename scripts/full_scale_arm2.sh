#!/bin/bash
# One full-scale arm: Stage A + eval bank + ladder, for $1=backbone $2=policy.
#
# A generalisation of full_scale.sh's `run_policy`, which hardcoded
# BB=dinov2regl and swept only the policy. The pairing that actually wins
# varies BOTH -- dinov2regl:crop + siglipso400m:band -- so the arm, not the
# policy, is the unit of work.
#
#   bash scripts/full_scale_arm.sh siglipso400m band
#   SHARDS_PER_GPU=2 bash scripts/full_scale_arm.sh siglipso400m band
#
# WHY MORE THAN ONE SHARD PER GPU. Measured 2026-08-31 mid-run: GPU SM sat at
# ~61% and OSCILLATED between 30% and 93%, on 6.7% of memory (1,636 of 24,564
# MiB), while the CPU idled at 2.8% of 320 cores with zero IO wait. Nothing was
# starving -- the pipeline simply never asks for enough work at once.
#
# The cause is structural, not a tuning miss. `extract_bank` loops ONE IMAGE at
# a time and calls `embed(model, spec, prepared["views"], batch_size=...)`,
# where `prepared["views"]` is that image's 11 views. `embed` then chunks with
# `range(0, len(imgs), batch_size)` -- so with n_views=11 < batch_size=32 it
# issues a SINGLE forward of 11 images, then a host sync and a disk write, then
# waits for the next image. `--batch-size` above 11 is inert and always was.
#
# Fixing that properly means batching across images, which changes the loop
# every bank tonight depends on. Running several extraction PROCESSES per GPU
# gets most of the same benefit for none of that risk: each process issues its
# own forwards and fills the others' idle gaps, and at 6.7% memory there is
# room for several. This is a launch-configuration change only; the shards it
# produces are the same contiguous shards `merge_banks` already concatenates.
set -uo pipefail
cd /workspace/robust-aigc-detection

BB=${1:?backbone}; MODE=${2:?band|crop}
DATA=/workspace/data
MANIFEST=$DATA/manifest_union.parquet
EVAL_MANIFEST=$DATA/eval_manifest_union.parquet
export AIGCDET_DATA_ROOT="$DATA/union/train"

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



# ONE THREAD PER WORKER -- see full_scale.sh. Without this, N shards x W
# workers each spin OpenBLAS to one thread per core and pthread_create fails
# before an image is read.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1

NGPU=4
# Shards are the unit of WORK; GPUs are the unit of HARDWARE. They were the
# same number while one process saturated a card, and are not any more.
SHARDS_PER_GPU=${SHARDS_PER_GPU:-2}
NSHARD=$((NGPU * SHARDS_PER_GPU))
# Workers are per SHARD, so the box-wide total is NSHARD x WORKERS. At the
# default 8 x 24 that is 192 processes on 320 cores -- deliberately below the
# 160-worker configuration that wedged this box on 2026-08-31, which failed
# because it ran with NO thread caps rather than because of its count. The caps
# above are what make this safe; the reduced count is the belt to that braces.
WORKERS=${WORKERS:-24}
BATCH=${BATCH:-32}
CROP_SIDE=200
extra=""; [ "$MODE" = crop ] && extra="--crop-side $CROP_SIDE"
tag="full_${MODE}_${BB}"; bank="data/banks/$tag"; ebank="data/banks/eval_$tag"

log() { echo "[$(date +%H:%M:%S)] $*"; }

finite_check() {
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

# The guard that exists because of 2026-08-31 07:00: a script exited without
# reaping its workers and the relaunch landed on top of ~160 orphans, after
# which sshd could no longer fork. Refusing by default is right.
#
# ALLOW_CONCURRENT=1 makes co-residency a DELIBERATE choice rather than an
# accident, and it is the whole point of running two arms at once: one process
# per GPU leaves the card at ~61% (see the header), so a second arm's shards
# fill the first's idle gaps with useful new work instead of re-running work
# already on disk. It is opt-in, it is logged, and it still refuses above a
# hard process ceiling -- the failure mode being guarded against is unbounded
# pile-up, not two named arms sharing four cards.
MAX_CONCURRENT_EXTRACT=${MAX_CONCURRENT_EXTRACT:-8}

# COUNT PYTHON PROCESSES, NOT PATTERN MATCHES. `pgrep -fc extract_features.py`
# matches ANY process whose command line contains that text -- including the
# shell that is running this very check, and including any `ssh ... pgrep`
# invocation used to inspect the box. That self-match reported 5 running
# extractions when 4 were live and refused a launch that should have gone
# ahead. Filtering on the executable name is what makes the count mean
# "extractions" rather than "mentions".
count_extract() {
  local n=0 p
  for p in $(pgrep -f "scripts/extract_(features|eval_bank)\.py" 2>/dev/null); do
    case "$(ps -o comm= -p "$p" 2>/dev/null)" in python*) n=$((n+1));; esac
  done
  echo "$n"
}
alive=$(count_extract)
if [ "${alive:-0}" -gt 0 ]; then
  if [ "${ALLOW_CONCURRENT:-0}" != "1" ]; then
    log "REFUSING: $alive extract process(es) already running."
    log "  (set ALLOW_CONCURRENT=1 to share the GPUs on purpose)"; exit 1
  fi
  if [ $((alive + NSHARD)) -gt "$MAX_CONCURRENT_EXTRACT" ]; then
    log "REFUSING: $alive running + $NSHARD requested exceeds "
    log "  MAX_CONCURRENT_EXTRACT=$MAX_CONCURRENT_EXTRACT."; exit 1
  fi
  log "CO-RESIDENT: $alive extract process(es) already running; adding $NSHARD."
fi

log "=== SMOKE (LIMIT=64) $BB:$MODE on the FULL manifest ==="
python3 -u scripts/extract_features.py --manifest "$MANIFEST" --backbone $BB \
  --out "data/banks/smoke_${tag}" --split train,val_internal \
  --limit 64 --device cuda --batch-size 16 --workers 8 \
  --canon-mode "$MODE" $extra > "logs/smoke_${tag}.log" 2>&1
finite_check "data/banks/smoke_${tag}" || { log "GATE FAILED -- see logs/smoke_${tag}.log"; exit 1; }
log "  GATE ok"

if [ -f "$bank/config.json" ] && finite_check "$bank" >/dev/null 2>&1; then
  log "$tag: stage A already complete, skipping"
else
  log "$tag: stage A, $NSHARD shards over $NGPU GPUs ($SHARDS_PER_GPU per GPU), $WORKERS workers each"
  for i in $(seq 0 $((NSHARD-1))); do
    CUDA_VISIBLE_DEVICES=$((i % NGPU)) nohup python3 -u scripts/extract_features.py \
      --manifest "$MANIFEST" --backbone $BB --out "${bank}_shard$i" \
      --split train,val_internal --shard "$i/$NSHARD" --device cuda \
      --batch-size $BATCH --workers $WORKERS --resume \
      --canon-mode "$MODE" $extra > "logs/${tag}_shard$i.log" 2>&1 &
  done
  wait
  for i in $(seq 0 $((NSHARD-1))); do
    finite_check "${bank}_shard$i" || { log "$tag: shard $i FAILED finite check"; exit 1; }
  done
  log "$tag: merging $NSHARD shards"
  python3 -u scripts/merge_banks.py --out "$bank" \
    $(for i in $(seq 0 $((NSHARD-1))); do echo -n "${bank}_shard$i "; done) \
    > "logs/${tag}_merge.log" 2>&1 || { log "$tag: MERGE FAILED"; exit 1; }
  finite_check "$bank" || { log "$tag: merged bank FAILED finite check"; exit 1; }
fi

if [ -f "$ebank/config.json" ]; then
  log "$tag: eval bank already present, skipping"
else
  log "$tag: eval bank, $NSHARD shards over $NGPU GPUs"
  for i in $(seq 0 $((NSHARD-1))); do
    CUDA_VISIBLE_DEVICES=$((i % NGPU)) nohup python3 -u scripts/extract_eval_bank.py \
      --manifest "$EVAL_MANIFEST" --root "$EVAL_ROOT" --backbone $BB --out "${ebank}_shard$i" \
      --tier ablation --shard "$i/$NSHARD" --device cuda \
      --batch-size $BATCH --resume \
      --canon-mode "$MODE" $extra > "logs/e${tag}_shard$i.log" 2>&1 &
  done
  wait
  python3 -u scripts/merge_banks.py --out "$ebank" \
    $(for i in $(seq 0 $((NSHARD-1))); do echo -n "${ebank}_shard$i "; done) \
    > "logs/e${tag}_merge.log" 2>&1 || { log "$tag: EVAL MERGE FAILED"; exit 1; }
  finite_check "$ebank" || { log "$tag: eval bank FAILED finite check"; exit 1; }
fi

log "$tag: ladder"
cfgs=(); for r in a0 a1 a2 a3 a7_norecon; do cfgs+=(configs/rungs/${r}.yaml); done
python3 -u scripts/run_ablation.py --bank "$bank" --eval-bank "$ebank" \
  --rungs "${cfgs[@]}" --tier ablation --device cuda \
  --out-dir "outputs/rungs_$tag" \
  --out "docs/robustness_table_$tag.md" \
  --selection "docs/selection_$tag.json" > "logs/ladder_$tag.log" 2>&1
log "  ladder exit $?"
log "=== $tag COMPLETE ==="
