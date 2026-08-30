#!/bin/bash
# Fan backbone-probe arms across every GPU on the pod, one arm per GPU at a
# time, and collect the ladders into one table.
#
# WHY THIS SHAPE. Stage A has no inter-GPU communication at all -- each arm is
# an independent process over the same 20,000 rows -- so N GPUs answer N
# questions in the time one GPU answers one. That is the entire reason a rented
# multi-GPU box beats a queue of Kaggle sessions here, and it is why the arms
# are assigned round-robin rather than sharded: sharding one arm across four
# cards would answer one question four times faster; this answers four.
#
# WHAT EACH ARM DOES, in order, resumable at every step:
#   stage A -> finite check -> eval bank -> ladder
# The finite check is not optional. On 2026-08-29 a five-hour DINOv3 bank came
# back 131,116 rows of NaN, produced at full speed with nothing raising,
# because the only post-condition checked was the row count.
#
#   LIMIT=64 scripts/run_pod_arms.sh                 # SMOKE FIRST. ~2 min.
#   scripts/run_pod_arms.sh                          # the default four
#   scripts/run_pod_arms.sh eva02l:band convnextv2h:band     # a second wave
#
# Each arm is "<backbone>:<canon_mode>".
#
# RUN WITH LIMIT=64 FIRST, every time, on a box you are paying for by the hour.
# It walks the whole chain -- corpus paths, every backbone, the bank writer, the
# finite check -- on 64 images per arm, and it writes to a SEPARATE `smoke_`
# directory so it can never be resumed from as if it were the real thing. A
# wrong --root discovered two minutes in costs two minutes; discovered by the
# eval bank forty minutes in, it costs the wave.
set -uo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-/data}"
MANIFEST="$DATA_DIR/manifest_union_probe.parquet"
EVAL_MANIFEST="$DATA_DIR/eval_manifest_union_probe.parquet"
RUNGS_LIST="${RUNGS_LIST:-a0 a1 a2 a3 a7_norecon}"
TIER=ablation
CROP_SIDE=200

# Wave 1. Two arms decide the POLICY, two rank the TOWERS.
#
# dinov2l in BOTH modes is the policy A/B, and it is deliberately one backbone
# rather than a comparison against the dinov3l band arm measured locally at a3
# 0.8667. Same tower, same box, same session, one variable -- where the
# cross-machine version would have varied the backbone, the hardware and the
# run all at once and called the difference "standardisation". It also needs no
# HF_TOKEN: dinov3l is gated behind Meta's per-account licence, and a rented
# host has root on the machine you would put that token on.
#
# The CPU content-blind control already favours crop hard on this corpus
# (SID_Set 0.9976 -> 0.6316, pooled 0.6105 -> 0.5081,
# docs/content_blind_probe_union.json). These two arms are the ladder half of
# the same question, and together they gate the full extraction: `canon_policy`
# is baked into the features, so a band bank correctly refuses to merge, resume
# or fuse with a crop one and the choice cannot be revisited from a cache.
#
# The other two are ungated candidates in band, which is the policy with bars
# to rank against. A RANKING survives a policy change far better than absolute
# numbers do, so if crop wins only the WINNING backbone needs re-running in
# crop -- one arm, not four.
#
# Wave 2 is the rest, and costs another ~40 min on four cards:
#   scripts/run_pod_arms.sh eva02l:band convnextv2h:band dinov2regl:crop siglipso400m:crop
DEFAULT_ARMS=(dinov2l:band dinov2l:crop dinov2regl:band siglipso400m:band)
ARMS=("${@:-}")
[ -z "${ARMS[0]:-}" ] && ARMS=("${DEFAULT_ARMS[@]}")

# Debian/Ubuntu images since 3.11 often ship `python3` and no `python` at all,
# and a bare `python` there is "command not found" three lines into a script on
# a box billed by the hour. Resolve it once.
PY_BIN="${PY_BIN:-$(command -v python || command -v python3)}"
[ -n "$PY_BIN" ] || { echo 'FATAL: no python or python3 on PATH' >&2; exit 1; }

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$NGPU" -ge 1 ] || { echo "FATAL: no GPUs" >&2; exit 1; }
NCPU=$(nproc)
# Workers per arm, leaving one core per arm for the parent and the writer. This
# is the knob that matters on a fat box: the Kaggle default of 4 exists because
# a Kaggle session has 4 cores, not because 4 is right.
WORKERS="${WORKERS:-$(( (NCPU / NGPU) - 2 ))}"
[ "$WORKERS" -lt 2 ] && WORKERS=2
# ...but capped. A 320-core box gives 78 per arm by that formula, and 4 x 78 is
# 312 spawned processes each importing numpy and cv2 -- real memory and real
# spawn cost for throughput that flattens long before it, since a 4090 is fed
# by far fewer decoders than that. Raise it with WORKERS= if the smoke run's
# measured rate says the cards are starving.
WORKER_CAP="${WORKER_CAP:-32}"
[ "$WORKERS" -gt "$WORKER_CAP" ] && WORKERS=$WORKER_CAP
BATCH="${BATCH:-32}"
LIMIT="${LIMIT:-}"          # set it to smoke the chain; unset for the real run

# ONE THREAD PER WORKER, and this is a correctness fix before it is a tuning
# one. Parallelism here comes from the WORKER PROCESSES; OpenMP, OpenCV, BLAS
# and torch each additionally size their own thread pool to nproc, so on a
# 320-core box every one of 32 workers x 4 arms tries to create ~320 threads
# and the box runs out:
#
#   libgomp: Thread creation failed: Resource temporarily unavailable
#   [ERROR] parallel_impl.cpp: WorkerThread 0: Can't spawn new thread
#
# It is a race, which is worse than a clean failure -- three arms took their
# threads and the fourth died. The pools were never wanted: they oversubscribe
# the cores the workers are already using, so capping them is faster as well as
# survivable. Nothing here is a big matmul; the GPU does that.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1

echo "arms:    ${ARMS[*]}"
echo "gpus:    $NGPU   cores: $NCPU   workers/arm: $WORKERS   batch: $BATCH"
echo "corpus:  $DATA_DIR"
echo "threads: 1 per worker (OMP/BLAS/OpenCV capped); nproc=$NCPU"
# The ceiling that was actually hit. Printed so the next failure is diagnosable
# from the header rather than from a wall of libgomp errors.
_maxproc=$(ulimit -u 2>/dev/null || echo "?")
_pidsmax=$(cat /sys/fs/cgroup/pids.max 2>/dev/null || echo "?")
echo "limits:  ulimit -u=$_maxproc  cgroup pids.max=$_pidsmax  "\
     "(this run wants ~$(( NGPU * (WORKERS + 2) )) processes)"
if [ -n "$LIMIT" ]; then
  echo "MODE:    SMOKE, $LIMIT images per arm, into data/banks/smoke_*"
else
  echo "MODE:    real run"
fi
[ -f "$MANIFEST" ] || { echo "FATAL: no $MANIFEST -- run pod_bootstrap.sh" >&2; exit 1; }
mkdir -p logs docs data/banks outputs

bank_done() {   # a bank is complete when its config's row count is reached
  "$PY_BIN" - "$1" <<'PY' 2>/dev/null
import json, os, sys
d = sys.argv[1]
cfg = os.path.join(d, "config.json")
meta = os.path.join(d, "meta.parquet")
if not (os.path.exists(cfg) and os.path.exists(meta)):
    raise SystemExit(1)
import pyarrow.parquet as pq
n = pq.ParquetFile(meta).metadata.num_rows
raise SystemExit(0 if n >= json.load(open(cfg))["n_images"] else 1)
# pandas/pyarrow can abort in a static destructor at interpreter shutdown --
# 'terminate called without an active exception' -- AFTER the work is done and
# printed. That non-zero exit then killed a bootstrap whose checks had all
# passed. os._exit skips the teardown entirely.
import os as _os; _os._exit(0)
PY
}

run_arm() {
  local gpu=$1 spec=$2
  local backbone="${spec%%:*}" mode="${spec##*:}"
  local tag="probe_${mode}_${backbone}"
  [ -n "$LIMIT" ] && tag="smoke_${tag}"
  local bank="data/banks/${tag}" ebank="data/banks/eval_${tag}"
  local sel="docs/selection_${tag}.json"
  local log="logs/arm_${tag}.log"
  local t0=$(date +%s)
  local extra=()
  [ "$mode" = "crop" ] && extra=(--crop-side "$CROP_SIDE")

  export CUDA_VISIBLE_DEVICES="$gpu"
  export AIGCDET_DATA_ROOT="$DATA_DIR"
  echo "[gpu $gpu] $tag: start" | tee -a "$log"

  # ---- stage A
  if bank_done "$bank"; then
    echo "[gpu $gpu] $tag: stage A already complete" | tee -a "$log"
  else
    "$PY_BIN" -u scripts/extract_features.py --manifest "$MANIFEST" \
      --backbone "$backbone" --out "$bank" --split train,val_internal \
      --device cuda --batch-size "$BATCH" --workers "$WORKERS" --resume \
      --canon-mode "$mode" "${extra[@]}" \
      ${LIMIT:+--limit "$LIMIT"} >> "$log" 2>&1 \
      || { echo "[gpu $gpu] $tag: STAGE A FAILED -- see $log"; return 1; }
  fi

  # ---- the finite check. See the header.
  "$PY_BIN" - "$bank" "$backbone" >> "$log" 2>&1 <<'PY' \
      || { echo "[gpu $gpu] $tag: NON-FINITE BANK -- see $log"; return 1; }
import os, sys
import numpy as np
sys.path.insert(0, "src")
from aigcdet.features.bank import FeatureBank
bank_dir, name = sys.argv[1], sys.argv[2]
FeatureBank(bank_dir).check_invariants()
f = np.load(os.path.join(bank_dir, "feats.npy"), mmap_mode="r")
# Evenly spaced, not the head: an overflow that begins part-way through a
# corpus (a brighter source, a larger image) leaves the first rows finite.
idx = np.linspace(0, f.shape[0] - 1, min(4096, f.shape[0])).astype(int)
s = np.asarray(f[idx], dtype=np.float32)
bad = int((~np.isfinite(s)).sum())
print(f"finite check {name}: {len(idx)}/{f.shape[0]} rows, {bad} non-finite, "
      f"max|x|={np.abs(s[np.isfinite(s)]).max():.2f}")
if bad:
    raise SystemExit(
        f"{bad} non-finite values. A DTYPE problem, not a bad image: fix "
        f"BackboneSpec.dtype for {name} and extract to a NEW directory.")
# pandas/pyarrow can abort in a static destructor at interpreter shutdown --
# 'terminate called without an active exception' -- AFTER the work is done and
# printed. That non-zero exit then killed a bootstrap whose checks had all
# passed. os._exit skips the teardown entirely.
import os as _os; _os._exit(0)
PY

  # ---- eval bank. --no-subsample: the eval manifest IS the probe cut already,
  # and subsampling it again would score different rows in each arm.
  if bank_done "$ebank"; then
    echo "[gpu $gpu] $tag: eval bank already complete" | tee -a "$log"
  else
    "$PY_BIN" -u scripts/extract_eval_bank.py --manifest "$EVAL_MANIFEST" \
      --backbone "$backbone" --out "$ebank" --tier "$TIER" --root "$DATA_DIR" \
      --device cuda --batch-size "$BATCH" --checkpoint-every 200 --resume \
      --no-subsample --canon-mode "$mode" "${extra[@]}" \
      ${LIMIT:+--limit "$LIMIT"} >> "$log" 2>&1 \
      || { echo "[gpu $gpu] $tag: EVAL BANK FAILED -- see $log"; return 1; }
  fi

  # ---- ladder. Not under LIMIT: a ladder over 64 rows is not a small result,
  # it is no result -- a split may not even carry both classes -- and printing
  # one invites somebody to read it.
  if [ -n "$LIMIT" ]; then
    echo "[gpu $gpu] $tag: smoke, skipping the ladder" | tee -a "$log"
  elif [ -f "$sel" ]; then
    echo "[gpu $gpu] $tag: ladder already done" | tee -a "$log"
  else
    local cfgs=()
    for r in $RUNGS_LIST; do cfgs+=("configs/rungs/${r}.yaml"); done
    "$PY_BIN" -u scripts/run_ablation.py --bank "$bank" --eval-bank "$ebank" \
      --rungs "${cfgs[@]}" --tier "$TIER" --device cuda \
      --out "docs/robustness_table_${tag}.md" --selection "$sel" \
      --heatmap "docs/robustness_heatmap_${tag}.png" \
      --out-dir "outputs/rungs_${tag}" >> "$log" 2>&1 \
      || { echo "[gpu $gpu] $tag: LADDER FAILED -- see $log"; return 1; }
  fi

  echo "[gpu $gpu] $tag: DONE in $(( ($(date +%s) - t0) / 60 )) min" | tee -a "$log"
}

# One worker per GPU, striding through the arm list. A worker takes the next
# arm only when its own is finished, so an arm never shares a card -- two
# extractions on one GPU is not twice the work, it is two slower extractions
# and a memory ceiling reached at an unpredictable moment.
FAILED_FILE=$(mktemp)
for g in $(seq 0 $((NGPU - 1))); do
  (
    i=$g
    while [ "$i" -lt "${#ARMS[@]}" ]; do
      run_arm "$g" "${ARMS[$i]}" || echo "${ARMS[$i]}" >> "$FAILED_FILE"
      i=$(( i + NGPU ))
    done
  ) &
done
wait

echo
echo "================ arms finished ================"
FAILED=$(cat "$FAILED_FILE" 2>/dev/null); rm -f "$FAILED_FILE"
[ -n "$FAILED" ] && echo "FAILED: $(echo "$FAILED" | tr '\n' ' ')"

if [ -n "$LIMIT" ]; then
  echo "SMOKE PASSED for: ${ARMS[*]}"
  echo "Now re-run WITHOUT LIMIT for the real arms. The smoke banks are in"
  echo "data/banks/smoke_* and are not resumed from."
else
  "$PY_BIN" scripts/backbone_probe_table.py --out docs/backbone_probe_table.md || true
fi

echo
echo "Read heldout_robust_tpr_at_1pct, NOT clean AUC. Two caveats travel with"
echo "every number: SCALE (20,000 rows vs the corpus's 375,358) and POPULATION"
echo "(the dinov3l bars were measured on the frozen 138,116-row corpus)."
[ -n "$FAILED" ] && exit 1 || exit 0
