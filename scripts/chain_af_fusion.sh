#!/bin/bash
# Experiment A, the CPU-only half: does the BEST RUNG survive fusion?
#
# THE GAP THIS CLOSES. Every fusion number in this project fuses `a3` heads --
# the 247-subset lattice, the fourteen combining rules, the 0.9247 four-way.
# Every rung above a3 is measured on ONE UNFUSED ARM. On crop_dinov2regl the
# ladder runs a3 0.7858 -> a4 0.8229 -> a4vq 0.8401 -> a4both 0.8422 ->
# aF 0.8487, a +0.0629 gain larger than the margin between most of the fusion
# combinations. Nobody has multiplied the two together.
#
# WHY aF AND NOT a4both. Not because it scores highest, though it does, but
# because `features/freq.py` imports numpy AND NOTHING ELSE. The reconstruction
# rungs need SD 1.5's AutoencoderKL, a VQModel and LPIPS, all fp16 on a device;
# the frequency block is pure numpy on the already-decoded view. This job
# therefore runs with CUDA_VISIBLE_DEVICES="" on the ~300 cores sitting at 6%
# while the GPUs are pinned at 100% by full-scale Stage A. It costs the
# extraction nothing and waits for nothing.
#
# THE ARM SET IS MIXED, AND THAT IS FORCED. `freq._require_crop` refuses a band
# bank, and the refusal is right: band resampling replaces the generator's
# native pixel grid with the resampler's, so the descriptor would measure the
# interpolation kernel -- and where source resolution is class-correlated it
# would leak instead (content-blind: band 0.6105 pooled, 0.9976 on SID_Set;
# crop 0.5081 / 0.6316). So the crop arms get aF and the band arms stay at a3.
#
#     crop_dinov2regl     aF     new head, freq block attached
#     crop_siglipso400m   aF     new head, freq block attached
#     band_dinov2regl     a3     existing head, unchanged
#     band_siglipso400m   a3     existing head, unchanged
#
# WHAT WOULD MAKE IT FAIL. If the frequency descriptor adds the SAME signal to
# every arm, the single-arm gains will not stack and the fused number moves by
# much less than +0.0629. That is a real possibility and it is precisely what
# this measures. The baseline to beat is the all-a3 four-way at 0.9247.
set -uo pipefail
# THE REPO ROOT IS A PARAMETER, not a constant. This job deliberately runs from
# an ISOLATED checkout carrying box 2's frequency branch, because the live tree
# is mid-extraction and replacing extract_features.py / train_head.py under it
# would be picked up by the next process full_scale.sh launches. Default to the
# directory this script lives in, so it is right wherever it is copied.
cd "${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
log() { echo "[$(date +%H:%M:%S)] $*"; }

# NO GPU. Not an optimisation -- an assertion. If any step here tries to reach
# for a device, it is not the job that was designed and it must fail loudly
# rather than quietly queue behind four saturated 4090s.
export CUDA_VISIBLE_DEVICES=""

# ONE THREAD PER SHARD. Without this each of N numpy processes spins OpenBLAS
# to one thread per core -- 32 x 320 -- and the box that wedged on 2026-08-31
# wedged for exactly that reason. The parallelism here is the shard, not the
# BLAS call: an FFT over a 200px view does not need 320 threads.
# The cap applies to the SHARD phase only. Sixty-four numpy processes each
# spinning OpenBLAS to one thread per core is how this box wedged on
# 2026-08-31; the parallelism there is the shard, not the BLAS call.
#
# Stage B is the opposite shape: ONE torch process, thirty epochs, and nothing
# else wants the cores. Left at 1 it pinned a single core at 99.9% and turned a
# minutes-long job into hours while ~250 cores idled. So the two phases get
# different budgets, and neither inherits the other's.
SHARD_THREADS=${SHARD_THREADS:-1}
TRAIN_THREADS=${TRAIN_THREADS:-32}
use_threads() {
  export OMP_NUM_THREADS=$1 OPENBLAS_NUM_THREADS=$1 MKL_NUM_THREADS=$1
  export NUMEXPR_NUM_THREADS=$1 OPENCV_FOR_THREADS_NUM=$1
}
use_threads "$SHARD_THREADS"

# THE EDITABLE INSTALL PINS AN ABSOLUTE PATH, so a bare `import aigcdet` in an
# isolated checkout still resolves to /workspace/robust-aigc-detection/src --
# the LIVE tree, which is at a commit with no frequency block. Without this the
# job silently tests the wrong code, or (as it did on first run) dies on
# `cannot import name AUX_BLOCKS`. Set it, and assert it took.
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

B=data/banks
O=outputs
MANIFEST=${MANIFEST:-/workspace/data/probe/manifest_union_probe.parquet}
EVAL_MANIFEST=${EVAL_MANIFEST:-/workspace/data/probe/eval_manifest_union_probe.parquet}
export AIGCDET_DATA_ROOT=${AIGCDET_DATA_ROOT:-/workspace/data/probe}

# The replay is embarrassingly parallel and nothing else wants these cores.
NSHARD=${NSHARD:-32}
# Overridable so a new arm can be added without editing the audited body.
# Every arm named in CROP_ARMS gets the frequency block and an a3+aF ladder;
# BAND_ARMS are reused as-is and never get the block (freq._require_crop).
CROP_ARMS="${CROP_ARMS:-crop_dinov2regl crop_siglipso400m}"
BAND_ARMS="${BAND_ARMS:-band_dinov2regl band_siglipso400m}"

# The four-arm lattice in step 3 names its arms explicitly, so it is only valid
# for the default set. Adding an arm means measuring the LADDER first and
# deciding what to fuse afterwards -- not silently fusing a new arm into a
# lattice whose --arm lines still describe the old one.
SKIP_LATTICE="${SKIP_LATTICE:-0}"

# --- 0. preflight -----------------------------------------------------------
log "=== preflight ==="
for f in src/aigcdet/features/freq.py src/aigcdet/features/replay.py \
         configs/rungs/aF.yaml scripts/fusion_lattice.py; do
  [ -f "$f" ] || { log "MISSING $f -- box 2's frequency branch is not on this box"; exit 1; }
done
python3 - <<'PYEOF' || exit 1
import sys
sys.path.insert(0, "src")
from aigcdet.features.bank import AUX_BLOCKS, FREQ_DIM
names = [n for _, n, _ in AUX_BLOCKS]
assert "freq" in names, f"bank.AUX_BLOCKS has no freq block: {names}"
import aigcdet.features.freq as fq
src = open(fq.__file__).read()
assert "import torch" not in src, "freq.py imports torch; this job assumed it does not"
print(f"preflight ok: freq block present, FREQ_DIM={FREQ_DIM}, no torch import")
PYEOF
for a in $CROP_ARMS $BAND_ARMS; do
  [ -d "$B/probe_$a" ] || { log "MISSING bank $B/probe_$a"; exit 1; }
  [ -d "$B/eval_probe_$a" ] || { log "MISSING eval bank $B/eval_probe_$a"; exit 1; }
done
for a in $BAND_ARMS; do
  [ -f "$O/rungs_probe_$a/a3/checkpoint.pt" ] || {
    log "MISSING a3 checkpoint for band arm $a -- this job reuses it, it does not train it"; exit 1; }
done
python3 - <<'PYEOF' || exit 1
import os, sys
sys.path.insert(0, "src")
import aigcdet.features.bank as b
want = os.path.join(os.getcwd(), "src")
if not b.__file__.startswith(want):
    raise SystemExit(f"aigcdet resolves to {b.__file__}, not {want} -- "
                     "PYTHONPATH did not take and this job would test the "
                     "wrong tree.")
print(f"aigcdet resolves to {b.__file__}")
PYEOF
log "  all banks and band-arm checkpoints present"

# --- 1. attach the frequency block to the CROP banks ------------------------
# Both the training bank and the EVAL bank. The eval bank is the one
# `score_grid` reads, and `replay_views` handles it deliberately -- its view
# axis is the CONDITION axis and it detects that from the config's
# `conditions` key rather than trusting the caller.
attach_freq() {   # $1 = bank dir, $2 = manifest
  local bank="$1" man="$2"
  if [ -f "$bank/freq.npy" ]; then log "  $bank: freq.npy present, skipping"; return 0; fi
  log "  $bank: computing freq over $NSHARD CPU shards"
  local pids=() i
  for i in $(seq 0 $((NSHARD-1))); do
    python3 -u scripts/extract_features.py --manifest "$man" --out "$bank" \
      --block freq --block-shard "$i/$NSHARD" \
      > "logs/freq_$(basename $bank)_$i.log" 2>&1 &
    pids+=($!)
  done
  local bad=0
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || { log "    shard $i FAILED"; bad=1; }; done
  [ "$bad" -eq 0 ] || { log "  $bank: a shard failed; NOT merging a partial block"; return 1; }
  python3 -u scripts/extract_features.py --manifest "$man" --out "$bank" \
    --block freq --block-merge "$NSHARD" > "logs/freq_$(basename $bank)_merge.log" 2>&1 \
    || { log "  $bank: MERGE FAILED"; return 1; }
  python3 - "$bank" <<'PYEOF' || return 1
import sys, numpy as np, os
p = os.path.join(sys.argv[1], "freq.npy")
a = np.load(p, mmap_mode="r")
bad = int((~np.isfinite(np.asarray(a, dtype=np.float32))).sum())
print(f"    FINITE {p}: shape={a.shape} nonfinite={bad} -> {'OK' if bad==0 else 'FAIL'}")
sys.exit(0 if bad == 0 else 1)
PYEOF
  log "  $bank: freq attached"
}

log "=== (1/3) attaching the frequency block (CPU, no model) ==="
for a in $CROP_ARMS; do
  attach_freq "$B/probe_$a"      "$MANIFEST"      || exit 1
  attach_freq "$B/eval_probe_$a" "$EVAL_MANIFEST" || exit 1
done

# --- 2. train the aF heads on the crop arms ---------------------------------
use_threads "$TRAIN_THREADS"
log "=== (2/3) training aF heads (Stage B on cached features, CPU x$TRAIN_THREADS) ==="
for a in $CROP_ARMS; do
  if [ -f "$O/rungs_af_probe_$a/aF/checkpoint.pt" ]; then
    log "  $a: aF checkpoint present, skipping"; continue
  fi
  log "  $a: ladder a3 + aF"
  # a3 is retrained ALONGSIDE aF, on this box, rather than reusing the existing
  # crop a3 checkpoint. The pair must differ by one flag AND by nothing else --
  # reusing a checkpoint trained elsewhere would put the machine into the
  # comparison, which is the one confound this project has not measured.
  python3 -u scripts/run_ablation.py \
    --bank "$B/probe_$a" --eval-bank "$B/eval_probe_$a" \
    --rungs configs/rungs/a3.yaml configs/rungs/aF.yaml \
    --tier ablation --device cpu \
    --out-dir "$O/rungs_af_probe_$a" \
    --out "docs/robustness_table_af_$a.md" \
    --selection "docs/selection_af_$a.json" > "logs/af_ladder_$a.log" 2>&1
  log "    exit $? ; $(grep -aE '^(a3|aF):' logs/af_ladder_$a.log | tr '\n' ' ')"
done

# --- 3. the mixed-rung fusion lattice ---------------------------------------
if [ "$SKIP_LATTICE" = "1" ]; then
  log "=== (3/3) SKIPPED (SKIP_LATTICE=1): the lattice's --arm lines describe"
  log "    the default four arms only; re-run without it once the ladder is read"
  log "=== aF FUSION CHAIN DONE ==="
  exit 0
fi
log "=== (3/3) fusion lattice over the MIXED arm set ==="
# `fusion_lattice` derives use_recon / use_recon_vq / use_freq from each
# checkpoint's own config (`aux_flags`), so a3 and aF arms can sit in the same
# lattice without the caller having to declare which is which. Before that fix
# an aF head was scored with use_freq=False -- fed a different vector from the
# one it was fitted on, silently.
python3 -u scripts/fusion_lattice.py \
  --arm crop_dinov2regl=dinov2regl:$B/eval_probe_crop_dinov2regl:$O/rungs_af_probe_crop_dinov2regl/aF/checkpoint.pt \
  --arm crop_siglipso400m=siglipso400m:$B/eval_probe_crop_siglipso400m:$O/rungs_af_probe_crop_siglipso400m/aF/checkpoint.pt \
  --arm band_dinov2regl=dinov2regl:$B/eval_probe_band_dinov2regl:$O/rungs_probe_band_dinov2regl/a3/checkpoint.pt \
  --arm band_siglipso400m=siglipso400m:$B/eval_probe_band_siglipso400m:$O/rungs_probe_band_siglipso400m/a3/checkpoint.pt \
  --max-arity 4 --simplex-top 6 --simplex-max-arity 4 \
  --boot-n 1000 --boot-top 8 --device cpu \
  --out docs/fusion_lattice_aF_mixed.json > logs/af_lattice.log 2>&1
log "  exit $?"; tail -26 logs/af_lattice.log

# --- the comparison this job exists for -------------------------------------
log "=== aF-mixed vs the all-a3 baseline ==="
python3 - <<'PYEOF'
import json, os
new = "docs/fusion_lattice_aF_mixed.json"
old = "docs/fusion_lattice.json"
if not os.path.exists(new):
    raise SystemExit("no mixed lattice -- step 3 did not finish")
n = json.load(open(new))
print("\nsingles, aF-mixed arm set:")
for k, v in sorted(n["singles"].items(), key=lambda t: -t[1]):
    print(f"  {k:26s} {v:.4f}")
best = {}
for k, v in n["combinations"].items():
    s = max(x for x in (v.get("equal"), v.get("fitted")) if x is not None)
    if v["arity"] not in best or s > best[v["arity"]][0]:
        best[v["arity"]] = (s, k)
print("\nbest per arity, aF-mixed:")
for a in sorted(best):
    print(f"  arity {a}: {best[a][0]:.4f}  {best[a][1]}")

# THE BASELINE. The all-a3 four-way over the SAME four arms, from the lattice
# box 1 already computed -- same banks, same manifest, same metric.
if os.path.exists(old):
    o = json.load(open(old))
    key = "band_dinov2regl+crop_dinov2regl+band_siglipso400m+crop_siglipso400m"
    base = o["combinations"].get(key)
    if base:
        b = max(x for x in (base.get("equal"), base.get("fitted")) if x is not None)
        top = max(s for s, _ in best.values())
        print(f"\n  all-a3 four-way   {b:.4f}")
        print(f"  aF-mixed best     {top:.4f}")
        print(f"  delta             {top - b:+.4f}")
        print("\n  Single-arm aF gained +0.0629 on crop_dinov2regl. If this delta is")
        print("  much smaller, the frequency signal is largely REDUNDANT with what")
        print("  fusion already recovers -- which is a real answer, not a failure.")
        print("  Read it against the bootstrap interval in the json: at probe scale")
        print("  every multi-arm margin measured so far has been a statistical tie.")
PYEOF
log "=== aF FUSION CHAIN DONE ==="
