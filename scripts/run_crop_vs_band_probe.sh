#!/bin/bash
# Decide crop vs band standardisation on the union corpus, both arms at once.
#
# WHY IT IS A PROBE AND NOT THE CORPUS
# Standardisation is baked into the features at extraction, so unlike dataset
# composition it cannot be revisited from a cached bank. Running both policies
# over the full 376,744-row union would be ~30 h of GPU to answer one binary
# question. 20,000 rows answers it in about two hours.
#
# WHY BOTH ARMS AT ONCE ON ONE GPU
# The pipeline is roughly half CPU-bound -- the dinov3l bank took 5 h 09 against
# a ~2.9 h GPU-only projection -- so two arms interleave one's decode/augment
# with the other's forward instead of queueing behind it. Two towers at fp16
# fit 20 GB with room to spare at batch 8.
#
# WHY dinov2l
# The verdict should come from the backbone we intend to ship, and dinov2l is
# the leading candidate: Apache-2.0, ungated, and float16-safe (so it runs at
# full speed on the fleet's T4s, where dinov3l's bfloat16 has no hardware and
# falls back to float32). It has never been run at scale, so this also returns
# its first real ladder. If it disappoints, the policy verdict still holds --
# the two arms differ in one flag and nothing else.
#
# WHAT IS HELD FIXED
# Same probe manifest, same eval manifest, same seed, same rungs, same
# backbone. `--geometric` is OFF in both: dihedral needs a square input so it
# is crop-only, and enabling it on one arm would test two changes at once.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=/mnt/berstorage/techjam/experiments/data
MANIFEST=$DATA/manifest_union.parquet
EVAL_MANIFEST=$DATA/eval_manifest_union.parquet
PROBE=$DATA/probe/manifest_union_probe.parquet
EVAL_PROBE=$DATA/probe/eval_manifest_union_probe.parquet
BACKBONE=dinov2l
RUNGS="configs/rungs/a0.yaml configs/rungs/a1.yaml configs/rungs/a2.yaml configs/rungs/a3.yaml configs/rungs/a7_norecon.yaml"
BUILD_PID="${1:-}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---- 1. wait for the manifest to freeze ---------------------------------
if [ -n "$BUILD_PID" ]; then
  log "waiting for build pid $BUILD_PID"
  while kill -0 "$BUILD_PID" 2>/dev/null; do sleep 60; done
fi
[ -f "$MANIFEST" ] || { log "FATAL: $MANIFEST does not exist -- the build did not finish"; exit 1; }
log "manifest frozen: $(python -c "import pandas as pd; d=pd.read_parquet('$MANIFEST'); print(len(d),'rows;',d['split'].value_counts().to_dict())")"

# ---- 2. the union's eval manifest ---------------------------------------
# Built from the frozen manifest plus the organisers' benchmark, which is
# untouched across every stream so numbers stay comparable.
if [ ! -f "$EVAL_MANIFEST" ]; then
  log "building the union eval manifest"
  python scripts/build_eval_manifest.py \
      --manifest "$MANIFEST" \
      --benchmark-manifest "$DATA/demo/benchmark_manifest.parquet" \
      --out "$EVAL_MANIFEST"
fi

# ---- 3. cut both probes -------------------------------------------------
# `uniform`, so the probe is a scale model of the corpus and its verdict
# transfers. The eval probe caps each split it scores; benchmark is kept
# small because the SELECTION metric does not read it -- it is there so the
# report is not missing a column, not because 500 rows measure the benchmark.
if [ ! -f "$PROBE" ]; then
  python scripts/cut_probe_manifest.py --manifest "$MANIFEST" --out "$PROBE" \
      --budget train=16000 --budget val_internal=4000 --split train,val_internal
fi
if [ ! -f "$EVAL_PROBE" ]; then
  python scripts/cut_probe_manifest.py --manifest "$EVAL_MANIFEST" --out "$EVAL_PROBE" \
      --budget val_internal=2000 --budget heldout_generator=1500 --budget benchmark=500
fi

# ---- 4. both arms, concurrently -----------------------------------------
arm () {
  local NAME=$1 MODE=$2; shift 2
  local EXTRA=("$@")
  local BANK=data/banks/probe_${NAME}_${BACKBONE}
  local EBANK=data/banks/eval_probe_${NAME}_${BACKBONE}
  log "[$NAME] stage A ($MODE)"
  python -u scripts/extract_features.py --manifest "$PROBE" --backbone "$BACKBONE" \
      --out "$BANK" --split train,val_internal --device cuda \
      --batch-size 8 --workers 8 --resume \
      --canon-mode "$MODE" "${EXTRA[@]}"
  log "[$NAME] eval bank ($MODE)"
  python -u scripts/extract_eval_bank.py --manifest "$EVAL_PROBE" --backbone "$BACKBONE" \
      --out "$EBANK" --tier ablation --device cuda \
      --batch-size 8 --checkpoint-every 200 --resume --no-subsample \
      --canon-mode "$MODE" "${EXTRA[@]}"
  log "[$NAME] ladder"
  python -u scripts/run_ablation.py --bank "$BANK" --eval-bank "$EBANK" \
      --rungs $RUNGS --tier ablation --device cuda \
      --out "docs/robustness_table_probe_${NAME}.md" \
      --selection "docs/selection_probe_${NAME}.json" \
      --out-dir "outputs/rungs_probe_${NAME}"
  log "[$NAME] DONE"
}

log "launching both arms on $BACKBONE"
arm crop crop --crop-side 200 > logs/probe_crop.log 2>&1 &
PID_CROP=$!
arm band band            > logs/probe_band.log 2>&1 &
PID_BAND=$!
log "crop pid $PID_CROP, band pid $PID_BAND"

FAIL=0
wait $PID_CROP || { log "CROP ARM FAILED (see logs/probe_crop.log)"; FAIL=1; }
wait $PID_BAND || { log "BAND ARM FAILED (see logs/probe_band.log)"; FAIL=1; }

# ---- 5. the verdict -----------------------------------------------------
log "=============== CROP vs BAND ==============="
python - <<'PY'
import json, os
for name in ("crop", "band"):
    p = f"docs/selection_probe_{name}.json"
    if not os.path.exists(p):
        print(f"{name:5s}: no selection.json -- arm did not finish"); continue
    d = json.load(open(p))
    print(f"{name:5s}: {json.dumps(d)[:400]}")
print()
print("Read the two on `heldout_robust_tpr_at_1pct` AT THE SAME RUNG. Not clean")
print("AUC: crop's whole claim is about DEGRADED generalisation, and clean AUC")
print("is the one condition where it is least likely to show.")
PY
exit $FAIL
