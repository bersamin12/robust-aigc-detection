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
#
# WHY THE PROBE'S IMAGES ARE COPIED TO THE SSD FIRST
# The corpus lives on sda, a spinning disk, and the Kaggle upload of the same
# corpus has to read ~128 GB off it. Measured 2026-08-30: three sequential
# readers on one head do not share, they seek -- adding two archivers beside
# `build_dataset` took average read latency from 222 ms to 319 ms and dropped
# normalisation from 73 img/s to 40. Copying the WHOLE corpus to the SSD would
# be pointless, because the copy pays exactly the read it is trying to avoid.
# The probe's own working set is different: 24,000 images, ~8 GB, read many
# times over two hours by two arms and a CPU control. Staged on the NVMe once,
# those reads leave sda entirely, and the upload can run beside the probe
# instead of behind it -- which is worth about two hours of critical path.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=/mnt/berstorage/techjam/experiments/data
MANIFEST=$DATA/manifest_union.parquet
EVAL_MANIFEST=$DATA/eval_manifest_union.parquet
PROBE=$DATA/probe/manifest_union_probe.parquet
EVAL_PROBE=$DATA/probe/eval_manifest_union_probe.parquet
# On the NVMe (/dev/nvme0n1p2), which is a different spindle from $DATA.
SSD=/home/administrator/aigc_probe_ssd
SSD_TRAIN=$SSD/train_root
SSD_EVAL=$SSD/eval_root
STAGED_MARKER=logs/probe_ssd_staged.done
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

# ---- 3a. stage the probe's images onto the SSD --------------------------
# TWO trees, because the two manifests measure rel_path from different roots:
# a TRAIN manifest's rel_path starts at the corpus top level
# (`sid_set/sid_set/x.png`), an EVAL manifest's starts one level up
# (`normalized_union/...`, `demo/...`) because it spans the corpus and the
# organisers' benchmark. Staging both into one tree would put some images at
# two rel_paths; giving each consumer its own root keeps every rel_path
# byte-identical to the manifest that names it, which is the half of bank
# identity that lives outside the parquet.
NEED_GB=12
FREE_GB=$(df -BG --output=avail "$(dirname "$SSD")" | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
  log "FATAL: only ${FREE_GB}G free on the SSD, need ~${NEED_GB}G"; exit 1
fi
log "staging the probe's images onto the SSD (${FREE_GB}G free)"
python scripts/stage_manifest_images.py --manifest "$PROBE" \
    --root "$DATA/normalized_union" --dest "$SSD_TRAIN" --mode copy
python scripts/stage_manifest_images.py --manifest "$EVAL_PROBE" \
    --root "$DATA" --dest "$SSD_EVAL" --mode copy
# The upload chain waits on THIS, not on the probe finishing. Staging is the
# last thing here that touches sda; once it is written, the probe reads only
# the NVMe and the archivers can have the spinning disk to themselves.
touch "$STAGED_MARKER"
log "staged; sda released to the upload chain"

# ---- 3b. the content-blind control, on CPU, beside the GPU arms ---------
# Crop's cost is a CONTENT confound, and no proxy in gate_confounds.py can see
# it: a 200x200 window is a whole frame for a 200px WildFake image and a
# detail for an 800px NTIRE photograph, so field of view tracks source, and
# two of the union's sources are authentic-only. This runs the 16x16
# thumbnail control over the CANONICALISED view -- the thing the model
# actually receives -- for both policies on identical rows, because the
# number is only meaningful as a difference. Pure CPU, so it costs no wall
# clock against the GPU arms. `--root` so it reads the SAME staged tree the
# arms read; without it the control would silently measure the original tree.
log "content-blind control (CPU, both policies) in the background"
python -u scripts/content_blind_probe.py --manifest "$PROBE" --root "$SSD_TRAIN" \
    --out docs/content_blind_probe_union.json --workers 12 \
    > logs/content_blind_probe.log 2>&1 &
PID_CONTROL=$!

# ---- 4. both arms, concurrently -----------------------------------------
# `extract_features` has no --root, so it takes the staged tree from
# $AIGCDET_DATA_ROOT; `extract_eval_bank` takes its own --root because its
# manifest is measured from the other level. Both are set per-command rather
# than exported, so nothing else in this script can pick up the wrong tree.
arm () {
  local NAME=$1 MODE=$2; shift 2
  local EXTRA=("$@")
  local BANK=data/banks/probe_${NAME}_${BACKBONE}
  local EBANK=data/banks/eval_probe_${NAME}_${BACKBONE}
  log "[$NAME] stage A ($MODE)"
  AIGCDET_DATA_ROOT="$SSD_TRAIN" \
  python -u scripts/extract_features.py --manifest "$PROBE" --backbone "$BACKBONE" \
      --out "$BANK" --split train,val_internal --device cuda \
      --batch-size 8 --workers 8 --resume \
      --canon-mode "$MODE" "${EXTRA[@]}"
  log "[$NAME] eval bank ($MODE)"
  python -u scripts/extract_eval_bank.py --manifest "$EVAL_PROBE" --backbone "$BACKBONE" \
      --out "$EBANK" --tier ablation --device cuda --root "$SSD_EVAL" \
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

wait $PID_CONTROL || log "content-blind control FAILED (see logs/content_blind_probe.log)"

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

# The control does not decide the arm; it qualifies it. A crop win alongside a
# materially higher content-blind AUC is a win bought with a shortcut, and the
# number to publish beside it -- not a reason to discard the arm unread.
p = "docs/content_blind_probe_union.json"
if os.path.exists(p):
    c = json.load(open(p))
    print()
    print("--- content-blind control (16x16 AFTER standardisation) ---")
    for m in ("crop", "band"):
        if m in c:
            pooled = c[m]["pooled"]
            auc = pooled.get("auc", pooled.get("auc_unverified_branch_provenance"))
            print(f"  {m:5s} pooled AUC {auc:.4f}   {pooled['verdict']}")
    if "crop" in c and "band" in c:
        d = c["crop"]["pooled"]["auc"] - c["band"]["pooled"]["auc"]
        print(f"  crop - band = {d:+.4f}"
              + ("   <-- crop bought its gain with a content shortcut"
                 if d > 0.03 else ""))
else:
    print("\n(content-blind control produced no output -- see logs/content_blind_probe.log)")
PY
exit $FAIL
