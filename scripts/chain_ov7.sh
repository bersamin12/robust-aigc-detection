#!/bin/bash
# AI-OV7 as a transfer test: stage from Kaggle, build eval banks, score.
#
# No head is trained. The four full-scale `a3` checkpoints are carried over and
# AI-OV7 is read once as an external test set -- 2024-25 generators (SDXL,
# SD 1.5, Klein-4B) over Open Images portraits, against a corpus whose every
# fake is 2017-2023. See scripts/ov7_transfer.py for the two readings and for
# why nothing is fitted here.
#
# --no-subsample IS REQUIRED, NOT AN OPTIMISATION. The ablation tier's plan
# carries `subsample={"benchmark": 5000}`, and `subsample_manifest` raises on a
# budget naming a split the selected rows do not contain -- "a budget that
# matches nothing silently caps nothing". AI-OV7 has no `benchmark` split, so
# the tier default would abort every shard. Refusing the cap is also correct on
# the merits: 5,300 rows is already under it.
#
# WHY val_internal + heldout_generator AND NOT THE WHOLE CORPUS. Those two
# splits hold 2,650 reals and 2,650 fakes covering ALL SEVEN generated
# families, already 50/50 by label -- so the cheap extraction is also the
# complete one. Adding `train` would add 14,656 rows of the same seven
# families and buy only tighter per-family intervals, at 3.8x the GPU cost.
set -uo pipefail
cd /workspace/robust-aigc-detection
log() { echo "[$(date +%H:%M:%S)] $*"; }

DATA=/workspace/data
OV7=$DATA/ov7
KDS=justinbersamin/techjam-aigc-ov7
B=data/banks
O=outputs
DEVICE=${DEVICE:-cuda}
NGPU=4
WORKERS=${WORKERS:-16}

# The audited manifest. `docs/ai_ov7_generation.md` §9a-§10 reports the gate
# and the freeze against THIS parquet; a different one is a different corpus
# and its numbers do not carry over.
MANIFEST_SHA=f6a5ba0c365885104c8dc5dcb39921b0adf5dcc99fc3c75d5d65a9183df9dc32

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1

ARM_SPECS=(
  "band_dinov2regl:dinov2regl:band"
  "crop_dinov2regl:dinov2regl:crop"
  "band_siglipso400m:siglipso400m:band"
  "crop_siglipso400m:siglipso400m:crop"
)

# --- 1. stage from Kaggle ---------------------------------------------------
if [ -f "$OV7/manifest_ov7.parquet" ]; then
  log "AI-OV7 already staged at $OV7"
else
  log "=== downloading $KDS (~9.2 GB) ==="
  mkdir -p "$OV7"
  kaggle datasets download "$KDS" -p "$OV7" --unzip > logs/ov7_download.log 2>&1 \
    || { log "DOWNLOAD FAILED"; tail -5 logs/ov7_download.log; exit 1; }
  log "  downloaded"
fi

got=$(sha256sum "$OV7/manifest_ov7.parquet" | cut -d' ' -f1)
if [ "$got" != "$MANIFEST_SHA" ]; then
  log "MANIFEST MISMATCH: got $got, expected $MANIFEST_SHA"
  log "  the published corpus is not the audited one; refusing."; exit 1
fi
log "manifest sha256 matches the audited freeze"

# THE ROOT IS ONE LEVEL DEEPER THAN IT LOOKS, and this is exactly the trap
# `docs/handover_tier2.md` §9 records for the Kaggle `union` stream: AI-OV7's
# rel_path is `real/0001800.png` with NO source-level prefix, so the root must
# be `normalized_ov7/open_images_v7`. At `normalized_ov7` zero paths resolve --
# and `extract_eval_bank` would report missing files rather than a wrong root.
ROOT=""
for cand in "$OV7/normalized_ov7/open_images_v7" "$OV7/normalized_ov7" "$OV7"; do
  [ -d "$cand" ] || continue
  n=$(python3 - "$OV7/manifest_ov7.parquet" "$cand" <<'PYEOF'
import os, sys, pandas as pd
m = pd.read_parquet(sys.argv[1])
print(sum(os.path.exists(os.path.join(sys.argv[2], r)) for r in m["rel_path"].head(50)))
PYEOF
)
  log "  $n/50 rel_paths resolve under $cand"
  [ "${n:-0}" -eq 50 ] && { ROOT="$cand"; break; }
done
[ -n "$ROOT" ] || { log "NO ROOT resolves all 50 sampled rel_paths; refusing."; exit 1; }
log "root: $ROOT"
export AIGCDET_DATA_ROOT="$ROOT"

# --- 2. wait for the four full-scale a3 checkpoints --------------------------
need=(); for spec in "${ARM_SPECS[@]}"; do
  tag=${spec%%:*}; need+=("$O/rungs_full_${tag}/a3/checkpoint.pt")
done
log "=== waiting for the four full-scale a3 checkpoints ==="
idle=0; IDLE_LIMIT=${IDLE_LIMIT:-6}
for i in $(seq 1 1500); do
  missing=(); for f in "${need[@]}"; do [ -f "$f" ] || missing+=("$f"); done
  [ ${#missing[@]} -eq 0 ] && break
  if pgrep -f "scripts/(full_scale(_arm)?|chain_siglip|queue_capacity)\.sh" >/dev/null; then
    idle=0
  else
    idle=$((idle+1))
    [ "$idle" -ge "$IDLE_LIMIT" ] && {
      log "NO DRIVER for $((IDLE_LIMIT*48))s and ${#missing[@]} checkpoint(s) missing:"
      printf '  %s\n' "${missing[@]}"; exit 1; }
  fi
  [ $((i % 20)) -eq 1 ] && log "  waiting; ${#missing[@]} checkpoint(s) missing"
  sleep 48
done
missing=(); for f in "${need[@]}"; do [ -f "$f" ] || missing+=("$f"); done
[ ${#missing[@]} -eq 0 ] || { log "TIMED OUT"; printf '  %s\n' "${missing[@]}"; exit 1; }
log "all four checkpoints present"

# --- 3. eval banks over AI-OV7 ----------------------------------------------
finite_check() {
  python3 - "$1" <<'PYEOF'
import sys, numpy as np, pandas as pd, json, os
d = sys.argv[1]
f = np.load(os.path.join(d, "feats.npy"), mmap_mode="r")
m = pd.read_parquet(os.path.join(d, "meta.parquet"))
bad = 0
for s in range(0, f.shape[0], 8192):
    bad += int((~np.isfinite(np.asarray(f[s:s+8192], dtype=np.float32))).sum())
ok = bad == 0 and len(m) == f.shape[0]
print(f"FINITE {d}: rows={f.shape[0]} meta={len(m)} nonfinite={bad} "
      f"-> {'OK' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PYEOF
}

for spec in "${ARM_SPECS[@]}"; do
  tag=${spec%%:*}; rest=${spec#*:}; BB=${rest%%:*}; MODE=${rest#*:}
  ebank="$B/eval_ov7_${tag}"
  extra=""; [ "$MODE" = crop ] && extra="--crop-side 200"
  if [ -f "$ebank/config.json" ]; then log "$tag: OV7 eval bank present, skipping"; continue; fi
  log "=== OV7 eval bank: $tag ($BB:$MODE), $NGPU shards ==="
  for i in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$i nohup python3 -u scripts/extract_eval_bank.py \
      --manifest "$OV7/manifest_ov7.parquet" --backbone "$BB" \
      --out "${ebank}_shard$i" --tier ablation \
      --split val_internal,heldout_generator --no-subsample \
      --root "$ROOT" --shard "$i/$NGPU" --device "$DEVICE" \
      --batch-size 32 --resume --canon-mode "$MODE" $extra \
      > "logs/ov7_e${tag}_shard$i.log" 2>&1 &
  done
  wait
  python3 -u scripts/merge_banks.py --out "$ebank" \
    $(for i in $(seq 0 $((NGPU-1))); do echo -n "${ebank}_shard$i "; done) \
    > "logs/ov7_e${tag}_merge.log" 2>&1 || { log "$tag: MERGE FAILED"; exit 1; }
  finite_check "$ebank" || { log "$tag: OV7 eval bank FAILED finite check"; exit 1; }
done

# --- 4. score ---------------------------------------------------------------
# Carry the union-fitted four-way weights across if chain_fourway produced
# them; otherwise equal. Fitting on AI-OV7 would spend the test set.
W=$(python3 - <<'PYEOF'
import json, os
p = "docs/fusion_lattice_full_x4.json"
if os.path.exists(p):
    d = json.load(open(p))
    for k, v in d.get("combinations", {}).items():
        if v.get("arity") == 4 and v.get("w_fitted"):
            order = ["band_dinov2regl","crop_dinov2regl","band_siglipso400m","crop_siglipso400m"]
            if sorted(v["arms"]) == sorted(order):
                w = dict(zip(v["arms"], v["w_fitted"]))
                print(",".join(str(w[a]) for a in order))
PYEOF
)
[ -n "$W" ] && log "carrying union-fitted weights: $W" || log "no union-fitted weights on disk; equal weighting"

log "=== scoring AI-OV7 with the union-trained heads ==="
python3 -u scripts/ov7_transfer.py \
  $(for spec in "${ARM_SPECS[@]}"; do tag=${spec%%:*}
      echo -n "--arm ${tag}=$B/eval_ov7_${tag}:$O/rungs_full_${tag}/a3/checkpoint.pt "; done) \
  --ov7-manifest "$OV7/manifest_ov7.parquet" \
  --union-manifest "$DATA/manifest_union.parquet" \
  ${W:+--weights "$W"} --device "$DEVICE" \
  --out docs/ov7_transfer.json 2>&1 | tee logs/ov7_transfer.log
log "=== OV7 CHAIN DONE ==="
