#!/usr/bin/env bash
# Phase 2 of the OV7 five-arm lattice: attach freq, then score the lattice.
#
# RUNS FROM af_repo, NOT the main checkout, and that is load-bearing twice:
#   1. `aigcdet.features.freq` only exists on this branch.
#   2. main's fusion_lattice.py passes ONLY `use_recon` to score_grid. af_repo's
#      has `aux_flags`, which derives use_recon/use_recon_vq/use_freq from the
#      checkpoint. Scoring an aF head with use_freq=False feeds it a different
#      vector from the one it was fitted on -- silently, with no shape error.
# af_repo/data and af_repo/outputs are symlinks into the main checkout, so the
# banks phase 1 is writing and every checkpoint are visible from here.
#
# NOTHING IS FITTED ON AI-OV7. Every head already exists. fusion weights are
# fitted on val_internal only (FIT_SPLITS_WHEN_FITTING_WEIGHT) exactly as on
# the primary split; the held-out number is read once.
set -uo pipefail
cd /workspace/af_repo
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -c "import aigcdet.features.freq" 2>/dev/null \
  || { echo "FATAL: freq branch not importable from $(pwd)/src"; exit 1; }
grep -q "def aux_flags" scripts/fusion_lattice.py \
  || { echo "FATAL: this fusion_lattice.py lacks aux_flags; aF would score wrong"; exit 1; }

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
B=/workspace/robust-aigc-detection/data/banks
O=/workspace/robust-aigc-detection/outputs
MAN=/workspace/data/ov7/eval_manifest_ov7_transfer.parquet
NFREQ=${NFREQ:-32}
CROP="crop_dinov2regl crop_dinov2l crop_siglipso400m"
ALL="crop_dinov2regl crop_dinov2l crop_siglipso400m band_dinov2regl band_siglipso400m"

# ---- 0. wait for phase 1 ---------------------------------------------------
log "waiting for phase 1 (ov7_lattice_banks.sh)"
while pgrep -f ov7_lattice_banks.sh > /dev/null; do sleep 60; done
log "phase 1 released"

for n in $ALL; do
  [ -f "$B/eval_ov7_${n}/config.json" ] || { log "FATAL: missing bank $B/eval_ov7_${n}"; exit 1; }
done
log "all 5 banks present"

# ---- 1. freq block, crop banks only ----------------------------------------
# `freq._require_crop` refuses a band bank, and the refusal is right: band
# resampling substitutes the resampler's pixel grid for the generator's, so
# the descriptor would measure the interpolation kernel.
log "=== (1/2) attaching freq to the 3 crop banks, $NFREQ CPU shards each ==="
for n in $CROP; do
  bank="$B/eval_ov7_${n}"
  if [ -f "$bank/freq.npy" ]; then log "  $n: freq.npy present, skipping"; continue; fi
  log "  $n: computing"
  bad=0; pids=()
  for i in $(seq 0 $((NFREQ-1))); do
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python3 -u scripts/extract_features.py \
      --manifest "$MAN" --out "$bank" --block freq --block-shard "$i/$NFREQ" \
      > "logs/ov7lat_freq_${n}_$i.log" 2>&1 &
    pids+=($!)
  done
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || { log "    shard $i FAILED"; bad=1; }; done
  [ "$bad" -eq 0 ] || { log "  $n: NOT merging a partial block -- missing rows become dim zeros that read as real measurements"; exit 1; }
  CUDA_VISIBLE_DEVICES="" python3 -u scripts/extract_features.py \
    --manifest "$MAN" --out "$bank" --block freq --block-merge "$NFREQ" \
    > "logs/ov7lat_freq_${n}_merge.log" 2>&1 || { log "  $n: FREQ MERGE FAILED"; exit 1; }
  log "  $n: freq attached"
done

# ---- 2. score the lattice, twice, one flag apart ---------------------------
# The band arms are a3 in BOTH runs by design: the only thing that moves
# between the two tables is the rung on the crop arms, which is the same
# control discipline the primary-split lattice used.
score() {   # $1 = tag, $2 = rung for the crop arms
  local tag=$1 rung=$2 args="" ck
  for n in $ALL; do
    case "$n" in
      crop_*) ck="$O/rungs_af_probe_${n}/${rung}/checkpoint.pt" ;;
      band_*) ck="$O/rungs_probe_${n}/a3/checkpoint.pt" ;;
    esac
    [ -f "$ck" ] || { log "  MISSING checkpoint $ck"; return 1; }
    local bb=${n#*_}
    args="$args --arm ${n}=${bb}:$B/eval_ov7_${n}:$ck"
  done
  log "  --- $tag (crop arms at $rung, band arms at a3) ---"
  python3 -u scripts/fusion_lattice.py $args --max-arity 5 --device cpu \
    --out "docs/ov7_lattice_${tag}.json" > "logs/ov7_lattice_${tag}.log" 2>&1
  log "    exit $? -> docs/ov7_lattice_${tag}.json"
  grep -aE "^single|best|legal" "logs/ov7_lattice_${tag}.log" | tail -14
}

log "=== (2/2) scoring ==="
score a3control a3
score aFmixed   aF
log "=== OV7 LATTICE DONE ==="
