#!/bin/bash
# The five-arm lattice, and the all-a3 control it must be read against.
#
# Adding crop_dinov2l makes this THREE backbones (dinov2regl, dinov2l,
# siglipso400m). `fusion_lattice.MAX_BACKBONES = 2` therefore tags most of
# these subsets BARRED rather than dropping them: the bar is our own spec line,
# not the track's, and the track's only hard constraint is the 2B parameter cap
# -- which 304,372,736 + 304,368,640 + 428,225,600 = 1,036,966,976 clears with
# room to spare. The tag is there so a barred number is never quoted as a legal
# one by accident.
#
# BOTH RUNS OR NEITHER. The measured cross-context spread on these very arms is
# -0.0308 to +0.0233, larger than any fusion margin this lattice can produce.
# The a3 control therefore uses the a3 heads trained in the SAME job as the aF
# heads, and the two band arms are byte-identical checkpoints in both runs, so
# the only thing that differs between the two lattices is the flag.
set -uo pipefail
cd /workspace/af_repo
export CUDA_VISIBLE_DEVICES="" PYTHONPATH=/workspace/af_repo/src
export OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 MKL_NUM_THREADS=32
B=data/banks; O=outputs
log() { echo "[$(date +%H:%M:%S)] $*"; }

run() {  # $1 = tag, $2 = rung dir for the three crop arms
  local tag="$1" r="$2"
  log "=== five-arm lattice: $tag ==="
  python3 -u scripts/fusion_lattice.py \
    --arm crop_dinov2regl=dinov2regl:$B/eval_probe_crop_dinov2regl:$O/rungs_af_probe_crop_dinov2regl/$r/checkpoint.pt \
    --arm crop_dinov2l=dinov2l:$B/eval_probe_crop_dinov2l:$O/rungs_af_probe_crop_dinov2l/$r/checkpoint.pt \
    --arm crop_siglipso400m=siglipso400m:$B/eval_probe_crop_siglipso400m:$O/rungs_af_probe_crop_siglipso400m/$r/checkpoint.pt \
    --arm band_dinov2regl=dinov2regl:$B/eval_probe_band_dinov2regl:$O/rungs_probe_band_dinov2regl/a3/checkpoint.pt \
    --arm band_siglipso400m=siglipso400m:$B/eval_probe_band_siglipso400m:$O/rungs_probe_band_siglipso400m/a3/checkpoint.pt \
    --max-arity 5 --simplex-top 10 --simplex-max-arity 4 --simplex-step 0.05 \
    --boot-n 1000 --boot-top 10 --device cpu \
    --out "docs/fusion_lattice_x5_$tag.json" > "logs/x5_$tag.log" 2>&1
  log "  exit $? -> docs/fusion_lattice_x5_$tag.json"
  tail -6 "logs/x5_$tag.log"
}

run a3control a3
run aFmixed  aF
log "=== FIVE-ARM DONE ==="
