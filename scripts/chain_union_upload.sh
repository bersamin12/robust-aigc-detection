#!/bin/bash
# Publish the union corpus to Kaggle ONE SOURCE AT A TIME, as each finishes.
#
# The union normalises to ~111 GiB and `build_dataset` writes it source by
# source in alphabetical order. Waiting for the whole tree before starting the
# upload serialises two multi-hour jobs that have no reason to be serial: the
# moment `coco_train2017/` stops growing, its 18 GiB can be on its way to
# Kaggle while `ntire/` is still being written.
#
# Splitting across several Datasets costs nothing on the far side. The
# notebook already unifies MULTIPLE image mounts -- that is the mechanism five
# teammates' shards use -- and `kb.unify_mounts` locates each corpus root by
# the top-level names the manifest reports, so N mounts behave as one tree.
#
# Completion signal is the SUCCESSOR, not a file count. Normalisation legally
# skips unreadable images, so "count == expected" may never become true and
# the chain would hang on a source that finished fine. A source is done when
# the next one alphabetically has appeared; the last one is done when the
# build process exits.
#
# Uploads run one at a time, LAST, and at idle I/O priority. The corpus volume
# is a spinning disk, and that is the whole reason this section exists:
# measured 2026-08-30, `build_dataset` alone holds sda at 99% utilisation with
# 222 ms average read latency. Adding two archivers took the read latency to
# 319 ms, dropped normalisation from 73 img/s to 40, and got the first archive
# to 954 MB in 22 minutes -- a 7-hour projection for one 18 GiB source. Three
# sequential readers on one head do not share; they seek.
#
# So the chain waits for the build AND for the crop-vs-band probe to finish
# before it archives anything, and runs under `ionice -c3` (idle class) so any
# later reader still takes priority. The probe is the time-critical job; the
# uploads are not.
#
# `--dir-mode tar`, not zip. The tree is PNGs, which are already deflate
# compressed: zip spends CPU re-compressing incompressible bytes for no size
# saving. tar is a straight sequential write, which is also the access pattern
# this disk is least bad at.
set -u
BUILD_PID="${1:?usage: chain_union_upload.sh <build_pid>}"
NORM="${2:-/mnt/berstorage/techjam/experiments/data/normalized_union}"
STAGE_ROOT="${3:-/mnt/berstorage/techjam/experiments/data}"
export TMPDIR="${TMPDIR:-/home/administrator/.cache/kaggle_tmp}"
mkdir -p "$TMPDIR"

# Alphabetical, matching the order build_dataset walks them.
SOURCES=(coco_train2017 ntire open_images sid_set wildfake)

log() { echo "[$(date +%H:%M:%S)] $*"; }

wait_for_quiet_disk() {
  # The probe script owns the disk (and the GPU) as soon as the build exits.
  # Wait it out rather than seeking against it.
  while pgrep -f "run_crop_vs_band_probe.sh" > /dev/null 2>&1; do
    sleep 120
  done
  log "probe finished; disk is quiet"
}

wait_for_source_done() {
  local idx=$1 name=${SOURCES[$1]} next=""
  [ $((idx + 1)) -lt ${#SOURCES[@]} ] && next="${SOURCES[$((idx + 1))]}"
  while true; do
    if [ -n "$next" ] && [ -d "$NORM/$next" ]; then
      # The successor exists, so `name` is finished. Give the writer a moment
      # to flush the last few files rather than racing its final batch.
      sleep 20; log "$name done (successor $next started)"; return 0
    fi
    if ! kill -0 "$BUILD_PID" 2>/dev/null; then
      log "$name done (build process exited)"; return 0
    fi
    sleep 60
  done
}

upload_source() {
  local name=$1
  local stage="$STAGE_ROOT/union_upload_$name"
  local slug="techjam-aigc-union-$(echo "$name" | tr '_' '-')"
  if [ -d "$stage" ]; then log "$name: staged already, skipping stage"; else
    mkdir -p "$stage"
    # Hardlinks: the bytes are already on this filesystem and the tree must
    # keep `<source>/<bucket>/...` intact, because that prefix IS the
    # manifest's rel_path and the notebook resolves <mount>/<rel_path>.
    cp -al "$NORM/$name" "$stage/$name" || { log "$name: cp -al FAILED"; return 1; }
    cat > "$stage/dataset-metadata.json" <<JSON
{
  "title": "TechJam Track5 AIGC Union - $name",
  "id": "justinbersamin/$slug",
  "licenses": [{"name": "other"}]
}
JSON
  fi
  log "$name: uploading as $slug ($(du -sh --apparent-size "$stage" | cut -f1))"
  set -a; . ~/.kaggle/env; set +a
  # `|| return 1` matters: without it this function returned 0 on a failed
  # upload and the caller wrote a .done marker for a Dataset that does not
  # exist, so a re-run would skip it forever.
  if ionice -c3 nice -n 19 kaggle datasets create -p "$stage" --dir-mode tar \
        >> "logs/upload_union_$name.log" 2>&1; then
    log "$name: upload command returned 0"
  else
    log "$name: upload FAILED -- see logs/upload_union_$name.log"
    return 1
  fi
}

# Every source must be normalised AND the probe must be done before the first
# archive starts. Waiting per-source here would put an archiver back beside the
# build, which is the thing that was measured to be a mistake.
for i in "${!SOURCES[@]}"; do wait_for_source_done "$i"; done
wait_for_quiet_disk

for name in "${SOURCES[@]}"; do
  if [ -f "logs/upload_union_$name.done" ]; then log "$name: already published"; continue; fi
  upload_source "$name" && touch "logs/upload_union_$name.done"
done
log "all sources published (or attempted); check logs/upload_union_*.log"
