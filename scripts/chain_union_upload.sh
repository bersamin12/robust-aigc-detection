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
# So the chain waits for the build, and then for the probe to finish COPYING
# its 24,000 images (~8 GB) onto the NVMe -- not for the probe itself. Once
# that copy lands, the probe's two arms and its CPU control read the SSD and
# touch sda not at all, so the archivers can have the spinning disk to
# themselves while the probe runs. Waiting for the whole probe instead would
# serialise two hours for nothing. `ionice -c3` (idle class) stays on anyway,
# so anything that DOES come back to sda still takes priority.
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
STAGED_MARKER=logs/probe_ssd_staged.done

log() { echo "[$(date +%H:%M:%S)] $*"; }

wait_for_probe_staging() {
  # The marker the probe writes once its images are on the SSD. If the probe
  # is not running at all -- it was never armed, or it already failed -- do
  # not block forever waiting for a marker nobody will write; the disk is
  # free in that case anyway.
  if ! pgrep -f "run_crop_vs_band_probe.sh" > /dev/null 2>&1; then
    log "probe is not running; nothing to wait for"; return 0
  fi
  log "waiting for the probe to stage its images onto the SSD"
  while [ ! -f "$STAGED_MARKER" ]; do
    if ! pgrep -f "run_crop_vs_band_probe.sh" > /dev/null 2>&1; then
      log "probe exited without staging -- proceeding, sda is free either way"
      return 0
    fi
    sleep 30
  done
  log "probe is on the SSD; sda is ours"
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

# Every source must be normalised, and the probe must have its images off sda,
# before the first archive starts. Waiting per-source here would put an
# archiver back beside the build, which is the thing that was measured to be a
# mistake.
for i in "${!SOURCES[@]}"; do wait_for_source_done "$i"; done
wait_for_probe_staging

for name in "${SOURCES[@]}"; do
  if [ -f "logs/upload_union_$name.done" ]; then log "$name: already published"; continue; fi
  upload_source "$name" && touch "logs/upload_union_$name.done"
done
log "all sources published (or attempted); check logs/upload_union_*.log"
