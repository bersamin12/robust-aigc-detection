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

# Alphabetical, matching the order build_dataset walks them. This is the
# order the COMPLETION WAIT uses -- a source is done when its successor
# appears, so the list has to match the writer's own order to mean anything.
SOURCES=(coco_train2017 ntire open_images sid_set wildfake)

# Smallest first, which is a different order and a deliberate one: the first
# transfer is the only cheap measurement of a path we are about to push 128 GB
# down, and open_images at 9.3 GB buys that measurement for a twentieth of
# ntire's 62 GB. It also lands usable mounts early -- five Datasets are
# attached as one tree, but four of five attached is four fewer to wait for.
UPLOAD_ORDER=(open_images wildfake coco_train2017 sid_set ntire)
STAGED_MARKER=logs/probe_ssd_staged.done
BUILD_LOG=logs/build_union.log

log() { echo "[$(date +%H:%M:%S)] $*"; }

normalisation_finished() {
  # `build_dataset` prints the sub-band drop report (build_dataset.py:435)
  # AFTER the last source is normalised and BEFORE the audit/dedupe/freeze
  # phases. From that line every image in $NORM is final, so the corpus cannot
  # change under an archiver any more.
  #
  # A log line and not a file count, for the same reason the per-source wait
  # uses a successor: normalisation legally skips unreadable images (14 of
  # them here), so "count == expected" may never become true.
  grep -q "below short side" "$BUILD_LOG" 2>/dev/null
}

wait_for_build_exit() {
  # CORRECTNESS is not the constraint here; THROUGHPUT is, and this was
  # measured the wrong way round first.
  #
  # `normalisation_finished` proves the archiver cannot read a half-written
  # image, so the first version of this script started uploading the moment it
  # went true -- with the build still running its audit. That is safe and it is
  # also three times slower. Measured 2026-08-30: open_images archived and
  # uploaded beside the audit at 4.7 MiB/s end-to-end (9.3 GiB in 2002 s),
  # against ~22 MB/s of measured network and a sequential archive that should
  # run several times that. The audit is a SERIAL pass over all 150 GiB of the
  # RAW tree at ~17 MB/s and 245 seeks/s; a sequential archiver interleaved
  # with it does not get its own head, it gets every other seek.
  #
  # At 4.7 MiB/s the remaining 119 GiB is ~7 h. Waiting ~40 min for the build
  # and then running uncontended is ~2 h. So the wait is not caution, it is the
  # faster of the two orders -- and it costs the probe nothing, because the
  # probe is gated on the same manifest and stages onto the NVMe anyway.
  [ -n "$BUILD_PID" ] || return 0
  kill -0 "$BUILD_PID" 2>/dev/null || return 0
  log "build $BUILD_PID still running; waiting -- archiving beside its audit"
  log "  was measured at 4.7 MiB/s against ~22 MB/s uncontended"
  while kill -0 "$BUILD_PID" 2>/dev/null; do sleep 60; done
  log "build exited; sda is free"
}

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
if normalisation_finished; then
  log "normalisation is complete; every image in $NORM is final"
else
  for i in "${!SOURCES[@]}"; do wait_for_source_done "$i"; done
fi
wait_for_build_exit
wait_for_probe_staging

for name in "${UPLOAD_ORDER[@]}"; do
  if [ -f "logs/upload_union_$name.done" ]; then log "$name: already published"; continue; fi
  bytes=$(du -sb "$NORM/$name" | cut -f1)
  t0=$(date +%s)
  if upload_source "$name"; then
    touch "logs/upload_union_$name.done"
    dt=$(( $(date +%s) - t0 )); dt=$(( dt > 0 ? dt : 1 ))
    # Archive AND transfer, because the CLI does both inside one call
    # (`upload_files` is a serial loop that wraps a directory in an archive
    # and then uploads it), so there is no honest way to separate them from
    # out here. Quoted as an end-to-end rate for that reason.
    log "$name: $(( bytes / 1024 / 1024 )) MiB in ${dt}s = $(( bytes / dt / 1024 / 1024 )) MiB/s end-to-end"
  fi
done
log "all sources published (or attempted); check logs/upload_union_*.log"
