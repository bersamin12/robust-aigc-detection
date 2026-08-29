#!/usr/bin/env bash
# Pull more SID_Set authentic images (spec 4.5; CC BY 4.0, commercial use
# permitted), so the manifest can drop WildFake's real half.
#
# WildFake is a COMPILATION: its authentic images are re-published FFHQ,
# CelebA-HQ, AFHQ, ImageNet and LSUN, several of which are non-commercial.
# The 28 Aug webinar rules out non-commercial datasets outright, which
# overrides the research-prototype argument in docs/dataset_licences.md.
# Masking those 55,000 rows alone leaves val_internal with 1,010 authentic
# images -- too thin to fit a calibrator or select a rung on -- so the
# authentic half has to be rebuilt from a source that is actually clear.
#
# --limit counts NON-TAMPERED images and includes ones already on disk, and
# the kept stream runs about half real / half synthetic. 80,000 therefore
# targets roughly +30k real on top of the 10,049 already here. Class balance
# is then a manifest decision (subsample the generated side), not a download
# decision.
#
# It re-walks the stream from index 0 to skip what exists, so the first
# ~21 GB is re-fetched. That is the cost of a resumable streaming ingest and
# is cheaper than any alternative at this hour.
set -eu
# The repo root, derived from this script's own location rather than
# pinned to an absolute path: the tree has moved once already, and a
# stale `cd` sends a multi-hour job at the wrong data.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Do NOT redirect HF_HOME. It would hide both the stored login token and the
# 161 GB model cache under ~/.cache/huggingface/hub -- including the gated
# DINOv3 weights, which would then be re-downloaded on the next extraction.
# `streaming=True` does not cache parquet shards, so the default cache barely
# grows here anyway, and / has 225 GB free.
#
# The token is exported (never printed, never committed) because unauthenticated
# Hub requests are rate-limited, and this stream re-walks ~21 GB of already-
# acquired records before it reaches new ones.
if [ -f "$HOME/.cache/huggingface/token" ]; then
  HF_TOKEN=$(cat "$HOME/.cache/huggingface/token"); export HF_TOKEN
fi
export HF_HUB_DISABLE_TELEMETRY=1

echo "[acquire] start $(date '+%F %T')"
df -h /mnt/berstorage | tail -1
python scripts/acquire_data.py --dataset sid_set --limit 80000 --out data/raw
echo "[acquire] done $(date '+%F %T')"
df -h /mnt/berstorage | tail -1
echo "[acquire] real=$(ls data/raw/sid_set/real | wc -l) fake=$(ls data/raw/sid_set/fake | wc -l)"
