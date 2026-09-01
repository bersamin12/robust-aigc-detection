#!/bin/bash
# Prepare a rented multi-GPU pod to run backbone-probe arms.
#
# Everything here is the work that does NOT depend on the corpus, plus the
# corpus pull. Run it first: on a box billed by the hour, the setup that needs
# no data should happen while the data is still moving.
#
# Assumes you have already cloned the repo and are sitting in it.
#
# SECURITY, and this is not boilerplate. A rented host has root on the machine
# you are renting. Any credential you put here is disclosed to a third party.
# Use tokens created FOR this pod and revoke them when you destroy it:
#   - HF_TOKEN   only needed for dinov3l (gated, Meta's licence). Every other
#                candidate is ungated and needs nothing.
#   - ~/.kaggle/kaggle.json  needed to pull the private probe Dataset.
# Neither is read from this repo, and neither is written anywhere by this
# script.
set -uo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-/data}"
SLUG="${SLUG:-justinbersamin/techjam-aigc-probe-union}"
# Every checkpoint any arm might load. Warmed once here rather than four times
# in parallel: four processes racing to populate one HF cache directory is a
# corrupted download, not a fast one.
MODELS=(
  facebook/dinov2-large
  facebook/dinov2-with-registers-large
  google/siglip-so400m-patch14-384
  timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k
  facebook/convnextv2-huge-22k-384
)
# Gated, and deliberately separate: a missing HF_TOKEN must not fail the five
# ungated ones. See docs/model_licences.md.
GATED_MODELS=(facebook/dinov3-vitl16-pretrain-lvd1689m)

# Debian/Ubuntu images since 3.11 often ship `python3` and no `python` at all,
# and a bare `python` there is "command not found" three lines into a script on
# a box billed by the hour. Resolve it once.
PY_BIN="${PY_BIN:-$(command -v python || command -v python3)}"
[ -n "$PY_BIN" ] || { echo 'FATAL: no python or python3 on PATH' >&2; exit 1; }

# `pip install kaggle` drops its console script next to $PY_BIN, not onto PATH.
# With PY_BIN overridden to a venv interpreter (a Vast image ships no torch for
# the system python, so it must be), the bare `kaggle` further down is "command
# not found" AFTER the deps are installed and the checkpoints are cached -- the
# latest, most expensive point in the script at which to lose the corpus.
export PATH="$(dirname "$PY_BIN"):$PATH"

# FIRST, before anything that can fail. These were created on the last line of
# this script, which meant a bootstrap that died anywhere -- the torch guard,
# the model download, a bad credential -- left no logs/ for the NEXT command to
# redirect into, and `nohup ... > logs/x.log &` then failed with "No such file
# or directory" for a reason that has nothing to do with the command being run.
mkdir -p logs docs data/banks outputs

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "FATAL: $*" >&2; exit 1; }

kaggle_creds_present() {
  # THREE FORMS, because all three are in use. KAGGLE_API_TOKEN comes first
  # because the CLI prefers it over everything else: a `KGAT_`-prefixed Kaggle
  # ACCESS TOKEN is not a legacy API key, and putting one in kaggle.json's
  # `key` field authenticates as nobody -- public endpoints still answer (they
  # need no auth at all), so it looks like it worked right up until a PRIVATE
  # dataset returns 403. `kaggle.json` is what "Create New
  # Token" downloads; KAGGLE_USERNAME/KAGGLE_KEY is what this project's own
  # ~/.kaggle/env uses and what a rented pod is easiest to configure with.
  # Requiring only the file turned the env-var form into a confusing refusal.
  [ -n "${KAGGLE_API_TOKEN:-}" ] && return 0
  [ -f "$HOME/.kaggle/kaggle.json" ] && { chmod 600 "$HOME/.kaggle/kaggle.json"; return 0; }
  [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ] && return 0
  return 1
}
KAGGLE_HELP='no Kaggle credentials. Either export KAGGLE_API_TOKEN (a KGAT_ access
     token), write ~/.kaggle/kaggle.json
     ({"username":"...","key":"..."}, chmod 600) or export KAGGLE_USERNAME and
     KAGGLE_KEY. The probe and union Datasets are PRIVATE -- they carry NTIRE
     rows, which may not be published. Use a token created for THIS pod and
     revoke it when you destroy the pod: a rented host has root on the machine.'

# ---------------------------------------------------------------- 1. hardware
log "hardware"
command -v nvidia-smi >/dev/null || die "no nvidia-smi; this is not a GPU pod"
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
           --format=csv,noheader | sed 's/^/    /'
NGPU=$(nvidia-smi -L | wc -l)
NCPU=$(nproc)
RAM_GB=$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo)
log "    ${NGPU} GPU, ${NCPU} cores, ${RAM_GB} GB RAM"
[ "$NGPU" -ge 1 ] || die "no GPUs visible"

# The pipeline is roughly half CPU-bound on decode, 11 augmentation recipes and
# the handcrafted proxies -- the measured dinov3l run took 5 h 09 against a
# ~2.9 h GPU-only projection. So cores per GPU is a real constraint, not a
# footnote, and a thin box will not feed these cards.
PER_GPU=$((NCPU / NGPU))
log "    ${PER_GPU} cores per GPU"
[ "$PER_GPU" -ge 8 ] || log "    ! thin on CPU; expect the decode half to bind"

# ------------------------------------------------------------- 2. environment
log "installing"
# NOT upgrading pip. On a Debian-managed interpreter that fails with "Cannot
# uninstall pip 24.0, RECORD file not found" -- harmless, since nothing here
# needs a newer pip, but it looks like a fatal error at the top of a log on a
# box that is costing money per hour, and a scary line nobody needs is a cost
# of its own at 2am.
# --no-deps on the project itself: the pod's torch is driver-matched and must
# not be replaced. Same rule as the Kaggle notebooks, same reason.
"$PY_BIN" -m pip install -q --no-deps -e . || die "editable install failed"
# torch is recorded BEFORE and compared AFTER. timm and transformers both
# declare a torch dependency, and a pip run that decides to "satisfy" it
# replaces the pod's driver-matched build with a generic wheel -- which either
# fails to see the GPUs at all or silently runs on CPU. It is the single most
# expensive mistake available on a rented box, and it is silent.
TORCH_BEFORE=$("$PY_BIN" -c "import torch;print(torch.__version__)" 2>/dev/null)
"$PY_BIN" -m pip install -q numpy pillow scipy opencv-python-headless pandas \
    pyarrow scikit-learn tqdm pyyaml transformers timm || die "deps failed"
TORCH_AFTER=$("$PY_BIN" -c "import torch;print(torch.__version__)" 2>/dev/null)
if [ -z "$TORCH_AFTER" ]; then
  die "torch is not importable after the install. Nothing here can run."
elif [ -z "$TORCH_BEFORE" ]; then
  # NOT a replacement -- there was nothing to replace. The image shipped no
  # torch for this interpreter, so pip resolved one as a dependency of timm and
  # transformers. That is usually fine, and the CUDA check below is what
  # decides: a wheel built for the wrong CUDA imports cleanly and then sees no
  # GPUs. Said out loud because "pip chose your torch" is worth knowing.
  log "    ! no torch was installed for $PY_BIN; pip resolved $TORCH_AFTER"
  log "      If this interpreter is not the one you meant, re-run with"
  log "      PY_BIN=/path/to/python (e.g. \$CONDA_PREFIX/bin/python)."
elif [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
  die "pip REPLACED torch: $TORCH_BEFORE -> $TORCH_AFTER.
     The image's build was driver-matched and this one may not be. Reinstall
     $TORCH_BEFORE before running anything on the GPUs."
else
  log "    torch unchanged at $TORCH_BEFORE"
fi

"$PY_BIN" - <<'PY' || exit 1
import sys
import torch
print(f"    torch {torch.__version__}  cuda {torch.version.cuda}  "
      f"devices {torch.cuda.device_count()}")
assert torch.cuda.is_available(), "torch cannot see a GPU"
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    # sm_120 (Blackwell / RTX 5090) needs a recent torch; an older wheel loads
    # and then fails to launch kernels, which is a confusing way to lose an
    # hour. Say it now.
    print(f"      cuda:{i} {p.name} sm_{p.major}{p.minor} "
          f"{p.total_memory / 1024**3:.0f} GB "
          f"bf16={torch.cuda.is_bf16_supported()}")
import aigcdet.features.backbones as bb
print(f"    aigcdet importable: {len(bb.BACKBONES)} backbones registered")
PY

# ---------------------------------------------------------- 3. warm the cache
log "warming the model cache (${#MODELS[@]} ungated checkpoints)"
"$PY_BIN" - "${MODELS[@]}" <<'PY'
import sys
from transformers import AutoModel
for m in sys.argv[1:]:
    AutoModel.from_pretrained(m)
    print("    cached", m)
PY
if [ -n "${HF_TOKEN:-}" ]; then
  log "HF_TOKEN present; warming the gated checkpoint(s)"
  "$PY_BIN" - "${GATED_MODELS[@]}" <<'PY'
import sys
from transformers import AutoModel
for m in sys.argv[1:]:
    try:
        AutoModel.from_pretrained(m)
        print("    cached", m)
    except Exception as e:
        print(f"    SKIPPED {m}: {type(e).__name__}: {e}")
PY
else
  log "no HF_TOKEN -- skipping dinov3l, which is the EXPECTED path here."
  log "  Every probe candidate is ungated (docs/model_licences.md), and a"
  log "  rented host has root on the machine a token would sit on. dinov3l"
  log "  stays in the registry because the frozen-stream bars are its, but"
  log "  nothing on this pod needs to load it."
fi

# ------------------------------------------------------------------ 4. corpus
if [ -f "$DATA_DIR/manifest_union_probe.parquet" ]; then
  log "corpus already present at $DATA_DIR"
else
  log "pulling $SLUG into $DATA_DIR"
  kaggle_creds_present || die "$KAGGLE_HELP"
  "$PY_BIN" -m pip install -q kaggle
  mkdir -p "$DATA_DIR"
  kaggle datasets download -d "$SLUG" -p "$DATA_DIR" --unzip || die "pull failed"
fi

# Kaggle may hand back the per-directory tars as files rather than as an
# extracted tree, depending on how it processed the upload. Handle both rather
# than assuming, because the difference only shows up as "no rows resolve" an
# hour into an arm.
shopt -s nullglob
for t in "$DATA_DIR"/*.tar; do
  log "extracting $(basename "$t")"
  tar -xf "$t" -C "$DATA_DIR" && rm -f "$t"
done
shopt -u nullglob

log "corpus root: $DATA_DIR"
ls "$DATA_DIR" | sed 's/^/    /'

# --------------------------------------------------- 5. prove the rows resolve
# A tree can list correctly and resolve to nothing -- a directory level shifted
# by an extract, a mount under a different name. 300 real rows per manifest,
# before any GPU is spent.
log "checking that both manifests resolve"
"$PY_BIN" - "$DATA_DIR" <<'PY' || exit 1
import os, sys
import pandas as pd
root = sys.argv[1]
for name in ("manifest_union_probe.parquet", "eval_manifest_union_probe.parquet"):
    path = os.path.join(root, name)
    assert os.path.exists(path), f"missing {path}"
    rel = pd.read_parquet(path, columns=["rel_path"])["rel_path"]
    miss = [x for x in rel.sample(min(300, len(rel)), random_state=20260827)
            if not os.path.exists(os.path.join(root, x))]
    assert not miss, f"{name}: {len(miss)} of 300 do not resolve, e.g. {miss[:3]}"
    print(f"    {name}: {len(rel)} rows, 300 sampled all resolve")
# pandas/pyarrow can abort in a static destructor at interpreter shutdown --
# 'terminate called without an active exception' -- AFTER the work is done and
# printed. That non-zero exit then killed a bootstrap whose checks had all
# passed. os._exit skips the teardown entirely.
import os as _os; _os._exit(0)
PY

# ------------------------------------------------------------- 6. the pin
# The arms are only comparable with each other and with the dinov3l bars if
# they score the SAME 20,000 rows. Checked here so a wrong Dataset fails during
# setup rather than after four arms have run.
log "checking the manifest fingerprint"
"$PY_BIN" - "$DATA_DIR" <<'PY' || exit 1
import os, sys
import pandas as pd
sys.path.insert(0, "src")
from aigcdet.features.bank import manifest_fingerprint
EXPECT = {"manifest_union_probe.parquet":
          "fdb2c38eab6b8664fb5043d47df0c8d65f383a6123e4d9c5e5ebcb15c60570e3",
          "eval_manifest_union_probe.parquet":
          "7f863cfcca12f4e76ea8cc3e64ea9d11ccac1246e5c9d537eef04e135d98c9b0"}
for name, want in EXPECT.items():
    got = manifest_fingerprint(pd.read_parquet(os.path.join(sys.argv[1], name)))
    assert got == want, (
        f"{name} fingerprint is {got}, expected {want}. This is a DIFFERENT "
        f"cut, so nothing run against it is comparable with the dinov3l bars.")
    print(f"    {name}: {got[:16]}... matches")
# pandas/pyarrow can abort in a static destructor at interpreter shutdown --
# 'terminate called without an active exception' -- AFTER the work is done and
# printed. That non-zero exit then killed a bootstrap whose checks had all
# passed. os._exit skips the teardown entirely.
import os as _os; _os._exit(0)
PY

log "READY. Next: scripts/run_pod_arms.sh"
