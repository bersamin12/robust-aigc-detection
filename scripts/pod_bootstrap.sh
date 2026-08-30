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

log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "FATAL: $*" >&2; exit 1; }

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
"$PY_BIN" -m pip install -q --upgrade pip
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
if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
  die "pip replaced torch: $TORCH_BEFORE -> $TORCH_AFTER.
     The pod's build was driver-matched and this one may not be. Reinstall the
     original ($TORCH_BEFORE) before running anything on the GPUs."
fi
log "    torch unchanged at ${TORCH_BEFORE:-<none>}"

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
  [ -f "$HOME/.kaggle/kaggle.json" ] || die \
    "no ~/.kaggle/kaggle.json. The probe Dataset is PRIVATE (it carries NTIRE
     rows, which may not be published). Create a token for this pod and revoke
     it when you destroy the pod."
  chmod 600 "$HOME/.kaggle/kaggle.json"
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
PY

mkdir -p logs docs data/banks outputs
log "READY. Next: scripts/run_pod_arms.sh"
