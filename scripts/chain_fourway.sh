#!/bin/bash
# The four-way fusion at full scale, and the nulls that make it a claim.
#
# WHY THIS EXISTS. `chain_siglip.sh` scripts the pairs and ONE three-way. The
# configuration the probe lattice actually recommends is the four-way --
#
#     band+crop dinov2regl  +  band+crop siglipso400m   0.9247   732,598,336 params
#
# -- and nothing at full scale computes it. Discovering that at 12:30, with the
# banks on disk and no job to consume them, is the avoidable failure this file
# removes. It waits its turn exactly as chain_siglip.sh does and touches no
# running job.
#
# WHY fusion_lattice AND NOT fuse_simplex. Both accept n parents, but their
# nulls differ and only one of them is the right null here. `fuse_simplex`
# compares the n-way against the best PAIR (its VERDICT string still says "the
# third parent", written when n was 3). For a four-way that is the wrong
# comparison: the honest question is whether the fourth arm beats the best
# THREE-way subset, and a four-way that merely beats a pair has not earned its
# fourth tower. `fusion_lattice` scores each parent once and then evaluates
# EVERY subset -- 6 pairs, 4 triples, 1 quad -- so the quad is read against the
# best triple, and the ladder of arities is visible rather than asserted.
#
# All eleven subsets draw on two backbones, so `MAX_BACKBONES = 2` tags every
# one of them legal and the constant does not need touching. That is a property
# of this arm set, not a general permission.
#
# ORDER IS DELIBERATE. Stages 1-2 are numpy over cached frames and answer the
# headline in minutes. Stage 3 RETRAINS eight heads on full-scale banks and
# costs hours. The cheap answer must not queue behind the expensive one.
set -uo pipefail
cd /workspace/robust-aigc-detection
log() { echo "[$(date +%H:%M:%S)] $*"; }

B=data/banks
O=outputs
RUN_SECOND_HOLDOUT=${RUN_SECOND_HOLDOUT:-1}

# DEVICE. Scoring loads checkpoints trained on cuda; second_holdout_lattice
# TRAINS. On 2026-08-31 the same banks, family and config gave crop_dinov2regl
# 0.2659 on sid_set under CPU init and 0.1698 under CUDA init -- ten points, from
# `torch.manual_seed` seeding two different init paths. A retrain that is to be
# read beside a cuda-trained primary number must itself be cuda-trained.
DEVICE=${DEVICE:-cuda}

# name=backbone:eval_bank:checkpoint -- the format both lattice scripts parse.
ARMS=(
  "band_dinov2regl=dinov2regl:$B/eval_full_band_dinov2regl:$O/rungs_full_band_dinov2regl/a3/checkpoint.pt"
  "crop_dinov2regl=dinov2regl:$B/eval_full_crop_dinov2regl:$O/rungs_full_crop_dinov2regl/a3/checkpoint.pt"
  "band_siglipso400m=siglipso400m:$B/eval_full_band_siglipso400m:$O/rungs_full_band_siglipso400m/a3/checkpoint.pt"
  "crop_siglipso400m=siglipso400m:$B/eval_full_crop_siglipso400m:$O/rungs_full_crop_siglipso400m/a3/checkpoint.pt"
)
# The TRAIN banks, needed by stage 3 only: it refits heads, so it needs the
# features the heads were fitted on, not just the eval grid.
TRAIN_ARMS=(
  "band_dinov2regl=dinov2regl:$B/full_band_dinov2regl:$B/eval_full_band_dinov2regl"
  "crop_dinov2regl=dinov2regl:$B/full_crop_dinov2regl:$B/eval_full_crop_dinov2regl"
  "band_siglipso400m=siglipso400m:$B/full_band_siglipso400m:$B/eval_full_band_siglipso400m"
  "crop_siglipso400m=siglipso400m:$B/full_crop_siglipso400m:$B/eval_full_crop_siglipso400m"
)

# --- wait for the four arms -------------------------------------------------
#
# Poll on the ARTEFACTS, not on a done-marker in someone's log. Two different
# drivers produce these four arms (full_scale.sh owns both dinov2regl arms,
# two separate full_scale_arm.sh invocations own the siglip ones), so there is
# no single log line that means "all four are ready". The eval bank's
# config.json and the a3 checkpoint are what stage 1 actually opens, so they
# are what this waits for.
need=()
for spec in "${ARMS[@]}"; do
  rest=${spec#*=}; ebank=${rest#*:}; ckpt=${ebank#*:}; ebank=${ebank%%:*}
  need+=("$ebank/config.json" "$ckpt")
done

idle=0; IDLE_LIMIT=${IDLE_LIMIT:-6}   # 6 x 48 s ~ 5 min of no driver at all
log "=== waiting for four full-scale arms (eval bank + a3 checkpoint each) ==="
for i in $(seq 1 1500); do        # up to ~20 h at 48 s
  missing=(); for f in "${need[@]}"; do [ -f "$f" ] || missing+=("$f"); done
  [ ${#missing[@]} -eq 0 ] && break
  # A DRIVER GAP IS NOT A DRIVER FAILURE. `chain_siglip.sh` waits for
  # full_scale.sh to exit and THEN launches its own full_scale_arm.sh, so
  # between those two events no process matches the extraction drivers even
  # though the work is proceeding normally. A single poll landing in that gap
  # must not abort a job that then has nothing to restart it. Require the
  # absence to persist, and count the chain scripts as drivers too.
  if pgrep -f "scripts/(full_scale(_arm)?|chain_siglip|queue_capacity)\.sh" > /dev/null; then
    idle=0
  else
    idle=$((idle + 1))
    log "  no driver visible (${idle}/${IDLE_LIMIT}), ${#missing[@]} artefact(s) missing"
    if [ "$idle" -ge "$IDLE_LIMIT" ]; then
      log "NO DRIVER for $((IDLE_LIMIT * 48))s and ${#missing[@]} artefact(s) still missing:"
      printf '  %s\n' "${missing[@]}"
      log "nothing will produce them; refusing to wait further."; exit 1
    fi
  fi
  [ $((i % 20)) -eq 1 ] && log "  still missing ${#missing[@]}: ${missing[0]##*/} ..."
  sleep 48
done
missing=(); for f in "${need[@]}"; do [ -f "$f" ] || missing+=("$f"); done
if [ ${#missing[@]} -ne 0 ]; then
  log "TIMED OUT with ${#missing[@]} missing:"; printf '  %s\n' "${missing[@]}"; exit 1
fi
log "all four arms present"

# The four eval banks must share a row set or a fused frame means nothing.
# `assert_fusion_parents` checks this inside both lattice scripts, but it
# checks it AFTER loading four full-scale banks; failing here costs seconds
# instead of minutes and names the mismatch in this log.
python3 - <<'PYEOF' || exit 1
import json, sys
keys = ("manifest_sha256", "n_views", "tier")
ref = None
for d in ("eval_full_band_dinov2regl", "eval_full_crop_dinov2regl",
          "eval_full_band_siglipso400m", "eval_full_crop_siglipso400m"):
    cfg = json.load(open(f"data/banks/{d}/config.json"))
    got = {k: cfg.get(k) for k in keys} | {"n_images": cfg.get("n_images")}
    if ref is None:
        ref, refd = got, d
    elif got != ref:
        print(f"ROW-SET MISMATCH: {d} {got} vs {refd} {ref}"); sys.exit(1)
print(f"row set agrees across four eval banks: {ref}")
PYEOF

# --- stage 1: every subset, equal and fitted, with the bootstrap -------------
log "=== (1/3) full-scale fusion lattice: 6 pairs + 4 triples + 1 quad ==="
# simplex-top 6 fits weights for EVERY subset at arity 2 and 3 (there are only
# 6 and 4), so no combination is reported equal-weighted merely because it fell
# outside a top-N cut. At arity 4 there is one subset and it is always fitted.
python3 -u scripts/fusion_lattice.py \
  $(printf -- "--arm %s " "${ARMS[@]}") \
  --max-arity 4 --simplex-top 6 --simplex-max-arity 4 --simplex-step 0.05 \
  --boot-n 1000 --boot-top 8 --device "$DEVICE" \
  --out docs/fusion_lattice_full_x4.json > logs/fourway_lattice.log 2>&1
log "  exit $? ; wrote docs/fusion_lattice_full_x4.json"
tail -30 logs/fourway_lattice.log

# --- stage 2: does anything beat the mean, on the arms we would ship ---------
log "=== (2/3) combining rules on the four full-scale arms ==="
# At probe scale no rule beat the mean on the shortlist (noisy_and was +0.0009
# at best-any and NEGATIVE on average over 247 subsets). Worth one confirmation
# on the exact four arms a bundle would carry, because that null is what
# justifies shipping the simplest possible combiner.
python3 -u scripts/fusion_rules.py \
  $(printf -- "--arm %s " "${ARMS[@]}") \
  --max-arity 4 --boot-n 1000 --boot-top 10 --device "$DEVICE" \
  --out docs/fusion_rules_full_x4.json > logs/fourway_rules.log 2>&1
log "  exit $? ; wrote docs/fusion_rules_full_x4.json"
tail -24 logs/fourway_rules.log

# --- stage 3: does the four-way transfer off the primary split ---------------
if [ "$RUN_SECOND_HOLDOUT" != "1" ]; then
  log "=== (3/3) SKIPPED (RUN_SECOND_HOLDOUT=0) ==="
else
  log "=== (3/3) second holdouts at full scale -- retrains 4 heads per family ==="
  # The primary split rests on two wildfake generators. Everything the probe
  # lattice recommends was chosen against them, and the four-way's whole claim
  # is that it is the combination that does NOT depend on which family is held
  # out -- it topped primary, mean AND maximin at probe scale. That claim is
  # only worth restating at full scale if it is re-measured at full scale.
  for fam in "sid_set:sid" \
             "styleGAN,GALIP,DF-GAN,GigaGAN,BigGAN,starGAN:gan"; do
    gens=${fam%:*}; short=${fam##*:}
    log "  --- family: $short ---"
    python3 -u scripts/second_holdout_lattice.py \
      $(printf -- "--arm %s " "${TRAIN_ARMS[@]}") \
      --holdout-generators "$gens" --config configs/rungs/a3.yaml \
      --max-arity 4 --max-backbones 2 \
      --simplex-top 2 --simplex-max-arity 3 --device "$DEVICE" \
      --out-dir "outputs/second_holdout_full_$short" \
      --out "docs/second_holdout_full_x4_$short.json" \
      > "logs/fourway_sh_$short.log" 2>&1
    log "    exit $? ; wrote docs/second_holdout_full_x4_$short.json"
    tail -16 "logs/fourway_sh_$short.log"
  done
fi

# --- summary ----------------------------------------------------------------
log "=== SUMMARY: the arity ladder at full scale ==="
python3 - <<'PYEOF'
import glob, json, os

lat = "docs/fusion_lattice_full_x4.json"
if not os.path.exists(lat):
    raise SystemExit("no lattice json -- stage 1 did not finish")
d = json.load(open(lat))
rows = d["combinations"] if "combinations" in d else d.get("rows", d)
if isinstance(rows, dict):
    rows = [{"name": k, **v} for k, v in rows.items()]

by_arity = {}
for r in rows:
    best = max(x for x in (r.get("equal"), r.get("fitted")) if x is not None)
    by_arity.setdefault(r["arity"], []).append((best, r))
print(f"{'arity':>5}  {'best':>7}  {'equal':>7}  {'fitted':>7}  combination")
prev = None
for k in sorted(by_arity):
    best, r = max(by_arity[k], key=lambda t: t[0])
    fit = r.get("fitted")
    delta = "" if prev is None else f"  ({best - prev:+.4f} over arity {k-1})"
    print(f"{k:>5}  {best:>7.4f}  {r['equal']:>7.4f}  "
          f"{'--' if fit is None else f'{fit:.4f}':>7}  {'+'.join(r['arms'])}{delta}")
    prev = best

# THE COMPARISON THIS SCRIPT WAS WRITTEN FOR.
if 4 in by_arity and 3 in by_arity:
    q = max(by_arity[4])[0]; t = max(by_arity[3])[0]
    verdict = ("the fourth arm earns its place"
               if q > t + 1e-9 else
               "NO gain over the best THREE-way: the fourth arm is cost "
               "without coverage")
    print(f"\nquad {q:.4f} vs best triple {t:.4f} -> {q - t:+.4f}  |  {verdict}")
    print("read that delta against the bootstrap interval in the json before "
          "believing it; at probe scale every multi-arm margin was a tie.")

for tag, path in (("sid_set", "docs/second_holdout_full_x4_sid.json"),
                  ("GAN family", "docs/second_holdout_full_x4_gan.json")):
    if os.path.exists(path):
        print(f"  {tag}: wrote {path}")
PYEOF
log "=== FOUR-WAY CHAIN DONE ==="
