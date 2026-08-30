#!/usr/bin/env python3
"""Generate the four backbone-probe notebooks, one per candidate tower.

ONE MODEL PER NOTEBOOK, and a generator rather than four hand-maintained
copies. The two are in tension and the generator is how they are reconciled:
a teammate should open exactly the notebook for the arm they were given and
find no knob that can silently make it a different arm, while the eleven cells
that are identical across all four must STAY identical or the comparison stops
being a comparison.

Cells 3-10 -- clone, install, environment, auth -- are lifted verbatim from
`kaggle_stage_a.ipynb` AT GENERATION TIME rather than pasted here, for the
reason its CNN sibling states: "that bootstrap path is the one that has already
been debugged at 2am". Copying by hand is how the fourth copy ends up one fix
behind.

Regenerate after editing this file:

    python scripts/build_probe_notebooks.py

and commit the notebooks it writes. They are committed, not generated on
Kaggle: a teammate imports a notebook, not a build step.
"""
from __future__ import annotations

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS = os.path.join(REPO, "notebooks")
SOURCE_NB = os.path.join(NOTEBOOKS, "kaggle_stage_a.ipynb")

#: The one Dataset every arm attaches. It carries the 20,000 training images,
#: the 4,000-row eval subsample, `demo/` (the organisers' benchmark rows the
#: eval manifest names) and both manifests -- so unlike the full streams there
#: is no second mount and no two-link farm to line up.
SLUG = "techjam-aigc-probe-union"
#: Fingerprints of the two probe manifests, pinned so an arm run against a
#: re-cut probe fails in the verify cell rather than producing a number that
#: looks fine and is not comparable with anyone else's.
TRAIN_SHA = "fdb2c38eab6b8664fb5043d47df0c8d65f383a6123e4d9c5e5ebcb15c60570e3"
EVAL_SHA = "7f863cfcca12f4e76ea8cc3e64ea9d11ccac1246e5c9d537eef04e135d98c9b0"
RUNGS = ["a0", "a1", "a2", "a3", "a7_norecon"]
#: The frozen dinov3l ladder every arm is read against, from
#: docs/robustness_table.md via scripts/run_crop_vs_band_probe.sh.
DINOV3L_BARS = {"a0": 0.8611, "a1": 0.9037, "a2": 0.9037,
                "a3": 0.9012, "a7_norecon": 0.0296}

MODELS = [
    dict(
        backbone="dinov2regl",
        title="DINOv2 with registers, ViT-L/14 @ 518",
        why=(
            "Registers exist to absorb the high-norm artefact tokens DINOv2 "
            "develops in low-information patches -- flat sky, blur, smooth "
            "gradients. Those are exactly the patches a generator's decoder "
            "leaves its trace in, so this is a directed hypothesis about THIS "
            "task and not a version bump. Read it against `dinov2l`, which is "
            "the same checkpoint family without them."),
        watch=(
            "num_prefix_tokens is 5 here and 1 for dinov2l (CLS + 4 "
            "registers). The registry test asserts it against the real "
            "architecture, so nothing to do -- but that is the number that "
            "would silently corrupt every pooled vector if it were wrong."),
        minutes=120,
    ),
    dict(
        backbone="eva02l",
        title="EVA-02 ViT-L/14 @ 448 (timm)",
        why=(
            "A third pretraining PARADIGM rather than a fourth ViT: masked "
            "image modelling distilled from a CLIP teacher, then supervised "
            "fine-tuning on IN-22k/IN-1k. Neither DINOv2's self-distillation "
            "nor SigLIP's contrastive image-text objective."),
        watch=(
            "This is the only arm whose input is DOWNSAMPLED. EVA-02's "
            "pretrained_cfg sets fixed_input_size, so the tower admits exactly "
            "448 while canonicalise emits a 512 nominal side. It is also the "
            "cheapest arm, at 1024 tokens against dinov2regl's 1369 -- the "
            "handicap and the discount are one fact. Quote both together.\n"
            "\n"
            "It is also the only arm loaded through timm. Kaggle's image ships "
            "timm, so the install cell has nothing to do; if it is ever "
            "missing, load_backbone fails with the install line rather than a "
            "bare ImportError."),
        minutes=90,
    ),
    dict(
        backbone="convnextv2h",
        title="ConvNeXt V2 Huge @ 384",
        why=(
            "The conv paradigm at a scale that can compete. `convnextt` lost "
            "the A5 comparison at a0 0.4244, and 'the conv paradigm is weak "
            "here' and 'we ran a 27.8M-parameter conv tower' are not the same "
            "claim. This arm separates them."),
        watch=(
            "TWO things, and neither is optional.\n"
            "\n"
            "1. THE HEAD IS BIGGER. This bank is dim 5632 against the ViT "
            "arms' 1024-1152, and `train_head` takes dim_feat=bank.config"
            "['dim'] -- so a win here has two explanations and the number "
            "alone cannot choose. stages=(4,) already cut it from 8448; 5.5x "
            "is what is left. If this arm wins, re-run the best ViT rung with "
            "`hidden` raised to match this head's parameter count before "
            "calling it a finding. That is a CPU-only Stage B run of minutes.\n"
            "\n"
            "2. FLOAT16 IS NOT YET MEASURED HERE. ConvNeXt V2's GRN layer "
            "takes a global L2 norm over all spatial positions, and at 384^2 "
            "with 2816 channels that is a far larger reduction than "
            "convnextt's -- so convnextt's 'LayerNorm throughout' safety "
            "argument does not transfer. The finite-check cell below is the "
            "one that catches it, and it is why that cell exists."),
        minutes=135,
    ),
    dict(
        backbone="siglipso400m",
        title="SigLIP SO400M/14 @ 384",
        why=(
            "A shape-optimised tower -- 27 layers at width 1152, sized by "
            "architecture search rather than by the ViT-L/H ladder -- and the "
            "strongest open contrastive image encoder available under "
            "Apache-2.0. Read against `siglip2l`, with the caveat below."),
        watch=(
            "THIS ARM IS NOT A CLEAN COMPARISON WITH siglip2l. It uses "
            "SigLIP's own [0.5]*3 normalisation, which maps to [-1, 1]; "
            "siglip2l keeps ImageNet's statistics because its banks were built "
            "that way and re-extracting them is a separate job. So a gap "
            "between the two confounds the tower with the preprocessing. "
            "Against dinov2regl, eva02l and the dinov3l bars it is a fair "
            "comparison; against siglip2l specifically it is not."),
        minutes=120,
    ),
]


def _cell(kind, cid, source):
    text = source.rstrip("\n")
    cell = {"cell_type": kind, "id": cid, "metadata": {},
            "source": [l + "\n" for l in text.split("\n")[:-1]]
                      + [text.split("\n")[-1]]}
    if kind == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell


def _verbatim(src_cells, index, cid):
    """A cell copied byte-for-byte out of kaggle_stage_a.ipynb."""
    c = json.loads(json.dumps(src_cells[index]))
    c["id"] = cid
    c["metadata"] = {}
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None
    return c


def build(model, src_cells):
    b = model["backbone"]
    cells = []
    n = [0]

    def add(kind, source):
        cells.append(_cell(kind, f"probe-{b}-{n[0]:02d}", source))
        n[0] += 1

    def add_verbatim(index):
        cells.append(_verbatim(src_cells, index, f"probe-{b}-{n[0]:02d}"))
        n[0] += 1

    roster = "\n".join(
        f"| {'**this one**' if m['backbone'] == b else ''} | "
        f"`{m['backbone']}` | {m['title']} | ~{m['minutes']} min |"
        for m in MODELS)

    add("markdown", f"""# Backbone probe: `{b}`

**{model['title']}**

{model['why']}

One model per notebook, on purpose. There is no BACKBONE knob here: an arm is
only comparable with the others if the manifest, the standardisation policy and
the seed are identical across all four, and the cheapest way to guarantee that
is to leave nothing switchable. Run the other three from their own notebooks,
in parallel sessions.

| | notebook | tower | projected |
|---|---|---|---|
{roster}

## What this arm has to watch

{model['watch']}

## Held fixed across all four arms

- **The same 20,000-row probe manifest**, fingerprint `{TRAIN_SHA[:16]}...`,
  asserted in the verify cell. A different digest is a different 20,000 rows
  and the arm is not comparable with anyone else's.
- **`CANON_MODE = "band"`**, matching the `dinov3l` and `dinov2l` bars. This is
  a BACKBONE comparison; standardisation is a separate question with its own
  probe.
- **`SEED = 20260827`**, which is what makes the augmented views bit-identical
  across arms.

## Before you start

- Settings -> Accelerator -> **GPU T4 x2** (or P100).
- Settings -> Internet -> **On**.
- Add data -> **`{SLUG}`**. One Dataset, and only one: it carries the training
  images, the eval subsample, `demo/` and both manifests.
- No HuggingFace token. All four candidates are ungated (`docs/model_licences.md`).

**Do not publish the bank this produces as a Dataset.** A probe manifest
fingerprints differently from the real one, so a probe bank cannot verify,
merge, resume or fuse against a real bank -- every one of those refusals is
correct. Download `selection_probe_band_{b}.json` instead; it is kilobytes.""")

    add("markdown", "## 0. Parameters")
    add("code", f'''# ============ ONE MODEL, NO BACKBONE KNOB ============
BACKBONE = "{b}"          # this notebook is {b} and nothing else
SMOKE    = True           # True first: proves the chain in minutes
PHASES   = "auto"         # "auto", or e.g. "stage_a" / "eval_bank,ladder"
# =====================================================

# The probe corpus and its standardisation policy travel together, and neither
# is a free knob: band mode is what the dinov3l bars were measured under.
CANON_MODE = "band"
SEED       = 20260827
SPLITS     = "train,val_internal"
RUNGS      = {RUNGS!r}
TIER       = "ablation"

# Fingerprints of the two probe manifests, pinned. Asserted in the verify cell
# below -- see the header for why an unpinned probe silently ruins the arm.
EXPECT_TRAIN_SHA = "{TRAIN_SHA}"
EXPECT_EVAL_SHA  = "{EVAL_SHA}"

DATASET_SLUG = "{SLUG}"
# Recursive, because Kaggle mounts at either /kaggle/input/<slug> or
# /kaggle/input/datasets/<owner>/<slug>.
MANIFEST_GLOB      = f"/kaggle/input/**/{{DATASET_SLUG}}*/manifest_union_probe.parquet"
EVAL_MANIFEST_GLOB = f"/kaggle/input/**/{{DATASET_SLUG}}*/eval_manifest_union_probe.parquet"
# No separate DATA_GLOB. The images live in the SAME Dataset as the manifests,
# so the mount is the manifest's own directory -- derived rather than globbed
# for a second time, which is one fewer pattern that can match a different
# number of things than the first one did.

WORKERS          = 4
BATCH_SIZE       = 16
CHECKPOINT_EVERY = 200            # images between flushes = work at risk

REPO_URL = "https://github.com/bersamin12/robust-aigc-detection"
BRANCH   = "feat/robust-aigc-detection"
REPO_DIR = "/kaggle/working/robust-aigc-detection"

WORK      = "/kaggle/working"
BANK_DIR  = f"{{WORK}}/banks/probe_band_{{BACKBONE}}"
EVAL_DIR  = f"{{WORK}}/banks/eval_probe_band_{{BACKBONE}}"
RUNS_DIR  = f"{{WORK}}/outputs/rungs_probe_band_{{BACKBONE}}"
DOCS_DIR  = f"{{WORK}}/outputs/docs"
TABLE     = f"{{DOCS_DIR}}/robustness_table_probe_band_{{BACKBONE}}.md"
SELECTION = f"{{DOCS_DIR}}/selection_probe_band_{{BACKBONE}}.json"

print(f"arm: {{BACKBONE}}  canon={{CANON_MODE}}  seed={{SEED}}  smoke={{SMOKE}}")''')

    for i in range(3, 11):        # clone, install, environment, auth
        add_verbatim(i)

    add("markdown", """## 4. Attach the data, and prove it is intact -- do not skip this

One mount. `unify_mounts` locates the corpus root by the top-level names the
manifest itself reports, so Kaggle's wrapper directories do not have to be
guessed at.""")

    add("code", '''import pandas as pd

MANIFEST = sorted(glob.glob(MANIFEST_GLOB, recursive=True))
EVAL_MANIFEST_PATHS = sorted(glob.glob(EVAL_MANIFEST_GLOB, recursive=True))
assert MANIFEST, f"no probe Dataset attached (looked for {MANIFEST_GLOB})"
assert len(MANIFEST) == 1, f"{len(MANIFEST)} manifests match: {MANIFEST}"
assert len(EVAL_MANIFEST_PATHS) == 1, f"eval manifests: {EVAL_MANIFEST_PATHS}"
MANIFEST, EVAL_MANIFEST = MANIFEST[0], EVAL_MANIFEST_PATHS[0]
# Kaggle mounts at either /kaggle/input/<slug> or
# /kaggle/input/datasets/<owner>/<slug>, and the recursive glob above already
# found whichever it is. Taking the dirname cannot disagree with it.
DATA_MOUNTS = [os.path.dirname(MANIFEST)]
print("train manifest:", MANIFEST)
print("eval manifest: ", EVAL_MANIFEST)

# What the manifest says its root contains -- this locates the root inside the
# mount instead of guessing at Kaggle's wrapper directories.
EXPECTED = kb.top_level_names(pd.read_parquet(MANIFEST, columns=["rel_path"]))
print("dataset root should contain:", sorted(EXPECTED))

UNIFIED = kb.unify_mounts(DATA_MOUNTS, "/kaggle/temp/aigcdet_probe_root", EXPECTED)
DATA_ROOT = UNIFIED.root
print("unified root:", DATA_ROOT, "->", sorted(os.listdir(DATA_ROOT)))

# The EVAL manifest is rooted one level up -- its rel_paths start with `demo/`
# or `normalized_union/`. Both are top-level entries of this same Dataset, so
# unlike the full streams there is no second mount and no two-link farm: the
# unified root already resolves both. Proven on real rows below rather than
# assumed, because a farm can list correctly and resolve to nothing.
EVAL_ROOT = DATA_ROOT
_rel = pd.read_parquet(EVAL_MANIFEST, columns=["rel_path"])["rel_path"]
_missing = [x for x in _rel.sample(200, random_state=SEED)
            if not os.path.exists(os.path.join(EVAL_ROOT, x))]
assert not _missing, (
    f"{len(_missing)} of 200 sampled eval rows do not resolve under "
    f"{EVAL_ROOT}, e.g. {_missing[:3]}")
print("200 sampled eval rows all resolve")''')

    add("code", f'''t0 = time.time()
manifest, GATE = kb.open_verified_manifest(
    MANIFEST, DATA_ROOT,
    sample=2000 if SMOKE else None,
    # os.walk does not follow the farm's symlinked directories, so an
    # "extra files: 0" from it would be unearned. Skipped and said so.
    check_extra=not UNIFIED.linked,
)
print(kb.describe_gate(GATE))

# THE PIN. Every arm of this probe must be scored on the same 20,000 rows, and
# a re-cut probe manifest is the failure that produces a number which looks
# fine and is not comparable with anyone else's. Two minutes here against a
# two-hour arm that has to be thrown away.
assert GATE.manifest_sha256 == EXPECT_TRAIN_SHA, (
    f"probe manifest fingerprint is {{GATE.manifest_sha256}}, expected "
    f"{{EXPECT_TRAIN_SHA}}. This is a DIFFERENT 20,000 rows, so the arm would "
    f"not be comparable with the other three or with the dinov3l bars. "
    f"Re-cut it with scripts/cut_probe_manifest.py, or attach the published "
    f"{SLUG} Dataset.")

from aigcdet.features.bank import manifest_fingerprint
_eval_sha = manifest_fingerprint(pd.read_parquet(EVAL_MANIFEST))
assert _eval_sha == EXPECT_EVAL_SHA, (
    f"eval manifest fingerprint is {{_eval_sha}}, expected {{EXPECT_EVAL_SHA}}")

print(f"\\nboth manifests match the pinned fingerprints")
print(f"verified in {{time.time() - t0:.0f}}s")
print(manifest["split"].value_counts().to_string())''')

    add("markdown", f"""## 5. What this arm actually loads

Printed from the registry rather than typed here, and the **effective** dtype
rather than the declared one -- a `bfloat16` spec on a T4 or P100 falls back to
`float32` and runs about 3x slower, which is worth seeing now and not at hour
two.""")

    add("code", '''from aigcdet.features.backbones import BACKBONES, run_dtype

spec = BACKBONES[BACKBONE]
DIM = spec.dim
print(f"{spec.name}: {spec.hf_id}")
print(f"  image_size {spec.image_size}   dim {DIM}   "
      f"prefix_tokens {spec.num_prefix_tokens}   pool {spec.pool}")
print(f"  params {spec.params:,}   gated {spec.gated}   loader {spec.loader}")
print(f"  normalisation mean={spec.mean} std={spec.std}")
print(f"  dtype declared {spec.dtype}, EFFECTIVE on this GPU "
      f"{run_dtype(spec, 'cuda')}")
if run_dtype(spec, "cuda") != spec.dtype:
    print("  ! this GPU has no hardware for the declared dtype; expect ~3x "
          "the projected time")

# 20,000 rows is one shard on any of these. Asserted rather than assumed: a
# bank that overflows /kaggle/working dies at the flush, hours in.
n_rows = int(len(kb.select_splits(manifest, SPLITS)))
ok, why = kb.fits_in_working(n_rows, DIM, n_views=11)
print(f"\\n{n_rows} rows x 11 views x dim {DIM}: {why}")
assert ok, why''')

    add("markdown", """## 6. Which phases this session runs

Resumable, because a two-hour arm that dies at 100 minutes should not restart
from zero. Re-running every cell from the top continues where it stopped.""")

    add("code", '''import json

def _bank_done(d):
    """A bank is complete when its metadata says every row was written."""
    st = kb.read_resume_state(d)
    return st.exists and st.n_images > 0 and st.n_done >= st.n_images

STATUS = {
    "stage_a":   _bank_done(BANK_DIR),
    "eval_bank": _bank_done(EVAL_DIR),
    "ladder":    os.path.exists(SELECTION),
}
ORDER = ["stage_a", "eval_bank", "ladder"]
TODO = ([p for p in ORDER if not STATUS[p]] if PHASES == "auto"
        else [p.strip() for p in PHASES.split(",") if p.strip()])

for p in ORDER:
    mark = "done" if STATUS[p] else ("TODO" if p in TODO else "skip")
    print(f"  {p:10s} {mark}")
print(f"\\nthis session will run: {TODO or '(nothing)'}")''')

    add("markdown", f"""## 7. Smoke run

Two timed runs, then the marginal rate between them -- the first includes the
model download and would flatter nothing if used alone. The projection for this
arm is **~{model['minutes']} minutes**; read the measured number below rather
than trusting it.""")

    add("code", '''def stage_a_argv(limit=None):
    return kb.run_shard_argv(
        GATE, manifest_path=MANIFEST, root=DATA_ROOT, backbone=BACKBONE,
        out_dir=BANK_DIR, splits=SPLITS, shard=0, n_shards=1,
        resume=True, workers=WORKERS, batch_size=BATCH_SIZE,
        checkpoint_every=CHECKPOINT_EVERY, limit=limit,
        canon_mode=CANON_MODE)

if SMOKE and "stage_a" in TODO:
    import shutil
    timings = {}

    def timed_smoke(n):
        # A throwaway directory: a smoke bank of 8 rows would otherwise be
        # resumed from as if it were the real one.
        out = f"/kaggle/temp/smoke_{BACKBONE}_{n}"
        shutil.rmtree(out, ignore_errors=True)
        argv = [a if a != BANK_DIR else out for a in stage_a_argv(limit=n)]
        t = time.time()
        rc = kb.run_streaming(argv)
        assert rc == 0, f"smoke run of {n} exited {rc}"
        timings[n] = time.time() - t
        print(f"  {n} images in {timings[n]:.0f}s")
        return timings[n]

    small, large = 8, 40
    timed_smoke(small)
    timed_smoke(large)
    # Run 2 reads the model from cache, so a cheap tower can measure a
    # NEGATIVE marginal rate and marginal_rate refuses it -- correctly.
    # Escalate the large sample until the two timings separate.
    ceiling = 8 * large
    while True:
        try:
            RATE = kb.marginal_rate(small, timings[small], large, timings[large])
            break
        except ValueError as e:
            if large >= ceiling:
                raise SystemExit(
                    f"could not measure a marginal rate up to {large} images: "
                    f"{e}") from None
            print(f"\\n  {e}\\n  -> doubling the large sample and re-measuring")
            large *= 2
            timed_smoke(large)

    n_rows = int(len(kb.select_splits(manifest, SPLITS)))
    plan = kb.session_plan(n_rows, RATE, checkpoint_every=CHECKPOINT_EVERY)
    print(f"\\n{RATE:.3f} s/image marginal (model download excluded)")
    print(f"stage A over {plan.n_images} images -> {plan.hours:.1f} h, "
          f"{plan.sessions_needed} session(s), "
          f"{plan.minutes_at_risk:.0f} min at risk per kill")
    for note in plan.notes:
        print("  !", note)
    print("\\nSet SMOKE = False and re-run from the top to start the real arm.")
else:
    print("smoke: skipped")''')

    add("markdown", "## 8. Stage A")

    add("code", '''if "stage_a" in TODO and not SMOKE:
    argv = stage_a_argv()
    print(" ".join(argv[1:]), "\\n")
    t0 = time.time()
    rc = kb.run_streaming(argv)
    assert rc == 0, f"stage A exited {rc}"
    print(f"\\nstage A finished in {(time.time() - t0)/60:.1f} min")
    st = kb.read_resume_state(BANK_DIR)
    print(f"{st.n_done}/{st.n_images} images ({st.fraction_done:.1%})")
elif SMOKE:
    print("stage_a: SMOKE is still True")
else:
    print("stage_a: skipped")''')

    add("markdown", f"""## 9. Is the bank finite?

**Do not skip this cell.** On 2026-08-29 a five-hour DINOv3 bank came back
131,116 rows of NaN -- produced at full speed, with nothing raising, and the
only post-condition checked was the row count. `float16` overflow is silent.

For `{b}` specifically, see "What this arm has to watch" at the top.""")

    add("code", '''if not SMOKE and _bank_done(BANK_DIR):
    import numpy as np

    from aigcdet.features.bank import FeatureBank

    bank = FeatureBank(BANK_DIR)
    bank.check_invariants()

    # Sampled evenly rather than from the head: an overflow that begins
    # part-way through a corpus (a brighter source, a larger image) leaves the
    # first rows finite.
    feats = np.load(os.path.join(BANK_DIR, "feats.npy"), mmap_mode="r")
    idx = np.linspace(0, feats.shape[0] - 1, min(4096, feats.shape[0])).astype(int)
    sample = np.asarray(feats[idx], dtype=np.float32)
    n_bad = int((~np.isfinite(sample)).sum())
    print(f"sampled {len(idx)} of {feats.shape[0]} rows: "
          f"{n_bad} non-finite values, max|x| {np.abs(sample).max():.2f}")
    assert n_bad == 0, (
        f"{n_bad} non-finite values in the bank. This is a DTYPE problem, not "
        f"a bad image. Fix BackboneSpec.dtype for {BACKBONE} (bfloat16, or "
        f"float32 on a GPU without it) and extract to a NEW directory -- do "
        f"not delete this one from under a resume.")
    print("bank is finite")
else:
    print("no completed bank to check yet")''')

    add("markdown", """## 10. The evaluation bank

4,000 rows x 20 conditions. `--no-subsample`: the probe eval manifest is
already the cut, and subsampling it again would score a different set of rows
in each arm.""")

    add("code", '''if "eval_bank" in TODO and not SMOKE:
    argv = [sys.executable, f"{REPO_DIR}/scripts/extract_eval_bank.py",
            "--manifest", EVAL_MANIFEST, "--backbone", BACKBONE,
            "--out", EVAL_DIR, "--tier", TIER, "--root", EVAL_ROOT,
            "--device", "cuda", "--batch-size", str(BATCH_SIZE),
            "--seed", str(SEED),
            "--checkpoint-every", str(CHECKPOINT_EVERY), "--resume",
            "--no-subsample", "--canon-mode", CANON_MODE]
    print(" ".join(argv[1:]), "\\n")
    rc = kb.run_streaming(argv)
    assert rc == 0, f"eval bank exited {rc}"
else:
    print("eval_bank: skipped")''')

    add("markdown", """## 11. The ladder""")

    add("code", '''if "ladder" in TODO and not SMOKE:
    os.makedirs(DOCS_DIR, exist_ok=True)
    rung_cfgs = [f"{REPO_DIR}/configs/rungs/{r}.yaml" for r in RUNGS]
    for c in rung_cfgs:
        assert os.path.exists(c), f"missing rung config {c}"

    argv = [sys.executable, f"{REPO_DIR}/scripts/run_ablation.py",
            "--bank", BANK_DIR, "--eval-bank", EVAL_DIR,
            "--rungs", *rung_cfgs,
            "--tier", TIER, "--device", "cuda",
            "--out", TABLE, "--selection", SELECTION,
            "--heatmap", f"{DOCS_DIR}/robustness_heatmap_probe_band_{BACKBONE}.png",
            "--out-dir", RUNS_DIR]
    print(" ".join(argv[1:]), "\\n")
    rc = kb.run_streaming(argv)
    assert rc == 0, f"ladder exited {rc}"
else:
    print("ladder: skipped")''')

    add("markdown", f"""## 12. Read the result

`heldout_robust_tpr_at_1pct`, at the same rung, against the frozen `dinov3l`
ladder. **Not clean AUC** -- degraded generalisation is what this project
selects on.

Two caveats travel with every number below, and neither is optional:

- **SCALE.** 20,000 rows against the corpus's 375,358. This ranks candidates;
  it does not settle the shipped choice.
- **POPULATION.** The dinov3l bars were measured on the frozen 138,116-row
  corpus, which is a different population from the union probe. A gap of a few
  points against them is not, on its own, evidence about the tower.""")

    add("code", f'''bars = {DINOV3L_BARS!r}

if os.path.exists(SELECTION):
    sel = json.load(open(SELECTION))
    print(f"arm: {{BACKBONE}}  (dim {{DIM}}, {{spec.params:,}} params, "
          f"{{run_dtype(spec, 'cuda')}})")
    print(f"headline: {{sel.get('headline')}}")
    print(f"metric:   {{sel.get('metric')}}\\n")
    print(f"{{'rung':14s}} {{'this arm':>10s}} {{'dinov3l':>10s}} {{'delta':>10s}}")
    for rung, v in sorted(sel.get("summary", {{}}).items()):
        got = v.get("heldout_robust_tpr_at_1pct")
        bar = bars.get(rung)
        if got is None:
            continue
        delta = f"{{got - bar:+.4f}}" if bar is not None else "--"
        print(f"{{rung:14s}} {{got:10.4f}} "
              f"{{(f'{{bar:.4f}}' if bar is not None else '--'):>10s}} {{delta:>10s}}")
    print(f"\\ntable: {{TABLE}}")
    print("\\nSCALE:      20,000 rows vs the corpus's 375,358.")
    print("POPULATION: the dinov3l bars are from the frozen 138,116-row corpus.")
    print("\\nDownload the selection JSON; do not publish the bank as a Dataset.")
else:
    print("no ladder output yet -- run the remaining phases in a new session")

for d in (BANK_DIR, EVAL_DIR):
    st = kb.read_resume_state(d)
    if st.exists:
        print(f"  {{d}}: {{st.n_done}}/{{st.n_images}} ({{st.fraction_done:.0%}})")''')

    add("markdown", """## The 2am playbook

| What you see | What to do |
|---|---|
| `CUDA out of memory` | Lower `BATCH_SIZE` to 8, then 4. **Retryable** -- nothing repeats. |
| `MemoryError`, kernel dies | Lower `WORKERS`. Retryable. |
| `ReadTimeout`, `ConnectionError` | The clone or the model download. Just re-run. |
| non-finite values in the bank | **Fatal for this dtype.** Extract to a NEW directory after fixing `BackboneSpec.dtype`; do not delete the old one from under a resume. |
| `probe manifest fingerprint is ...` | **Fatal.** You attached a different probe cut. This arm would not be comparable. |
| `cannot resume the bank at ...` | **Fatal.** A parameter moved between sessions. Restore it, or extract to a new directory. |
| `no probe Dataset attached` | You attached the wrong Dataset, or the share has not reached you. |
| `gated repo` / `401` / `403` | Should be impossible here -- all four candidates are ungated. It means the mirror, not auth. |

Do **not**: change `BACKBONE`, `SEED`, `CANON_MODE` or `SPLITS` between
sessions of the same arm; delete a bank that refuses to resume; or skip the
finite-check cell. A `pip install torch` would replace Kaggle's driver-matched
build and cost the session.

Paste an error below and it will say whether re-running can possibly help.""")

    add("code", '''ERROR = """paste the error here"""
print(kb.explain(ERROR))''')

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.13",
                              "mimetype": "text/x-python",
                              "file_extension": ".py",
                              "pygments_lexer": "ipython3",
                              "nbconvert_exporter": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    src_cells = json.load(open(SOURCE_NB, encoding="utf-8"))["cells"]
    for model in MODELS:
        nb = build(model, src_cells)
        path = os.path.join(NOTEBOOKS, f"kaggle_probe_{model['backbone']}.ipynb")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {os.path.relpath(path, REPO)}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
