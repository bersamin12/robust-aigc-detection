"""audit -> normalise -> dedupe -> split -> manifest (spec §4).

This is the seam where every Plan 1 module meets: it is run once, by a human,
against real acquired data, to produce the manifest every later plan indexes
against positionally. The manifest is frozen the moment it is written —
re-running the split after feature banks exist would silently misalign
labels against features (spec §4.6, project constraints). `build_dataset`
therefore refuses to overwrite an existing manifest unless `force=True`.

Usage:
    python scripts/build_dataset.py --raw data/raw --out data/normalized \
        --demo-dir data/demo --manifest data/manifest.parquet

`--demo-dir` must sit OUTSIDE `--raw`: it is the organisers' benchmark, and
anything under `--raw` is scanned as training data.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from aigcdet.data.audit import audit_flags, audit_table
from aigcdet.data.dedupe import build_hash_index, find_leaks
from aigcdet.data.manifest import MANIFEST_COLUMNS, validate_manifest, write_manifest
from aigcdet.data.normalize import normalize_many
from aigcdet.data.presets import DEFAULT_CAP_KEY, DatasetPreset, load_preset
from aigcdet.data.sources import (
    classify, is_excluded_from_training, is_restricted_bucket, restriction_reason,
)
from aigcdet.data.splits import (
    DEFAULT_SEED,
    assign_splits,
    choose_heldout_generators,
    split_report,
)

IMG_EXT = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")


def _scan(root: str) -> list[str]:
    out = []
    for ext in IMG_EXT:
        out += glob.glob(os.path.join(root, "**", ext), recursive=True)
    return sorted(out)


def _to_markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table without the
    `tabulate` package, which is not part of this project's dependency set
    (pandas.DataFrame.to_markdown requires it and would raise ImportError)."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    )
    return "\n".join([header, sep, body])


def _load_licences(raw_dir: str) -> dict[str, str]:
    """Read <raw_dir>/LICENCES.json, written by scripts/acquire_data.py at
    acquisition time (JSON-lines, one `{dataset: licence_string}` object per
    line; the licence string embeds the source URL, e.g. "CC BY 4.0 ... -
    https://...").

    Spec §4.5 requires every dataset's licence and source URL to reach the
    manifest. A Task 8 review flagged that nothing verified the hand-off
    actually happens, so a missing file fails loudly here (FileNotFoundError)
    rather than the caller silently writing blank provenance someone has to
    reconstruct later. Missing or blank *entries* for a specific source being
    ingested are checked separately, by the caller, once it knows which
    sources are actually present.
    """
    path = os.path.join(raw_dir, "LICENCES.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run scripts/acquire_data.py first so every "
            "dataset's licence and source URL are recorded at acquisition "
            "time (spec §4.5) — this script refuses to fabricate them.")
    licences: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                licences.update(json.loads(line))
    return licences


def _sample_keep(keep, idx, cap: int, rng) -> int:
    """Thin positions `idx` down to `cap`, in place on `keep`. Returns the
    number dropped.

    `choice` runs on the POSITIONS and the complement is then set False:
    sampling which to KEEP rather than which to drop holds the draw's size
    fixed at `cap` however large the group is.
    """
    if cap <= 0 or len(idx) <= cap:
        return 0
    chosen = rng.choice(idx, size=cap, replace=False)
    keep[idx] = False
    keep[chosen] = True
    return int(len(idx) - cap)


def _cap_per_generator(
    df: pd.DataFrame, caps: dict[str, int], seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Thin each generator family to the cap that applies to it.

    `caps` maps a family name to its cap, with `DEFAULT_CAP_KEY` ("*") standing
    for every family without its own entry; 0 means uncapped. A mapping rather
    than one number because `sid_set` is a PSEUDO-generator -- it names a
    source, not a family -- so a preset that caps the 17 real WildFake families
    must be able to say "and leave that one alone" (see `data.presets`).

    Returns the thinned frame in its original row order and the per-family
    dropped counts. Authentic rows (generator "") are never touched here;
    `_cap_real_per_source` is the knob for those.

    Row order is preserved rather than regrouped because `_scan` sorts, and
    that sorted order is what makes a rebuild with the same seed produce the
    same manifest -- which matters more here than usual, since the manifest is
    indexed positionally by every feature bank built against it.
    """
    keep = np.ones(len(df), dtype=bool)
    dropped: dict[str, int] = {}
    rng = np.random.default_rng(seed)
    generated = df["generator"].to_numpy()
    for g in sorted({x for x in generated if x}):
        cap = caps.get(g, caps.get(DEFAULT_CAP_KEY, 0))
        n = _sample_keep(keep, np.flatnonzero(generated == g), cap, rng)
        if n:
            dropped[g] = n
    return df[keep].reset_index(drop=True), dropped


def _cap_real_per_source(
    df: pd.DataFrame, caps: dict[str, int], seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Thin each source's AUTHENTIC rows to `caps[source]`; 0 = uncapped.

    This is the source-balancing lever `augment.canonical` asks for by name and
    nothing implemented. The two authentic sources leak the label through
    different low-level channels -- WildFake through sharpness (var-Laplacian
    AUC 0.694 within the source), SID_Set through its noise floor (0.731) --
    and each dilutes the other, so the pooled figure sits below both. WildFake
    supplies 55,000 of 65,049 authentic rows in the frozen manifest, which is
    why the pooled statistics are very nearly WildFake's own. Capping the
    dominant source is what turns "two sources" into a mix rather than a
    rounding error. See `docs/dataset_presets.md`.

    A separate seed stream from `_cap_per_generator` would be redundant here:
    the two act on disjoint row sets (generator "" versus not), so one
    generator advanced by both cannot make either draw depend on the other's
    group sizes -- but they are drawn from one `default_rng(seed)` in call
    order, so changing the ORDER of the two calls changes the manifest. It is
    fixed: authentic first, then generated.
    """
    keep = np.ones(len(df), dtype=bool)
    dropped: dict[str, int] = {}
    rng = np.random.default_rng(seed)
    source = df["source"].to_numpy()
    label = df["label"].to_numpy()
    for src in sorted(caps):
        idx = np.flatnonzero((source == src) & (label == 0))
        n = _sample_keep(keep, idx, caps[src], rng)
        if n:
            dropped[src] = n
    return df[keep].reset_index(drop=True), dropped


def _drop_below_short_side(
    df: pd.DataFrame, floor: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop rows whose normalised short side is below `floor`.

    The residue `augment.canonical` calls irreducible: it band-limits every
    image to CANON_BAND_SIDE and then upscales, which equalises the band for
    everything AT or ABOVE that ceiling and can do nothing for what is below
    it. In the frozen manifest 1,308 images sit under 200px and every one of
    them is generated (1,260 are BigGAN at exactly 128px), so they are a
    permanent label leak with no authentic counterpart -- index balancing
    cannot remove them, because there is nothing to balance them against.

    Applied to the NORMALISED width/height rather than the native ones:
    normalisation caps the short side at `normalize.SHORT_SIDE` and never
    upscales, so below that cap the two agree, and the normalised image is the
    one the canonicaliser actually sees. The cost is that the images are
    normalised before being dropped, which is why this runs after step 1 and
    the caps above run before it -- a family's surviving count can therefore
    land slightly under its cap, and the counts written to splits.json are the
    ones actually kept.
    """
    if floor <= 0:
        return df, {}
    short = np.minimum(df["width"].to_numpy(), df["height"].to_numpy())
    below = short < floor
    dropped = (df.loc[below]
                 .assign(_k=lambda d: d["source"] + "/"
                         + d["generator"].where(d["generator"] != "", "real"))
                 ["_k"].value_counts().to_dict())
    return df[~below].reset_index(drop=True), {k: int(v) for k, v in dropped.items()}


def build_dataset(
    raw: str,
    out: str,
    demo_dir: str,
    manifest: str,
    workers: int = 16,
    docs_dir: str = "docs",
    seed: int = DEFAULT_SEED,
    force: bool = False,
    heldout_generators: list[str] | None = None,
    max_per_generator: int | dict[str, int] = 0,
    preset: DatasetPreset | None = None,
) -> pd.DataFrame:
    """Run the full audit -> normalise -> dedupe -> split -> manifest
    pipeline and write the manifest. Returns the DataFrame it wrote.

    Refuses to run at all if `manifest` already exists and `force` is not
    set: the manifest is frozen the instant it is written, because Plan 2's
    feature banks index against it positionally, and a silent re-split
    would misalign labels against features without ever raising an error.

    `heldout_generators` pins the held-out families instead of drawing them
    with `choose_heldout_generators`; the automatic choice is restricted to
    genuine generator families (spec §4.6), so a human who wants a specific
    pair says so here rather than reseeding until the draw obliges.

    `max_per_generator` keeps at most that many images per generator family,
    drawn deterministically from `seed`. It exists because the corpus is
    lopsided by construction: barring WildFake's authentic bucket (see
    `aigcdet.data.sources`) leaves every real image coming from SID_Set while
    the generated side keeps every WildFake family. Thinning the families
    rebalances the classes WITHOUT deleting a family -- `heldout_generator`
    and the LOTO rung both need every family to survive -- and without
    touching the raw tree, so the decision stays a rebuild flag rather than an
    irreversible `rm`. Authentic images are never capped BY THIS KNOB: they
    were the scarce side in the corpus it was written for. A mapping is also
    accepted, so a preset can cap the real generator families and leave a
    dataset-level pseudo-generator alone (see `data.presets`).

    `preset` supplies all four composition knobs at once from a file under
    `configs/datasets/`, and its name and note are written into
    `docs/splits.json` so a bank found on disk can be traced back to the
    corpus it came from. Passing a preset AND an overlapping argument raises:
    two sources of truth for one knob is the thing presets exist to remove.
    """
    real_caps: dict[str, int] = {}
    min_short_side = 0
    excluded_prefixes: list[str] = []
    if preset is not None:
        conflicts = sorted(
            k for k, v in (("heldout_generators", heldout_generators),
                           ("max_per_generator", max_per_generator)) if v)
        if conflicts:
            raise ValueError(
                f"preset {preset.name!r} already sets {conflicts}; passing "
                "both leaves no record of which one built the corpus. Edit "
                "the preset file instead.")
        max_per_generator = dict(preset.max_per_generator)
        real_caps = dict(preset.max_real_per_source)
        min_short_side = preset.min_short_side
        excluded_prefixes = preset.excluded_prefixes
        heldout_generators = list(preset.heldout_generators)
        print(f"preset {preset.name}: {' '.join(preset.note.split())}")
    gen_caps = ({DEFAULT_CAP_KEY: max_per_generator} if max_per_generator
                else {}) if isinstance(max_per_generator, int) else dict(
                    max_per_generator)
    if os.path.exists(manifest) and not force:
        raise FileExistsError(
            f"{manifest} already exists and would be overwritten. Any "
            "feature bank built against it indexes rows positionally, so "
            "silently rebuilding the manifest would misalign labels against "
            "features without raising an error. Pass force=True "
            "(--force on the CLI) only if you intend to discard it and "
            "everything built against it.")

    licences = _load_licences(raw)

    # `raw/<source>/<bucket>/...`, mapped to label/generator by the registry
    # in aigcdet.data.sources -- the same module acquire_data.py writes the
    # tree with. Inferring the label from the directory name here is what
    # labelled all ~5,000 COCO val2017 photographs AI-generated: their bucket
    # is `val2017`, not `real`. An unregistered source or bucket now raises.
    rows = []
    restricted: dict[str, int] = {}
    excluded_subpaths: dict[str, int] = {}
    for p in _scan(raw):
        rel = os.path.relpath(p, raw).split(os.sep)
        source = rel[0]
        bucket = rel[1] if len(rel) > 1 else ""
        # classify FIRST, so an unregistered source or bucket still raises
        # even when the bucket would then have been barred: a tree we cannot
        # read is a different problem from one we may not use.
        label, generator = classify(source, bucket)
        # A preset's sub-bucket exclusion. Dropped in the same place and for
        # the same reason as a licence-restricted bucket -- before
        # normalisation, because normalising images we then discard costs an
        # hour and ~17 GB. This is the level `restricted_buckets` cannot
        # reach: WildFake's authentic images are nested one BELOW the bucket
        # (`wildfake/real/<subset>/`), and `classify` reads `rel[1]`, so
        # barring five of those six subsets while keeping the sixth is not
        # expressible in the registry. It is also not a licence bar in force
        # for every corpus, which is what the registry means -- it is one
        # composition's decision, so it lives in the composition's file.
        rel_posix = "/".join(rel)
        hit = next((pre for pre in excluded_prefixes
                    if rel_posix.startswith(pre)), None)
        if hit is not None:
            excluded_subpaths[hit] = excluded_subpaths.get(hit, 0) + 1
            continue
        if is_restricted_bucket(source, bucket):
            # Barred here, before normalisation, not filtered out of the
            # manifest afterwards. Normalising images we may not use costs an
            # hour and ~17 GB, and the licence is about the copy on our disk,
            # not about the manifest row that points at it.
            key = f"{source}/{bucket}"
            restricted[key] = restricted.get(key, 0) + 1
            continue
        rows.append({"src": p, "label": label, "generator": generator, "source": source})
    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        raise ValueError(f"no images found under {raw}")
    for key, n in sorted(restricted.items()):
        print(f"barred {n} images in {key} by licence: "
              f"{restriction_reason(key.split('/')[0])}")
    # A sub-path that matches nothing excludes nothing, silently -- the same
    # failure `named_generators` is checked against below, and the reason
    # `SourceSpec.__post_init__` refuses a restriction on an undeclared
    # bucket. The path is untestable against a static registry (it names a
    # directory in someone's raw tree), so the scan is the only place it can
    # be checked.
    missed = [pre for pre in excluded_prefixes if pre not in excluded_subpaths]
    if missed:
        raise ValueError(
            f"exclude_subpaths matched no images under {raw} for "
            f"{[m.rstrip('/') for m in missed]}. An exclusion that excludes "
            "nothing leaves a corpus that disagrees with the preset "
            "describing it; check the path against the staged tree.")
    for key, n in sorted(excluded_subpaths.items()):
        print(f"excluded {n} images under {key.rstrip('/')} (preset)")
    # A preset that names a family the corpus does not have caps nothing and
    # holds nothing out, silently. The source registry is static and
    # `DatasetPreset` checks against it; generator names come from the data,
    # so they can only be checked here, against what was actually scanned.
    if preset is not None:
        present = set(raw_df["generator"]) - {""}
        missing = [g for g in preset.named_generators if g not in present]
        if missing:
            raise ValueError(
                f"preset {preset.name!r} names generator families {missing} "
                f"that are not in {raw}. A cap on a family that is not there "
                "caps nothing, and a hold-out on one holds nothing out; "
                f"present families are {sorted(present)}.")

    # Authentic first, then generated. The two act on disjoint row sets, but
    # they share one seeded generator, so the ORDER is part of the manifest.
    capped_real: dict[str, int] = {}
    if real_caps:
        raw_df, capped_real = _cap_real_per_source(raw_df, real_caps, seed)
        for src, n in sorted(capped_real.items()):
            print(f"capped authentic {src}: dropped {n} to {real_caps[src]}")
    capped: dict[str, int] = {}
    if gen_caps:
        raw_df, capped = _cap_per_generator(raw_df, gen_caps, seed)
        for g, n in sorted(capped.items()):
            print(f"capped {g}: dropped {n} to "
                  f"{gen_caps.get(g, gen_caps.get(DEFAULT_CAP_KEY, 0))}")
    print(f"scanned {len(raw_df)} raw images")

    # Reject a missing key, JSON `null`, `""`, AND a whitespace-only value
    # (e.g. `"   "`) -- all four flow through to a blank-in-substance
    # `licence` column otherwise. Falsiness is tested on the RAW value
    # first, before any str() coercion: str(None) == "None" is truthy, so
    # coercing first (as an earlier round of this fix did) would silently
    # accept a JSON-null licence. Only a truthy value is then checked for
    # being whitespace-only once stringified.
    missing_licences = sorted(
        s for s in set(raw_df["source"])
        if not licences.get(s) or not str(licences.get(s)).strip()
    )
    if missing_licences:
        raise ValueError(
            f"LICENCES.json has no usable (non-blank) entry for source(s) "
            f"{missing_licences}. Record every dataset's licence at "
            "acquisition time (spec §4.5) before building the manifest.")

    os.makedirs(docs_dir, exist_ok=True)

    # 1. Normalise. Unreadable files are skipped and recorded rather than
    # ending the run: at ~100k images streamed from third-party hosts, at
    # least one truncated or zero-byte file is near-certain.
    pairs, dsts = [], []
    for i, r in raw_df.iterrows():
        # Absolute: manifest.py documents `path` as absolute, and Plans 2
        # and 3 open row["path"] directly from notebooks and from Kaggle,
        # where the working directory is not this one.
        dst = os.path.abspath(
            os.path.join(out, r["source"], r["generator"] or "real", f"{i:07d}.png"))
        pairs.append((r["src"], dst))
        dsts.append(dst)
    sizes, failures = normalize_many(pairs, workers=workers)
    if failures:
        skipped_path = os.path.join(docs_dir, "normalize_skipped.json")
        with open(skipped_path, "w") as f:
            json.dump([{"src": src, "reason": reason} for src, reason in failures],
                      f, indent=2)
        print(f"skipped {len(failures)} of {len(pairs)} images that could not "
              f"be read; the full list is in {skipped_path}")
    ok = [i for i, size in enumerate(sizes) if size is not None]
    if not ok:
        raise ValueError(
            f"every one of the {len(pairs)} images under {raw} failed to "
            f"normalise; see {docs_dir}/normalize_skipped.json")
    raw_df = raw_df.iloc[ok].reset_index(drop=True)
    raw_df["path"] = [dsts[i] for i in ok]
    raw_df["width"] = [sizes[i][0] for i in ok]
    raw_df["height"] = [sizes[i][1] for i in ok]

    # 1b. Sub-band floor. Runs here because it needs the normalised
    # dimensions, and before the audit so the audit table describes the corpus
    # that was actually kept.
    dropped_sub: dict[str, int] = {}
    if min_short_side:
        before = len(raw_df)
        raw_df, dropped_sub = _drop_below_short_side(raw_df, min_short_side)
        print(f"dropped {before - len(raw_df)} images below short side "
              f"{min_short_side} (band floor): {dropped_sub}")
        if raw_df.empty:
            raise ValueError(
                f"min_short_side={min_short_side} dropped every one of the "
                f"{before} normalised images")

    # 2. Audit the RAW files -- the table profiles the two classes before
    # normalisation removes their container differences, and it is the
    # figure for the README. It runs after step 1 only so that it reads the
    # raw files normalisation has just proved decodable: audit_table opens
    # every image, so an undecodable one would otherwise abort the ~20-minute
    # audit pass before the skip list above could ever be produced.
    at = audit_table(raw_df["src"].tolist(), raw_df["label"].tolist(),
                     raw_df["source"].tolist())
    flags = audit_flags(at)
    with open(os.path.join(docs_dir, "data_audit.md"), "w") as f:
        f.write("# Pre-normalisation data audit\n\n")
        f.write(_to_markdown_table(at) + "\n\n## Flags\n\n")
        f.write("\n".join(f"- {x}" for x in flags) if flags else "- none")
    print(f"audit flags: {flags}")

    # 3. Leakage guard against the demo set (spec §4.1). find_leaks removes
    # nothing itself; we drop only the training-side matches it reports,
    # never a demo path — the demo set is never added to raw_df at all.
    # The demo side is STRICT: a demo image that cannot be hashed is a hole in
    # the guard, on the side the guard exists to protect.
    demo = build_hash_index(_scan(demo_dir), workers=workers)
    # The candidate side skips, and then DROPS what it skipped. A row that
    # could not be hashed was never checked against the demo set, so keeping
    # it would mean training on an image this step never saw -- the one thing
    # it exists to prevent. Dropping is the conservative reading, and the
    # count is recorded rather than left in a log.
    #
    # This is not hypothetical. A single COCO photograph carried an ICC
    # profile large enough that Pillow would write it into the normalised PNG
    # and then refuse to read it back, and the raise killed a 95-minute build
    # at its last step. The root cause is fixed in `data.normalize`, which now
    # strips the profile; this is the part that stops the NEXT odd file among
    # 180,000 from costing the same.
    cand = build_hash_index(raw_df["path"].tolist(), skip_unreadable=True,
                            workers=workers)
    unhashable = [p for p in raw_df["path"] if p not in cand]
    if unhashable:
        print(f"dropping {len(unhashable)} images that could not be hashed for "
              "the demo-leak check (an unchecked row is not trained on): "
              f"{unhashable[:5]}")
        raw_df = raw_df[~raw_df["path"].isin(set(unhashable))].reset_index(drop=True)
    leaks = find_leaks(cand, demo, max_distance=4)
    print(f"dropping {len(leaks)} images that near-duplicate the demo set")
    raw_df = raw_df[~raw_df["path"].isin(leaks)].reset_index(drop=True)

    # Spec §4.1(2): the organisers' demo benchmark may never be trained on.
    # Keyed on the SOURCE alone, never on the label: gating this on
    # `label == 0` meant a source mislabelled upstream slipped straight
    # through the exclusion it exists to enforce.
    excluded = raw_df["source"].map(is_excluded_from_training)
    if excluded.any():
        by_source = raw_df.loc[excluded, "source"].value_counts().to_dict()
        print(f"excluding {int(excluded.sum())} images from demo-benchmark "
              f"sources (spec §4.1): {by_source}")
    raw_df = raw_df[~excluded].reset_index(drop=True)

    raw_df["licence"] = raw_df["source"].map(licences)
    raw_df["split"] = ""

    df = raw_df[MANIFEST_COLUMNS]
    if heldout_generators:
        held = sorted(heldout_generators)
        print(f"held-out generators (pinned): {held}")
    else:
        held = choose_heldout_generators(df, n=2, seed=seed)
        print(f"held-out generators: {held}")
    df = assign_splits(df, heldout_generators=held, seed=seed)
    validate_manifest(df)
    write_manifest(df, manifest)
    print(split_report(df).to_string(index=False))
    with open(os.path.join(docs_dir, "splits.json"), "w") as f:
        json.dump(
            {"heldout_generators": held, "seed": seed, "leaked_dropped": len(leaks),
             "unhashable_dropped": len(unhashable),
             "normalize_skipped": len(failures),
             # The licence audit trail. The counts say what was dropped; the
             # reasons say why, which is the half a reader six months from now
             # cannot reconstruct.
             "restricted_dropped": restricted,
             "restriction_reasons": {
                 src: restriction_reason(src)
                 for src in sorted({k.split("/")[0] for k in restricted})},
             "max_per_generator": max_per_generator,
             "capped_dropped": capped,
             "max_real_per_source": real_caps,
             "capped_real_dropped": capped_real,
             "min_short_side": min_short_side,
             "below_short_side_dropped": dropped_sub,
             "excluded_subpath_dropped": excluded_subpaths,
             # The composition's identity. Without it a manifest records the
             # numbers it was built with but not which decision they were.
             "preset": preset.as_record() if preset is not None else None},
            f, indent=2,
        )
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--demo-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--docs-dir", default="docs",
                    help="where the audit table, the skip list and splits.json "
                         "are written. A second corpus MUST point this "
                         "somewhere else: these files are the frozen stream's "
                         "provenance and the default would overwrite them "
                         "with a different corpus's, silently.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--force", action="store_true",
                     help="overwrite an existing manifest at --manifest")
    ap.add_argument("--max-per-generator", type=int, default=0,
                    help="keep at most N images per generator family, drawn "
                         "deterministically from --seed; 0 (default) keeps "
                         "every image. Authentic images are never capped.")
    ap.add_argument("--heldout-generators", default="",
                    help="comma-separated generator families to hold out, "
                         "pinning the choice instead of drawing it")
    ap.add_argument("--preset", default="",
                    help="path to a corpus preset under configs/datasets/. "
                         "Supplies the caps, the sub-band floor and the "
                         "held-out families together, and records which "
                         "composition was built in docs/splits.json. Cannot "
                         "be combined with --max-per-generator or "
                         "--heldout-generators.")
    a = ap.parse_args()
    build_dataset(a.raw, a.out, a.demo_dir, a.manifest, workers=a.workers,
                  docs_dir=a.docs_dir, seed=a.seed, force=a.force,
                  max_per_generator=a.max_per_generator,
                  preset=load_preset(a.preset) if a.preset else None,
                  heldout_generators=[g for g in a.heldout_generators.split(",") if g])


if __name__ == "__main__":
    main()
