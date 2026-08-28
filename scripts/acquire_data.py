"""Subset download for WildFake, SID_Set, and COCO val2017 (spec §4.1).

Pulls selected generator subsets and parquet shards rather than whole
repositories — download volume is the binding risk, not compute (spec §4.4).
Records each dataset's licence, which the manifest and README require (§4.5).

Usage:
    python scripts/acquire_data.py --dataset sid_set --limit 30000 --out data/raw
    python scripts/acquire_data.py --dataset wildfake --limit 30000 --out data/raw \
        --generators adm,styleGAN,real_ffhq,personalizedSD_finetune
    python scripts/acquire_data.py --dataset coco_val2017 --out data/raw

    # The organisers' demo benchmark. A DIFFERENT VERB and a DIFFERENT
    # DESTINATION ARGUMENT, because it must never land in the training tree:
    python scripts/acquire_data.py --dataset wildfake_benchmark \
        --benchmark-dir data/demo

Two properties every routine here has, because these are hour-scale downloads
from third-party hosts that WILL be interrupted or hit a bad file:

- **Resumable.** Destination names are pure functions of the source identity
  (a stream index for SID_Set, a hash of the archive-relative path for
  WildFake), never of how many images happened to succeed before. Re-running
  skips what is already on disk instead of restarting or renumbering. Every
  image is written through a `.part` file and renamed, so an interrupted write
  cannot leave a truncated file that a resume mistakes for a finished one.
- **A single bad image does not end the run.** One CMYK JPEG killed a real
  SID_Set run 206 images in. Failures are recorded and reported, in the same
  spirit as `normalize_many`.

This script is not exercised against the network in CI: it pulls tens of
gigabytes from third-party hosts and is meant to be run by a human who can
confirm each source's licence before the download starts. Its logic IS tested,
against archives planted on disk.
"""
from __future__ import annotations

import argparse
import collections
import fnmatch
import json
import os
import re
import shutil
import zipfile

from aigcdet.data import wildfake as wf
from aigcdet.data.normalize import save_png
from aigcdet.data.sources import LICENCES, SOURCES, is_safe_generator, raw_subdir
from aigcdet.data.splits import DEFAULT_SEED

#: Record fields SID_Set may carry the generating model under. Where one is
#: present the image is filed under that generator, so it stays a real
#: generator family in the manifest instead of collapsing into the
#: dataset-level pseudo-generator "sid_set" (which spec §4.6 cannot hold out).
_GENERATOR_FIELDS = ("generator", "model", "generator_name", "source_model")
_SAFE_GENERATOR = re.compile(r"^[A-Za-z0-9._-]+$")

#: Where WildFake archives and label CSVs land inside `--out`. Archives are
#: deleted the moment they are extracted (they run to 53 GB), the CSVs are
#: kept: they are ~10 MB each and a resume needs them again.
WILDFAKE_CACHE = "_wildfake_cache"


def _record_generator(rec: dict, source: str) -> str:
    """The generator this record names, or "" if it does not name a usable one.

    The value is untrusted third-party data that becomes a directory name, so
    it must clear two separate gates: a plain-identifier character class (no
    separators, no whitespace), and `sources.is_safe_generator`, which also
    rejects `"."`/`".."` — which contain no separator yet still escape a
    level — and any name that aliases one of the source's declared buckets,
    since a record whose generator field read "real" would otherwise be
    written into the authentic bucket and read back as label 0.

    Returns "" rather than raising: an odd record must fall back to the
    unattributed bucket, not end a 30,000-image streaming download.
    """
    for key in _GENERATOR_FIELDS:
        value = rec.get(key)
        if not isinstance(value, str):
            continue
        value = value.strip()
        if _SAFE_GENERATOR.match(value) and is_safe_generator(source, value):
            return value
    return ""


# ---------------------------------------------------------------------------
# SID_Set
# ---------------------------------------------------------------------------

def sid_set_dest(out: str, rec: dict, index: int) -> str:
    """Where record `index` of the SID_Set stream is written.

    Named by the STREAM POSITION, not by a counter of successful saves. The
    counter version renumbered every subsequent image the moment one record
    was skipped, so an interrupted run and its restart — which skip a
    different set, because "skipped" now includes images that failed to
    decode — disagreed about which file is which, and resume was impossible.
    The stream is ordered, so `index` is stable across runs.
    """
    label = 0 if rec["label"] == 0 else 1
    # raw_subdir is the inverse of the mapping build_dataset.py reads the
    # tree back with, so the two scripts cannot drift apart.
    sub = raw_subdir("sid_set", label,
                     "" if label == 0 else _record_generator(rec, "sid_set"))
    return os.path.join(out, "sid_set", sub, f"{index:07d}.png")


def _write_ingest_report(out: str, source: str, modes: collections.Counter,
                         failures: list[dict]) -> str:
    """Record what acquisition changed about each image, merging with any
    earlier run's report.

    Spec §4.2 wants the ingest audit to profile native format and encoding per
    class per source, and `aigcdet.data.audit` reads the RAW TREE — so for a
    source that streams already-decoded images (SID_Set), every native mode and
    container is gone before the audit can see it. `audit_table` now carries a
    `mode_top` column, which covers sources whose raw files are the originals
    (WildFake, COCO); this file is the equivalent evidence for the one source
    where it cannot be. Without it, "3 CMYK JPEGs were converted to RGB with no
    ICC transform" is invisible.
    """
    path = os.path.join(out, source, "ingest_report.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prior_modes: dict[str, int] = {}
    prior_failures: list[dict] = []
    if os.path.exists(path):
        with open(path) as f:
            prior = json.load(f)
        prior_modes = prior.get("modes", {})
        prior_failures = prior.get("failures", [])
    merged = collections.Counter(prior_modes)
    merged.update(modes)
    seen = {(f.get("index"), f.get("reason")) for f in prior_failures}
    failures = prior_failures + [f for f in failures
                                 if (f.get("index"), f.get("reason")) not in seen]
    with open(path, "w") as f:
        json.dump({"source": source, "modes": dict(sorted(merged.items())),
                   "failures": failures}, f, indent=2)
    return path


def acquire_sid_set(out: str, limit: int) -> int:
    """Stream SID_Set into `out/sid_set/<bucket>/`. Returns the image count.

    Already-present images count against `limit`, so a resumed run converges on
    the same set as an uninterrupted one rather than acquiring `limit` MORE.
    """
    from datasets import load_dataset  # pip install datasets
    ds = load_dataset("saberzl/SID_Set", split="train", streaming=True)
    os.makedirs(out, exist_ok=True)
    modes: collections.Counter = collections.Counter()
    failures: list[dict] = []
    n = 0
    for index, rec in enumerate(ds):
        if n >= limit:
            break
        # SID_Set labels: 0 real, 1 fully synthetic, 2 tampered.
        # Tampered is out of scope for the binary task (spec §4.1).
        if rec.get("label") == 2:
            continue
        dest = sid_set_dest(out, rec, index)
        if os.path.exists(dest):
            n += 1                       # resume: already acquired
            continue
        try:
            im = rec["image"]
            # PNG cannot hold every mode PIL can decode: a CMYK JPEG raises
            # "cannot write mode CMYK as PNG" and used to end the whole run.
            # save_png converts only what PNG cannot store and reports what it
            # wrote, so the conversion is recorded rather than silent.
            written = save_png(im, dest)
            modes[f"{im.mode}->{written}"] += 1
        except Exception as e:  # one bad record must not end a 30k download
            failures.append({"index": index, "reason": f"{type(e).__name__}: {e}"})
            continue
        n += 1
    report = _write_ingest_report(out, "sid_set", modes, failures)
    converted = {k: v for k, v in modes.items() if k.split("->")[0] != k.split("->")[1]}
    print(f"sid_set: wrote {n}"
          + (f", converted modes {converted}" if converted else "")
          + (f", skipped {len(failures)} unreadable records" if failures else "")
          + f"; ingest report {report}")
    return n


# ---------------------------------------------------------------------------
# WildFake
# ---------------------------------------------------------------------------

def _hub_download(member: str, cache_dir: str) -> str:
    """Fetch one repo-relative file from the WildFake dataset repo.

    Skips the download when the file is already cached, which is what makes an
    interrupted multi-part run resumable and what lets the tests exercise this
    module's real code path against archives planted on disk.
    """
    local = os.path.join(cache_dir, *wf.split_segments(member))
    if os.path.exists(local):
        return local
    try:
        from modelscope.hub.file_download import dataset_file_download
    except ImportError:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "modelscope is required to download WildFake: pip install modelscope")
    return dataset_file_download(dataset_id=wf.DATASET_ID, file_path=member,
                                 local_dir=cache_dir)


def _hub_list_files() -> list[str]:  # pragma: no cover - network
    """Every repo-relative path in the WildFake repo, for expanding the
    `part_*.zip` globs. Only called for a subset whose archive is a glob."""
    from modelscope.hub.api import HubApi
    out = []
    for entry in HubApi().get_dataset_files(wf.DATASET_ID):
        if isinstance(entry, str):
            out.append(entry)
        elif isinstance(entry, dict):
            path = entry.get("Path") or entry.get("Name") or entry.get("path")
            if path:
                out.append(path)
    return out


def _expand_archives(pattern: str, list_files) -> list[str]:
    if "*" not in pattern:
        return [pattern]
    names = sorted(n for n in list_files() if fnmatch.fnmatch(n, pattern))
    if not names:
        raise SystemExit(
            f"no file in {wf.DATASET_ID} matches {pattern!r}. The multi-part "
            "archive layout has changed; update aigcdet.data.wildfake.SUBSETS.")
    return names


def _extract_wanted(archive: str, wanted: dict[str, tuple[str, str]],
                    subsets: set[str], guard) -> collections.Counter:
    """Extract exactly the members of `archive` that `wanted` asks for.

    Streams member by member (`copyfileobj`) and never calls `extractall`:
    these archives run to 53 GB against 268 GB of free disk, and only a few
    thousand of their members are wanted.

    Nothing from the archive reaches the filesystem path — destinations were
    computed by `wildfake.training_dest`/`benchmark_dest` from the CSV, so a
    member named `../../etc/x` cannot escape. And every member that matches a
    wanted key is put through `guard`, the plan's per-path benchmark gate: the
    CSV said this image belonged here, and this asks the ARCHIVE the same
    question. The two plans answer it in opposite directions, which is why the
    gate is passed in rather than hard-coded here.
    """
    got: collections.Counter = collections.Counter()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            hit = wanted.get(wf.tail_key(info.filename))
            # `hit[0] not in subsets` is not redundant: `wanted` spans every
            # requested subset, and a member of THIS archive can match the
            # tail of a row belonging to a subset that lives in a different
            # archive. Writing it would put one generator's image into
            # another's bucket, with the count still looking right.
            if hit is None or hit[0] not in subsets:
                continue
            name, dest = hit
            guard(info.filename, name)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            # Write through `.part` and rename. Resume decides what is already
            # done by `os.path.exists(dest)`, so an interruption mid-copy must
            # leave nothing at `dest` — a truncated image that a later run
            # counts as finished is worse than a missing one, because it
            # reaches the manifest.
            tmp = dest + ".part"
            try:
                with z.open(info) as src, open(tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                os.replace(tmp, dest)
            except BaseException:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise
            del wanted[wf.tail_key(info.filename)]
            got[name] += 1
    return got


def _run_extraction(wanted: dict[str, tuple[str, str]],
                    archives: list[tuple[str, set[str]]],
                    cache: str, fetch, list_files, guard,
                    already: set[str]) -> collections.Counter:
    """Download, extract, and DELETE one archive at a time.

    Disk is the binding constraint (268 GB free against archives of up to
    53 GB), so at most one archive exists on disk at any moment and it is
    removed before the next is fetched — including when extraction raises,
    since a half-downloaded archive must be re-fetched rather than reused.

    Stops dead if the FIRST archive fetched for a subset yields none of that
    subset's images. `Advanced`/`Typical` is a version axis, so a subset lives
    in exactly one tree; pointing it at the other matches nothing across
    every part of a ~372 GB glob. Warning about a shortfall at the end would
    mean paying for all seven parts to learn the registry is wrong. A subset
    that has already produced images (this run or a previous one) is exempt:
    a later part of a multi-part tree holding none of the capped selection is
    ordinary, not a misdeclaration.
    """
    got: collections.Counter = collections.Counter()
    for pattern, subsets in archives:
        for member in _expand_archives(pattern, list_files):
            pending = {name for name, _ in wanted.values() if name in subsets}
            if not pending:
                break  # every subset this archive serves is complete
            archive = fetch(member, cache)
            try:
                got += _extract_wanted(archive, wanted, subsets, guard)
            finally:
                if os.path.exists(archive):
                    os.remove(archive)
            barren = sorted(n for n in pending
                            if got[n] == 0 and n not in already)
            if barren:
                subset = wf.SUBSETS[barren[0]]
                gb = wf.subset_gb(subset)
                raise SystemExit(
                    f"{member} held none of the images {barren[0]}.csv lists "
                    f"(declared prefix {subset.prefix!r}). The registry points "
                    f"that subset at the wrong tree — `Advanced` and `Typical` "
                    "are a version axis, so a subset lives in exactly one of "
                    "them. Stopping after the first archive"
                    + (f" rather than downloading the remaining ~{gb:.0f} GB "
                       "to find out." if gb is not None else "."))
    return got


def _plan(subsets, plan, cache, fetch, limit, seed):
    """`(wanted, resumed)`: the tail-key -> (subset, destination) map still to
    extract, and how many images each subset already had on disk.

    `plan` is a `_TrainingPlan` or a `_BenchmarkPlan`: calling it selects the
    rows, and `plan.dest` names the file. Both of those are where the
    benchmark gate lives, so the two destinations cannot share one by
    accident.
    """
    wanted: dict[str, tuple[str, str]] = {}
    keys: dict[str, str] = {}
    resumed: collections.Counter = collections.Counter()
    for subset in subsets:
        paths = wf.read_subset_csv(subset, fetch(wf.csv_member(subset), cache))
        for rel in plan(subset, paths, limit, seed):
            dest = plan.dest(subset, rel)
            key = wf.tail_key(rel)
            if key in keys and keys[key] != rel:
                raise SystemExit(
                    f"WildFake rows {keys[key]!r} and {rel!r} share the "
                    f"archive-matching key {key!r}. Raise TAIL_COMPONENTS in "
                    "aigcdet.data.wildfake — matching them by that key would "
                    "extract one image into the other's place.")
            keys[key] = rel
            if os.path.exists(dest):
                resumed[subset.name] += 1   # resume: already extracted
                continue
            wanted[key] = (subset.name, dest)
    return wanted, resumed


class _TrainingPlan:
    """Row selection and destination naming for the training tree."""

    def __init__(self, out: str):
        self.out = out

    def __call__(self, subset, paths, limit, seed):
        return wf.select_paths(subset, paths, limit, seed)

    def dest(self, subset, rel):
        return wf.training_dest(self.out, subset, rel)

    def guard(self, member: str, subset_name: str) -> None:
        """Layer 2, asked of the ARCHIVE rather than of the CSV."""
        reason = wf.forbidden_reason(member)
        if reason:
            raise SystemExit(
                f"archive member {member!r} (matched for subset "
                f"{subset_name!r}) is benchmark data and was about to be "
                f"written to the training tree. {reason}")


class _BenchmarkPlan:
    """Row selection and destination naming for the demo benchmark. `limit`
    and `seed` are ignored on purpose: a subsampled benchmark is not the
    benchmark, and every reported number names its own tier (spec §4.4a)."""

    def __init__(self, benchmark_dir: str, halves):
        self.benchmark_dir = benchmark_dir
        self.halves = {h.subset: h for h in halves}

    def __call__(self, subset, paths, limit, seed):
        return wf.benchmark_rows(self.halves[subset.name], paths)

    def dest(self, subset, rel):
        return wf.benchmark_dest(self.benchmark_dir, self.halves[subset.name], rel)

    def guard(self, member: str, subset_name: str) -> None:
        """The inverted gate: a member reaching the benchmark tree must BE
        benchmark data. `real_coco.csv` spans train2017 and test2017 too, and
        those are not the benchmark."""
        half = self.halves[subset_name]
        if not wf.is_benchmark_path(half, member):
            raise SystemExit(
                f"archive member {member!r} matched the {half.source} "
                f"benchmark half but does not contain "
                f"{'/'.join(half.marker)!r}; it is not benchmark data and "
                "must not be materialised as though it were.")


def _archive_order(subsets) -> list[tuple[str, set[str]]]:
    order: list[str] = []
    served: dict[str, set[str]] = {}
    for subset in subsets:
        for pattern in subset.zips:
            if pattern not in served:
                served[pattern] = set()
                order.append(pattern)
            served[pattern].add(subset.name)
    return [(pattern, served[pattern]) for pattern in order]


def acquire_wildfake(out: str, limit: int, generators: list[str], *,
                     seed: int = DEFAULT_SEED, cache_dir: str | None = None,
                     allow_large: bool = False,
                     fetch=None, list_files=None) -> dict[str, int]:
    """Acquire WildFake generator subsets into the TRAINING tree `out`.

    Per requested subset: download that subset's label CSV, select up to
    `limit` of its rows deterministically from `seed`, download the archive
    holding them, extract only those images, delete the archive.

    This function can only ever write training data. `wildfake.training_dest`
    refuses the benchmark subsets by name and any image whose own path is
    benchmark data, and `_extract_wanted` asks the second question again of the
    archive member. The demo benchmark has its own verb —
    `acquire_wildfake_benchmark` — with its own destination argument, so
    "acquire training data" and "materialise the benchmark" are not the same
    call with a different string in it.

    Returns `{subset: images on disk}`.
    """
    subsets = [wf.resolve_for_training(g) for g in generators]
    if not subsets:
        raise SystemExit(
            "--generators is required for wildfake: the repository is ~1.2 TB "
            "and is acquired one subset at a time. Known subsets are "
            f"{sorted(wf.SUBSETS)}.")
    wf.check_download_budget(subsets, allow_large=allow_large)
    _report_download_volume(subsets)
    cache = cache_dir or os.path.join(out, WILDFAKE_CACHE)
    os.makedirs(cache, exist_ok=True)
    fetch = fetch or _hub_download
    list_files = list_files or _hub_list_files

    plan = _TrainingPlan(out)
    wanted, resumed = _plan(subsets, plan, cache, fetch, limit, seed)
    got = _run_extraction(wanted, _archive_order(subsets), cache, fetch,
                          list_files, plan.guard, set(resumed))

    totals = {s.name: got[s.name] + resumed[s.name] for s in subsets}
    missing = collections.Counter(name for name, _ in wanted.values())
    if missing:
        print(f"warning: {dict(missing)} listed images were not found in the "
              "declared archives; the registry may be incomplete")
    print(f"wildfake: {totals} (resumed {dict(resumed)})")
    return totals


def _report_download_volume(subsets) -> None:
    """Say what this run will transfer before it starts. Archives are shared,
    so the total is over DISTINCT archives, not summed per subset."""
    archives = {z for s in subsets for z in s.zips}
    known = [wf.ARCHIVE_GB[a] for a in archives if a in wf.ARCHIVE_GB]
    unknown = sorted(a for a in archives if a not in wf.ARCHIVE_GB)
    note = f"~{sum(known):.0f} GB across {len(archives)} archive(s)"
    if unknown:
        # Unrecorded is not "small": say which, so nobody reads a low total as
        # the whole cost.
        note += f"; size unrecorded for {unknown}"
    print(f"wildfake: will download {note}")


def acquire_wildfake_benchmark(benchmark_dir: str, *, halves=None,
                               cache_dir: str | None = None,
                               fetch=None, list_files=None) -> dict[str, int]:
    """Materialise the organisers' demo benchmark out of WildFake.

    A DIFFERENT VERB with a DIFFERENT DESTINATION ARGUMENT from
    `acquire_wildfake`, because the two are different acts: this one produces
    data that may never be trained on. What that structurally guarantees, and
    what it does not:

    - It guarantees the benchmark cannot be filed under a training source.
      Images land under `coco_val2017/` and `dalle_advanced/`, and
      `wildfake.benchmark_dest` refuses to write under a source that is not
      `exclude_from_training` in `aigcdet.data.sources` — so wherever a human
      points `benchmark_dir`, `build_dataset` drops these rows by SOURCE
      before it ever looks at a label. That is the end-to-end property.
    - It does NOT guarantee `benchmark_dir` is outside `--raw`; nothing here
      can know what `--raw` will be. The cheap check below catches the obvious
      mistake, and the source-level exclusion above catches the rest.

    Returns `{source: images on disk}`.
    """
    halves = tuple(halves) if halves is not None else wf.BENCHMARK_HALVES
    _refuse_training_tree(benchmark_dir)
    cache = cache_dir or os.path.join(benchmark_dir, WILDFAKE_CACHE)
    os.makedirs(cache, exist_ok=True)
    fetch = fetch or _hub_download
    list_files = list_files or _hub_list_files

    subsets = [wf.SUBSETS[h.subset] for h in halves]
    plan = _BenchmarkPlan(benchmark_dir, halves)
    wanted, resumed = _plan(subsets, plan, cache, fetch, 0, DEFAULT_SEED)
    got = _run_extraction(wanted, _archive_order(subsets), cache, fetch,
                          list_files, plan.guard, set(resumed))

    totals = {h.source: got[h.subset] + resumed[h.subset] for h in halves}
    for half in halves:
        n = totals[half.source]
        if n != half.expected:
            print(f"warning: {half.source} has {n} of {half.expected} images; "
                  "re-run to finish before reporting any benchmark number")
    print(f"wildfake_benchmark: {totals}")
    return totals


def _refuse_training_tree(benchmark_dir: str) -> None:
    """Refuse a `benchmark_dir` that is visibly the `--raw` training tree.

    A heuristic, and labelled as one: it recognises a directory that already
    holds a source `build_dataset` DOES train on. The real guarantee is the
    source-level `exclude_from_training` check in `benchmark_dest`.
    """
    if not os.path.isdir(benchmark_dir):
        return
    for entry in sorted(os.listdir(benchmark_dir)):
        spec = SOURCES.get(entry)
        if (spec is not None and not spec.exclude_from_training
                and os.path.isdir(os.path.join(benchmark_dir, entry))):
            raise SystemExit(
                f"{benchmark_dir} already holds {entry!r}, a source that IS "
                "trained on — this looks like the --raw training tree. The "
                "demo benchmark must live outside it; build_dataset.py takes "
                "it as --demo-dir and scans --raw as training data.")


# ---------------------------------------------------------------------------
# COCO val2017
# ---------------------------------------------------------------------------

def acquire_coco_val2017(out: str) -> None:
    import urllib.request
    os.makedirs(out, exist_ok=True)
    zp = os.path.join(out, "val2017.zip")
    if not os.path.exists(zp):
        urllib.request.urlretrieve("http://images.cocodataset.org/zips/val2017.zip", zp)
    dest = os.path.join(out, "coco_val2017")
    with zipfile.ZipFile(zp) as z:
        z.extractall(dest)
    # The zip's own top-level directory is the bucket build_dataset.py reads
    # this tree back through. It is "val2017", not "real" -- assuming
    # otherwise is what labelled every COCO photograph AI-generated. If the
    # archive layout ever changes, say so here rather than 5,000 mislabelled
    # rows later.
    buckets = {e for e in os.listdir(dest) if os.path.isdir(os.path.join(dest, e))}
    declared = SOURCES["coco_val2017"].real_buckets
    if not buckets <= declared:
        raise SystemExit(
            f"coco_val2017 extracted to {sorted(buckets)}, but "
            f"aigcdet.data.sources declares {sorted(declared)}. Update the "
            "registry before building a manifest from this tree.")
    print(f"coco_val2017: extracted to {sorted(buckets)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

#: `--dataset` values that are not raw-tree sources. The benchmark verb is one
#: of these rather than a `--dataset dalle_advanced` spelling, so that asking
#: for the benchmark and asking for training data are visibly different
#: commands with different destination flags.
BENCHMARK_DATASET = "wildfake_benchmark"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=sorted(SOURCES) + [BENCHMARK_DATASET])
    ap.add_argument("--out", default="data/raw",
                    help="the TRAINING tree; scanned wholesale by build_dataset.py")
    ap.add_argument("--benchmark-dir", default="",
                    help=f"destination for --dataset {BENCHMARK_DATASET}; must "
                         "be outside --out (build_dataset.py --demo-dir)")
    ap.add_argument("--limit", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="fixes WHICH images each WildFake subset contributes, "
                         "so an interrupted run resumes onto the same set")
    ap.add_argument("--generators", default="",
                    help=f"comma-separated WildFake subsets: {sorted(wf.SUBSETS)}")
    ap.add_argument("--allow-large-archives", action="store_true",
                    help="acquire a subset whose archives exceed "
                         f"{wf.DOWNLOAD_BUDGET_GB:.0f} GB (sdxl ~322, mjv5 "
                         "~372, mjv4 ~196, originsd ~119)")
    a = ap.parse_args()

    if a.dataset == BENCHMARK_DATASET:
        if not a.benchmark_dir:
            raise SystemExit(
                f"--dataset {BENCHMARK_DATASET} requires --benchmark-dir, and "
                "it must not be inside --out: build_dataset.py scans --out as "
                "training data and takes the benchmark separately as "
                "--demo-dir (spec §4.1).")
        acquire_wildfake_benchmark(a.benchmark_dir)
        # Both halves come from WildFake, so its terms apply to these bytes
        # too; recording all three is the honest provenance (spec §4.5).
        _record_licences(a.benchmark_dir,
                         ["coco_val2017", "dalle_advanced", "wildfake"])
        return

    if a.dataset == "sid_set":
        acquire_sid_set(a.out, a.limit)
    elif a.dataset == "wildfake":
        acquire_wildfake(a.out, a.limit,
                         [g for g in a.generators.split(",") if g], seed=a.seed,
                         allow_large=a.allow_large_archives)
    elif a.dataset == "coco_val2017":
        acquire_coco_val2017(a.out)
    else:
        # Registered so it can never be trained on, but there is no routine to
        # fetch it INTO THE TRAINING TREE. An `else` that fell through to COCO
        # would have downloaded the wrong dataset silently.
        raise SystemExit(
            f"{a.dataset} is the organisers' demo benchmark and is not "
            f"acquired into --out. Run `--dataset {BENCHMARK_DATASET} "
            "--benchmark-dir <dir>` instead, and pass that directory to "
            "build_dataset.py as --demo-dir, never as --raw.")

    _record_licences(a.out, [a.dataset])


def _record_licences(target: str, datasets: list[str]) -> None:
    """Append each dataset's licence and source URL (spec §4.5). JSON-lines,
    which build_dataset.py reads back as `{source: licence}`."""
    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, "LICENCES.json"), "a") as f:
        for name in datasets:
            f.write(json.dumps({name: LICENCES[name]}) + "\n")


if __name__ == "__main__":
    main()
