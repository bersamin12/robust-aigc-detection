"""Which WildFake archive holds which generator, and which of them may never
reach the training tree (spec §4.1).

WildFake is ~1.2 TB on ModelScope, so it is acquired one generator subset at a
time: download the archive that holds a subset, extract only the images that
subset's `label_csv_files/<name>.csv` lists, delete the archive. This module is
the DATA half of that — the declared registry plus the pure functions that turn
a requested subset name into a destination path. `scripts/acquire_data.py` is
the I/O half. The split mirrors `sources.py` (contract) vs `acquire_data.py`
(download), and it exists so every gate below is unit-testable without a byte
of network traffic.

Why a registry at all, rather than "just extract the zip":

    The organisers' demo benchmark is INSIDE WildFake, exactly. `dalle3.csv`
    is 8,843 rows, every one of them `IsAdvanced=1` and under
    `Diffusion_based/DALLE/Advanced/DALLE3/` — the organisers' "AIGC: DALL·E
    Advanced, 8843". `real_coco.csv` contains train2017 (118,202),
    test2017 (40,646) and val2017 (4,998) — and 4,998 is the organisers'
    "Non-AIGC: COCO val2017". Competition rules forbid training on either.
    A whole-archive extraction of DALLE.zip would put the benchmark into the
    training tree while looking like an ordinary acquisition.

So the exclusion is structural, in the same spirit as `sources.py`: it is a
property of the SOURCE PATH, not of anything a caller says. Two independent
layers, either of which alone would stop the mistake:

1. `WildFakeSubset.training_forbidden` — declared per subset, checked when a
   requested NAME is resolved. Every spelling of "dalle3" resolves to the one
   canonical entry, so a near-miss cannot route around it (an unrecognised
   name is refused outright, never silently skipped).
2. `forbidden_reason()` — checked per IMAGE PATH, on the CSV row AND on the
   archive member name, case-insensitively, for every subset. Even if layer 1
   were mis-declared, an image whose own path says `DALLE/Advanced/DALLE3/`
   or `coco/val2017/` still cannot be written under `raw/wildfake/`.

`training_dest()` is the ONLY way to name a file under the training tree, and
it runs layer 2 itself, so the check cannot be forgotten at a call site.

The benchmark is a different verb: `benchmark_dest()`, which INVERTS the same
marker table — a path must MATCH a benchmark marker to be materialised there.
One table drives both, so the two halves cannot drift apart.
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
from dataclasses import dataclass

import numpy as np

from aigcdet.data.sources import SOURCES, raw_subdir

#: ModelScope dataset id (spec §4.1). The licence lives in `sources.LICENCES`.
DATASET_ID = "hy2628982280/WildFake"

#: Repo-relative directory holding one `<subset>.csv` per generator subset.
CSV_DIR = "label_csv_files"

#: Column of `label_csv_files/*.csv` carrying the repo-relative image path,
#: e.g. `./Diffusion_based/DALLE/Advanced/DALLE3/dalle3/<hash>/<hash>.jpg`
#: (relative to `Images/`, which is also what `WildFakeSubset.prefix` is
#: relative to).
PATH_COLUMN = "Image_path"
FAKE_COLUMN = "IsFake"

#: Number of trailing path components used to match a CSV row against an
#: archive member. The archives may or may not carry an extra top-level
#: directory, and a tail match is agnostic to that where a full-path match is
#: not. Three components is `<model>/<hash>/<hash>.jpg` in WildFake's layout;
#: `plan_subset` refuses to proceed if that is not unique within a subset.
TAIL_COMPONENTS = 3


@dataclass(frozen=True)
class WildFakeSubset:
    """One `label_csv_files/<name>.csv` and the archive(s) holding its images."""

    #: CSV stem. Doubles as the generator bucket for fakes, so it must satisfy
    #: `sources.is_safe_generator("wildfake", name)` — asserted below.
    name: str
    #: 0 authentic / 1 generated. Cross-checked against the CSV's IsFake column.
    label: int
    #: Row count of `label_csv_files/<name>.csv`, counted upstream. A CSV that
    #: does not match is a changed dataset, and raises rather than silently
    #: acquiring something else.
    rows: int
    #: Every `Image_path` in the CSV must start with this (relative to
    #: `Images/`). Derived from the archive path except where a tighter prefix
    #: has been verified upstream (dalle3).
    prefix: str
    #: Repo-relative archive paths. `*` is a glob expanded against the Hub's
    #: file listing (the multi-part `part_*.zip` trees). Empty means the
    #: subset's archive has NOT been established: it raises rather than guesses.
    zips: tuple[str, ...]
    #: Extra spellings that resolve to this subset. Matching is already
    #: case- and hyphen-insensitive, so these are only for the rest.
    aliases: tuple[str, ...] = ()
    #: Non-empty => never acquirable into the training tree; the string is the
    #: reason, and is what the raise says.
    training_forbidden: str = ""
    note: str = ""


#: Spec §4.1(2). The AIGC half of the organisers' demo benchmark: competition
#: rules forbid training on it, and it is a subdirectory of a WildFake archive
#: that also holds perfectly trainable images (dalle2), so the exclusion has to
#: survive someone acquiring its neighbour.
_DALLE3_FORBIDDEN = (
    "`Diffusion_based/DALLE/Advanced/DALLE3/` IS the organisers' demo "
    "benchmark (8,843 images, all IsAdvanced=1). Competition rules forbid "
    "training on it (spec §4.1). Materialise it with the benchmark verb — "
    "`acquire_wildfake_benchmark(benchmark_dir=...)`, or "
    "`--dataset wildfake_benchmark --benchmark-dir ...` — which files it "
    "under the `dalle_advanced` source, where `exclude_from_training` keeps "
    "it out of any manifest."
)

#: Spec §4.1(2) exclusion 2, which is BROADER than the literal overlap: not
#: just val2017 (the benchmark's own 4,998 photographs) but every COCO-derived
#: real. The point is not that the same files would appear twice — the pHash
#: leakage guard already removes literal duplicates. It is that COCO val2017,
#: train2017 and test2017 are one photographic distribution, so training on
#: train2017 would let the model memorise "what a COCO photo looks like" and
#: score on the demo set by distribution recall rather than by detecting
#: generation. The benchmark number has to measure generalisation.
_COCO_FORBIDDEN = (
    "COCO-derived reals are excluded from TRAINING ENTIRELY (spec §4.1(2)), "
    "not merely deduplicated against the benchmark: train2017/test2017/"
    "val2017 are one photographic distribution, so training on any of them "
    "would let the demo-set score measure distribution memorisation instead "
    "of generalisation. `real_coco.csv` is 163,846 rows of which 4,998 are "
    "val2017 — the organisers' own Non-AIGC half."
)

#: Path-segment sequences that may never appear in an image written under the
#: training tree, longest (most specific) first so the raise names the tighter
#: reason. Matched case-insensitively, on the CSV row AND on the archive member
#: name, and against a requested subset name that looks like a path — so
#: `dalle3`, `DALLE-3`, and `DALLE/Advanced/DALLE3` all reach the same refusal.
FORBIDDEN_PATH_MARKERS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("DALLE", "Advanced", "DALLE3"), _DALLE3_FORBIDDEN),
    (("coco", "val2017"), _COCO_FORBIDDEN),
    (("Real", "coco"), _COCO_FORBIDDEN),
)


SUBSETS: dict[str, WildFakeSubset] = {
    s.name: s
    for s in (
        # --- Images/Real/*.zip -------------------------------------------
        WildFakeSubset("real_afhq", 0, 31933, "Real/afhq/", ("Images/Real/afhq.zip",)),
        WildFakeSubset("real_celebahq", 0, 30000, "Real/celebahq/",
                       ("Images/Real/celebahq.zip",)),
        WildFakeSubset("real_church", 0, 83352, "Real/church/",
                       ("Images/Real/church.zip",)),
        WildFakeSubset("real_coco", 0, 163846, "Real/coco/", ("Images/Real/coco.zip",),
                       aliases=("coco", "coco_val2017"),
                       training_forbidden=_COCO_FORBIDDEN),
        WildFakeSubset("real_ffhq", 0, 70000, "Real/ffhq/", ("Images/Real/ffhq.zip",)),
        WildFakeSubset("real_imagenet", 0, 96788, "Real/imagenet/",
                       ("Images/Real/imagenet.zip",)),
        WildFakeSubset("real_laion5b", 0, 271831, "Real/laion5b/",
                       ("Images/Real/laion5b.zip",)),
        WildFakeSubset("real_wukong", 0, 265696, "Real/wukong/",
                       ("Images/Real/wukong.zip",)),
        # --- Images/Diffusion_based/*.zip --------------------------------
        WildFakeSubset("adm", 1, 155022, "Diffusion_based/ADM/",
                       ("Images/Diffusion_based/ADM.zip",)),
        WildFakeSubset("ddim", 1, 65713, "Diffusion_based/DDIM/",
                       ("Images/Diffusion_based/DDIM.zip",)),
        WildFakeSubset("ddpm", 1, 76561, "Diffusion_based/DDPM/",
                       ("Images/Diffusion_based/DDPM.zip",)),
        WildFakeSubset("imagen", 1, 47435, "Diffusion_based/Imagen/",
                       ("Images/Diffusion_based/Imagen.zip",)),
        WildFakeSubset("vqdm", 1, 153479, "Diffusion_based/VQDM/",
                       ("Images/Diffusion_based/VQDM.zip",)),
        WildFakeSubset("dalle2", 1, 55638, "Diffusion_based/DALLE/",
                       ("Images/Diffusion_based/DALLE.zip",),
                       aliases=("dalle_2",),
                       note="shares DALLE.zip with dalle3, whose Advanced/DALLE3/ "
                            "subtree is the demo benchmark and is refused per image"),
        WildFakeSubset("dalle3", 1, 8843, "Diffusion_based/DALLE/Advanced/DALLE3/",
                       ("Images/Diffusion_based/DALLE.zip",),
                       aliases=("dalle_3", "dalle_advanced", "dalle3_advanced"),
                       training_forbidden=_DALLE3_FORBIDDEN),
        # --- Images/Diffusion_based/SD/*.zip -----------------------------
        WildFakeSubset("SDwithAdaptor_controlnet", 1, 86991,
                       "Diffusion_based/SD/SDwithAdaptor/",
                       ("Images/Diffusion_based/SD/SDwithAdaptor.zip",)),
        WildFakeSubset("SDwithAdaptor_lora", 1, 56545,
                       "Diffusion_based/SD/SDwithAdaptor/",
                       ("Images/Diffusion_based/SD/SDwithAdaptor.zip",)),
        WildFakeSubset("SDwithAdaptor_lycris", 1, 56445,
                       "Diffusion_based/SD/SDwithAdaptor/",
                       ("Images/Diffusion_based/SD/SDwithAdaptor.zip",)),
        WildFakeSubset("personalizedSD_finetune", 1, 153274,
                       "Diffusion_based/SD/personalizedSD/",
                       ("Images/Diffusion_based/SD/personalizedSD.zip",)),
        WildFakeSubset("personalizedSD_dreambooth", 1, 56593,
                       "Diffusion_based/SD/personalizedSD/",
                       ("Images/Diffusion_based/SD/personalizedSD.zip",)),
        WildFakeSubset("originsd", 1, 271267, "Diffusion_based/SD/originalSD/",
                       ("Images/Diffusion_based/SD/originalSD/Typical/part_*.zip",
                        "Images/Diffusion_based/SD/originalSD/Advanced/part_*.zip"),
                       note="~51 GB per quality tier, split across part_*.zip; "
                            "parts are fetched and deleted one at a time"),
        # --- Images/Diffusion_based/Midjourney/**/part_*.zip -------------
        # Advanced/Typical is a QUALITY axis (the CSVs' IsAdvanced column), not
        # a version axis, so both mjv4 and mjv5 span both trees.
        WildFakeSubset("mjv4", 1, 202046, "Diffusion_based/Midjourney/",
                       ("Images/Diffusion_based/Midjourney/Typical/part_*.zip",
                        "Images/Diffusion_based/Midjourney/Advanced/part_*.zip"),
                       aliases=("midjourney_v4", "midjourneyv4")),
        WildFakeSubset("mjv5", 1, 236578, "Diffusion_based/Midjourney/",
                       ("Images/Diffusion_based/Midjourney/Typical/part_*.zip",
                        "Images/Diffusion_based/Midjourney/Advanced/part_*.zip"),
                       aliases=("midjourney_v5", "midjourneyv5")),
        # --- Images/GAN_based.zip (47.33 GB) -----------------------------
        WildFakeSubset("DF-GAN", 1, 191980, "GAN_based/", ("Images/GAN_based.zip",)),
        WildFakeSubset("GALIP", 1, 162646, "GAN_based/", ("Images/GAN_based.zip",)),
        WildFakeSubset("styleGAN", 1, 80000, "GAN_based/", ("Images/GAN_based.zip",)),
        WildFakeSubset("GigaGAN", 1, 27610, "GAN_based/", ("Images/GAN_based.zip",)),
        WildFakeSubset("BigGAN", 1, 15540, "GAN_based/", ("Images/GAN_based.zip",)),
        WildFakeSubset("starGAN", 1, 15442, "GAN_based/", ("Images/GAN_based.zip",)),
        WildFakeSubset("VQGAN", 1, 14000, "GAN_based/", ("Images/GAN_based.zip",)),
        # --- Images/Other_based.zip (13.34 GB) ---------------------------
        WildFakeSubset("MAGE", 1, 100000, "Other_based/", ("Images/Other_based.zip",)),
        WildFakeSubset("VQVAE", 1, 55000, "Other_based/", ("Images/Other_based.zip",)),
        WildFakeSubset("MAE", 1, 8390, "Other_based/", ("Images/Other_based.zip",)),
        # --- archive not established -------------------------------------
        # `sdxl.csv` exists (204,240 rows) but which archive holds it was not
        # confirmed upstream: it is plausibly under originalSD and plausibly
        # its own tree. Declaring a guess would spend a ~51 GB download to find
        # out, so this raises with instructions instead — the same discipline
        # the whole-function SystemExit used to enforce, now scoped to the one
        # entry that actually needs it.
        WildFakeSubset("sdxl", 1, 204240, "", (),
                       note="archive path unconfirmed; run the Hub file listing "
                            "and read the first Image_path of sdxl.csv"),
    )
}

#: `sources.SOURCES` name each half of the organisers' benchmark is filed
#: under. Both carry `exclude_from_training=True`; `benchmark_dest` asserts it.
BENCHMARK_SOURCES: dict[str, str] = {
    "dalle3": "dalle_advanced",
    "real_coco": "coco_val2017",
}


@dataclass(frozen=True)
class BenchmarkHalf:
    """One half of the organisers' demo benchmark, carved out of a WildFake
    subset. `marker` is a FORBIDDEN_PATH_MARKERS key: what is refused in the
    training tree is exactly what is required here, from one table."""

    subset: str
    source: str
    #: Generator bucket, "" for the authentic half.
    generator: str
    marker: tuple[str, ...]
    #: The organisers' own count, verified after filtering.
    expected: int


BENCHMARK_HALVES: tuple[BenchmarkHalf, ...] = (
    BenchmarkHalf("real_coco", "coco_val2017", "", ("coco", "val2017"), 4998),
    BenchmarkHalf("dalle3", "dalle_advanced", "dalle3",
                  ("DALLE", "Advanced", "DALLE3"), 8843),
)


# --------------------------------------------------------------------------
# path helpers
# --------------------------------------------------------------------------

def split_segments(path: str) -> list[str]:
    """Path components, separator-agnostic, dropping "" and "." segments.

    Archive members use "/" whatever the host OS; CSV rows arrive as "./a/b".
    """
    return [s for s in re.split(r"[/\\]+", path.strip()) if s not in ("", ".")]


def normalise_path(path: str) -> str:
    """`./Diffusion_based/DALLE/x.jpg` -> `Diffusion_based/DALLE/x.jpg`."""
    return "/".join(split_segments(path))


def _contains_marker(segments: list[str], marker: tuple[str, ...]) -> bool:
    low = [s.lower() for s in segments]
    want = [s.lower() for s in marker]
    return any(low[i:i + len(want)] == want
               for i in range(len(low) - len(want) + 1))


def forbidden_reason(path: str) -> str:
    """Why `path` may never be written under the training tree, or "".

    Layer 2 of the benchmark gate: it reads the PATH, so it does not care
    which subset was requested, whether the registry flagged it, or how the
    name was spelled. Case-insensitive, because a mis-cased archive member
    would otherwise sail past a check that a correctly cased one fails.
    """
    segments = split_segments(path)
    for marker, reason in FORBIDDEN_PATH_MARKERS:
        if _contains_marker(segments, marker):
            return reason
    return ""


def tail_key(path: str) -> str:
    """The key a CSV row and an archive member are matched on.

    A full-path match would break the moment an archive carries (or drops) a
    top-level directory relative to the CSV's `Image_path`; the last
    `TAIL_COMPONENTS` components survive either. Lower-cased for the same
    reason `forbidden_reason` is.
    """
    return "/".join(split_segments(path)[-TAIL_COMPONENTS:]).lower()


# --------------------------------------------------------------------------
# name resolution
# --------------------------------------------------------------------------

def _normalise_name(name: str) -> str:
    name = name.strip().strip("/\\")
    if name.lower().endswith(".csv"):
        name = name[:-len(".csv")]
    return name.lower().replace("-", "_")


def _build_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for subset in SUBSETS.values():
        for spelling in (subset.name,) + subset.aliases:
            key = _normalise_name(spelling)
            if index.get(key, subset.name) != subset.name:
                raise ValueError(
                    f"WildFake subset spelling {spelling!r} normalises to "
                    f"{key!r}, which already resolves to {index[key]!r}. Two "
                    "subsets sharing a spelling would make a request "
                    "ambiguous, and one of them is benchmark-forbidden.")
            index[key] = subset.name
    return index


#: Every accepted spelling -> canonical subset name. Built at import so an
#: ambiguous alias is a startup error, not a silent mis-resolution.
NAME_INDEX: dict[str, str] = _build_index()


def resolve(name: str) -> WildFakeSubset:
    """The subset `name` names, however it was spelled. Raises for anything
    unrecognised — a request that cannot be resolved must never be quietly
    skipped, because "skipped" and "acquired" look identical afterwards."""
    key = _normalise_name(name)
    if key in NAME_INDEX:
        return SUBSETS[NAME_INDEX[key]]
    reason = forbidden_reason(name)
    if reason:
        # A path-shaped request such as "DALLE/Advanced/DALLE3": refuse it for
        # the reason that actually applies, not as an unknown name.
        raise ValueError(f"{name!r} names benchmark data. {reason}")
    raise ValueError(
        f"unknown WildFake subset {name!r}. Known subsets are "
        f"{sorted(SUBSETS)} (spellings are case- and hyphen-insensitive, and "
        "a trailing .csv is accepted).")


def resolve_for_training(name: str) -> WildFakeSubset:
    """`resolve`, then layer 1 of the benchmark gate and the "archive not
    established" refusal. This is what the training acquisition path calls;
    nothing else may."""
    subset = resolve(name)
    if subset.training_forbidden:
        raise ValueError(
            f"refusing to acquire WildFake subset {subset.name!r} into the "
            f"training tree. {subset.training_forbidden}")
    if not subset.zips:
        raise ValueError(
            f"WildFake subset {subset.name!r} has no established archive path "
            f"in aigcdet.data.wildfake.SUBSETS ({subset.note}). Confirm it and "
            "add it to the registry rather than guessing: a wrong archive is a "
            "tens-of-gigabytes download that ends in an empty extraction.")
    return subset


# --------------------------------------------------------------------------
# CSV reading and deterministic selection
# --------------------------------------------------------------------------

def _parse_is_fake(value: str | None) -> int | None:
    if value is None:
        return None
    v = value.strip().lower()
    if v in ("1", "true", "yes"):
        return 1
    if v in ("0", "false", "no"):
        return 0
    return None


def read_subset_csv(subset: WildFakeSubset, csv_path: str) -> list[str]:
    """The normalised image paths `label_csv_files/<subset>.csv` lists.

    Verifies the CSV against the registry on three axes — row count, path
    prefix, and IsFake against the declared label — and raises on any
    disagreement. The registry is a claim about the upstream dataset; a claim
    that is never checked is a comment.
    """
    paths: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or PATH_COLUMN not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} has no {PATH_COLUMN!r} column (columns: "
                f"{reader.fieldnames}). WildFake's label CSVs are "
                "Generator,Architecture,Weight,Category,IsAdvanced,IsFake,"
                "Image_path,Num.")
        for row in reader:
            rel = normalise_path(row[PATH_COLUMN] or "")
            if not rel:
                continue
            if not rel.startswith(subset.prefix):
                raise ValueError(
                    f"{csv_path} row {rel!r} is outside the prefix "
                    f"{subset.prefix!r} declared for subset {subset.name!r}. "
                    "Update the registry before acquiring: the prefix is what "
                    "ties a subset to the archive that is downloaded for it.")
            is_fake = _parse_is_fake(row.get(FAKE_COLUMN))
            if is_fake is not None and is_fake != subset.label:
                raise ValueError(
                    f"{csv_path} row {rel!r} has {FAKE_COLUMN}={is_fake} but "
                    f"subset {subset.name!r} is declared label={subset.label}. "
                    "Acquiring it would write an image into the wrong bucket, "
                    "which is exactly the class of defect sources.py exists to "
                    "prevent.")
            paths.append(rel)
    if len(paths) != subset.rows:
        raise ValueError(
            f"{csv_path} has {len(paths)} rows, but subset {subset.name!r} "
            f"declares {subset.rows}. Upstream has changed; re-count and "
            "update aigcdet.data.wildfake.SUBSETS before acquiring, rather "
            "than acquiring something the registry does not describe.")
    return paths


def _subset_seed(subset_name: str, seed: int) -> np.random.Generator:
    """A per-subset generator derived from `seed` by a STABLE hash.

    Python's `hash()` is salted per process, so using it would make a resumed
    run select different images from a fresh one — which is precisely the
    property this seeding exists to provide.
    """
    tag = int.from_bytes(
        hashlib.sha256(subset_name.encode("utf-8")).digest()[:8], "big")
    return np.random.default_rng([seed, tag])


def select_paths(subset: WildFakeSubset, paths: list[str], limit: int,
                 seed: int) -> list[str]:
    """At most `limit` of `paths`, chosen deterministically from (name, seed).

    Determinism is what makes the acquisition resumable: an interrupted run
    and its restart must agree on WHICH images the subset contributes, or the
    restart quietly grows the subset past its cap. `limit <= 0` means all.
    """
    if limit <= 0 or limit >= len(paths):
        return list(paths)
    rng = _subset_seed(subset.name, seed)
    chosen = rng.permutation(len(paths))[:limit]
    return [paths[i] for i in sorted(chosen.tolist())]


def dest_filename(rel_path: str) -> str:
    """A destination name that is a pure function of the source path.

    Deterministic so a resumed run recomputes the same name and skips the file
    (a running counter would rename everything the moment the cap or the
    ordering changed), and derived rather than copied so a hostile archive
    member name can never steer the write — nothing from the archive reaches
    the filesystem path.
    """
    ext = os.path.splitext(rel_path)[1].lower() or ".jpg"
    digest = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:16]
    return digest + ext


def training_dest(out: str, subset: WildFakeSubset, rel_path: str) -> str:
    """Where `rel_path` is written under the training tree `out`.

    The only function that names a file under `out/wildfake/`, and it runs the
    per-path benchmark gate itself, so no call site can forget to. Buckets
    come from `sources.raw_subdir`, the same function `classify()` inverts.
    """
    if subset.training_forbidden:
        raise ValueError(
            f"refusing to place WildFake subset {subset.name!r} in the "
            f"training tree. {subset.training_forbidden}")
    reason = forbidden_reason(rel_path)
    if reason:
        raise ValueError(
            f"refusing to place {rel_path!r} in the training tree. {reason}")
    if subset.label == 0:
        # Nested one level BELOW the bucket: `classify` reads `rel[1]`, so the
        # bucket is still "real" and the label is still 0, while the directory
        # name keeps which real source the images came from — spec §4.1(2)
        # asks for exactly that to be recorded.
        bucket = os.path.join(raw_subdir("wildfake", 0), subset.name)
    else:
        bucket = raw_subdir("wildfake", 1, subset.name)
    return os.path.join(out, "wildfake", bucket, dest_filename(rel_path))


def is_benchmark_path(half: BenchmarkHalf, path: str) -> bool:
    """Whether `path` belongs to this benchmark half.

    The exact inverse of `forbidden_reason`'s test, over the same marker
    table: a path the training tree refuses for `half.marker` is a path the
    benchmark requires, and there is no second rule to keep in step.
    """
    return _contains_marker(split_segments(path), half.marker)


def benchmark_dest(benchmark_dir: str, half: BenchmarkHalf, rel_path: str) -> str:
    """Where `rel_path` is written when materialising the demo benchmark.

    The inverse gate: `rel_path` must MATCH `half.marker` — the same segments
    `forbidden_reason` refuses in the training tree — so an image that would
    be legal to train on cannot be smuggled into the benchmark either, and the
    two rules cannot drift because there is one table.

    Also asserts the destination source is `exclude_from_training`. That is
    the guarantee that actually holds end to end: whatever directory a human
    points this at, `build_dataset` drops these rows by SOURCE, before any
    label is consulted.
    """
    if not is_benchmark_path(half, rel_path):
        raise ValueError(
            f"{rel_path!r} is not part of the {half.source} benchmark half: "
            f"it does not contain {'/'.join(half.marker)!r}.")
    spec = SOURCES[half.source]
    if not spec.exclude_from_training:
        raise ValueError(
            f"source {half.source!r} is not marked exclude_from_training in "
            "aigcdet.data.sources; materialising the demo benchmark under it "
            "would let build_dataset train on the benchmark.")
    bucket = (raw_subdir(half.source, 0) if not half.generator
              else raw_subdir(half.source, 1, half.generator))
    return os.path.join(benchmark_dir, half.source, bucket,
                        dest_filename(rel_path))


def csv_member(subset: WildFakeSubset) -> str:
    """Repo-relative path of this subset's label CSV."""
    return f"{CSV_DIR}/{subset.name}.csv"


def benchmark_rows(half: BenchmarkHalf, paths: list[str]) -> list[str]:
    """The rows of a subset's CSV that make up `half`, verified against the
    organisers' own count.

    `real_coco.csv` is 163,846 rows spanning train2017/test2017/val2017; only
    the 4,998 val2017 rows are the benchmark. `dalle3.csv` is entirely the
    benchmark, and the count check is what proves that rather than assuming it.
    """
    rows = [p for p in paths if is_benchmark_path(half, p)]
    if len(rows) != half.expected:
        raise ValueError(
            f"{half.source}: {len(rows)} rows of {SUBSETS[half.subset].name}.csv "
            f"contain {'/'.join(half.marker)!r}, but the organisers' benchmark "
            f"half is {half.expected} images. Materialising a different set "
            "would silently change what every reported benchmark number means.")
    return rows
