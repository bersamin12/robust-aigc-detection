"""The single source of truth for how a raw directory maps to a manifest row.

`scripts/acquire_data.py` WRITES the raw tree and `scripts/build_dataset.py`
READS it. Those two lived apart, and the reader inferred `label` from the
directory name with `0 if bucket == "real" else 1`. COCO val2017's zip
extracts to `coco_val2017/val2017/`, so every one of its ~5,000 authentic
photographs was labelled AI-generated, and the §4.1 exclusion — gated on
`label == 0` — never fired. Both scripts now go through this module, so the
layout is declared once and a change on the writing side that the reading
side does not know about raises instead of silently mislabelling a class.

Three properties are deliberately NOT derived from the label:

- `exclude_from_training` is a property of the SOURCE. Spec §4.1(2) forbids
  training on the organisers' demo benchmark whatever a row claims to be, so
  a mislabel cannot route around the exclusion.
- Held-out eligibility is a property of the GENERATOR NAME. A source that
  does not attribute its fakes to a generator gets a dataset-level
  *pseudo*-generator (its own name), and holding that out would remove an
  entire source — measuring dataset shift, not the unseen-generator
  generalisation spec §4.6 defines.
- `restricted_buckets` is a property of a BUCKET. A compilation can be usable
  in part: WildFake's generated images are the authors' own, its authentic
  images are re-published from datasets several of which are non-commercial,
  and the 28 Aug webinar Q&A bars those outright. Source-level exclusion is
  too coarse to say that, and label-level exclusion says it about the wrong
  thing — SID_Set's authentic images are CC BY 4.0 and must stay.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSpec:
    """How one raw `<source>/<bucket>/...` tree maps to manifest columns."""

    name: str
    #: Licence string recorded at acquisition time (spec §4.5); embeds the URL.
    licence: str
    #: Bucket directories holding authentic images (label 0).
    real_buckets: frozenset[str]
    #: Bucket holding fakes whose generator the source does not report. Its
    #: rows get `generator = name` (a pseudo-generator, never held out).
    #: "" when the source always attributes its fakes.
    pseudo_bucket: str = ""
    #: True when any other bucket directory names a real generator family.
    generator_buckets: bool = True
    #: Spec §4.1(2): never train on this source, whatever a row's label says.
    exclude_from_training: bool = False
    #: Declared buckets barred from ingestion outright, whatever their label.
    #: This is separate from `exclude_from_training`, which is a property of a
    #: whole SOURCE: a compilation can be usable in part, and WildFake is
    #: exactly that -- its generated buckets are the authors' own work, its
    #: authentic bucket is re-published from datasets with their own terms.
    #: Only DECLARED buckets may be named, so a typo cannot silently restrict
    #: nothing (see `__post_init__`).
    restricted_buckets: frozenset[str] = frozenset()
    #: Why those buckets are barred. Required alongside them and forbidden
    #: without them: a count in a log is not an audit trail, and a reason
    #: recorded against nothing reads as a restriction that is not in force.
    restriction: str = ""

    def __post_init__(self):
        declared = set(self.real_buckets) | (
            {self.pseudo_bucket} if self.pseudo_bucket else set())
        unknown = sorted(self.restricted_buckets - declared)
        if unknown:
            raise ValueError(
                f"{self.name}: restricted_buckets names {unknown}, which this "
                f"source does not declare (declared: {sorted(declared)}). A "
                "restriction on a bucket that does not exist restricts "
                "nothing while every image in the bucket it was meant to name "
                "flows through -- name a declared bucket, or declare it first.")
        if self.restricted_buckets and not self.restriction.strip():
            raise ValueError(
                f"{self.name}: restricted_buckets needs a restriction reason. "
                "The dropped count reaches a log; the reason is what makes it "
                "an audit trail.")
        if self.restriction.strip() and not self.restricted_buckets:
            raise ValueError(
                f"{self.name}: a restriction reason is recorded but no bucket "
                "is restricted, which reads as a rule in force when it is not.")

    @property
    def pseudo_generator(self) -> str:
        return self.name if self.pseudo_bucket else ""


SOURCES: dict[str, SourceSpec] = {
    # SID_Set streams `label` (0 real / 1 synthetic / 2 tampered) but does not
    # reliably carry the generating model. Attributed records go to a
    # generator bucket; the rest to "fake", which is ineligible for held-out
    # selection because "sid_set" is a dataset, not a generator family.
    "sid_set": SourceSpec(
        name="sid_set",
        licence="CC BY 4.0 — https://huggingface.co/datasets/saberzl/SID_Set "
                "(derived from COCO, OpenImages V7 and Flickr30k; attribution required). See docs/dataset_licences.md",
        real_buckets=frozenset({"real"}),
        pseudo_bucket="fake",
        generator_buckets=True,
    ),
    # WildFake is organised by generator upstream, so every non-real bucket
    # is a genuine generator family.
    #
    # Its `real/` bucket was barred on 28 Aug (the webinar's "non-commercial
    # datasets cannot be used", read against the re-published FFHQ, CelebA-HQ,
    # AFHQ, ImageNet, LSUN and LAION-5B subsets inside it). The organisers'
    # rules slide of 29 Aug names WildFake and SID_Set as examples of approved
    # "public/licensed datasets", which settles the reading: the rule bars
    # datasets whose OWN licence is non-commercial, not the upstream
    # provenance of sets the organisers themselves listed. The bar is lifted;
    # the upstream terms stay documented in docs/dataset_licences.md, and the
    # `restricted_buckets` mechanism stays for any source that needs it. The
    # frozen manifest (and every bank extracted against it) includes the
    # bucket, so a bar reappearing here would make the next rebuild disagree
    # with every bank on disk.
    "wildfake": SourceSpec(
        name="wildfake",
        licence="Apache-2.0 (ModelScope hub metadata) — covers the COMPILATION; "
                "constituent real subsets keep their own upstream terms, several of "
                "which are non-commercial. Organiser-listed as an approved dataset "
                "(rules slide, 29 Aug). See docs/dataset_licences.md",
        real_buckets=frozenset({"real"}),
        generator_buckets=True,
    ),
    # NTIRE 2026 "Robust AI-Generated Image Detection in the Wild" training
    # set. Three of six published shards are on disk: 150,000 images, 96,000
    # generated and 54,000 authentic. The card states the shards share a
    # distribution and may be used independently, so a partial download is a
    # smaller corpus and not a skewed one.
    #
    # It does not ship in bucket layout -- every image sits in one `images/`
    # directory with the class in a sibling `labels.csv` -- so it must be
    # restaged into `real/` and `generated/` by `scripts/stage_ntire.py`
    # before it can be ingested. That script hardlinks, so the restaging is
    # free and the original tree is left alone.
    #
    # `generator_buckets=False` and a pseudo bucket, because NTIRE publishes
    # no generator attribution at all. The consequence is worth stating
    # plainly rather than discovering during a split: NTIRE contributes
    # training MASS and no generator diversity. Its fakes cannot appear in a
    # held-out generator family, so §4.6's protocol still rests entirely on
    # WildFake's and SID_Set's attributed buckets.
    #
    # Label polarity was verified twice before any of this was built, because
    # an inverted 150,000-row corpus does not fail loudly -- it trains, and it
    # produces a confidently wrong detector. The card says "0 corresponds to a
    # real image, and 1 to a generated one"; scoring 300 images per class with
    # the dinov3l a3 head (2026-08-30) gave mean logits -10.63 for label 0 and
    # +8.87 for label 1, AUC 0.9454 in the card's direction.
    "ntire": SourceSpec(
        name="ntire",
        licence="NTIRE 2026 challenge training set — no `license:` tag on the HF "
                "repo; terms are the challenge's own and must be accepted per HF "
                "ACCOUNT. Usable for training locally and in a PRIVATE Kaggle "
                "Dataset; do NOT publish an NTIRE-derived Dataset. "
                "https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train "
                "See docs/dataset_licences.md",
        real_buckets=frozenset({"real"}),
        pseudo_bucket="generated",
        generator_buckets=False,
    ),
    # Open Images V7 portrait-orientation photographs, harvested by
    # `scripts/acquire_open_images_portrait.py` under a hard CC BY 2.0 filter
    # with per-image attribution written to `attribution.csv`.
    #
    # Authentic only, and that is a confound to measure rather than a fact to
    # note. A source with no generated half lets a model reach the right
    # answer through "this looks like an Open Images photo, so it is real",
    # which would flatter every metric here while transferring to nothing.
    # Two things are meant to catch it: `scripts/gate_confounds.py` before the
    # corpus is trusted, and `scripts/stratified_auc.py --stratify-by source`
    # reporting the false positive rate per source beside any headline. The
    # real fix is the generated half -- handoff 02 puts open-weight generators
    # on these same prompts -- and until that lands this source should be
    # treated as on probation.
    "open_images": SourceSpec(
        name="open_images",
        licence="CC BY 2.0 — https://creativecommons.org/licenses/by/2.0/ ; only "
                "rows whose Open Images V7 metadata declares exactly that licence "
                "were kept, and per-image Author/Title/OriginalURL attribution is "
                "recorded in attribution.csv. Attribution is REQUIRED on "
                "redistribution. See docs/dataset_licences.md",
        real_buckets=frozenset({"portrait"}),
        generator_buckets=False,
    ),
    # AI-OV7: open-weight generators run over the Open Images V7 reals above,
    # built by `scripts/generate_ov7.py`. Separate from `open_images` and NOT a
    # widening of it: that source is registered `generator_buckets=False` and is
    # already frozen into `manifest_union.parquet`, where every feature bank
    # fingerprints `manifest_sha256` over `rel_path` in row order. Inserting
    # generated rows into it would orphan every bank on disk, so this is its own
    # stream with its own manifest, exactly as `coco_crop` was.
    #
    # Its reason to exist: every fake in the union corpus comes from a 2017-2023
    # generator, and published results put detection at ~79% on 2020-21
    # generators against ~38% on 2024 ones. Every public dataset that would close
    # that gap is licence-barred, because their reals are web scrapes nobody can
    # relicense downstream. Generating over reals we may redistribute is the way
    # around it (docs/02).
    #
    # Each fake is generated FROM one real, at that real's own MCU-aligned crop
    # dimensions and through that real's own JPEG quantisation tables, so the
    # pair differs in the generator and as little else as we can manage. The
    # bucket is the family name and family names are precise -- `sdxl_t2i` and
    # `sdxl_img2img` are two families, because the held-out design groups by
    # DECODER LINEAGE and needs the name to mean something.
    "open_images_v7": SourceSpec(
        name="open_images_v7",
        licence="Reals: CC BY 2.0 — https://creativecommons.org/licenses/by/2.0/ ; "
                "per-image Author/Title/OriginalURL attribution is recorded in "
                "attribution.csv and is REQUIRED on redistribution. Generated "
                "buckets: our own outputs, from weights under apache-2.0 "
                "(FLUX.2-klein-4B), openrail++ (SDXL) and creativeml-openrail-m "
                "(SD 1.5) — all three grant ownership of the outputs and permit "
                "commercial use; their use-based restrictions bind model use, "
                "not image redistribution. Per-row `licence_tag` is recorded so "
                "an Apache-only subset can be cut without regenerating. See "
                "docs/02 and src/aigcdet/generate/registry.py",
        real_buckets=frozenset({"real"}),
        generator_buckets=True,
    ),
    # The authentic half of the organisers' demo benchmark. val2017/ is the
    # directory name inside the official zip — the mapping that C1 got wrong.
    # Authentic throughout, and excluded from training whatever any row claims.
    "coco_val2017": SourceSpec(
        name="coco_val2017",
        licence="CC BY 4.0 (images: Flickr terms) — https://cocodataset.org/#termsofuse",
        real_buckets=frozenset({"val2017"}),
        generator_buckets=False,
        exclude_from_training=True,
    ),
    # COCO train2017, as a TRAINING source. This reverses a rule the project
    # states explicitly elsewhere, so the reversal is recorded here rather
    # than left to be inferred.
    #
    # `data/wildfake.py`'s `_COCO_FORBIDDEN` bars COCO-derived reals from
    # training entirely -- not merely deduplicated against the benchmark --
    # on the grounds that "train2017/test2017/val2017 are one photographic
    # distribution, so training on any of them would let the demo-set score
    # measure distribution memorisation instead of generalisation". The
    # organisers' benchmark real half IS COCO val2017, so that concern is
    # real and it has not gone away.
    #
    # It is reversed for ONE experiment stream (configs/datasets/coco_crop.yaml),
    # deliberately, because that stream's question is what a genuinely
    # in-the-wild photographic real class does to a detector -- and because
    # the alternative on offer was WildFake's authentic half, 40,000 of whose
    # 55,000 images come from upstreams with non-commercial terms. The
    # protection is not the registry, it is the measurement:
    # `scripts/stratified_auc.py --stratify-by source` reports the false
    # positive rate separately for COCO, LAION and SID_Set reals, and a model
    # that has memorised the COCO distribution shows a far lower rate on COCO
    # than on the other two while the benchmark looks excellent. That gap is
    # the number to publish beside any headline from this stream. See
    # docs/dataset_presets.md and docs/dataset_licences.md.
    #
    # `coco_val2017` stays `exclude_from_training=True` and the `wildfake.py`
    # markers stay as they are: they bar WildFake's own re-published COCO
    # copy, which would now duplicate this one.
    "coco_train2017": SourceSpec(
        name="coco_train2017",
        licence="CC BY 4.0 (images: Flickr terms) — https://cocodataset.org/#termsofuse",
        real_buckets=frozenset({"train2017"}),
        generator_buckets=False,
        exclude_from_training=False,
    ),
    # The generated half of the same demo benchmark. Spec §4.1 forbids
    # training on it exactly as it forbids COCO val2017, so it is registered
    # for the same reason: `exclude_from_training` is what makes "the demo set
    # is excluded wholesale by the registry" true of BOTH halves, not one.
    #
    # No bucket layout is declared, because there is none to declare: this set
    # comes from the organisers rather than from an acquisition routine, and
    # it belongs under `--demo-dir`, which build_dataset never classifies. The
    # entry is the backstop for a copy that lands under `--raw` anyway — any
    # directory name there reads as generated and is then excluded by source,
    # and images sitting directly in the source root raise rather than being
    # guessed at.
    "dalle_advanced": SourceSpec(
        name="dalle_advanced",
        licence="TikTok TechJam organisers' demo benchmark — terms per the "
                "competition brief; never used for training (spec §4.1)",
        real_buckets=frozenset(),
        generator_buckets=True,
        exclude_from_training=True,
    ),
}

#: Dataset-level stand-ins for a real generator family. Holding one out would
#: remove a whole source rather than an unseen generator (spec §4.6).
PSEUDO_GENERATORS: frozenset[str] = frozenset(
    spec.pseudo_generator for spec in SOURCES.values() if spec.pseudo_generator
)

LICENCES: dict[str, str] = {name: spec.licence for name, spec in SOURCES.items()}


def spec_for(source: str) -> SourceSpec:
    try:
        return SOURCES[source]
    except KeyError:
        raise ValueError(
            f"unregistered raw source directory {source!r}. Every source must "
            f"declare how its buckets map to label/generator in "
            f"aigcdet.data.sources.SOURCES before it can be ingested; known "
            f"sources are {sorted(SOURCES)}."
        ) from None


def classify(source: str, bucket: str) -> tuple[int, str]:
    """Map `<source>/<bucket>` to `(label, generator)`.

    Raises ValueError for an unregistered source or an unrecognised bucket
    rather than guessing — guessing is what produced C1.
    """
    spec = spec_for(source)
    if bucket in spec.real_buckets:
        return 0, ""
    if bucket and bucket == spec.pseudo_bucket:
        return 1, spec.pseudo_generator
    if bucket and spec.generator_buckets:
        return 1, bucket
    raise ValueError(
        f"source {source!r} has no rule for bucket {bucket!r}. Its authentic "
        f"buckets are {sorted(spec.real_buckets)}"
        + (f", its unattributed-fake bucket is {spec.pseudo_bucket!r}"
           if spec.pseudo_bucket else "")
        + ("; any other bucket names a generator family."
           if spec.generator_buckets else "; it declares no generator buckets.")
    )


def is_safe_generator(source: str, generator: str) -> bool:
    """Whether `generator` may be used as a bucket directory for `source`.

    Generator names are UNTRUSTED third-party data — `acquire_data.py` lifts
    them from a dataset record field — and they become directory names, so
    two things must be impossible:

    1. Aliasing a declared bucket. A record whose generator field is the
       literal string "real" would be written into the authentic bucket and
       read back as label 0: the writer and the reader disagreeing about a
       directory name, which is exactly C1.
    2. Escaping the source's own tree. `".."` contains no separator, so a
       character-class check alone does not prevent it.
    """
    spec = spec_for(source)
    if not generator or not spec.generator_buckets:
        return False
    if generator.strip(".") == "":                        # ".", "..", "..."
        return False
    if any(sep in generator for sep in ("/", "\\", os.sep)):
        return False
    return generator not in spec.real_buckets and generator != spec.pseudo_bucket


def raw_subdir(source: str, label: int, generator: str = "") -> str:
    """The bucket directory `acquire_data.py` must write into. Inverse of
    `classify` on every branch, so the writer and the reader cannot drift
    apart: any generator this accepts, `classify` maps back to
    `(1, generator)`, and anything it would not, this refuses."""
    spec = spec_for(source)
    if label == 0:
        if len(spec.real_buckets) != 1:
            raise ValueError(
                f"source {source!r} declares {sorted(spec.real_buckets)} as "
                "authentic buckets; the writer must name one explicitly.")
        return next(iter(spec.real_buckets))
    if generator:
        if not spec.generator_buckets:
            raise ValueError(f"source {source!r} declares no generator buckets")
        if not is_safe_generator(source, generator):
            raise ValueError(
                f"generator {generator!r} is not usable as a bucket directory "
                f"for source {source!r}: it either aliases a declared bucket "
                f"(authentic {sorted(spec.real_buckets)}, unattributed "
                f"{spec.pseudo_bucket!r}), which would make classify() read it "
                f"back with the wrong label, or it escapes the source's tree.")
        return generator
    if not spec.pseudo_bucket:
        raise ValueError(
            f"source {source!r} has no unattributed-fake bucket; pass the "
            "generator this image came from.")
    return spec.pseudo_bucket


def is_excluded_from_training(source: str) -> bool:
    """Spec §4.1(2). Label-agnostic on purpose: a mislabelled row must still
    be excluded."""
    return source in SOURCES and SOURCES[source].exclude_from_training


def is_restricted_bucket(source: str, bucket: str) -> bool:
    """Whether `<source>/<bucket>` is barred from ingestion by its licence.

    Total over unregistered sources, like `is_excluded_from_training`: the
    scan loop in `build_dataset` asks this of every row it has already
    classified, and a raise here would only duplicate the one `classify`
    already makes on an unregistered source.
    """
    spec = SOURCES.get(source)
    return spec is not None and bucket in spec.restricted_buckets


def restriction_reason(source: str) -> str:
    """Why `source`'s restricted buckets are barred; "" if none are."""
    spec = SOURCES.get(source)
    return spec.restriction if spec is not None else ""


def is_heldout_eligible(generator: str) -> bool:
    """Whether `generator` names a real generator family that may be held out
    (spec §4.6), as opposed to "" (authentic) or a dataset-level pseudo."""
    return bool(generator) and generator not in PSEUDO_GENERATORS
