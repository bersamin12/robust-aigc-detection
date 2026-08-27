"""The single source of truth for how a raw directory maps to a manifest row.

`scripts/acquire_data.py` WRITES the raw tree and `scripts/build_dataset.py`
READS it. Those two lived apart, and the reader inferred `label` from the
directory name with `0 if bucket == "real" else 1`. COCO val2017's zip
extracts to `coco_val2017/val2017/`, so every one of its ~5,000 authentic
photographs was labelled AI-generated, and the §4.1 exclusion — gated on
`label == 0` — never fired. Both scripts now go through this module, so the
layout is declared once and a change on the writing side that the reading
side does not know about raises instead of silently mislabelling a class.

Two properties are deliberately NOT derived from the label:

- `exclude_from_training` is a property of the SOURCE. Spec §4.1(2) forbids
  training on the organisers' demo benchmark whatever a row claims to be, so
  a mislabel cannot route around the exclusion.
- Held-out eligibility is a property of the GENERATOR NAME. A source that
  does not attribute its fakes to a generator gets a dataset-level
  *pseudo*-generator (its own name), and holding that out would remove an
  entire source — measuring dataset shift, not the unseen-generator
  generalisation spec §4.6 defines.
"""
from __future__ import annotations

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
        licence="see https://huggingface.co/datasets/saberzl/SID_Set — confirm before use",
        real_buckets=frozenset({"real"}),
        pseudo_bucket="fake",
        generator_buckets=True,
    ),
    # WildFake is organised by generator upstream, so every non-real bucket
    # is a genuine generator family.
    "wildfake": SourceSpec(
        name="wildfake",
        licence="see https://modelscope.cn/datasets/hy2628982280/WildFake — confirm before use",
        real_buckets=frozenset({"real"}),
        generator_buckets=True,
    ),
    # The organisers' demo benchmark. val2017/ is the directory name inside
    # the official zip — the mapping that C1 got wrong. Authentic throughout,
    # and excluded from training whatever any row claims.
    "coco_val2017": SourceSpec(
        name="coco_val2017",
        licence="CC BY 4.0 (images: Flickr terms) — https://cocodataset.org/#termsofuse",
        real_buckets=frozenset({"val2017"}),
        generator_buckets=False,
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


def raw_subdir(source: str, label: int, generator: str = "") -> str:
    """The bucket directory `acquire_data.py` must write into. Inverse of
    `classify`, so the writer and the reader cannot drift apart."""
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


def is_heldout_eligible(generator: str) -> bool:
    """Whether `generator` names a real generator family that may be held out
    (spec §4.6), as opposed to "" (authentic) or a dataset-level pseudo."""
    return bool(generator) and generator not in PSEUDO_GENERATORS
