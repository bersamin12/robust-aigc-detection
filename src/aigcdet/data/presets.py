"""Named corpus compositions, so "which dataset" is a config file rather than
a remembered command line.

`configs/rungs/*.yaml` already makes the MODEL side of every ablation a file
under version control. The DATA side was a set of flags someone typed once:
the frozen manifest of 29 Aug was built with a particular cap, a particular
pair of held-out families and no source balancing at all, and none of that is
recoverable from the manifest itself. A preset is the same discipline applied
one stage earlier -- `build_dataset --preset configs/datasets/<x>.yaml` -- and
its `name` and `note` are copied into `docs/splits.json`, so a bank found on
disk six months from now can be traced back to the composition it came from.

WHY THE THREE KNOBS ARE THE THREE KNOBS
---------------------------------------
Measured from the frozen manifest and the cached view-0 proxies of
`data/banks/siglip2l` on 2026-08-30 (orientation-corrected AUC against the
label; see `docs/low_level_confounds.md` for the pooled figures):

                    pooled      wildfake only    sid_set only
    jpeg_quality    0.5532         0.5414           0.6212
    laplacian_var   0.6721       **0.6944**         0.5548
    noise_floor     0.6374         0.6214         **0.7314**
    short side      0.5992         0.6525           0.5047

Neither source is clean, and they are not dirty in the same way: WildFake
leaks the label through sharpness, SID_Set through its noise floor, and each
one DILUTES the other's leak -- every pooled figure sits below the worse of
the two singles. That is the argument for balancing the two sources against
each other rather than for finding a cleaner one, and `max_real_per_source`
is what makes it expressible: WildFake supplies 55,000 of the 65,049
authentic images in the frozen manifest, so the pooled statistics are very
nearly WildFake's own. `augment.canonical` names this remedy in its own
docstring ("Addressing it needs source-balanced sampling") and had no
mechanism behind it until now.

`min_short_side` covers the residue `canonical.py` calls irreducible. The
canonicaliser band-limits every image to `CANON_BAND_SIDE` (200) and then
upscales, which equalises the band for everything AT or ABOVE that ceiling.
Images below it cannot be raised to it, so they stay measurably softer than
the rest of the corpus for ever. In the frozen manifest there are 1,308 such
images and **every one of them is generated** -- 1,260 are BigGAN at exactly
128px. They are a free, permanent label leak with no authentic counterpart,
and no amount of index balancing removes them, because there is nothing on
the other side of the class to balance against.

`max_per_generator` is the existing cap, widened from one number to a mapping
because one number cannot say the thing P1 needs to say. `sid_set` is a
PSEUDO-generator: it names a source, not a family (`sources.PSEUDO_GENERATORS`),
and capping it "per family" is the same category error as holding it out. The
mapping lets a preset cap the 17 real WildFake families and leave the
pseudo-generator alone, in a file where the decision is visible.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from aigcdet.data.normalize import SHORT_SIDE
from aigcdet.data.sources import SOURCES, is_heldout_eligible

#: Key in `max_per_generator` that applies to every family without its own
#: entry. "0 means uncapped" everywhere, matching the CLI flag's convention.
DEFAULT_CAP_KEY = "*"


@dataclass(frozen=True)
class DatasetPreset:
    """One named corpus composition. Loaded from YAML, never constructed by
    hand in a script -- the file IS the record of what was built."""

    name: str
    #: One line saying what this composition is FOR. Copied into splits.json.
    #: Required, because a preset without one is a set of numbers whose
    #: motivation lives only in whoever typed them.
    note: str
    #: Per-generator-family caps. `{"*": 1800, "sid_set": 0}` reads as "cap
    #: every family at 1800, leave sid_set uncapped". 0 anywhere means
    #: uncapped. An int is accepted and means `{"*": n}`.
    max_per_generator: dict[str, int] = field(default_factory=dict)
    #: Per-source caps on AUTHENTIC rows. The generator cap cannot express
    #: this: authentic rows carry generator "" and are deliberately exempt
    #: from it, being the scarce side in the corpus it was written for.
    max_real_per_source: dict[str, int] = field(default_factory=dict)
    #: Drop images whose short side, after `data.normalize`, is below this.
    #: Applied on the NORMALISED dimensions on purpose: normalisation caps the
    #: short side at 512 and never upscales, so for any threshold at or below
    #: 512 the normalised short side and the native one agree, and the
    #: normalised image is the one the canonicaliser actually sees.
    min_short_side: int = 0
    #: Directory sub-paths, relative to the raw root, to drop at scan time.
    #: `"wildfake/real/real_ffhq"` reads as "every image under that
    #: directory". Deeper than a bucket, which is the whole reason it exists.
    exclude_subpaths: list[str] = field(default_factory=list)
    #: Pin the held-out families instead of drawing them from the seed.
    heldout_generators: list[str] = field(default_factory=list)
    #: Held-out LINEAGES, each a list of families that share a decoder and so
    #: must be held out together or not at all.
    #:
    #: The frozen manifest's pair was drawn at random and landed on
    #: `SDwithAdaptor_controlnet` and `VQGAN`, while
    #: `SDwithAdaptor_lora`/`SDwithAdaptor_lycris` and `VQVAE`/`vqdm` stayed in
    #: training. An adapter changes the conditioning, not the decoder that
    #: leaves the forensic trace, so "unseen generator" there means "unseen
    #: adapter on a decoder the model was trained on" -- which is a much easier
    #: question than the name suggests, and inflates the headline.
    #:
    #: Flattened into `heldout_generators` at validation time, so everything
    #: downstream -- `assign_splits`, `splits.json`, the bank's split column --
    #: sees one list of families and needs no changes. This field records WHY
    #: those families travel together, which a flat list cannot.
    heldout_groups: list[list[str]] = field(default_factory=list)
    #: Draw the train/val split once per IMAGE ID rather than once per row.
    #:
    #: For a paired corpus -- AI-OV7, where every fake is generated FROM one
    #: real and both rows carry that real's ImageID as their filename stem --
    #: the per-row draw puts a real and its own fake on opposite sides of the
    #: val boundary about 18% of the time, and the model then trains on a scene
    #: it is validated against under the other label. Held-out membership
    #: propagates too: without it the real paired with a held-out fake stays in
    #: training, and the held-out rung evaluates on scenes already memorised.
    #:
    #: Off by default, and it must stay off for every preset frozen before this
    #: existed: turning it on changes the RNG stream, and every feature bank on
    #: disk fingerprints the manifest it was extracted against.
    pair_split_by_stem: bool = False

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("a preset needs a name; it is written into "
                             "docs/splits.json as the corpus's identity")
        if not self.note.strip():
            raise ValueError(
                f"preset {self.name!r} needs a `note` saying what the "
                "composition is for. Every knob below is a judgement call, "
                "and the numbers alone do not record which one was being made.")

        # An int is the CLI's shape; normalise it here so callers never have
        # to branch. object.__setattr__ because the dataclass is frozen.
        if isinstance(self.max_per_generator, int):
            object.__setattr__(
                self, "max_per_generator",
                {DEFAULT_CAP_KEY: self.max_per_generator}
                if self.max_per_generator else {})

        for label, caps in (("max_per_generator", self.max_per_generator),
                            ("max_real_per_source", self.max_real_per_source)):
            for key, n in caps.items():
                if not isinstance(n, int) or isinstance(n, bool) or n < 0:
                    raise ValueError(
                        f"preset {self.name!r}: {label}[{key!r}] is {n!r}; a "
                        "cap must be a non-negative int (0 = uncapped)")

        # A source typo must not silently cap nothing. The source registry is
        # static, so this is checkable here; generator names come from the
        # data and are checked against the scan in build_dataset instead.
        unknown = sorted(set(self.max_real_per_source) - set(SOURCES))
        if unknown:
            raise ValueError(
                f"preset {self.name!r}: max_real_per_source names {unknown}, "
                f"which are not registered sources (known: {sorted(SOURCES)}). "
                "A cap on a source that does not exist caps nothing while "
                "every image of the source it was meant to name flows through.")

        if not 0 <= self.min_short_side <= SHORT_SIDE:
            raise ValueError(
                f"preset {self.name!r}: min_short_side={self.min_short_side} is "
                f"outside 0..{SHORT_SIDE}. data.normalize caps the short side "
                f"at {SHORT_SIDE}, so a higher floor would drop the entire "
                "corpus rather than its sub-band tail.")

        # Holding out a pseudo-generator removes a whole SOURCE and measures
        # dataset shift, not the unseen-generator generalisation of spec §4.6.
        # `choose_heldout_generators` already refuses to DRAW one; a pinned
        # list is the way round it, so refuse here too.
        ineligible = sorted(g for g in self.heldout_generators
                            if not is_heldout_eligible(g))
        if ineligible:
            raise ValueError(
                f"preset {self.name!r}: heldout_generators names {ineligible}, "
                "which are dataset-level pseudo-generators (or empty). Holding "
                "one out removes an entire source, so the held-out score would "
                "measure dataset shift rather than an unseen generator family "
                "(spec §4.6).")
        for raw in self.exclude_subpaths:
            parts = [p for p in str(raw).replace("\\", "/").split("/") if p]
            if len(parts) < 3:
                raise ValueError(
                    f"preset {self.name!r}: exclude_subpaths entry {raw!r} names "
                    f"{len(parts)} path component(s). It must name something "
                    "BELOW a bucket -- `source/bucket/subdir` at least. A whole "
                    "SOURCE is excluded by not staging it, and a whole BUCKET "
                    "by `SourceSpec.restricted_buckets`, which also records the "
                    "licence reason; this field is only for the case neither "
                    "can express.")
            if parts[0] not in SOURCES:
                raise ValueError(
                    f"preset {self.name!r}: exclude_subpaths entry {raw!r} starts "
                    f"with {parts[0]!r}, which is not a registered source "
                    f"(known: {sorted(SOURCES)}). An exclusion under a source "
                    "that does not exist excludes nothing while every image it "
                    "was meant to name flows through.")
        for grp in self.heldout_groups:
            if isinstance(grp, str) or not isinstance(grp, (list, tuple)):
                raise ValueError(
                    f"preset {self.name!r}: heldout_groups must be a list of "
                    f"LISTS of family names, got {grp!r}. A bare string would "
                    "silently iterate as characters.")
            if len(grp) < 2:
                raise ValueError(
                    f"preset {self.name!r}: heldout_groups entry {list(grp)!r} "
                    "has fewer than two families. A lineage of one is just a "
                    "heldout_generators entry; use that field, so the grouping "
                    "here always means 'these share a decoder'.")
        # Flatten AFTER the per-group checks, so downstream code and
        # splits.json see exactly one list of families.
        grouped = [g for grp in self.heldout_groups for g in grp]
        if grouped:
            object.__setattr__(
                self, "heldout_generators",
                list(self.heldout_generators) + [
                    g for g in grouped if g not in self.heldout_generators])
        ineligible_grouped = sorted(g for g in grouped
                                    if not is_heldout_eligible(g))
        if ineligible_grouped:
            raise ValueError(
                f"preset {self.name!r}: heldout_groups names "
                f"{ineligible_grouped}, which are dataset-level "
                "pseudo-generators (or empty).")
        seen_groups: dict[str, int] = {}
        for i, grp in enumerate(self.heldout_groups):
            for g in grp:
                if g in seen_groups and seen_groups[g] != i:
                    raise ValueError(
                        f"preset {self.name!r}: {g!r} appears in two "
                        "heldout_groups. A family belongs to one lineage.")
                seen_groups[g] = i
        dupes = sorted({g for g in self.heldout_generators
                        if self.heldout_generators.count(g) > 1})
        if dupes:
            raise ValueError(
                f"preset {self.name!r}: heldout_generators repeats {dupes}")

    def cap_for(self, generator: str) -> int:
        """The cap that applies to `generator`; 0 means uncapped."""
        if generator in self.max_per_generator:
            return self.max_per_generator[generator]
        return self.max_per_generator.get(DEFAULT_CAP_KEY, 0)

    @property
    def excluded_prefixes(self) -> list[str]:
        """`exclude_subpaths` as POSIX prefixes ending in a separator.

        The trailing separator is what stops `wildfake/real/real_ffhq` also
        matching a sibling directory called `real_ffhq_v2`: a bare
        `startswith` on the un-terminated string would.
        """
        return [
            "/".join(p for p in str(raw).replace("\\", "/").split("/") if p) + "/"
            for raw in self.exclude_subpaths
        ]

    @property
    def named_generators(self) -> list[str]:
        """Every generator family this preset names explicitly -- caps and
        hold-outs both. `build_dataset` checks these against the scan, so a
        typo raises instead of capping nothing."""
        named = {g for g in self.max_per_generator if g != DEFAULT_CAP_KEY}
        return sorted(named | set(self.heldout_generators))

    def as_record(self) -> dict:
        """The provenance blob written into `docs/splits.json`."""
        return {
            "name": self.name,
            "note": " ".join(self.note.split()),
            "max_per_generator": dict(self.max_per_generator),
            "max_real_per_source": dict(self.max_real_per_source),
            "min_short_side": self.min_short_side,
            "exclude_subpaths": list(self.exclude_subpaths),
            "heldout_generators": list(self.heldout_generators),
            "heldout_groups": [list(g) for g in self.heldout_groups],
        }


def load_preset(path: str) -> DatasetPreset:
    """Read a preset from `configs/datasets/<name>.yaml`.

    An unknown key raises (TypeError from the dataclass) rather than being
    ignored: a misspelled `min_short_size` that silently did nothing would
    produce a corpus that disagrees with the file describing it, and the file
    is the only record there is.
    """
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping of preset fields, "
                         f"got {type(raw).__name__}")
    try:
        preset = DatasetPreset(**raw)
    except TypeError as e:
        raise TypeError(f"{path}: {e}") from None
    stem = os.path.splitext(os.path.basename(path))[0]
    if preset.name != stem:
        raise ValueError(
            f"{path}: preset name is {preset.name!r} but the file is named "
            f"{stem!r}. The name reaches docs/splits.json and the filename is "
            "what a human types, so a mismatch makes the record untraceable.")
    return preset
