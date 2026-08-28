import numpy as np
from PIL import Image

from aigcdet.data.audit import audit_flags, audit_table


def _write(p, size, fmt, quality=None):
    arr = np.random.default_rng(0).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(arr).save(p, format=fmt, **({"quality": quality} if quality else {}))
    return str(p)


def test_audit_table_reports_per_class_shape(tmp_path):
    paths, labels, sources = [], [], []
    for i in range(4):
        paths.append(_write(tmp_path / f"r{i}.jpg", (640, 480), "JPEG", 75))
        labels.append(0); sources.append("coco")
    for i in range(4):
        paths.append(_write(tmp_path / f"f{i}.png", (1024, 1024), "PNG"))
        labels.append(1); sources.append("sdxl")
    df = audit_table(paths, labels, sources)
    assert len(df) == 2
    real = df[df["label"] == 0].iloc[0]
    fake = df[df["label"] == 1].iloc[0]
    assert real["fmt_top"] == "JPEG" and fake["fmt_top"] == "PNG"
    assert fake["width_median"] > real["width_median"]


def test_audit_table_reports_the_dominant_colour_mode(tmp_path):
    """Spec §4.2's "encoding history" is not only the container: a class that
    is part greyscale facing one that is entirely RGB is exploitable without
    looking at content. Sources that keep their original files (WildFake,
    COCO) are profiled on mode here; SID_Set, which is decoded before it ever
    reaches disk, records it in its acquisition ingest report instead."""
    paths, labels, sources = [], [], []
    for i in range(3):
        p = tmp_path / f"g{i}.png"
        Image.fromarray(
            np.random.default_rng(i).integers(0, 256, (32, 32), dtype=np.uint8)
        ).save(p, format="PNG")
        paths.append(str(p)); labels.append(0); sources.append("grey")
    for i in range(3):
        paths.append(_write(tmp_path / f"c{i}.png", (32, 32), "PNG"))
        labels.append(1); sources.append("colour")
    df = audit_table(paths, labels, sources)
    assert set(df["mode_top"]) == {"L", "RGB"}
    assert df[df["source"] == "grey"].iloc[0]["mode_top"] == "L"


def test_audit_flags_detects_the_confound(tmp_path):
    paths, labels, sources = [], [], []
    for i in range(4):
        paths.append(_write(tmp_path / f"r{i}.jpg", (640, 480), "JPEG", 75))
        labels.append(0); sources.append("coco")
    for i in range(4):
        paths.append(_write(tmp_path / f"f{i}.png", (1024, 1024), "PNG"))
        labels.append(1); sources.append("sdxl")
    flags = audit_flags(audit_table(paths, labels, sources))
    assert any("format" in f.lower() for f in flags)
    assert any("resolution" in f.lower() for f in flags)


def test_audit_flags_empty_when_classes_match(tmp_path):
    paths, labels, sources = [], [], []
    for i in range(4):
        paths.append(_write(tmp_path / f"r{i}.png", (512, 512), "PNG"))
        labels.append(0); sources.append("a")
    for i in range(4):
        paths.append(_write(tmp_path / f"f{i}.png", (512, 512), "PNG"))
        labels.append(1); sources.append("b")
    assert audit_flags(audit_table(paths, labels, sources)) == []


def test_audit_flags_catch_confound_diluted_by_pooling_across_sources(tmp_path):
    """A within-class confound that a naive class-pooled comparison misses.

    The real class mixes a ~500px source with a ~2000px source. Their median
    of medians is 1250, which sits within 1.5x of the fake class's 1024 --
    close enough that the class-level (pooled) comparison alone reports
    nothing, even though *each* real source individually differs sharply
    from the fake class, and the two real sources differ sharply from each
    other. That per-source and within-class shape is exactly what a
    classifier could exploit, and exactly what the class-level check cannot
    see once it has already pooled the class.
    """
    paths, labels, sources = [], [], []
    for i in range(4):
        paths.append(_write(tmp_path / f"coco{i}.jpg", (500, 375), "JPEG", 85))
        labels.append(0); sources.append("coco_val2017")
    for i in range(4):
        paths.append(_write(tmp_path / f"big{i}.jpg", (2000, 1500), "JPEG", 85))
        labels.append(0); sources.append("big_real_src")
    for i in range(8):
        paths.append(_write(tmp_path / f"sdxl{i}.jpg", (1024, 768), "JPEG", 85))
        labels.append(1); sources.append("sdxl")

    flags = audit_flags(audit_table(paths, labels, sources))

    # The pooled class-level check stays silent: median([500, 2000]) = 1250
    # is within 1.5x of the fake class's 1024.
    assert not any("Resolution confound: median width" in f for f in flags)

    # Per-source vs. the opposite class, pooled: each real source alone is
    # more than 1.5x off from the fake class's 1024. Assert the comparison
    # target itself (1024, the fake class's pooled width), not merely that
    # a flag mentioning the source exists -- a comparison against the
    # source's own class would also produce a flag mentioning the source,
    # but it would not name 1024.
    assert any("coco_val2017" in f and "resolution" in f.lower() and "1024" in f
               for f in flags)
    assert any("big_real_src" in f and "resolution" in f.lower() and "1024" in f
               for f in flags)

    # Within-class heterogeneity: the two real sources are more than 1.5x
    # apart from each other, invisible to any comparison that pools first.
    assert any("coco_val2017" in f and "big_real_src" in f for f in flags)
