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
