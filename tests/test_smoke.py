def test_package_imports():
    import aigcdet
    assert aigcdet.__version__


def test_declared_dependencies_cover_everything_src_imports():
    """A clean `pip install -e .` must yield an importable package. torch and
    transformers are imported from src/ and were both missing from
    pyproject.toml, so the installed package was unimportable."""
    import pathlib
    import tomllib

    root = pathlib.Path(__file__).resolve().parents[1]
    with open(root / "pyproject.toml", "rb") as f:
        declared = tomllib.load(f)["project"]["dependencies"]
    names = {d.split(">=")[0].split("==")[0].split("[")[0].strip() for d in declared}
    assert {"torch", "transformers"} <= names
