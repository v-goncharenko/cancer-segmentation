"""Test that package __version__ matches pyproject.toml version."""

import importlib
import tomllib
from pathlib import Path


def test_version_matches_pyproject() -> None:
    """Package __version__ must equal version in pyproject.toml."""
    pkg = importlib.import_module("cancer-segmentation")

    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    pyproject_version = pyproject["project"]["version"]
    package_version = pkg.__version__

    assert package_version == pyproject_version, (
        f"Version mismatch: cancer-segmentation.__version__ is {package_version!r}, "
        f"but pyproject.toml has version = {pyproject_version!r}"
    )
