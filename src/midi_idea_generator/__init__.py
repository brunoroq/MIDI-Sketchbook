"""Tools for building small symbolic-music sketch datasets."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("midi-idea-generator")
except PackageNotFoundError:  # Running directly from a source checkout.
    __version__ = "0.1.0"

__all__ = ["__version__"]
