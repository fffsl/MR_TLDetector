"""Project and runtime path helpers.

The helpers work both from source and from a PyInstaller bundle.
"""

from pathlib import Path
import os
import sys


def resource_root() -> Path:
    """Return the read-only directory containing bundled application assets."""
    bundle_root = getattr(sys, "_MEIPASS", None)
    return Path(bundle_root) if bundle_root else Path(__file__).resolve().parent.parent


def resource_path(*parts: str) -> Path:
    """Return an absolute path to an application asset."""
    return resource_root().joinpath(*parts)


def application_root() -> Path:
    """Return the directory containing the executable or source project."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def application_path(*parts: str) -> Path:
    """Return a path to an editable file stored beside the application."""
    return application_root().joinpath(*parts)


def output_root() -> Path:
    """Return a writable output directory, creating it when necessary."""
    configured = os.environ.get("MR_TLDETECTOR_OUTPUT_DIR")
    root = Path(configured).expanduser() if configured else application_path("output")
    root.mkdir(parents=True, exist_ok=True)
    return root
