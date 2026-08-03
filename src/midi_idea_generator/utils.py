"""Small shared utilities for deterministic files and clear command-line logs."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any


class JsonWriteError(OSError):
    """Raised when a JSON report cannot be serialized or written safely."""


def configure_logging(verbose: bool = False) -> None:
    """Configure concise process-wide console logging."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def relative_label(path: Path, root: Path) -> str:
    """Represent a path relative to the project when possible."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def write_json(path: Path, payload: Any) -> None:
    """Atomically write deterministic, human-readable JSON."""

    destination = path.expanduser().resolve()
    temporary: Path | None = None
    descriptor: int | None = None
    try:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}-",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = None
            handle.write(f"{serialized}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    except (OSError, TypeError, ValueError) as exc:
        raise JsonWriteError(f"Could not write JSON '{destination}': {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
