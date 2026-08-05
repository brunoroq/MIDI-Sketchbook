#!/usr/bin/env python3
"""Print review-friendly tonic/mode labels from the Stage 1 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


def build_parser() -> argparse.ArgumentParser:
    """Build the tonality-report command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "List compatible source tonalities, with the least-confident "
            "automatic estimates first."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/splits/manifest.json"),
        help="Stage 1 manifest (default: data/splits/manifest.json)",
    )
    parser.add_argument(
        "--sort",
        choices=("confidence", "source"),
        default="confidence",
        help="Row ordering (default: confidence)",
    )
    return parser


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _confidence(value: object) -> float:
    if value is None:
        return 1.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("tonality confidence must be numeric or null")
    return float(value)


def load_rows(path: Path) -> list[tuple[float, float, str, str, str, str]]:
    """Load compatible-source labels from one Stage 1 manifest."""

    payload = _mapping(json.loads(path.read_text(encoding="utf-8")), "manifest")
    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise ValueError("manifest.sources must be a JSON array")
    rows: list[tuple[float, float, str, str, str, str]] = []
    for index, raw_source in enumerate(sources):
        source = _mapping(raw_source, f"sources[{index}]")
        if source.get("compatible") is not True:
            continue
        tonality = _mapping(source.get("tonality"), f"sources[{index}].tonality")
        source_file = source.get("source_file")
        tonic = tonality.get("tonic")
        mode = tonality.get("mode")
        method = tonality.get("method")
        if not all(isinstance(value, str) and value for value in (source_file, tonic, mode, method)):
            raise ValueError(f"sources[{index}] contains invalid tonality metadata")
        rows.append(
            (
                _confidence(tonality.get("mode_confidence")),
                _confidence(tonality.get("tonic_confidence")),
                source_file,
                tonic,
                mode,
                method,
            )
        )
    return rows


def _render_confidence(value: float, method: str) -> str:
    return "-" if method in {"MANUAL", "UNKNOWN"} else f"{value:.4f}"


def main(argv: list[str] | None = None) -> int:
    """Print the report and return a process exit code."""

    args = build_parser().parse_args(argv)
    try:
        rows = load_rows(args.manifest.expanduser().resolve())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR | {exc}", file=sys.stderr)
        return 2
    if args.sort == "source":
        rows.sort(key=lambda row: row[2])
    else:
        rows.sort(key=lambda row: (row[0], row[1], row[2]))
    print("mode_conf\ttonic_conf\tsource\ttonic\tmode\tmethod")
    for mode_conf, tonic_conf, source, tonic, mode, method in rows:
        print(
            "\t".join(
                (
                    _render_confidence(mode_conf, method),
                    _render_confidence(tonic_conf, method),
                    source,
                    tonic,
                    mode,
                    method,
                )
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
