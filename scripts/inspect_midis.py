#!/usr/bin/env python3
"""Inspect a directory of MIDI files and report stage-one compatibility."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from midi_idea_generator.config import TrackSelectionConfig, ValidationConfig
from midi_idea_generator.midi_io import discover_midi_files, inspect_midi
from midi_idea_generator.utils import (
    JsonWriteError,
    configure_logging,
    relative_label,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Inspect MIDI files without aborting on incompatible inputs."
    )
    parser.add_argument("--input", type=Path, required=True, help="MIDI directory")
    parser.add_argument("--pitch-min", type=int, default=21, help="Lowest allowed MIDI pitch")
    parser.add_argument("--pitch-max", type=int, default=108, help="Highest allowed MIDI pitch")
    parser.add_argument(
        "--output", type=Path, help="Optional path for a machine-readable JSON report"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the MIDI inspector and return a process exit code."""

    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    try:
        validation = ValidationConfig(pitch_min=args.pitch_min, pitch_max=args.pitch_max)
        if not 0 <= validation.pitch_min <= validation.pitch_max <= 127:
            raise ValueError("Pitch range must satisfy 0 <= pitch_min <= pitch_max <= 127")
        files = discover_midi_files(args.input)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"ERROR | {exc}", file=sys.stderr)
        return 2

    inspections = [
        inspect_midi(path, validation, TrackSelectionConfig()) for path in files
    ]
    compatible = sum(result.compatible for result in inspections)
    for result in inspections:
        label = relative_label(result.source_file, PROJECT_ROOT)
        status = "OK" if result.compatible else "DISCARD"
        details = (
            f"track={result.selected_track} tempo={result.tempo_bpm:.3f}"
            if result.compatible and result.tempo_bpm is not None
            else result.discard_reason
        )
        print(f"{status:7} {label} | {details}")
    print(
        f"Inspected {len(inspections)} file(s): {compatible} compatible, "
        f"{len(inspections) - compatible} discarded."
    )

    if args.output:
        output_path = args.output.expanduser().resolve()
        input_path = args.input.expanduser().resolve()
        if output_path.suffix.lower() != ".json" or output_path.is_relative_to(
            input_path
        ):
            print(
                "ERROR | --output must be a .json file outside the MIDI input directory",
                file=sys.stderr,
            )
            return 2
        payload = {
            "summary": {
                "inspected_files": len(inspections),
                "compatible_files": compatible,
                "discarded_files": len(inspections) - compatible,
            },
            "files": [
                result.to_dict(
                    source_label=relative_label(result.source_file, PROJECT_ROOT)
                )
                for result in inspections
            ],
        }
        try:
            write_json(output_path, payload)
        except JsonWriteError as exc:
            print(f"ERROR | {exc}", file=sys.stderr)
            return 2
        print(f"Report written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
