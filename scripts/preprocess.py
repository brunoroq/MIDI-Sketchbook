#!/usr/bin/env python3
"""Run stage-one preprocessing from a validated YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from midi_idea_generator.config import ConfigError, load_preprocess_config
from midi_idea_generator.pipeline import run_preprocessing
from midi_idea_generator.utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Validate, normalize, split, and augment a MIDI collection."
    )
    parser.add_argument("--config", type=Path, required=True, help="Preprocessing YAML")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run preprocessing and return a process exit code."""

    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    try:
        config = load_preprocess_config(args.config)
        report = run_preprocessing(config)
    except (ConfigError, OSError) as exc:
        print(f"ERROR | {exc}", file=sys.stderr)
        return 2
    print(
        f"Preprocessing complete: {report.compatible_sources} compatible source(s), "
        f"{report.discarded_sources} discarded, "
        f"{report.generated_fragments} MIDI phrase(s)."
    )
    print(f"Manifest: {report.manifest_path}")
    print(f"Immutable processed run: {report.processed_run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
