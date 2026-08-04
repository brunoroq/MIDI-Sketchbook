#!/usr/bin/env python3
"""Run deterministic Stage 2 REMI tokenization.

The filename deliberately avoids ``tokenize.py``, which would shadow Python's
standard-library :mod:`tokenize` module when this script is executed directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from midi_idea_generator.config import ConfigError
from midi_idea_generator.tokenization_config import load_tokenization_config
from midi_idea_generator.tokenization_pipeline import (
    TokenizationPipelineError,
    run_tokenization,
)
from midi_idea_generator.utils import JsonWriteError, configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Tokenize the authoritative Stage 1 MIDI manifest with REMI."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Stage 2 tokenization YAML"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run tokenization and return a process exit code."""

    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    try:
        config = load_tokenization_config(args.config)
        report = run_tokenization(config)
    except (ConfigError, TokenizationPipelineError, JsonWriteError, OSError) as exc:
        print(f"ERROR | {exc}", file=sys.stderr)
        return 2

    action = "Reused" if report.reused_run else "Created"
    print(
        f"Tokenization complete: {report.num_sequences} sequence(s), "
        f"vocabulary {report.vocabulary_size}, lengths "
        f"{report.min_tokens}-{report.max_tokens} tokens."
    )
    print(f"{action} immutable tokenized run: {report.tokenized_run_dir}")
    print(f"Manifest: {report.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
