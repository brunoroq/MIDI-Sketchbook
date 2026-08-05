#!/usr/bin/env python3
"""Generate symbolic guitar ideas from a Stage 3 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from midi_idea_generator.config import ConfigError
from midi_idea_generator.generation import GenerationError, run_generation
from midi_idea_generator.generation_artifacts import GenerationArtifactError
from midi_idea_generator.generation_checkpoint import GenerationCheckpointError
from midi_idea_generator.generation_config import load_generation_config
from midi_idea_generator.model import ModelConfigurationError, ModelInputError
from midi_idea_generator.tokenizer import TokenizationError
from midi_idea_generator.utils import JsonWriteError, configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the symbolic-generation command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Sample guitar-token sequences and export MIDI, "
            "metadata, technique annotations, and piano-roll images."
        )
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Generation YAML configuration"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run conditioned or legacy generation and return a process exit code."""

    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    try:
        config = load_generation_config(args.config)
        report = run_generation(config)
    except (
        ConfigError,
        GenerationArtifactError,
        GenerationCheckpointError,
        GenerationError,
        JsonWriteError,
        ModelConfigurationError,
        ModelInputError,
        TokenizationError,
        OSError,
    ) as exc:
        print(f"ERROR | {exc}", file=sys.stderr)
        return 2

    mode = (
        "tonality-conditioned"
        if config.conditioning is not None
        else "unconditional"
    )
    print(
        f"Generation complete: {len(report.samples)} {mode} sample(s) "
        f"on {report.device}."
    )
    if config.conditioning is not None:
        print(
            "Conditioning: "
            f"{config.conditioning.tonic} {config.conditioning.mode}."
        )
    print(
        f"Checkpoint: {report.training_run_id}, epoch {report.epoch} "
        f"({report.checkpoint_sha256[:12]})."
    )
    for sample in report.samples:
        extras = []
        if sample.num_pitch_bends:
            extras.append(f"{sample.num_pitch_bends} bend events")
        if sample.num_techniques:
            extras.append(f"{sample.num_techniques} techniques")
        detail = f" | {', '.join(extras)}" if extras else ""
        print(
            f"Sample {sample.sample_index:03d}: {sample.num_notes} notes, "
            f"{sample.num_tokens} tokens, {sample.attempts_used} attempt(s){detail}"
        )
        print(f"  MIDI: {sample.midi_path}")
        if sample.visualization_path is not None:
            print(f"  Piano roll: {sample.visualization_path}")
    print(f"Run directory: {report.output_dir}")
    print(f"Manifest: {report.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
