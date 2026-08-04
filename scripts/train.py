#!/usr/bin/env python3
"""Train the Stage 3 autoregressive GRU from the token manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from midi_idea_generator.config import ConfigError
from midi_idea_generator.dataset import DatasetContractError
from midi_idea_generator.model import ModelConfigurationError, ModelInputError
from midi_idea_generator.trainer import TrainingError, run_training
from midi_idea_generator.training_config import load_training_config
from midi_idea_generator.utils import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the training command-line parser."""

    parser = argparse.ArgumentParser(
        description="Train or resume the small autoregressive MIDI GRU."
    )
    parser.add_argument("--config", type=Path, required=True, help="Training YAML")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override total epochs (useful for a smoke run)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run model training and return a process exit code."""

    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    try:
        config = load_training_config(args.config)
        report = run_training(config, epochs_override=args.epochs)
    except (
        ConfigError,
        DatasetContractError,
        ModelConfigurationError,
        ModelInputError,
        TrainingError,
        OSError,
    ) as exc:
        print(f"ERROR | {exc}", file=sys.stderr)
        return 2

    final = report.history[-1]
    print(
        f"Training complete: epoch {report.completed_epochs}, "
        f"train loss {final.train_loss:.4f}, "
        f"validation loss {final.validation_loss:.4f}."
    )
    print(
        f"Best validation: {report.best_validation_loss:.4f} "
        f"at epoch {report.best_epoch}."
    )
    print(f"Device: {report.device} | Parameters: {report.num_parameters:,}")
    print(f"Best checkpoint: {report.best_checkpoint}")
    print(f"Latest checkpoint: {report.latest_checkpoint}")
    print(f"Training report: {report.training_report_path}")
    print(f"TensorBoard: {report.tensorboard_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
