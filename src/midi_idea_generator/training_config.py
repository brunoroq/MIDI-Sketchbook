"""Strict YAML configuration contract for Stage 3 model training.

This module deliberately does not import PyTorch.  Configuration can therefore
be validated before expensive training dependencies or accelerator state are
initialised, and its unit tests remain fast and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import ConfigError


@dataclass(frozen=True, slots=True)
class TrainingPathsConfig:
    """Resolved token input and mutable training-output locations."""

    tokenization_manifest_path: Path
    checkpoints_dir: Path
    tensorboard_log_dir: Path
    resume_from: Path | None = None


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Small autoregressive GRU architecture used by the baseline."""

    architecture: str = "gru"
    embedding_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Batching limits for already-tokenized sequences."""

    max_sequence_length: int = 512
    batch_size: int = 4
    num_workers: int = 0


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    """Optimizer, validation, and checkpoint scheduling settings."""

    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    mixed_precision: str = "auto"
    checkpoint_every_epochs: int = 1
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Complete and validated Stage 3 training configuration."""

    seed: int
    device: str
    project_root: Path
    paths: TrainingPathsConfig
    model: ModelConfig
    data: DataConfig
    training: OptimizationConfig


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"Configuration section '{name}' must be a mapping.")
    return value


def _section(root: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if name not in root:
        raise ConfigError(f"Missing required configuration section: '{name}'.")
    return _mapping(root[name], name)


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], name: str) -> None:
    non_string_keys = [key for key in mapping if not isinstance(key, str)]
    if non_string_keys:
        rendered = ", ".join(repr(key) for key in non_string_keys)
        raise ConfigError(
            f"Configuration keys in '{name}' must be strings: {rendered}."
        )
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"Unknown key(s) in '{name}': {', '.join(unknown)}.")


def _as_int(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{key}' must be an integer.")
    return value


def _as_float(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{key}' must be a number.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ConfigError(f"'{key}' must be finite.")
    return converted


def _as_choice(value: object, key: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        rendered = ", ".join(sorted(choices))
        raise ConfigError(f"'{key}' must be one of: {rendered}.")
    return value


def _project_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return config_path.parent


def _resolve_path(value: object, key: str, project_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{key}' must be a non-empty path string.")
    try:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (project_root / path).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfigError(f"'{key}' is not a valid path: {exc}") from exc


def _resolve_optional_path(
    value: object, key: str, project_root: Path
) -> Path | None:
    if value is None:
        return None
    return _resolve_path(value, key, project_root)


def _is_within(path: Path, directory: Path) -> bool:
    return path == directory or path.is_relative_to(directory)


def _load_paths(
    root: Mapping[str, Any], project_root: Path
) -> TrainingPathsConfig:
    values = _section(root, "paths")
    allowed = {
        "tokenization_manifest",
        "checkpoints_dir",
        "tensorboard_log_dir",
        "resume_from",
    }
    _reject_unknown(values, allowed, "paths")
    required = {"tokenization_manifest", "checkpoints_dir", "tensorboard_log_dir"}
    missing = sorted(required - set(values))
    if missing:
        raise ConfigError(f"Missing path setting(s): {', '.join(missing)}.")
    return TrainingPathsConfig(
        tokenization_manifest_path=_resolve_path(
            values["tokenization_manifest"],
            "paths.tokenization_manifest",
            project_root,
        ),
        checkpoints_dir=_resolve_path(
            values["checkpoints_dir"], "paths.checkpoints_dir", project_root
        ),
        tensorboard_log_dir=_resolve_path(
            values["tensorboard_log_dir"],
            "paths.tensorboard_log_dir",
            project_root,
        ),
        resume_from=_resolve_optional_path(
            values.get("resume_from"), "paths.resume_from", project_root
        ),
    )


def _validate_output_directory(
    path: Path,
    key: str,
    project_root: Path,
) -> None:
    if _is_within(project_root, path):
        raise ConfigError(f"'{key}' cannot contain the project root.")

    protected_relative_paths = (
        ".git",
        "configs",
        "docs",
        "scripts",
        "src",
        "tests",
        "data",
    )
    for relative_path in protected_relative_paths:
        protected = (project_root / relative_path).resolve()
        if _is_within(path, protected) or _is_within(protected, path):
            raise ConfigError(
                f"'{key}' must not overlap protected project path "
                f"'{relative_path}'."
            )


def _validate_path_safety(
    paths: TrainingPathsConfig, project_root: Path, config_path: Path
) -> None:
    manifest = paths.tokenization_manifest_path
    checkpoints = paths.checkpoints_dir
    tensorboard = paths.tensorboard_log_dir

    if manifest.suffix.lower() != ".json":
        raise ConfigError("'paths.tokenization_manifest' must use a .json extension.")
    if manifest == config_path:
        raise ConfigError(
            "'paths.tokenization_manifest' cannot be its own configuration file."
        )

    _validate_output_directory(
        checkpoints, "paths.checkpoints_dir", project_root
    )
    _validate_output_directory(
        tensorboard, "paths.tensorboard_log_dir", project_root
    )

    if _is_within(checkpoints, tensorboard) or _is_within(tensorboard, checkpoints):
        raise ConfigError(
            "'paths.checkpoints_dir' and 'paths.tensorboard_log_dir' must not "
            "overlap."
        )
    if _is_within(manifest, checkpoints) or _is_within(manifest, tensorboard):
        raise ConfigError(
            "'paths.tokenization_manifest' must be outside training output "
            "directories."
        )

    resume_from = paths.resume_from
    if resume_from is not None:
        if resume_from.suffix.lower() not in {".pt", ".pth", ".ckpt"}:
            raise ConfigError(
                "'paths.resume_from' must use a .pt, .pth, or .ckpt extension."
            )
        if resume_from.name != "latest.pt":
            raise ConfigError(
                "'paths.resume_from' must point to a file named 'latest.pt'."
            )
        if not _is_within(resume_from, checkpoints):
            raise ConfigError(
                "'paths.resume_from' must be inside 'paths.checkpoints_dir'."
            )


def _load_model(root: Mapping[str, Any]) -> ModelConfig:
    values = _section(root, "model")
    allowed = {
        "architecture",
        "embedding_dim",
        "hidden_dim",
        "num_layers",
        "dropout",
    }
    _reject_unknown(values, allowed, "model")

    config = ModelConfig(
        architecture=_as_choice(
            values.get("architecture", "gru"),
            "model.architecture",
            {"gru"},
        ),
        embedding_dim=_as_int(
            values.get("embedding_dim", 64), "model.embedding_dim"
        ),
        hidden_dim=_as_int(values.get("hidden_dim", 128), "model.hidden_dim"),
        num_layers=_as_int(values.get("num_layers", 2), "model.num_layers"),
        dropout=_as_float(values.get("dropout", 0.2), "model.dropout"),
    )
    if config.embedding_dim <= 0:
        raise ConfigError("'model.embedding_dim' must be positive.")
    if config.hidden_dim <= 0:
        raise ConfigError("'model.hidden_dim' must be positive.")
    if config.num_layers < 2:
        raise ConfigError("'model.num_layers' must be at least 2.")
    if not 0.0 <= config.dropout < 1.0:
        raise ConfigError("'model.dropout' must satisfy 0 <= dropout < 1.")
    return config


def _load_data(root: Mapping[str, Any]) -> DataConfig:
    values = _section(root, "data")
    allowed = {"max_sequence_length", "batch_size", "num_workers"}
    _reject_unknown(values, allowed, "data")

    config = DataConfig(
        max_sequence_length=_as_int(
            values.get("max_sequence_length", 512),
            "data.max_sequence_length",
        ),
        batch_size=_as_int(values.get("batch_size", 4), "data.batch_size"),
        num_workers=_as_int(
            values.get("num_workers", 0), "data.num_workers"
        ),
    )
    if config.max_sequence_length < 3:
        raise ConfigError("'data.max_sequence_length' must be at least 3.")
    if config.batch_size <= 0:
        raise ConfigError("'data.batch_size' must be positive.")
    if config.num_workers < 0:
        raise ConfigError("'data.num_workers' cannot be negative.")
    return config


def _load_training(root: Mapping[str, Any]) -> OptimizationConfig:
    values = _section(root, "training")
    allowed = {
        "epochs",
        "learning_rate",
        "weight_decay",
        "gradient_clip",
        "mixed_precision",
        "checkpoint_every_epochs",
        "early_stopping_patience",
        "early_stopping_min_delta",
    }
    _reject_unknown(values, allowed, "training")

    config = OptimizationConfig(
        epochs=_as_int(values.get("epochs", 50), "training.epochs"),
        learning_rate=_as_float(
            values.get("learning_rate", 3e-4), "training.learning_rate"
        ),
        weight_decay=_as_float(
            values.get("weight_decay", 1e-4), "training.weight_decay"
        ),
        gradient_clip=_as_float(
            values.get("gradient_clip", 1.0), "training.gradient_clip"
        ),
        mixed_precision=_as_choice(
            values.get("mixed_precision", "auto"),
            "training.mixed_precision",
            {"auto", "off", "on"},
        ),
        checkpoint_every_epochs=_as_int(
            values.get("checkpoint_every_epochs", 1),
            "training.checkpoint_every_epochs",
        ),
        early_stopping_patience=_as_int(
            values.get("early_stopping_patience", 8),
            "training.early_stopping_patience",
        ),
        early_stopping_min_delta=_as_float(
            values.get("early_stopping_min_delta", 1e-4),
            "training.early_stopping_min_delta",
        ),
    )
    if config.epochs <= 0:
        raise ConfigError("'training.epochs' must be positive.")
    if config.learning_rate <= 0:
        raise ConfigError("'training.learning_rate' must be positive.")
    if config.weight_decay < 0:
        raise ConfigError("'training.weight_decay' cannot be negative.")
    if config.gradient_clip <= 0:
        raise ConfigError("'training.gradient_clip' must be positive.")
    if config.checkpoint_every_epochs <= 0:
        raise ConfigError("'training.checkpoint_every_epochs' must be positive.")
    if config.early_stopping_patience <= 0:
        raise ConfigError("'training.early_stopping_patience' must be positive.")
    if config.early_stopping_min_delta < 0:
        raise ConfigError("'training.early_stopping_min_delta' cannot be negative.")
    return config


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load and validate an executable Stage 3 YAML configuration.

    Relative paths are resolved against the nearest parent containing
    ``pyproject.toml`` so invocation does not depend on the current directory.
    """

    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigError(
            f"Could not read configuration '{config_path}': {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in '{config_path}': {exc}") from exc

    root = _mapping(raw, "root")
    _reject_unknown(
        root,
        {"seed", "device", "paths", "model", "data", "training"},
        "root",
    )
    project_root = _project_root(config_path)
    seed = _as_int(root.get("seed", 42), "seed")
    if not 0 <= seed <= 2**32 - 1:
        raise ConfigError("'seed' must be between 0 and 4294967295.")
    device = _as_choice(
        root.get("device", "auto"), "device", {"auto", "cpu", "cuda"}
    )
    paths = _load_paths(root, project_root)
    _validate_path_safety(paths, project_root, config_path)

    return TrainingConfig(
        seed=seed,
        device=device,
        project_root=project_root,
        paths=paths,
        model=_load_model(root),
        data=_load_data(root),
        training=_load_training(root),
    )


__all__ = [
    "DataConfig",
    "ModelConfig",
    "OptimizationConfig",
    "TrainingConfig",
    "TrainingPathsConfig",
    "load_training_config",
]
