"""Strict YAML configuration contract for unconditional MIDI generation.

The module intentionally stays independent of PyTorch and the generation
runtime.  It validates sampling parameters and filesystem boundaries before a
checkpoint is loaded or an output directory is created.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import ConfigError


@dataclass(frozen=True, slots=True)
class GenerationPathsConfig:
    """Resolved immutable inputs and mutable generation output location."""

    checkpoint_path: Path
    tokenization_manifest_path: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Limits and probability controls for unconditional token sampling."""

    max_tokens: int = 256
    min_tokens: int = 32
    temperature: float = 0.9
    top_k: int = 20
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    max_simultaneous_notes: int = 3
    num_samples: int = 4
    max_attempts_per_sample: int = 25


@dataclass(frozen=True, slots=True)
class MidiConfig:
    """MIDI rendering settings."""

    program: int = 29


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    """Optional piano-roll rendering settings."""

    enabled: bool = True
    dpi: int = 160


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Complete validated configuration for unconditional generation."""

    seed: int
    device: str
    project_root: Path
    paths: GenerationPathsConfig
    generation: SamplingConfig
    midi: MidiConfig
    visualization: VisualizationConfig


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


def _as_bool(value: object, key: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"'{key}' must be true or false.")
    return value


def _as_choice(value: object, key: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        rendered = ", ".join(sorted(choices))
        raise ConfigError(f"'{key}' must be one of: {rendered}.")
    return value


def _project_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate.resolve()
    return config_path.parent.resolve()


def _resolve_path(value: object, key: str, project_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"'{key}' must be a non-empty path string.")
    try:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (project_root / path).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfigError(f"'{key}' is not a valid path: {exc}") from exc


def _is_within(path: Path, directory: Path) -> bool:
    return path == directory or path.is_relative_to(directory)


def _load_paths(
    root: Mapping[str, Any], project_root: Path
) -> GenerationPathsConfig:
    values = _section(root, "paths")
    allowed = {"checkpoint", "tokenization_manifest", "output_dir"}
    _reject_unknown(values, allowed, "paths")
    missing = sorted(allowed - set(values))
    if missing:
        raise ConfigError(f"Missing path setting(s): {', '.join(missing)}.")
    return GenerationPathsConfig(
        checkpoint_path=_resolve_path(
            values["checkpoint"], "paths.checkpoint", project_root
        ),
        tokenization_manifest_path=_resolve_path(
            values["tokenization_manifest"],
            "paths.tokenization_manifest",
            project_root,
        ),
        output_dir=_resolve_path(
            values["output_dir"], "paths.output_dir", project_root
        ),
    )


def _validate_input_file(path: Path, key: str, suffix: str) -> None:
    if path.suffix.lower() != suffix:
        raise ConfigError(f"'{key}' must use a {suffix} extension.")
    if not path.is_file():
        raise ConfigError(f"'{key}' must reference an existing regular file: {path}")


def _validate_output_directory(
    output_dir: Path,
    project_root: Path,
    config_path: Path,
) -> None:
    if not _is_within(output_dir, project_root) or output_dir == project_root:
        raise ConfigError(
            "'paths.output_dir' must be inside the project and cannot be the "
            "project root."
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise ConfigError("'paths.output_dir' must not reference an existing file.")

    protected_relative_paths = (
        ".git",
        "checkpoints",
        "configs",
        "data",
        "docs",
        "scripts",
        "src",
        "tests",
    )
    for relative_path in protected_relative_paths:
        protected = (project_root / relative_path).resolve()
        if _is_within(output_dir, protected) or _is_within(protected, output_dir):
            raise ConfigError(
                "'paths.output_dir' must not overlap protected project path "
                f"'{relative_path}'."
            )
    if _is_within(config_path, output_dir):
        raise ConfigError(
            "'paths.output_dir' must not contain its generation configuration."
        )


def _validate_path_safety(
    paths: GenerationPathsConfig,
    project_root: Path,
    config_path: Path,
) -> None:
    _validate_input_file(paths.checkpoint_path, "paths.checkpoint", ".pt")
    _validate_input_file(
        paths.tokenization_manifest_path,
        "paths.tokenization_manifest",
        ".json",
    )
    _validate_output_directory(paths.output_dir, project_root, config_path)

    for input_path, key in (
        (paths.checkpoint_path, "paths.checkpoint"),
        (paths.tokenization_manifest_path, "paths.tokenization_manifest"),
    ):
        if _is_within(input_path, paths.output_dir):
            raise ConfigError(
                f"'{key}' must be outside 'paths.output_dir'."
            )


def _load_sampling(root: Mapping[str, Any]) -> SamplingConfig:
    values = _section(root, "generation")
    allowed = {
        "max_tokens",
        "min_tokens",
        "temperature",
        "top_k",
        "top_p",
        "repetition_penalty",
        "max_simultaneous_notes",
        "num_samples",
        "max_attempts_per_sample",
    }
    _reject_unknown(values, allowed, "generation")
    config = SamplingConfig(
        max_tokens=_as_int(
            values.get("max_tokens", 256), "generation.max_tokens"
        ),
        min_tokens=_as_int(
            values.get("min_tokens", 32), "generation.min_tokens"
        ),
        temperature=_as_float(
            values.get("temperature", 0.9), "generation.temperature"
        ),
        top_k=_as_int(values.get("top_k", 20), "generation.top_k"),
        top_p=_as_float(values.get("top_p", 0.95), "generation.top_p"),
        repetition_penalty=_as_float(
            values.get("repetition_penalty", 1.05),
            "generation.repetition_penalty",
        ),
        max_simultaneous_notes=_as_int(
            values.get("max_simultaneous_notes", 3),
            "generation.max_simultaneous_notes",
        ),
        num_samples=_as_int(
            values.get("num_samples", 4), "generation.num_samples"
        ),
        max_attempts_per_sample=_as_int(
            values.get("max_attempts_per_sample", 25),
            "generation.max_attempts_per_sample",
        ),
    )
    if config.min_tokens <= 0:
        raise ConfigError("'generation.min_tokens' must be positive.")
    if config.max_tokens < config.min_tokens + 5:
        raise ConfigError(
            "'generation.max_tokens' must be at least five greater than "
            "'generation.min_tokens' so a REMI sequence can close safely."
        )
    if config.temperature <= 0:
        raise ConfigError("'generation.temperature' must be positive.")
    if config.top_k < 0:
        raise ConfigError("'generation.top_k' cannot be negative (use 0 to disable).")
    if not 0.0 < config.top_p <= 1.0:
        raise ConfigError("'generation.top_p' must satisfy 0 < top_p <= 1.")
    if config.repetition_penalty < 1.0:
        raise ConfigError(
            "'generation.repetition_penalty' must be greater than or equal to 1."
        )
    if not 1 <= config.max_simultaneous_notes <= 6:
        raise ConfigError(
            "'generation.max_simultaneous_notes' must be between 1 and 6."
        )
    if config.num_samples <= 0:
        raise ConfigError("'generation.num_samples' must be positive.")
    if config.max_attempts_per_sample <= 0:
        raise ConfigError(
            "'generation.max_attempts_per_sample' must be positive."
        )
    return config


def _load_midi(root: Mapping[str, Any]) -> MidiConfig:
    values = _section(root, "midi")
    _reject_unknown(values, {"program"}, "midi")
    config = MidiConfig(program=_as_int(values.get("program", 29), "midi.program"))
    if not 0 <= config.program <= 127:
        raise ConfigError("'midi.program' must be between 0 and 127.")
    return config


def _load_visualization(root: Mapping[str, Any]) -> VisualizationConfig:
    values = _section(root, "visualization")
    _reject_unknown(values, {"enabled", "dpi"}, "visualization")
    config = VisualizationConfig(
        enabled=_as_bool(
            values.get("enabled", True), "visualization.enabled"
        ),
        dpi=_as_int(values.get("dpi", 160), "visualization.dpi"),
    )
    if config.dpi <= 0:
        raise ConfigError("'visualization.dpi' must be positive.")
    return config


def load_generation_config(path: str | Path) -> GenerationConfig:
    """Load and validate an unconditional-generation YAML configuration.

    Relative paths are resolved against the nearest parent containing
    ``pyproject.toml``.  Input artifacts must already exist; the output
    directory may be absent, but it must be a safe location inside the project.
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
        {"seed", "device", "paths", "generation", "midi", "visualization"},
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

    return GenerationConfig(
        seed=seed,
        device=device,
        project_root=project_root,
        paths=paths,
        generation=_load_sampling(root),
        midi=_load_midi(root),
        visualization=_load_visualization(root),
    )


__all__ = [
    "GenerationConfig",
    "GenerationPathsConfig",
    "MidiConfig",
    "SamplingConfig",
    "VisualizationConfig",
    "load_generation_config",
]
