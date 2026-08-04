"""YAML configuration loading and validation for REMI tokenization.

Stage 2 deliberately has its own configuration module.  Keeping this contract
separate from preprocessing lets the tokenization pipeline evolve without
changing the already-published Stage 1 configuration schema.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import ConfigError


@dataclass(frozen=True, slots=True)
class TokenizationPathsConfig:
    """Resolved Stage 1 input and Stage 2 output locations."""

    preprocessing_manifest_path: Path
    tokenized_dir: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class BeatResolutionConfig:
    """MidiTok resolution for the half-open beat range ``[start, end)``."""

    start_beat: int
    end_beat: int
    resolution: int


@dataclass(frozen=True, slots=True)
class RemiTokenizerConfig:
    """Validated REMI vocabulary and quantization settings.

    ``pitch_min`` and ``pitch_max`` are inclusive limits, matching the exact
    MidiTok 3.0.6.post1 implementation used by this project.
    """

    pitch_min: int = 21
    pitch_max: int = 108
    beat_res: tuple[BeatResolutionConfig, ...] = (
        BeatResolutionConfig(start_beat=0, end_beat=4, resolution=24),
        BeatResolutionConfig(start_beat=4, end_beat=16, resolution=4),
    )
    special_tokens: tuple[str, ...] = ("PAD", "BOS", "EOS")
    encode_ids_split: str = "bar"
    use_velocities: bool = False
    num_velocities: int = 16
    use_tempos: bool = True
    num_tempos: int = 32
    tempo_min: float = 40.0
    tempo_max: float = 250.0
    add_trailing_bars: bool = True
    max_bar_embedding: int | None = None


@dataclass(frozen=True, slots=True)
class TokenizationConfig:
    """Complete, validated Stage 2 configuration."""

    project_root: Path
    paths: TokenizationPathsConfig
    tokenizer: RemiTokenizerConfig


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
        raise ConfigError(f"Configuration keys in '{name}' must be strings: {rendered}.")
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"Unknown key(s) in '{name}': {', '.join(unknown)}.")


def _as_bool(value: object, key: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"'{key}' must be true or false.")
    return value


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


def _is_within(path: Path, directory: Path) -> bool:
    return path == directory or path.is_relative_to(directory)


def _load_paths(
    root: Mapping[str, Any], project_root: Path
) -> TokenizationPathsConfig:
    values = _section(root, "paths")
    allowed = {"preprocessing_manifest_path", "tokenized_dir", "manifest_path"}
    _reject_unknown(values, allowed, "paths")
    missing = sorted(allowed - set(values))
    if missing:
        raise ConfigError(f"Missing path setting(s): {', '.join(missing)}.")
    return TokenizationPathsConfig(
        preprocessing_manifest_path=_resolve_path(
            values["preprocessing_manifest_path"],
            "paths.preprocessing_manifest_path",
            project_root,
        ),
        tokenized_dir=_resolve_path(
            values["tokenized_dir"], "paths.tokenized_dir", project_root
        ),
        manifest_path=_resolve_path(
            values["manifest_path"], "paths.manifest_path", project_root
        ),
    )


def _validate_path_safety(
    paths: TokenizationPathsConfig, project_root: Path, config_path: Path
) -> None:
    input_manifest = paths.preprocessing_manifest_path
    output_dir = paths.tokenized_dir
    output_manifest = paths.manifest_path

    if input_manifest.suffix.lower() != ".json":
        raise ConfigError(
            "'paths.preprocessing_manifest_path' must use a .json extension."
        )
    if output_manifest.suffix.lower() != ".json":
        raise ConfigError("'paths.manifest_path' must use a .json extension.")
    if input_manifest == output_manifest:
        raise ConfigError(
            "The preprocessing and tokenization manifests must be different files."
        )
    if _is_within(project_root, output_dir):
        raise ConfigError("'paths.tokenized_dir' cannot contain the project root.")

    protected_relative_paths = (
        ".git",
        "configs",
        "docs",
        "scripts",
        "src",
        "tests",
        "checkpoints",
        "outputs",
        "data/raw",
        "data/processed",
        "data/splits",
    )
    for relative_path in protected_relative_paths:
        protected = (project_root / relative_path).resolve()
        if _is_within(output_dir, protected) or _is_within(protected, output_dir):
            raise ConfigError(
                "'paths.tokenized_dir' must not overlap protected project "
                f"path '{relative_path}'."
            )
    if _is_within(input_manifest, output_dir):
        raise ConfigError(
            "'paths.preprocessing_manifest_path' must be outside "
            "'paths.tokenized_dir'."
        )
    if not _is_within(output_manifest, output_dir):
        raise ConfigError("'paths.manifest_path' must be inside 'paths.tokenized_dir'.")
    if _is_within(output_manifest, output_dir / "runs"):
        raise ConfigError(
            "'paths.manifest_path' must be outside the immutable 'runs' directory."
        )
    if output_manifest == config_path:
        raise ConfigError("'paths.manifest_path' cannot replace its own configuration file.")


def _load_beat_res(values: object) -> tuple[BeatResolutionConfig, ...]:
    if not isinstance(values, list) or not values:
        raise ConfigError("'tokenizer.beat_res' must be a non-empty list.")

    resolutions: list[BeatResolutionConfig] = []
    for index, raw_resolution in enumerate(values):
        name = f"tokenizer.beat_res[{index}]"
        resolution_values = _mapping(raw_resolution, name)
        allowed = {"start_beat", "end_beat", "resolution"}
        _reject_unknown(resolution_values, allowed, name)
        missing = sorted(allowed - set(resolution_values))
        if missing:
            raise ConfigError(
                f"Missing setting(s) in '{name}': {', '.join(missing)}."
            )
        resolution = BeatResolutionConfig(
            start_beat=_as_int(
                resolution_values["start_beat"], f"{name}.start_beat"
            ),
            end_beat=_as_int(resolution_values["end_beat"], f"{name}.end_beat"),
            resolution=_as_int(
                resolution_values["resolution"], f"{name}.resolution"
            ),
        )
        if not 0 <= resolution.start_beat < resolution.end_beat <= 64:
            raise ConfigError(
                f"'{name}' must satisfy 0 <= start_beat < end_beat <= 64."
            )
        if not 1 <= resolution.resolution <= 64:
            raise ConfigError(f"'{name}.resolution' must be between 1 and 64.")
        resolutions.append(resolution)

    if resolutions[0].start_beat != 0:
        raise ConfigError("'tokenizer.beat_res' must start at beat 0.")
    for previous, current in zip(resolutions, resolutions[1:], strict=False):
        if previous.end_beat != current.start_beat:
            raise ConfigError(
                "'tokenizer.beat_res' ranges must be ordered, contiguous, and "
                "non-overlapping."
            )
    return tuple(resolutions)


def _load_special_tokens(values: object) -> tuple[str, ...]:
    if not isinstance(values, list) or not all(
        isinstance(token, str) and token for token in values
    ):
        raise ConfigError(
            "'tokenizer.special_tokens' must be a list of non-empty strings."
        )
    tokens = tuple(values)
    if len(set(tokens)) != len(tokens):
        raise ConfigError("'tokenizer.special_tokens' cannot contain duplicates.")
    if tokens != ("PAD", "BOS", "EOS"):
        raise ConfigError(
            "Stage 2 requires 'tokenizer.special_tokens' in the order "
            "[PAD, BOS, EOS]."
        )
    return tokens


def _optional_positive_int(value: object, key: str) -> int | None:
    if value is None:
        return None
    converted = _as_int(value, key)
    if converted <= 0:
        raise ConfigError(f"'{key}' must be null or a positive integer.")
    return converted


def _load_tokenizer(root: Mapping[str, Any]) -> RemiTokenizerConfig:
    values = _section(root, "tokenizer")
    allowed = {
        "type",
        "pitch_min",
        "pitch_max",
        "beat_res",
        "special_tokens",
        "encode_ids_split",
        "use_velocities",
        "num_velocities",
        "use_tempos",
        "num_tempos",
        "tempo_min",
        "tempo_max",
        "add_trailing_bars",
        "max_bar_embedding",
    }
    _reject_unknown(values, allowed, "tokenizer")

    tokenizer_type = values.get("type", "remi")
    if tokenizer_type != "remi":
        raise ConfigError("'tokenizer.type' must be 'remi' in Stage 2.")

    raw_encode_ids_split = values.get("encode_ids_split", "bar")
    if not isinstance(raw_encode_ids_split, str) or raw_encode_ids_split not in {
        "bar",
        "beat",
        "no",
    }:
        raise ConfigError("'tokenizer.encode_ids_split' must be 'bar', 'beat', or 'no'.")

    config = RemiTokenizerConfig(
        pitch_min=_as_int(values.get("pitch_min", 21), "tokenizer.pitch_min"),
        pitch_max=_as_int(values.get("pitch_max", 108), "tokenizer.pitch_max"),
        beat_res=_load_beat_res(
            values.get(
                "beat_res",
                [
                    {"start_beat": 0, "end_beat": 4, "resolution": 24},
                    {"start_beat": 4, "end_beat": 16, "resolution": 4},
                ],
            )
        ),
        special_tokens=_load_special_tokens(
            values.get("special_tokens", ["PAD", "BOS", "EOS"])
        ),
        encode_ids_split=raw_encode_ids_split,
        use_velocities=_as_bool(
            values.get("use_velocities", False), "tokenizer.use_velocities"
        ),
        num_velocities=_as_int(
            values.get("num_velocities", 16), "tokenizer.num_velocities"
        ),
        use_tempos=_as_bool(values.get("use_tempos", True), "tokenizer.use_tempos"),
        num_tempos=_as_int(values.get("num_tempos", 32), "tokenizer.num_tempos"),
        tempo_min=_as_float(values.get("tempo_min", 40), "tokenizer.tempo_min"),
        tempo_max=_as_float(values.get("tempo_max", 250), "tokenizer.tempo_max"),
        add_trailing_bars=_as_bool(
            values.get("add_trailing_bars", True), "tokenizer.add_trailing_bars"
        ),
        max_bar_embedding=_optional_positive_int(
            values.get("max_bar_embedding"), "tokenizer.max_bar_embedding"
        ),
    )

    if not 0 <= config.pitch_min <= config.pitch_max <= 127:
        raise ConfigError(
            "Tokenizer pitch range must satisfy "
            "0 <= pitch_min <= pitch_max <= 127 (inclusive)."
        )
    if not 1 <= config.num_velocities <= 128:
        raise ConfigError("'tokenizer.num_velocities' must be between 1 and 128.")
    if not 1 <= config.num_tempos <= 512:
        raise ConfigError("'tokenizer.num_tempos' must be between 1 and 512.")
    if config.tempo_min <= 0 or config.tempo_min >= config.tempo_max:
        raise ConfigError(
            "Tokenizer tempo range must satisfy 0 < tempo_min < tempo_max."
        )
    if config.use_velocities:
        raise ConfigError(
            "Stage 2 requires 'tokenizer.use_velocities: false'; the inspected "
            "corpus has no useful within-riff velocity variation."
        )
    if not config.use_tempos:
        raise ConfigError("Stage 2 requires 'tokenizer.use_tempos: true'.")
    if not config.add_trailing_bars:
        raise ConfigError("Stage 2 requires 'tokenizer.add_trailing_bars: true'.")
    return config


def load_tokenization_config(path: str | Path) -> TokenizationConfig:
    """Load and validate the executable Stage 2 YAML configuration.

    Relative paths are resolved against the nearest parent containing
    ``pyproject.toml``, independently of the process's current directory.
    """

    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigError(f"Could not read configuration '{config_path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in '{config_path}': {exc}") from exc

    root = _mapping(raw, "root")
    _reject_unknown(root, {"paths", "tokenizer"}, "root")
    project_root = _project_root(config_path)
    paths = _load_paths(root, project_root)
    _validate_path_safety(paths, project_root, config_path)
    return TokenizationConfig(
        project_root=project_root,
        paths=paths,
        tokenizer=_load_tokenizer(root),
    )
