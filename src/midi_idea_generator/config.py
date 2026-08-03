"""YAML configuration loading and validation for preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Resolved filesystem locations used by preprocessing."""

    input_dir: Path
    processed_dir: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Rules used to decide whether a MIDI file and track are usable."""

    pitch_min: int = 21
    pitch_max: int = 108
    allowed_time_signature: tuple[int, int] = (4, 4)
    allow_missing_time_signature: bool = True
    reject_pitch_bends: bool = True
    exclude_drums: bool = True
    min_notes_per_track: int = 1
    tempo_tolerance: float = 0.01


@dataclass(frozen=True, slots=True)
class TrackSelectionConfig:
    """Strategy for choosing one instrumental track from a MIDI file."""

    mode: str = "most_notes"
    track_index: int | None = None


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    """Musical transformations applied before writing phrase MIDIs."""

    phrase_bars: int = 4
    remove_initial_silence: bool = True
    quantize: bool = True
    subdivisions_per_beat: int = 4
    include_partial_final_phrase: bool = True
    min_notes_per_phrase: int = 1


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    """Pitch-transposition augmentation settings."""

    enabled: bool = True
    min_semitones: int = -5
    max_semitones: int = 6
    apply_to_splits: tuple[str, ...] = ("train",)


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Source-file-level dataset split ratios."""

    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """Complete, validated preprocessing configuration."""

    random_seed: int
    project_root: Path
    paths: PathsConfig
    validation: ValidationConfig
    track_selection: TrackSelectionConfig
    preprocessing: ProcessingConfig
    augmentation: AugmentationConfig
    splits: SplitConfig


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
        joined = ", ".join(unknown)
        raise ConfigError(f"Unknown key(s) in '{name}': {joined}.")


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
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _load_paths(root: Mapping[str, Any], project_root: Path) -> PathsConfig:
    values = _section(root, "paths")
    allowed = {"input_dir", "processed_dir", "manifest_path"}
    _reject_unknown(values, allowed, "paths")
    missing = sorted(allowed - set(values))
    if missing:
        raise ConfigError(f"Missing path setting(s): {', '.join(missing)}.")
    return PathsConfig(
        input_dir=_resolve_path(values["input_dir"], "paths.input_dir", project_root),
        processed_dir=_resolve_path(
            values["processed_dir"], "paths.processed_dir", project_root
        ),
        manifest_path=_resolve_path(
            values["manifest_path"], "paths.manifest_path", project_root
        ),
    )


def _is_within(path: Path, directory: Path) -> bool:
    return path == directory or path.is_relative_to(directory)


def _validate_path_safety(
    paths: PathsConfig, project_root: Path, config_path: Path
) -> None:
    if _is_within(paths.processed_dir, paths.input_dir) or _is_within(
        paths.input_dir, paths.processed_dir
    ):
        raise ConfigError(
            "'paths.input_dir' and 'paths.processed_dir' must not overlap."
        )
    if _is_within(project_root, paths.processed_dir):
        raise ConfigError(
            "'paths.processed_dir' cannot contain the project root."
        )
    protected_directories = (
        ".git",
        "configs",
        "docs",
        "scripts",
        "src",
        "tests",
    )
    for directory_name in protected_directories:
        protected = (project_root / directory_name).resolve()
        if _is_within(paths.processed_dir, protected) or _is_within(
            protected, paths.processed_dir
        ):
            raise ConfigError(
                "'paths.processed_dir' must not overlap protected project "
                f"directory '{directory_name}'."
            )
    if _is_within(paths.manifest_path, paths.input_dir):
        raise ConfigError("'paths.manifest_path' cannot be inside the MIDI input directory.")
    if _is_within(paths.manifest_path, paths.processed_dir):
        raise ConfigError("'paths.manifest_path' must be outside the processed MIDI directory.")
    if paths.manifest_path == config_path:
        raise ConfigError("'paths.manifest_path' cannot replace its own configuration file.")
    if paths.manifest_path.suffix.lower() != ".json":
        raise ConfigError("'paths.manifest_path' must use a .json extension.")
    for directory_name in protected_directories:
        protected = (project_root / directory_name).resolve()
        if _is_within(paths.manifest_path, protected):
            raise ConfigError(
                "'paths.manifest_path' must not be inside protected project "
                f"directory '{directory_name}'."
            )


def _load_validation(root: Mapping[str, Any]) -> ValidationConfig:
    values = _section(root, "validation")
    allowed = {
        "pitch_min",
        "pitch_max",
        "allowed_time_signature",
        "allow_missing_time_signature",
        "reject_pitch_bends",
        "exclude_drums",
        "min_notes_per_track",
        "tempo_tolerance",
    }
    _reject_unknown(values, allowed, "validation")
    signature = values.get("allowed_time_signature", [4, 4])
    if (
        not isinstance(signature, (list, tuple))
        or len(signature) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in signature)
    ):
        raise ConfigError("'validation.allowed_time_signature' must be [numerator, denominator].")
    config = ValidationConfig(
        pitch_min=_as_int(values.get("pitch_min", 21), "validation.pitch_min"),
        pitch_max=_as_int(values.get("pitch_max", 108), "validation.pitch_max"),
        allowed_time_signature=(signature[0], signature[1]),
        allow_missing_time_signature=_as_bool(
            values.get("allow_missing_time_signature", True),
            "validation.allow_missing_time_signature",
        ),
        reject_pitch_bends=_as_bool(
            values.get("reject_pitch_bends", True), "validation.reject_pitch_bends"
        ),
        exclude_drums=_as_bool(
            values.get("exclude_drums", True), "validation.exclude_drums"
        ),
        min_notes_per_track=_as_int(
            values.get("min_notes_per_track", 1), "validation.min_notes_per_track"
        ),
        tempo_tolerance=_as_float(
            values.get("tempo_tolerance", 0.01), "validation.tempo_tolerance"
        ),
    )
    if not 0 <= config.pitch_min <= config.pitch_max <= 127:
        raise ConfigError("Pitch range must satisfy 0 <= pitch_min <= pitch_max <= 127.")
    numerator, denominator = config.allowed_time_signature
    if (numerator, denominator) != (4, 4):
        raise ConfigError("The stage-one MVP only supports a 4/4 time signature.")
    if config.min_notes_per_track <= 0:
        raise ConfigError("'validation.min_notes_per_track' must be positive.")
    if config.tempo_tolerance < 0:
        raise ConfigError("'validation.tempo_tolerance' cannot be negative.")
    if not config.reject_pitch_bends:
        raise ConfigError("The stage-one MVP requires 'reject_pitch_bends: true'.")
    if not config.exclude_drums:
        raise ConfigError("The stage-one MVP requires 'exclude_drums: true'.")
    return config


def _load_track_selection(root: Mapping[str, Any]) -> TrackSelectionConfig:
    values = _section(root, "track_selection")
    _reject_unknown(values, {"mode", "track_index"}, "track_selection")
    mode = values.get("mode", "most_notes")
    if not isinstance(mode, str) or mode not in {"most_notes", "index"}:
        raise ConfigError("'track_selection.mode' must be 'most_notes' or 'index'.")
    raw_index = values.get("track_index")
    track_index = None if raw_index is None else _as_int(raw_index, "track_selection.track_index")
    if track_index is not None and track_index < 0:
        raise ConfigError("'track_selection.track_index' cannot be negative.")
    if mode == "index" and track_index is None:
        raise ConfigError("'track_selection.track_index' is required when mode is 'index'.")
    if mode == "most_notes" and track_index is not None:
        raise ConfigError(
            "'track_selection.track_index' must be null when mode is 'most_notes'."
        )
    return TrackSelectionConfig(mode=mode, track_index=track_index)


def _load_processing(root: Mapping[str, Any]) -> ProcessingConfig:
    values = _section(root, "preprocessing")
    allowed = {
        "phrase_bars",
        "remove_initial_silence",
        "quantize",
        "subdivisions_per_beat",
        "include_partial_final_phrase",
        "min_notes_per_phrase",
    }
    _reject_unknown(values, allowed, "preprocessing")
    config = ProcessingConfig(
        phrase_bars=_as_int(values.get("phrase_bars", 4), "preprocessing.phrase_bars"),
        remove_initial_silence=_as_bool(
            values.get("remove_initial_silence", True),
            "preprocessing.remove_initial_silence",
        ),
        quantize=_as_bool(values.get("quantize", True), "preprocessing.quantize"),
        subdivisions_per_beat=_as_int(
            values.get("subdivisions_per_beat", 4),
            "preprocessing.subdivisions_per_beat",
        ),
        include_partial_final_phrase=_as_bool(
            values.get("include_partial_final_phrase", True),
            "preprocessing.include_partial_final_phrase",
        ),
        min_notes_per_phrase=_as_int(
            values.get("min_notes_per_phrase", 1),
            "preprocessing.min_notes_per_phrase",
        ),
    )
    if config.phrase_bars not in {2, 4, 8}:
        raise ConfigError("'preprocessing.phrase_bars' must be 2, 4, or 8.")
    if config.subdivisions_per_beat <= 0:
        raise ConfigError("'preprocessing.subdivisions_per_beat' must be positive.")
    if config.min_notes_per_phrase <= 0:
        raise ConfigError("'preprocessing.min_notes_per_phrase' must be positive.")
    return config


def _load_augmentation(root: Mapping[str, Any]) -> AugmentationConfig:
    values = _section(root, "augmentation")
    allowed = {"enabled", "min_semitones", "max_semitones", "apply_to_splits"}
    _reject_unknown(values, allowed, "augmentation")
    raw_splits = values.get("apply_to_splits", ["train"])
    if not isinstance(raw_splits, list) or not all(isinstance(item, str) for item in raw_splits):
        raise ConfigError("'augmentation.apply_to_splits' must be a list of split names.")
    valid_splits = {"train", "validation", "test"}
    if not raw_splits or not set(raw_splits) <= valid_splits:
        raise ConfigError(
            "'augmentation.apply_to_splits' must contain train, validation, and/or test."
        )
    config = AugmentationConfig(
        enabled=_as_bool(values.get("enabled", True), "augmentation.enabled"),
        min_semitones=_as_int(
            values.get("min_semitones", -5), "augmentation.min_semitones"
        ),
        max_semitones=_as_int(
            values.get("max_semitones", 6), "augmentation.max_semitones"
        ),
        apply_to_splits=tuple(dict.fromkeys(raw_splits)),
    )
    if config.min_semitones > config.max_semitones:
        raise ConfigError("Augmentation min_semitones cannot exceed max_semitones.")
    if not -127 <= config.min_semitones <= config.max_semitones <= 127:
        raise ConfigError("Augmentation semitone offsets must stay within [-127, 127].")
    return config


def _load_splits(root: Mapping[str, Any]) -> SplitConfig:
    values = _section(root, "splits")
    _reject_unknown(values, {"train", "validation", "test"}, "splits")
    config = SplitConfig(
        train=_as_float(values.get("train", 0.8), "splits.train"),
        validation=_as_float(values.get("validation", 0.1), "splits.validation"),
        test=_as_float(values.get("test", 0.1), "splits.test"),
    )
    ratios = (config.train, config.validation, config.test)
    if any(ratio < 0 for ratio in ratios):
        raise ConfigError("Split ratios cannot be negative.")
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ConfigError("Split ratios must sum to 1.0.")
    if config.train <= 0:
        raise ConfigError("The training split ratio must be positive.")
    return config


def load_preprocess_config(path: str | Path) -> PreprocessConfig:
    """Load a preprocessing YAML file and return a validated configuration.

    Relative data paths are resolved against the nearest parent containing
    ``pyproject.toml``. This makes invocation independent of the current working
    directory while keeping project configuration portable.
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
    allowed = {
        "random_seed",
        "paths",
        "validation",
        "track_selection",
        "preprocessing",
        "augmentation",
        "splits",
    }
    _reject_unknown(root, allowed, "root")
    project_root = _project_root(config_path)
    random_seed = _as_int(root.get("random_seed", 42), "random_seed")
    if random_seed < 0:
        raise ConfigError("'random_seed' cannot be negative.")
    paths = _load_paths(root, project_root)
    _validate_path_safety(paths, project_root, config_path)
    return PreprocessConfig(
        random_seed=random_seed,
        project_root=project_root,
        paths=paths,
        validation=_load_validation(root),
        track_selection=_load_track_selection(root),
        preprocessing=_load_processing(root),
        augmentation=_load_augmentation(root),
        splits=_load_splits(root),
    )
