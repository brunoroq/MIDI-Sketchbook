"""Validation and path-resolution tests for preprocessing configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from midi_idea_generator.config import ConfigError, load_preprocess_config


def _valid_payload() -> dict[str, Any]:
    return {
        "random_seed": 17,
        "paths": {
            "input_dir": "data/raw",
            "processed_dir": "data/processed",
            "manifest_path": "data/splits/manifest.json",
        },
        "validation": {},
        "track_selection": {},
        "preprocessing": {},
        "augmentation": {},
        "splits": {},
    }


def _write_config(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _set_nested(payload: dict[str, Any], dotted_key: str, value: object) -> None:
    section_name, key = dotted_key.split(".", maxsplit=1)
    section = payload[section_name]
    assert isinstance(section, dict)
    section[key] = value


def test_load_config_resolves_relative_paths_from_nearest_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    absolute_processed = tmp_path / "outside" / "processed"
    payload = _valid_payload()
    payload["paths"]["processed_dir"] = str(absolute_processed)
    config_path = _write_config(
        project_root / "nested" / "configs" / "preprocess.yaml",
        payload,
    )
    unrelated_cwd = tmp_path / "somewhere-else"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    config = load_preprocess_config(config_path)

    assert config.project_root == project_root.resolve()
    assert config.paths.input_dir == (project_root / "data/raw").resolve()
    assert config.paths.processed_dir == absolute_processed.resolve()
    assert config.paths.manifest_path == (
        project_root / "data/splits/manifest.json"
    ).resolve()
    assert config.random_seed == 17
    assert config.validation.pitch_min == 21
    assert config.preprocessing.phrase_bars == 4
    assert config.splits.train == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("dotted_key", "value", "message"),
    [
        ("validation.pitch_min", True, "must be an integer"),
        ("validation.pitch_max", 128, "Pitch range must satisfy"),
        (
            "validation.allowed_time_signature",
            [3, 4],
            "only supports a 4/4 time signature",
        ),
        ("validation.tempo_tolerance", -0.01, "cannot be negative"),
        ("validation.reject_pitch_bends", False, "requires 'reject_pitch_bends: true'"),
        ("validation.exclude_drums", False, "requires 'exclude_drums: true'"),
        ("track_selection.mode", "index", "track_index.*is required"),
        ("preprocessing.phrase_bars", 3, "must be 2, 4, or 8"),
        ("preprocessing.subdivisions_per_beat", 0, "must be positive"),
        ("augmentation.apply_to_splits", [], "must contain train"),
        ("augmentation.min_semitones", 7, "cannot exceed max_semitones"),
        ("splits.validation", 0.2, "must sum to 1.0"),
        ("paths.input_dir", "", "must be a non-empty path string"),
    ],
)
def test_load_config_rejects_invalid_values(
    tmp_path: Path,
    dotted_key: str,
    value: object,
    message: str,
) -> None:
    payload = deepcopy(_valid_payload())
    _set_nested(payload, dotted_key, value)
    config_path = _write_config(tmp_path / "preprocess.yaml", payload)

    with pytest.raises(ConfigError, match=message):
        load_preprocess_config(config_path)


def test_load_config_rejects_boolean_random_seed(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["random_seed"] = True
    config_path = _write_config(tmp_path / "preprocess.yaml", payload)

    with pytest.raises(ConfigError, match="random_seed.*must be an integer"):
        load_preprocess_config(config_path)


@pytest.mark.parametrize("missing_section", ["paths", "validation", "splits"])
def test_load_config_requires_all_sections(
    tmp_path: Path,
    missing_section: str,
) -> None:
    payload = _valid_payload()
    del payload[missing_section]
    config_path = _write_config(tmp_path / "preprocess.yaml", payload)

    with pytest.raises(ConfigError, match=f"Missing required.*'{missing_section}'"):
        load_preprocess_config(config_path)


def test_load_config_rejects_missing_path_and_unknown_keys(tmp_path: Path) -> None:
    missing_path = _valid_payload()
    del missing_path["paths"]["manifest_path"]
    with pytest.raises(ConfigError, match="Missing path setting.*manifest_path"):
        load_preprocess_config(
            _write_config(tmp_path / "missing-path.yaml", missing_path)
        )

    unknown_key = _valid_payload()
    unknown_key["validation"]["mystery"] = 1
    with pytest.raises(ConfigError, match="Unknown key.*mystery"):
        load_preprocess_config(
            _write_config(tmp_path / "unknown-key.yaml", unknown_key)
        )


@pytest.mark.parametrize("contents", ["- not\n- a\n- mapping\n", "paths: [\n"])
def test_load_config_reports_non_mapping_or_invalid_yaml(
    tmp_path: Path,
    contents: str,
) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError):
        load_preprocess_config(config_path)


def test_load_config_wraps_missing_file_error(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.yaml"

    with pytest.raises(ConfigError, match="Could not read configuration"):
        load_preprocess_config(missing)


def test_load_config_reports_non_string_keys_as_config_errors(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["validation"][1] = "not a valid YAML key"

    with pytest.raises(ConfigError, match="keys.*must be strings"):
        load_preprocess_config(
            _write_config(tmp_path / "numeric-key.yaml", payload)
        )


@pytest.mark.parametrize(
    ("path_key", "unsafe_value", "message"),
    [
        ("processed_dir", "data/raw/derived", "must not overlap"),
        ("processed_dir", "src/generated", "protected project directory 'src'"),
        ("manifest_path", "data/raw/manifest.json", "MIDI input directory"),
        ("manifest_path", "data/processed/manifest.json", "processed MIDI directory"),
        ("manifest_path", "data/splits/manifest.csv", "must use a .json extension"),
        ("manifest_path", "configs/manifest.json", "protected project directory 'configs'"),
    ],
)
def test_load_config_rejects_unsafe_output_paths(
    tmp_path: Path,
    path_key: str,
    unsafe_value: str,
    message: str,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='fixture'\n", encoding="utf-8"
    )
    payload = _valid_payload()
    payload["paths"][path_key] = unsafe_value

    with pytest.raises(ConfigError, match=message):
        load_preprocess_config(
            _write_config(project_root / "configs/preprocess.yaml", payload)
        )
