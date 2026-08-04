"""Validation and path-safety tests for Stage 2 configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from midi_idea_generator.config import ConfigError
from midi_idea_generator.tokenization_config import (
    BeatResolutionConfig,
    load_tokenization_config,
)


def _valid_payload() -> dict[str, Any]:
    return {
        "paths": {
            "preprocessing_manifest_path": "data/splits/manifest.json",
            "tokenized_dir": "data/tokenized",
            "manifest_path": "data/tokenized/manifest.json",
        },
        "tokenizer": {
            "type": "remi",
            "pitch_min": 21,
            "pitch_max": 108,
            "beat_res": [
                {"start_beat": 0, "end_beat": 4, "resolution": 24},
                {"start_beat": 4, "end_beat": 16, "resolution": 4},
            ],
            "special_tokens": ["PAD", "BOS", "EOS"],
            "encode_ids_split": "bar",
            "use_velocities": False,
            "num_velocities": 16,
            "use_tempos": True,
            "num_tempos": 32,
            "tempo_min": 40,
            "tempo_max": 250,
            "add_trailing_bars": True,
            "max_bar_embedding": None,
        },
    }


def _project(tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='tokenization-fixture'\n", encoding="utf-8"
    )
    return project_root


def _write_config(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _set_nested(payload: dict[str, Any], dotted_key: str, value: object) -> None:
    section_name, key = dotted_key.split(".", maxsplit=1)
    section = payload[section_name]
    assert isinstance(section, dict)
    section[key] = value


def test_load_tokenization_config_resolves_paths_and_preserves_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = _project(tmp_path)
    config_path = _write_config(
        project_root / "nested/configs/tokenize.yaml", _valid_payload()
    )
    unrelated_cwd = tmp_path / "somewhere-else"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    config = load_tokenization_config(config_path)

    assert config.project_root == project_root.resolve()
    assert config.paths.preprocessing_manifest_path == (
        project_root / "data/splits/manifest.json"
    ).resolve()
    assert config.paths.tokenized_dir == (project_root / "data/tokenized").resolve()
    assert config.paths.manifest_path == (
        project_root / "data/tokenized/manifest.json"
    ).resolve()
    assert config.tokenizer.pitch_min == 21
    assert config.tokenizer.pitch_max == 108
    assert config.tokenizer.beat_res == (
        BeatResolutionConfig(0, 4, 24),
        BeatResolutionConfig(4, 16, 4),
    )
    assert config.tokenizer.special_tokens == ("PAD", "BOS", "EOS")
    assert config.tokenizer.use_velocities is False
    assert config.tokenizer.use_tempos is True
    assert config.tokenizer.add_trailing_bars is True
    assert config.tokenizer.max_bar_embedding is None


def test_tokenizer_section_uses_documented_defaults(tmp_path: Path) -> None:
    project_root = _project(tmp_path)
    payload = _valid_payload()
    payload["tokenizer"] = {}

    config = load_tokenization_config(
        _write_config(project_root / "configs/tokenize.yaml", payload)
    )

    assert config.tokenizer.pitch_min == 21
    assert config.tokenizer.pitch_max == 108
    assert config.tokenizer.beat_res[0] == BeatResolutionConfig(0, 4, 24)
    assert config.tokenizer.encode_ids_split == "bar"
    assert config.tokenizer.tempo_min == pytest.approx(40.0)
    assert config.tokenizer.tempo_max == pytest.approx(250.0)


@pytest.mark.parametrize(
    ("dotted_key", "value", "message"),
    [
        ("tokenizer.type", "midi_like", "must be 'remi'"),
        ("tokenizer.pitch_min", True, "must be an integer"),
        ("tokenizer.pitch_max", 128, "pitch range must satisfy"),
        ("tokenizer.beat_res", [], "must be a non-empty list"),
        (
            "tokenizer.beat_res",
            [{"start_beat": 1, "end_beat": 4, "resolution": 12}],
            "must start at beat 0",
        ),
        (
            "tokenizer.beat_res",
            [
                {"start_beat": 0, "end_beat": 4, "resolution": 12},
                {"start_beat": 5, "end_beat": 16, "resolution": 4},
            ],
            "ordered, contiguous, and non-overlapping",
        ),
        (
            "tokenizer.beat_res",
            [{"start_beat": 0, "end_beat": 4, "resolution": 0}],
            "resolution.*between 1 and 64",
        ),
        (
            "tokenizer.special_tokens",
            ["BOS", "EOS", "PAD"],
            "order.*PAD, BOS, EOS",
        ),
        ("tokenizer.encode_ids_split", "phrase", "bar.*beat.*no"),
        ("tokenizer.use_velocities", True, "use_velocities: false"),
        ("tokenizer.num_velocities", 0, "between 1 and 128"),
        ("tokenizer.use_tempos", False, "use_tempos: true"),
        ("tokenizer.num_tempos", 0, "between 1 and 512"),
        ("tokenizer.tempo_min", 250, "0 < tempo_min < tempo_max"),
        ("tokenizer.tempo_max", float("inf"), "must be finite"),
        ("tokenizer.add_trailing_bars", False, "add_trailing_bars: true"),
        ("tokenizer.max_bar_embedding", 0, "null or a positive integer"),
    ],
)
def test_load_tokenization_config_rejects_invalid_tokenizer_values(
    tmp_path: Path,
    dotted_key: str,
    value: object,
    message: str,
) -> None:
    project_root = _project(tmp_path)
    payload = deepcopy(_valid_payload())
    _set_nested(payload, dotted_key, value)

    with pytest.raises(ConfigError, match=message):
        load_tokenization_config(
            _write_config(project_root / "configs/tokenize.yaml", payload)
        )


@pytest.mark.parametrize("missing_section", ["paths", "tokenizer"])
def test_load_tokenization_config_requires_sections(
    tmp_path: Path, missing_section: str
) -> None:
    project_root = _project(tmp_path)
    payload = _valid_payload()
    del payload[missing_section]

    with pytest.raises(ConfigError, match=f"Missing required.*'{missing_section}'"):
        load_tokenization_config(
            _write_config(project_root / "configs/tokenize.yaml", payload)
        )


def test_load_tokenization_config_rejects_missing_and_unknown_settings(
    tmp_path: Path,
) -> None:
    project_root = _project(tmp_path)
    missing_path = _valid_payload()
    del missing_path["paths"]["manifest_path"]
    with pytest.raises(ConfigError, match="Missing path setting.*manifest_path"):
        load_tokenization_config(
            _write_config(project_root / "configs/missing.yaml", missing_path)
        )

    unknown_tokenizer_key = _valid_payload()
    unknown_tokenizer_key["tokenizer"]["use_pitch_bends"] = False
    with pytest.raises(ConfigError, match="Unknown key.*use_pitch_bends"):
        load_tokenization_config(
            _write_config(
                project_root / "configs/unknown.yaml", unknown_tokenizer_key
            )
        )

    unknown_beat_key = _valid_payload()
    unknown_beat_key["tokenizer"]["beat_res"][0]["extra"] = 1
    with pytest.raises(ConfigError, match="Unknown key.*extra"):
        load_tokenization_config(
            _write_config(project_root / "configs/beat.yaml", unknown_beat_key)
        )


@pytest.mark.parametrize(
    ("path_key", "unsafe_value", "message"),
    [
        ("tokenized_dir", ".", "cannot contain the project root"),
        ("tokenized_dir", "src/tokens", "protected project path 'src'"),
        ("tokenized_dir", "data/raw/tokens", "protected project path 'data/raw'"),
        (
            "tokenized_dir",
            "data/processed/tokens",
            "protected project path 'data/processed'",
        ),
        ("tokenized_dir", "data", "protected project path 'data/raw'"),
        ("tokenized_dir", "outputs/tokens", "protected project path 'outputs'"),
        (
            "preprocessing_manifest_path",
            "data/tokenized/input.json",
            "must be outside.*tokenized_dir",
        ),
        ("preprocessing_manifest_path", "data/splits/manifest.yaml", r"\.json"),
        ("manifest_path", "data/splits/tokens.json", "must be inside"),
        ("manifest_path", "data/tokenized/runs/latest.json", "outside.*runs"),
        ("manifest_path", "data/tokenized/manifest.yaml", r"\.json"),
    ],
)
def test_load_tokenization_config_rejects_unsafe_paths(
    tmp_path: Path, path_key: str, unsafe_value: str, message: str
) -> None:
    project_root = _project(tmp_path)
    payload = deepcopy(_valid_payload())
    payload["paths"][path_key] = unsafe_value

    with pytest.raises(ConfigError, match=message):
        load_tokenization_config(
            _write_config(project_root / "configs/tokenize.yaml", payload)
        )


def test_load_tokenization_config_allows_external_output_directory(
    tmp_path: Path,
) -> None:
    project_root = _project(tmp_path)
    external_dir = tmp_path / "tokenized-output"
    payload = _valid_payload()
    payload["paths"]["tokenized_dir"] = str(external_dir)
    payload["paths"]["manifest_path"] = str(external_dir / "manifest.json")

    config = load_tokenization_config(
        _write_config(project_root / "configs/tokenize.yaml", payload)
    )

    assert config.paths.tokenized_dir == external_dir.resolve()
    assert config.paths.manifest_path == (external_dir / "manifest.json").resolve()


@pytest.mark.parametrize("contents", ["- not\n- a\n- mapping\n", "paths: [\n"])
def test_load_tokenization_config_reports_non_mapping_or_invalid_yaml(
    tmp_path: Path, contents: str
) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError):
        load_tokenization_config(config_path)


def test_load_tokenization_config_wraps_missing_file_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Could not read configuration"):
        load_tokenization_config(tmp_path / "missing.yaml")


def test_load_tokenization_config_rejects_non_string_keys(tmp_path: Path) -> None:
    project_root = _project(tmp_path)
    payload = _valid_payload()
    payload["tokenizer"][1] = "invalid YAML key"

    with pytest.raises(ConfigError, match="keys.*must be strings"):
        load_tokenization_config(
            _write_config(project_root / "configs/tokenize.yaml", payload)
        )
