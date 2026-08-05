"""Validation and path-safety tests for unconditional generation config."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from midi_idea_generator.config import ConfigError
from midi_idea_generator.generation_config import load_generation_config


def _valid_payload() -> dict[str, Any]:
    return {
        "seed": 42,
        "device": "auto",
        "paths": {
            "checkpoint": "checkpoints/run/best.pt",
            "tokenization_manifest": "data/tokenized/manifest.json",
            "output_dir": "outputs/generated",
        },
        "generation": {
            "min_tokens": 32,
            "max_tokens": 256,
            "temperature": 0.9,
            "top_k": 20,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
            "max_simultaneous_notes": 3,
            "num_samples": 4,
            "max_attempts_per_sample": 25,
        },
        "midi": {"program": 29},
        "visualization": {"enabled": True, "dpi": 160},
    }


def _project(tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='generation-fixture'\n", encoding="utf-8"
    )
    checkpoint = project_root / "checkpoints/run/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic checkpoint")
    manifest = project_root / "data/tokenized/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    return project_root


def _write_config(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _set_nested(payload: dict[str, Any], dotted_key: str, value: object) -> None:
    if "." not in dotted_key:
        payload[dotted_key] = value
        return
    section_name, key = dotted_key.split(".", maxsplit=1)
    section = payload[section_name]
    assert isinstance(section, dict)
    section[key] = value


def test_load_generation_config_resolves_paths_and_preserves_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = _project(tmp_path)
    config_path = _write_config(
        project_root / "nested/configs/generate.yaml", _valid_payload()
    )
    unrelated_cwd = tmp_path / "somewhere-else"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    config = load_generation_config(config_path)

    assert config.project_root == project_root.resolve()
    assert config.seed == 42
    assert config.device == "auto"
    assert config.paths.checkpoint_path == (
        project_root / "checkpoints/run/best.pt"
    ).resolve()
    assert config.paths.tokenization_manifest_path == (
        project_root / "data/tokenized/manifest.json"
    ).resolve()
    assert config.paths.output_dir == (project_root / "outputs/generated").resolve()
    assert config.generation.min_tokens == 32
    assert config.generation.max_tokens == 256
    assert config.generation.temperature == pytest.approx(0.9)
    assert config.generation.top_k == 20
    assert config.generation.top_p == pytest.approx(0.95)
    assert config.generation.repetition_penalty == pytest.approx(1.05)
    assert config.generation.max_simultaneous_notes == 3
    assert config.generation.num_samples == 4
    assert config.generation.max_attempts_per_sample == 25
    assert config.midi.program == 29
    assert config.visualization.enabled is True
    assert config.visualization.dpi == 160
    assert not config.paths.output_dir.exists()


def test_sections_use_safe_documented_defaults(tmp_path: Path) -> None:
    project_root = _project(tmp_path)
    payload = _valid_payload()
    del payload["seed"]
    del payload["device"]
    payload["generation"] = {}
    payload["midi"] = {}
    payload["visualization"] = {}

    config = load_generation_config(
        _write_config(project_root / "configs/generate.yaml", payload)
    )

    assert config.seed == 42
    assert config.device == "auto"
    assert config.generation.min_tokens == 32
    assert config.generation.max_tokens == 256
    assert config.generation.temperature == pytest.approx(0.9)
    assert config.generation.top_k == 20
    assert config.generation.top_p == pytest.approx(0.95)
    assert config.generation.repetition_penalty == pytest.approx(1.05)
    assert config.generation.max_simultaneous_notes == 3
    assert config.generation.num_samples == 4
    assert config.generation.max_attempts_per_sample == 25
    assert config.midi.program == 29
    assert config.visualization.enabled is True
    assert config.visualization.dpi == 160


@pytest.mark.parametrize(
    ("dotted_key", "value", "message"),
    [
        ("seed", True, "seed.*must be an integer"),
        ("seed", -1, "between 0 and 4294967295"),
        ("device", "mps", "device.*must be one of"),
        ("generation.min_tokens", 0, "min_tokens.*must be positive"),
        ("generation.min_tokens", True, "min_tokens.*must be an integer"),
        (
            "generation.max_tokens",
            31,
            "max_tokens.*at least five greater",
        ),
        ("generation.max_tokens", 32.5, "max_tokens.*must be an integer"),
        ("generation.temperature", 0, "temperature.*must be positive"),
        ("generation.temperature", float("inf"), "temperature.*must be finite"),
        ("generation.top_k", -1, "top_k.*cannot be negative"),
        ("generation.top_k", False, "top_k.*must be an integer"),
        ("generation.top_p", 0, "0 < top_p <= 1"),
        ("generation.top_p", 1.01, "0 < top_p <= 1"),
        ("generation.repetition_penalty", 0.99, "greater than or equal to 1"),
        ("generation.max_simultaneous_notes", 0, "between 1 and 6"),
        ("generation.max_simultaneous_notes", 7, "between 1 and 6"),
        ("generation.num_samples", 0, "num_samples.*must be positive"),
        (
            "generation.max_attempts_per_sample",
            0,
            "max_attempts_per_sample.*must be positive",
        ),
        ("midi.program", -1, "program.*between 0 and 127"),
        ("midi.program", 128, "program.*between 0 and 127"),
        ("midi.program", True, "program.*must be an integer"),
        ("visualization.enabled", 1, "enabled.*must be true or false"),
        ("visualization.dpi", 0, "dpi.*must be positive"),
        ("visualization.dpi", True, "dpi.*must be an integer"),
    ],
)
def test_load_generation_config_rejects_invalid_values(
    tmp_path: Path, dotted_key: str, value: object, message: str
) -> None:
    project_root = _project(tmp_path)
    payload = deepcopy(_valid_payload())
    _set_nested(payload, dotted_key, value)

    with pytest.raises(ConfigError, match=message):
        load_generation_config(
            _write_config(project_root / "configs/generate.yaml", payload)
        )


def test_sampling_disable_and_boundary_values_are_explicit(tmp_path: Path) -> None:
    project_root = _project(tmp_path)
    payload = _valid_payload()
    payload["generation"]["top_k"] = 0
    payload["generation"]["top_p"] = 1.0
    payload["generation"]["min_tokens"] = 59
    payload["generation"]["max_tokens"] = 64
    payload["midi"]["program"] = 127

    config = load_generation_config(
        _write_config(project_root / "configs/generate.yaml", payload)
    )

    assert config.generation.top_k == 0
    assert config.generation.top_p == 1.0
    assert config.generation.min_tokens == 59
    assert config.generation.max_tokens == 64
    assert config.midi.program == 127


@pytest.mark.parametrize(
    "missing_section", ["paths", "generation", "midi", "visualization"]
)
def test_load_generation_config_requires_all_sections(
    tmp_path: Path, missing_section: str
) -> None:
    project_root = _project(tmp_path)
    payload = _valid_payload()
    del payload[missing_section]

    with pytest.raises(ConfigError, match=f"Missing required.*'{missing_section}'"):
        load_generation_config(
            _write_config(project_root / "configs/generate.yaml", payload)
        )


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("root", "implemented"),
        ("paths", "tokenizer"),
        ("generation", "beam_width"),
        ("midi", "tempo"),
        ("visualization", "format"),
    ],
)
def test_unknown_settings_are_rejected(
    tmp_path: Path, section: str, key: str
) -> None:
    project_root = _project(tmp_path)
    payload = _valid_payload()
    if section == "root":
        payload[key] = True
    else:
        payload[section][key] = True

    with pytest.raises(ConfigError, match=f"Unknown key.*{key}"):
        load_generation_config(
            _write_config(project_root / "configs/generate.yaml", payload)
        )


@pytest.mark.parametrize(
    "missing", ["checkpoint", "tokenization_manifest", "output_dir"]
)
def test_all_path_settings_are_required(tmp_path: Path, missing: str) -> None:
    project_root = _project(tmp_path)
    payload = _valid_payload()
    del payload["paths"][missing]

    with pytest.raises(ConfigError, match=f"Missing path setting.*{missing}"):
        load_generation_config(
            _write_config(project_root / "configs/generate.yaml", payload)
        )


@pytest.mark.parametrize(
    ("path_key", "value", "message"),
    [
        ("checkpoint", "checkpoints/run/best.pth", r"\.pt extension"),
        (
            "tokenization_manifest",
            "data/tokenized/manifest.yaml",
            r"\.json extension",
        ),
        ("checkpoint", "checkpoints/missing.pt", "existing regular file"),
        (
            "tokenization_manifest",
            "data/tokenized/missing.json",
            "existing regular file",
        ),
    ],
)
def test_input_artifacts_require_exact_extensions_and_existing_files(
    tmp_path: Path, path_key: str, value: str, message: str
) -> None:
    project_root = _project(tmp_path)
    payload = _valid_payload()
    payload["paths"][path_key] = value

    with pytest.raises(ConfigError, match=message):
        load_generation_config(
            _write_config(project_root / "configs/generate.yaml", payload)
        )


def test_input_artifact_cannot_be_a_directory(tmp_path: Path) -> None:
    project_root = _project(tmp_path)
    fake_checkpoint = project_root / "artifacts/directory.pt"
    fake_checkpoint.mkdir(parents=True)
    payload = _valid_payload()
    payload["paths"]["checkpoint"] = "artifacts/directory.pt"

    with pytest.raises(ConfigError, match="existing regular file"):
        load_generation_config(
            _write_config(project_root / "configs/generate.yaml", payload)
        )


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (".", "inside the project.*project root"),
        ("data/generated", "protected project path 'data'"),
        ("checkpoints/generated", "protected project path 'checkpoints'"),
        ("src/generated", "protected project path 'src'"),
        ("tests/generated", "protected project path 'tests'"),
    ],
)
def test_output_rejects_project_root_and_protected_paths(
    tmp_path: Path, output: str, message: str
) -> None:
    project_root = _project(tmp_path)
    payload = _valid_payload()
    payload["paths"]["output_dir"] = output

    with pytest.raises(ConfigError, match=message):
        load_generation_config(
            _write_config(project_root / "configs/generate.yaml", payload)
        )


def test_output_must_remain_inside_project(tmp_path: Path) -> None:
    project_root = _project(tmp_path)
    payload = _valid_payload()
    payload["paths"]["output_dir"] = str(tmp_path / "external-output")

    with pytest.raises(ConfigError, match="must be inside the project"):
        load_generation_config(
            _write_config(project_root / "configs/generate.yaml", payload)
        )


def test_output_cannot_be_an_existing_file(tmp_path: Path) -> None:
    project_root = _project(tmp_path)
    output_file = project_root / "outputs/generated"
    output_file.parent.mkdir()
    output_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ConfigError, match="existing file"):
        load_generation_config(
            _write_config(
                project_root / "configs/generate.yaml", _valid_payload()
            )
        )


@pytest.mark.parametrize("input_key", ["checkpoint", "tokenization_manifest"])
def test_output_cannot_contain_an_input_artifact(
    tmp_path: Path, input_key: str
) -> None:
    project_root = _project(tmp_path)
    artifacts = project_root / "artifacts"
    artifacts.mkdir()
    checkpoint = artifacts / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = artifacts / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    payload = _valid_payload()
    if input_key == "checkpoint":
        payload["paths"]["checkpoint"] = "artifacts/model.pt"
    else:
        payload["paths"]["tokenization_manifest"] = "artifacts/manifest.json"
    payload["paths"]["output_dir"] = "artifacts"

    with pytest.raises(ConfigError, match=f"paths.{input_key}.*outside.*output_dir"):
        load_generation_config(
            _write_config(project_root / "configs/generate.yaml", payload)
        )


def test_output_cannot_contain_the_loaded_configuration(tmp_path: Path) -> None:
    project_root = _project(tmp_path)
    payload = _valid_payload()
    payload["paths"]["output_dir"] = "outputs/generated"
    config_path = project_root / "outputs/generated/generate.yaml"

    with pytest.raises(ConfigError, match="must not contain its generation"):
        load_generation_config(_write_config(config_path, payload))


@pytest.mark.parametrize("contents", ["- not\n- a\n- mapping\n", "paths: [\n"])
def test_loader_reports_non_mapping_or_invalid_yaml(
    tmp_path: Path, contents: str
) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError):
        load_generation_config(config_path)


def test_loader_wraps_missing_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Could not read configuration"):
        load_generation_config(tmp_path / "missing.yaml")
