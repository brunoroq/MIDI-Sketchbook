"""Validation and path-safety tests for the Stage 3 training contract."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from midi_idea_generator.config import ConfigError
from midi_idea_generator.training_config import load_training_config


def _valid_payload() -> dict[str, Any]:
    return {
        "seed": 42,
        "device": "auto",
        "paths": {
            "tokenization_manifest": "data/tokenized/manifest.json",
            "checkpoints_dir": "checkpoints",
            "tensorboard_log_dir": "outputs/logs/training",
            "resume_from": None,
        },
        "model": {
            "architecture": "gru",
            "embedding_dim": 64,
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.2,
        },
        "data": {
            "max_sequence_length": 384,
            "batch_size": 4,
            "num_workers": 0,
        },
        "training": {
            "epochs": 50,
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "gradient_clip": 1.0,
            "mixed_precision": "auto",
            "checkpoint_every_epochs": 1,
            "early_stopping_patience": 8,
            "early_stopping_min_delta": 0.0001,
        },
    }


def _project(tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='training-fixture'\n", encoding="utf-8"
    )
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


def test_load_training_config_resolves_paths_and_preserves_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = _project(tmp_path)
    config_path = _write_config(
        project_root / "nested/configs/train.yaml", _valid_payload()
    )
    unrelated_cwd = tmp_path / "somewhere-else"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    config = load_training_config(config_path)

    assert config.project_root == project_root.resolve()
    assert config.seed == 42
    assert config.device == "auto"
    assert config.paths.tokenization_manifest_path == (
        project_root / "data/tokenized/manifest.json"
    ).resolve()
    assert config.paths.checkpoints_dir == (project_root / "checkpoints").resolve()
    assert config.paths.tensorboard_log_dir == (
        project_root / "outputs/logs/training"
    ).resolve()
    assert config.paths.resume_from is None
    assert config.model.architecture == "gru"
    assert config.model.embedding_dim == 64
    assert config.model.hidden_dim == 128
    assert config.model.num_layers == 2
    assert config.model.dropout == pytest.approx(0.2)
    assert config.data.max_sequence_length == 384
    assert config.data.batch_size == 4
    assert config.data.num_workers == 0
    assert config.training.epochs == 50
    assert config.training.learning_rate == pytest.approx(3e-4)
    assert config.training.weight_decay == pytest.approx(1e-4)
    assert config.training.gradient_clip == pytest.approx(1.0)
    assert config.training.mixed_precision == "auto"
    assert config.training.checkpoint_every_epochs == 1
    assert config.training.early_stopping_patience == 8
    assert config.training.early_stopping_min_delta == pytest.approx(1e-4)


def test_sections_use_documented_defaults(tmp_path: Path) -> None:
    project_root = _project(tmp_path)
    payload = _valid_payload()
    payload["model"] = {}
    payload["data"] = {}
    payload["training"] = {}
    del payload["seed"]
    del payload["device"]

    config = load_training_config(
        _write_config(project_root / "configs/train.yaml", payload)
    )

    assert config.seed == 42
    assert config.device == "auto"
    assert config.model.embedding_dim == 64
    assert config.model.hidden_dim == 128
    assert config.data.max_sequence_length == 384
    assert config.training.epochs == 50
    assert config.training.mixed_precision == "auto"


@pytest.mark.parametrize(
    ("dotted_key", "value", "message"),
    [
        ("seed", True, "seed.*must be an integer"),
        ("seed", -1, "between 0 and 4294967295"),
        ("device", "mps", "device.*must be one of"),
        ("model.architecture", "lstm", "architecture.*must be one of"),
        ("model.embedding_dim", 0, "embedding_dim.*must be positive"),
        ("model.hidden_dim", 0, "hidden_dim.*must be positive"),
        ("model.num_layers", 1, "num_layers.*must be at least 2"),
        ("model.dropout", 1.0, "0 <= dropout < 1"),
        ("model.dropout", float("inf"), "must be finite"),
        ("data.max_sequence_length", 2, "must be at least 3"),
        ("data.batch_size", 0, "batch_size.*must be positive"),
        ("data.num_workers", -1, "num_workers.*cannot be negative"),
        ("training.epochs", 0, "epochs.*must be positive"),
        ("training.learning_rate", 0, "learning_rate.*must be positive"),
        ("training.weight_decay", -0.1, "weight_decay.*cannot be negative"),
        ("training.gradient_clip", 0, "gradient_clip.*must be positive"),
        ("training.mixed_precision", True, "mixed_precision.*must be one of"),
        (
            "training.checkpoint_every_epochs",
            0,
            "checkpoint_every_epochs.*must be positive",
        ),
        (
            "training.early_stopping_patience",
            0,
            "early_stopping_patience.*must be positive",
        ),
        (
            "training.early_stopping_min_delta",
            -0.1,
            "early_stopping_min_delta.*cannot be negative",
        ),
    ],
)
def test_load_training_config_rejects_invalid_values(
    tmp_path: Path, dotted_key: str, value: object, message: str
) -> None:
    project_root = _project(tmp_path)
    payload = deepcopy(_valid_payload())
    _set_nested(payload, dotted_key, value)

    with pytest.raises(ConfigError, match=message):
        load_training_config(
            _write_config(project_root / "configs/train.yaml", payload)
        )


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda"])
def test_device_modes_are_explicit_and_torch_independent(
    tmp_path: Path, device: str
) -> None:
    project_root = _project(tmp_path)
    payload = _valid_payload()
    payload["device"] = device

    config = load_training_config(
        _write_config(project_root / "configs/train.yaml", payload)
    )

    assert config.device == device


@pytest.mark.parametrize("mixed_precision", ["auto", "on", "off"])
def test_mixed_precision_modes_are_explicit(
    tmp_path: Path, mixed_precision: str
) -> None:
    project_root = _project(tmp_path)
    payload = _valid_payload()
    payload["training"]["mixed_precision"] = mixed_precision

    config = load_training_config(
        _write_config(project_root / "configs/train.yaml", payload)
    )

    assert config.training.mixed_precision == mixed_precision


@pytest.mark.parametrize("missing_section", ["paths", "model", "data", "training"])
def test_load_training_config_requires_all_sections(
    tmp_path: Path, missing_section: str
) -> None:
    project_root = _project(tmp_path)
    payload = _valid_payload()
    del payload[missing_section]

    with pytest.raises(ConfigError, match=f"Missing required.*'{missing_section}'"):
        load_training_config(
            _write_config(project_root / "configs/train.yaml", payload)
        )


def test_load_training_config_rejects_missing_and_unknown_settings(
    tmp_path: Path,
) -> None:
    project_root = _project(tmp_path)
    missing_path = _valid_payload()
    del missing_path["paths"]["checkpoints_dir"]
    with pytest.raises(ConfigError, match="Missing path setting.*checkpoints_dir"):
        load_training_config(
            _write_config(project_root / "configs/missing.yaml", missing_path)
        )

    unknown_root = _valid_payload()
    unknown_root["implemented"] = True
    with pytest.raises(ConfigError, match="Unknown key.*implemented"):
        load_training_config(
            _write_config(project_root / "configs/root.yaml", unknown_root)
        )

    unknown_model = _valid_payload()
    unknown_model["model"]["bidirectional"] = True
    with pytest.raises(ConfigError, match="Unknown key.*bidirectional"):
        load_training_config(
            _write_config(project_root / "configs/model.yaml", unknown_model)
        )


@pytest.mark.parametrize(
    ("path_key", "unsafe_value", "message"),
    [
        ("tokenization_manifest", "data/tokenized/manifest.yaml", r"\.json"),
        ("checkpoints_dir", ".", "cannot contain the project root"),
        (
            "checkpoints_dir",
            "src/checkpoints",
            "protected project path 'src'",
        ),
        ("checkpoints_dir", "data/models", "protected project path 'data'"),
        (
            "tensorboard_log_dir",
            "tests/events",
            "protected project path 'tests'",
        ),
        (
            "tensorboard_log_dir",
            "checkpoints/events",
            "must not overlap",
        ),
        (
            "tokenization_manifest",
            "checkpoints/manifest.json",
            "outside training output directories",
        ),
        ("resume_from", "other/latest.pt", "must be inside.*checkpoints_dir"),
        (
            "resume_from",
            "checkpoints/latest.json",
            r"\.pt, \.pth, or \.ckpt",
        ),
        (
            "resume_from",
            "checkpoints/best.pt",
            r"named 'latest\.pt'",
        ),
    ],
)
def test_load_training_config_rejects_unsafe_paths(
    tmp_path: Path, path_key: str, unsafe_value: str, message: str
) -> None:
    project_root = _project(tmp_path)
    payload = deepcopy(_valid_payload())
    payload["paths"][path_key] = unsafe_value

    with pytest.raises(ConfigError, match=message):
        load_training_config(
            _write_config(project_root / "configs/train.yaml", payload)
        )


def test_resume_path_is_resolved_inside_checkpoint_directory(tmp_path: Path) -> None:
    project_root = _project(tmp_path)
    payload = _valid_payload()
    payload["paths"]["resume_from"] = "checkpoints/latest.pt"

    config = load_training_config(
        _write_config(project_root / "configs/train.yaml", payload)
    )

    assert config.paths.resume_from == (
        project_root / "checkpoints/latest.pt"
    ).resolve()


def test_external_output_directories_are_allowed_when_separate(
    tmp_path: Path,
) -> None:
    project_root = _project(tmp_path)
    checkpoints = tmp_path / "training-output" / "checkpoints"
    tensorboard = tmp_path / "training-output" / "tensorboard"
    payload = _valid_payload()
    payload["paths"]["checkpoints_dir"] = str(checkpoints)
    payload["paths"]["tensorboard_log_dir"] = str(tensorboard)
    payload["paths"]["resume_from"] = str(checkpoints / "latest.pt")

    config = load_training_config(
        _write_config(project_root / "configs/train.yaml", payload)
    )

    assert config.paths.checkpoints_dir == checkpoints.resolve()
    assert config.paths.tensorboard_log_dir == tensorboard.resolve()
    assert config.paths.resume_from == (checkpoints / "latest.pt").resolve()


@pytest.mark.parametrize("contents", ["- not\n- a\n- mapping\n", "paths: [\n"])
def test_load_training_config_reports_non_mapping_or_invalid_yaml(
    tmp_path: Path, contents: str
) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError):
        load_training_config(config_path)


def test_load_training_config_wraps_missing_file_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Could not read configuration"):
        load_training_config(tmp_path / "missing.yaml")


def test_load_training_config_rejects_non_string_keys(tmp_path: Path) -> None:
    project_root = _project(tmp_path)
    payload = _valid_payload()
    payload["training"][1] = "invalid YAML key"

    with pytest.raises(ConfigError, match="keys.*must be strings"):
        load_training_config(
            _write_config(project_root / "configs/train.yaml", payload)
        )


def test_repository_training_config_matches_small_corpus_baseline() -> None:
    project_root = Path(__file__).resolve().parents[1]

    config = load_training_config(project_root / "configs/train.yaml")

    assert config.model.embedding_dim == 64
    assert config.model.hidden_dim == 128
    assert config.model.num_layers == 2
    assert config.data.max_sequence_length == 384
    assert config.data.batch_size == 4
    assert config.training.early_stopping_patience == 8
