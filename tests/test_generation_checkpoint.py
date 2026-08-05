"""Tests for safe, provenance-bound generation checkpoint loading."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from midi_idea_generator import generation_checkpoint
from midi_idea_generator.generation_checkpoint import (
    GenerationCheckpointError,
    load_generation_bundle,
)
from midi_idea_generator.model import GRUModel


VOCABULARY_SIZE = 12
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
TOKENIZATION_RUN_ID = "a" * 20
CONFIGURATION_SHA256 = "b" * 64
IMPLEMENTATION_SHA256 = "c" * 64
RUN_ID = "synthetic-generation-run"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model() -> GRUModel:
    torch.manual_seed(37)
    return GRUModel(
        VOCABULARY_SIZE,
        PAD_TOKEN_ID,
        embedding_dim=4,
        hidden_dim=6,
        num_layers=2,
        dropout=0.15,
    )


def _history_entry() -> dict[str, Any]:
    return {
        "epoch": 1,
        "train_loss": 2.5,
        "validation_loss": 2.25,
        "train_perplexity": 12.0,
        "validation_perplexity": 9.5,
        "train_tokens": 80,
        "validation_tokens": 20,
        "mean_gradient_norm": 0.75,
        "duration_seconds": 0.1,
    }


def _payload(model: GRUModel, manifest: Path, tokenizer: Path) -> dict[str, Any]:
    model_config = {
        "architecture": "gru",
        "embedding_dim": 4,
        "hidden_dim": 6,
        "num_layers": 2,
        "dropout": 0.15,
    }
    compatibility = {
        "tokenization_run_id": TOKENIZATION_RUN_ID,
        "tokenization_manifest_sha256": _sha256(manifest),
        "tokenizer_sha256": _sha256(tokenizer),
        "tokenization_configuration_sha256": CONFIGURATION_SHA256,
        "vocabulary_size": VOCABULARY_SIZE,
        "pad_token_id": PAD_TOKEN_ID,
        "model": model_config,
        "data": {"max_sequence_length": 32},
        "optimizer": {"learning_rate": 0.001},
        "seed": 42,
        "num_parameters": model.num_parameters,
        "execution": {"resolved_device_type": "cpu", "amp_enabled": False},
        "training_implementation_sha256": IMPLEMENTATION_SHA256,
        "torch_version": str(torch.__version__),
    }
    return {
        "schema_version": 1,
        "training_run_id": RUN_ID,
        "epoch": 1,
        "model_state_dict": deepcopy(model.state_dict()),
        "optimizer_state_dict": {},
        "scaler_state_dict": {},
        "loader_generator_state": torch.Generator().get_state(),
        "rng_state": {},
        "configuration": {"model": deepcopy(model_config)},
        "requested_total_epochs": 1,
        "compatibility": compatibility,
        "best_validation_loss": 2.25,
        "best_epoch": 1,
        "epochs_without_improvement": 0,
        "history": [_history_entry()],
        "torch_version": str(torch.__version__),
    }


def _install_fake_corpus(
    monkeypatch: pytest.MonkeyPatch,
    manifest: Path,
    tokenizer: Path,
) -> None:
    class FakeDataset:
        def __init__(self, manifest_path: Path, split: str, **kwargs: object) -> None:
            assert Path(manifest_path).resolve() == manifest.resolve()
            assert split == "train"
            assert kwargs["verify_hashes"] is True
            self.manifest_path = manifest.resolve()
            self.tokenization_manifest_sha256 = _sha256(manifest)
            self.tokenization_run_id = TOKENIZATION_RUN_ID
            self.configuration_sha256 = CONFIGURATION_SHA256
            self.tokenizer_path = tokenizer.resolve()
            self.tokenizer_sha256 = _sha256(tokenizer)
            self.vocabulary_size = VOCABULARY_SIZE
            self.pad_token_id = PAD_TOKEN_ID
            self.bos_token_id = BOS_TOKEN_ID
            self.eos_token_id = EOS_TOKEN_ID

    fake_tokenizer = SimpleNamespace(
        vocab={f"Token_{index}": index for index in range(VOCABULARY_SIZE)}
    )
    monkeypatch.setattr(
        generation_checkpoint, "TokenizedSequenceDataset", FakeDataset
    )
    monkeypatch.setattr(
        generation_checkpoint, "load_tokenizer", lambda _path: fake_tokenizer
    )
    monkeypatch.setattr(
        generation_checkpoint,
        "get_special_token_ids",
        lambda _tokenizer: SimpleNamespace(
            pad=PAD_TOKEN_ID,
            bos=BOS_TOKEN_ID,
            eos=EOS_TOKEN_ID,
        ),
    )


def _artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, GRUModel, dict[str, Any]]:
    manifest = tmp_path / "data" / "tokenized" / "manifest.json"
    tokenizer = tmp_path / "data" / "tokenized" / "runs" / TOKENIZATION_RUN_ID / "tokenizer.json"
    checkpoint = tmp_path / "checkpoints" / RUN_ID / "best.pt"
    manifest.parent.mkdir(parents=True)
    tokenizer.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    manifest.write_bytes(b'{"schema_version":2}\n')
    tokenizer.write_bytes(b'{"tokenization":"GuitarREMI"}\n')
    model = _model()
    payload = _payload(model, manifest, tokenizer)
    torch.save(payload, checkpoint)
    _install_fake_corpus(monkeypatch, manifest, tokenizer)
    return checkpoint, manifest, tokenizer, model, payload


def test_load_generation_bundle_restores_eval_model_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, manifest, tokenizer, original, _payload_value = _artifacts(
        tmp_path, monkeypatch
    )

    bundle = load_generation_bundle(
        checkpoint,
        manifest,
        tmp_path,
        "cpu",
    )

    assert bundle.device == torch.device("cpu")
    assert bundle.model.training is False
    assert bundle.model.num_parameters == original.num_parameters
    assert bundle.checkpoint_sha256 == _sha256(checkpoint)
    assert bundle.manifest_sha256 == _sha256(manifest)
    assert bundle.tokenizer_sha256 == _sha256(tokenizer)
    assert bundle.training_run_id == RUN_ID
    assert bundle.tokenization_run_id == TOKENIZATION_RUN_ID
    assert (bundle.epoch, bundle.best_epoch, bundle.best_validation_loss) == (
        1,
        1,
        2.25,
    )
    assert (
        bundle.pad_token_id,
        bundle.bos_token_id,
        bundle.eos_token_id,
    ) == (PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID)
    for name, parameter in bundle.model.state_dict().items():
        torch.testing.assert_close(parameter, original.state_dict()[name], rtol=0, atol=0)


def test_checkpoint_is_deserialized_in_restricted_cpu_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, manifest, _tokenizer, _model_value, _payload_value = _artifacts(
        tmp_path, monkeypatch
    )
    original_load = generation_checkpoint.torch.load
    observed: dict[str, object] = {}

    def recording_load(source: object, **kwargs: object) -> object:
        observed["source"] = source
        observed.update(kwargs)
        return original_load(source, **kwargs)

    monkeypatch.setattr(generation_checkpoint.torch, "load", recording_load)

    load_generation_bundle(checkpoint, manifest, tmp_path, "cpu")

    assert observed["weights_only"] is True
    assert observed["map_location"] == torch.device("cpu")
    assert not isinstance(observed["source"], (str, Path))


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("tokenization_manifest_sha256", "0" * 64, "manifest hash"),
        ("tokenizer_sha256", "0" * 64, "tokenizer hash"),
        ("tokenization_configuration_sha256", "0" * 64, "configuration hash"),
        ("tokenization_run_id", "0" * 20, "run ID"),
        ("vocabulary_size", VOCABULARY_SIZE + 1, "vocabulary size"),
        ("pad_token_id", 3, "PAD token ID"),
    ],
)
def test_checkpoint_must_match_explicit_tokenization_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
    message: str,
) -> None:
    checkpoint, manifest, _tokenizer, _model_value, payload = _artifacts(
        tmp_path, monkeypatch
    )
    payload["compatibility"][field] = replacement
    torch.save(payload, checkpoint)

    with pytest.raises(GenerationCheckpointError, match=message):
        load_generation_bundle(checkpoint, manifest, tmp_path, "cpu")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing"),
        ("shape", "shape or dtype"),
        ("nonfinite", "non-finite"),
    ],
)
def test_model_state_must_be_exact_and_finite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    checkpoint, manifest, _tokenizer, _model_value, payload = _artifacts(
        tmp_path, monkeypatch
    )
    state = payload["model_state_dict"]
    if mutation == "missing":
        del state["output_projection.bias"]
    elif mutation == "shape":
        state["output_projection.bias"] = torch.zeros(VOCABULARY_SIZE + 1)
    else:
        state["embedding.weight"][0, 0] = float("nan")
    torch.save(payload, checkpoint)

    with pytest.raises(GenerationCheckpointError, match=message):
        load_generation_bundle(checkpoint, manifest, tmp_path, "cpu")


def test_incorrect_parameter_count_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, manifest, _tokenizer, _model_value, payload = _artifacts(
        tmp_path, monkeypatch
    )
    payload["compatibility"]["num_parameters"] += 1
    torch.save(payload, checkpoint)

    with pytest.raises(GenerationCheckpointError, match="num_parameters"):
        load_generation_bundle(checkpoint, manifest, tmp_path, "cpu")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("schema", "schema"),
        ("unknown", "unknown keys"),
        ("progress", "best_epoch"),
        ("model", "configuration.model"),
    ],
)
def test_inconsistent_checkpoint_metadata_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    checkpoint, manifest, _tokenizer, _model_value, payload = _artifacts(
        tmp_path, monkeypatch
    )
    if mutation == "schema":
        payload["schema_version"] = 2
    elif mutation == "unknown":
        payload["unexpected"] = True
    elif mutation == "progress":
        payload["best_epoch"] = 2
    else:
        payload["configuration"]["model"]["hidden_dim"] = 7
    torch.save(payload, checkpoint)

    with pytest.raises(GenerationCheckpointError, match=message):
        load_generation_bundle(checkpoint, manifest, tmp_path, "cpu")


def test_corrupt_checkpoint_is_rejected_before_manifest_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, manifest, _tokenizer, _model_value, _payload_value = _artifacts(
        tmp_path, monkeypatch
    )
    checkpoint.write_bytes(b"not a checkpoint")

    with pytest.raises(GenerationCheckpointError, match="deserialize"):
        load_generation_bundle(checkpoint, manifest, tmp_path, "cpu")


def test_checkpoint_symlink_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, manifest, _tokenizer, _model_value, _payload_value = _artifacts(
        tmp_path, monkeypatch
    )
    link = tmp_path / "checkpoints" / "linked.pt"
    link.symlink_to(checkpoint)

    with pytest.raises(GenerationCheckpointError, match="symlink"):
        load_generation_bundle(link, manifest, tmp_path, "cpu")


def test_unavailable_cuda_and_unknown_devices_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, manifest, _tokenizer, _model_value, _payload_value = _artifacts(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(GenerationCheckpointError, match="CUDA was requested"):
        load_generation_bundle(checkpoint, manifest, tmp_path, "cuda")
    with pytest.raises(GenerationCheckpointError, match="device must be"):
        load_generation_bundle(checkpoint, manifest, tmp_path, "mps")
