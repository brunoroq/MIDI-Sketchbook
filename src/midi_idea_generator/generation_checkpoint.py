"""Safe, provenance-checked loading of Stage 3 models for generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import io
import math
import os
from pathlib import Path
import pickle
import re
import stat
from typing import Any

import torch
from miditok import REMI
from torch import Tensor

from .dataset import DatasetContractError, TokenizedSequenceDataset
from .model import GRUModel, ModelConfigurationError
from .tokenizer import (
    TokenizationError,
    get_special_token_ids,
    load_tokenizer,
)


CHECKPOINT_SCHEMA_VERSION = 1
_MAX_CHECKPOINT_SIZE_BYTES = 512 * 1024 * 1024
_MAX_MANIFEST_SIZE_BYTES = 64 * 1024 * 1024
_MAX_TOKENIZER_SIZE_BYTES = 16 * 1024 * 1024
_MAX_SEQUENCE_LENGTH_FOR_VALIDATION = 2**31 - 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_TOKENIZATION_RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{20}$")

_CHECKPOINT_KEYS = {
    "schema_version",
    "training_run_id",
    "epoch",
    "model_state_dict",
    "optimizer_state_dict",
    "scaler_state_dict",
    "loader_generator_state",
    "rng_state",
    "configuration",
    "requested_total_epochs",
    "compatibility",
    "best_validation_loss",
    "best_epoch",
    "epochs_without_improvement",
    "history",
    "torch_version",
}
_COMPATIBILITY_KEYS = {
    "tokenization_run_id",
    "tokenization_manifest_sha256",
    "tokenizer_sha256",
    "tokenization_configuration_sha256",
    "vocabulary_size",
    "pad_token_id",
    "model",
    "data",
    "optimizer",
    "seed",
    "num_parameters",
    "execution",
    "training_implementation_sha256",
    "torch_version",
}
_MODEL_KEYS = {
    "architecture",
    "embedding_dim",
    "hidden_dim",
    "num_layers",
    "dropout",
}
_HISTORY_KEYS = {
    "epoch",
    "train_loss",
    "validation_loss",
    "train_perplexity",
    "validation_perplexity",
    "train_tokens",
    "validation_tokens",
    "mean_gradient_norm",
    "duration_seconds",
}


class GenerationCheckpointError(RuntimeError):
    """Raised when a trained model cannot be loaded safely for inference."""


@dataclass(frozen=True, slots=True)
class GenerationBundle:
    """A validated model/tokenizer pair plus immutable generation provenance."""

    model: GRUModel
    tokenizer: REMI
    device: torch.device
    checkpoint_path: Path
    checkpoint_sha256: str
    manifest_path: Path
    manifest_sha256: str
    tokenizer_path: Path
    tokenizer_sha256: str
    training_run_id: str
    tokenization_run_id: str
    epoch: int
    best_epoch: int
    best_validation_loss: float
    vocabulary_size: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int


def load_generation_bundle(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    project_root: str | Path,
    device: str = "auto",
) -> GenerationBundle:
    """Load a Stage 3 GRU only after validating its exact Stage 2 provenance.

    The checkpoint is always deserialized on CPU with PyTorch's restricted
    ``weights_only`` loader.  Its model is moved to the requested inference
    device only after the checkpoint, manifest, tokenizer and state tensors
    have passed validation.
    """

    root = _resolve_project_root(project_root)
    checkpoint = _resolve_project_file(
        checkpoint_path, root, "checkpoint", allowed_suffixes={".pt", ".pth", ".ckpt"}
    )
    checkpoint_raw = _read_regular_file(
        checkpoint,
        "checkpoint",
        maximum_size=_MAX_CHECKPOINT_SIZE_BYTES,
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint_raw).hexdigest()
    payload = _deserialize_checkpoint(checkpoint_raw, checkpoint)
    metadata = _validate_checkpoint_payload(payload, checkpoint)

    manifest = _resolve_project_file(
        manifest_path, root, "tokenization manifest", allowed_suffixes={".json"}
    )
    manifest_raw = _read_regular_file(
        manifest,
        "tokenization manifest",
        maximum_size=_MAX_MANIFEST_SIZE_BYTES,
    )
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()

    try:
        dataset = TokenizedSequenceDataset(
            manifest,
            "train",
            project_root=root,
            max_sequence_length=_MAX_SEQUENCE_LENGTH_FOR_VALIDATION,
            verify_hashes=True,
        )
    except (DatasetContractError, OSError, ValueError) as exc:
        raise GenerationCheckpointError(
            f"Could not validate the tokenization manifest: {exc}"
        ) from exc

    compatibility = metadata["compatibility"]
    _validate_dataset_provenance(
        dataset,
        compatibility,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )

    tokenizer_path = dataset.tokenizer_path
    _ensure_inside_root(tokenizer_path, root, "tokenizer")
    tokenizer_raw = _read_regular_file(
        tokenizer_path,
        "tokenizer",
        maximum_size=_MAX_TOKENIZER_SIZE_BYTES,
    )
    tokenizer_sha256 = hashlib.sha256(tokenizer_raw).hexdigest()
    if tokenizer_sha256 != dataset.tokenizer_sha256:
        raise GenerationCheckpointError(
            "Tokenizer bytes do not match the authoritative manifest."
        )
    if tokenizer_sha256 != compatibility["tokenizer_sha256"]:
        raise GenerationCheckpointError(
            "Checkpoint tokenizer hash does not match the explicit tokenizer."
        )
    try:
        tokenizer = load_tokenizer(tokenizer_path)
        special_ids = get_special_token_ids(tokenizer)
    except (TokenizationError, OSError, TypeError, ValueError) as exc:
        raise GenerationCheckpointError(f"Could not load the tokenizer: {exc}") from exc
    if not isinstance(tokenizer.vocab, Mapping):
        raise GenerationCheckpointError("Tokenizer must contain one vocabulary.")
    if len(tokenizer.vocab) != dataset.vocabulary_size:
        raise GenerationCheckpointError(
            "Loaded tokenizer vocabulary size changed after manifest validation."
        )
    if (special_ids.pad, special_ids.bos, special_ids.eos) != (
        dataset.pad_token_id,
        dataset.bos_token_id,
        dataset.eos_token_id,
    ):
        raise GenerationCheckpointError(
            "Loaded tokenizer special-token IDs changed after manifest validation."
        )

    # Recheck both mutable JSON artifacts after the tokenizer constructor has
    # read them. This prevents a mixed bundle if either path changed mid-load.
    if (
        _read_regular_file(
            manifest,
            "tokenization manifest",
            maximum_size=_MAX_MANIFEST_SIZE_BYTES,
        )
        != manifest_raw
    ):
        raise GenerationCheckpointError(
            "Tokenization manifest changed while the generation bundle was loaded."
        )
    if (
        _read_regular_file(
            tokenizer_path,
            "tokenizer",
            maximum_size=_MAX_TOKENIZER_SIZE_BYTES,
        )
        != tokenizer_raw
    ):
        raise GenerationCheckpointError(
            "Tokenizer changed while the generation bundle was loaded."
        )

    model = _restore_model(payload, metadata)
    resolved_device = _resolve_device(device)
    try:
        model.to(resolved_device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise GenerationCheckpointError(
            f"Could not move the model to {resolved_device}: {exc}"
        ) from exc
    model.eval()

    return GenerationBundle(
        model=model,
        tokenizer=tokenizer,
        device=resolved_device,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        manifest_path=manifest,
        manifest_sha256=manifest_sha256,
        tokenizer_path=tokenizer_path,
        tokenizer_sha256=tokenizer_sha256,
        training_run_id=metadata["training_run_id"],
        tokenization_run_id=compatibility["tokenization_run_id"],
        epoch=metadata["epoch"],
        best_epoch=metadata["best_epoch"],
        best_validation_loss=metadata["best_validation_loss"],
        vocabulary_size=dataset.vocabulary_size,
        pad_token_id=dataset.pad_token_id,
        bos_token_id=dataset.bos_token_id,
        eos_token_id=dataset.eos_token_id,
    )


def _resolve_project_root(value: str | Path) -> Path:
    unresolved = Path(value).expanduser()
    if unresolved.is_symlink():
        raise GenerationCheckpointError("project_root cannot be a symlink.")
    root = unresolved.resolve()
    if not root.is_dir():
        raise GenerationCheckpointError(f"project_root is not a directory: {root}")
    return root


def _resolve_project_file(
    value: str | Path,
    root: Path,
    name: str,
    *,
    allowed_suffixes: set[str],
) -> Path:
    unresolved = Path(value).expanduser()
    candidate = unresolved if unresolved.is_absolute() else root / unresolved
    try:
        relative = candidate.absolute().relative_to(root)
    except ValueError as exc:
        raise GenerationCheckpointError(f"{name} must be inside project_root.") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise GenerationCheckpointError(f"{name} cannot use symlinks.")
    resolved = candidate.resolve()
    _ensure_inside_root(resolved, root, name)
    if resolved.suffix.lower() not in allowed_suffixes:
        rendered = ", ".join(sorted(allowed_suffixes))
        raise GenerationCheckpointError(f"{name} must use one of: {rendered}.")
    return resolved


def _ensure_inside_root(path: Path, root: Path, name: str) -> None:
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise GenerationCheckpointError(f"{name} must be inside project_root.") from exc
    relative = path.resolve().relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise GenerationCheckpointError(f"{name} cannot use symlinks.")


def _read_regular_file(path: Path, name: str, *, maximum_size: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise GenerationCheckpointError(f"{name} is not a regular file: {path}")
        if details.st_size <= 0:
            raise GenerationCheckpointError(f"{name} is empty: {path}")
        if details.st_size > maximum_size:
            raise GenerationCheckpointError(
                f"{name} exceeds the {maximum_size}-byte safety limit."
            )
        chunks: list[bytes] = []
        remaining = details.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) != details.st_size:
            raise GenerationCheckpointError(f"{name} changed while it was read.")
        return raw
    except GenerationCheckpointError:
        raise
    except OSError as exc:
        raise GenerationCheckpointError(f"Could not read {name} '{path}': {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _deserialize_checkpoint(raw: bytes, path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(
            io.BytesIO(raw),
            map_location=torch.device("cpu"),
            weights_only=True,
        )
    except (RuntimeError, ValueError, TypeError, EOFError, pickle.UnpicklingError) as exc:
        raise GenerationCheckpointError(
            f"Could not deserialize checkpoint '{path}': {exc}"
        ) from exc
    return _mapping(payload, "checkpoint")


def _validate_checkpoint_payload(
    payload: Mapping[str, Any], checkpoint_path: Path
) -> dict[str, Any]:
    _exact_keys(payload, _CHECKPOINT_KEYS, "checkpoint")
    if _integer(payload["schema_version"], "schema_version", minimum=1) != (
        CHECKPOINT_SCHEMA_VERSION
    ):
        raise GenerationCheckpointError(
            f"Checkpoint schema must be {CHECKPOINT_SCHEMA_VERSION}."
        )
    training_run_id = _string(payload["training_run_id"], "training_run_id")
    if not _RUN_ID_PATTERN.fullmatch(training_run_id) or training_run_id in {".", ".."}:
        raise GenerationCheckpointError("training_run_id is not a safe run identifier.")
    if checkpoint_path.parent.name != training_run_id:
        raise GenerationCheckpointError(
            "Checkpoint parent directory does not match training_run_id."
        )

    epoch = _integer(payload["epoch"], "epoch", minimum=1)
    requested_total_epochs = _integer(
        payload["requested_total_epochs"], "requested_total_epochs", minimum=1
    )
    if epoch > requested_total_epochs:
        raise GenerationCheckpointError("epoch exceeds requested_total_epochs.")
    best_epoch = _integer(payload["best_epoch"], "best_epoch", minimum=1)
    if best_epoch > epoch:
        raise GenerationCheckpointError("best_epoch exceeds checkpoint epoch.")
    if checkpoint_path.name == "best.pt" and epoch != best_epoch:
        raise GenerationCheckpointError(
            "best.pt must contain the state from its recorded best_epoch."
        )
    best_validation_loss = _number(
        payload["best_validation_loss"], "best_validation_loss", minimum=0.0
    )
    epochs_without_improvement = _integer(
        payload["epochs_without_improvement"],
        "epochs_without_improvement",
        minimum=0,
    )
    if epochs_without_improvement != epoch - best_epoch:
        raise GenerationCheckpointError(
            "Checkpoint early-stopping progress is inconsistent with best_epoch."
        )

    compatibility = _mapping(payload["compatibility"], "compatibility")
    _exact_keys(compatibility, _COMPATIBILITY_KEYS, "compatibility")
    tokenization_run_id = _string(
        compatibility["tokenization_run_id"], "compatibility.tokenization_run_id"
    )
    if not _TOKENIZATION_RUN_ID_PATTERN.fullmatch(tokenization_run_id):
        raise GenerationCheckpointError(
            "compatibility.tokenization_run_id must be 20 lowercase hexadecimal digits."
        )
    for key in (
        "tokenization_manifest_sha256",
        "tokenizer_sha256",
        "tokenization_configuration_sha256",
        "training_implementation_sha256",
    ):
        _sha256(compatibility[key], f"compatibility.{key}")
    vocabulary_size = _integer(
        compatibility["vocabulary_size"],
        "compatibility.vocabulary_size",
        minimum=2,
    )
    pad_token_id = _integer(
        compatibility["pad_token_id"], "compatibility.pad_token_id", minimum=0
    )
    if pad_token_id >= vocabulary_size:
        raise GenerationCheckpointError(
            "compatibility.pad_token_id is outside the vocabulary."
        )
    expected_num_parameters = _integer(
        compatibility["num_parameters"],
        "compatibility.num_parameters",
        minimum=1,
    )
    _integer(compatibility["seed"], "compatibility.seed", minimum=0)
    for key in ("data", "optimizer", "execution"):
        _mapping(compatibility[key], f"compatibility.{key}")
    compatibility_torch = _string(
        compatibility["torch_version"], "compatibility.torch_version"
    )
    if _string(payload["torch_version"], "torch_version") != compatibility_torch:
        raise GenerationCheckpointError(
            "Checkpoint torch_version metadata is internally inconsistent."
        )

    model_config = _validate_model_config(compatibility["model"])
    configuration = _mapping(payload["configuration"], "configuration")
    configured_model = _mapping(configuration.get("model"), "configuration.model")
    if dict(configured_model) != dict(model_config):
        raise GenerationCheckpointError(
            "configuration.model disagrees with compatibility.model."
        )

    history = payload["history"]
    if not isinstance(history, list) or len(history) != epoch:
        raise GenerationCheckpointError(
            "Checkpoint history must contain every completed epoch."
        )
    validation_losses: list[float] = []
    for index, value in enumerate(history, start=1):
        entry = _mapping(value, f"history[{index - 1}]")
        _exact_keys(entry, _HISTORY_KEYS, f"history[{index - 1}]")
        if _integer(entry["epoch"], f"history[{index - 1}].epoch", minimum=1) != index:
            raise GenerationCheckpointError("Checkpoint history epochs are not contiguous.")
        for key in (
            "train_loss",
            "validation_loss",
            "train_perplexity",
            "validation_perplexity",
            "mean_gradient_norm",
            "duration_seconds",
        ):
            _number(entry[key], f"history[{index - 1}].{key}", minimum=0.0)
        for key in ("train_tokens", "validation_tokens"):
            _integer(entry[key], f"history[{index - 1}].{key}", minimum=1)
        validation_losses.append(float(entry["validation_loss"]))
    if validation_losses[best_epoch - 1] != best_validation_loss:
        raise GenerationCheckpointError(
            "best_validation_loss does not match history at best_epoch."
        )

    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "rng_state",
    ):
        _mapping(payload[key], key)
    if not isinstance(payload["loader_generator_state"], Tensor):
        raise GenerationCheckpointError("loader_generator_state must be a tensor.")

    return {
        "training_run_id": training_run_id,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "compatibility": compatibility,
        "model_config": model_config,
        "expected_num_parameters": expected_num_parameters,
    }


def _validate_model_config(value: object) -> Mapping[str, Any]:
    model = _mapping(value, "compatibility.model")
    _exact_keys(model, _MODEL_KEYS, "compatibility.model")
    if model["architecture"] != "gru":
        raise GenerationCheckpointError("Only GRU checkpoints can be generated from.")
    _integer(model["embedding_dim"], "compatibility.model.embedding_dim", minimum=1)
    _integer(model["hidden_dim"], "compatibility.model.hidden_dim", minimum=1)
    _integer(model["num_layers"], "compatibility.model.num_layers", minimum=2)
    _number(model["dropout"], "compatibility.model.dropout", minimum=0.0, maximum=1.0)
    if float(model["dropout"]) >= 1.0:
        raise GenerationCheckpointError(
            "compatibility.model.dropout must be less than one."
        )
    return model


def _validate_dataset_provenance(
    dataset: TokenizedSequenceDataset,
    compatibility: Mapping[str, Any],
    *,
    manifest: Path,
    manifest_sha256: str,
) -> None:
    expected = {
        "manifest path": (dataset.manifest_path, manifest),
        "manifest hash": (
            dataset.tokenization_manifest_sha256,
            manifest_sha256,
        ),
        "checkpoint manifest hash": (
            compatibility["tokenization_manifest_sha256"],
            manifest_sha256,
        ),
        "tokenization run ID": (
            compatibility["tokenization_run_id"],
            dataset.tokenization_run_id,
        ),
        "tokenization configuration hash": (
            compatibility["tokenization_configuration_sha256"],
            dataset.configuration_sha256,
        ),
        "tokenizer hash": (
            compatibility["tokenizer_sha256"],
            dataset.tokenizer_sha256,
        ),
        "vocabulary size": (
            compatibility["vocabulary_size"],
            dataset.vocabulary_size,
        ),
        "PAD token ID": (
            compatibility["pad_token_id"],
            dataset.pad_token_id,
        ),
    }
    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            raise GenerationCheckpointError(
                f"Checkpoint {name} does not match the explicit tokenization manifest."
            )


def _restore_model(
    payload: Mapping[str, Any], metadata: Mapping[str, Any]
) -> GRUModel:
    compatibility = metadata["compatibility"]
    config = metadata["model_config"]
    try:
        model = GRUModel(
            compatibility["vocabulary_size"],
            compatibility["pad_token_id"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        )
    except ModelConfigurationError as exc:
        raise GenerationCheckpointError(f"Invalid checkpoint model config: {exc}") from exc
    if model.num_parameters != metadata["expected_num_parameters"]:
        raise GenerationCheckpointError(
            "Checkpoint num_parameters does not match the reconstructed GRU."
        )

    state = _mapping(payload["model_state_dict"], "model_state_dict")
    expected_state = model.state_dict()
    _exact_keys(state, set(expected_state), "model_state_dict")
    for name, expected in expected_state.items():
        value = state[name]
        if not isinstance(value, Tensor):
            raise GenerationCheckpointError(
                f"model_state_dict.{name} must be a tensor."
            )
        if value.device.type != "cpu":
            raise GenerationCheckpointError(
                f"model_state_dict.{name} was not loaded onto CPU."
            )
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise GenerationCheckpointError(
                f"model_state_dict.{name} has an incompatible shape or dtype."
            )
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise GenerationCheckpointError(
                f"model_state_dict.{name} contains non-finite values."
            )
    try:
        model.load_state_dict(dict(state), strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise GenerationCheckpointError(
            f"Checkpoint model state is incompatible with the GRU: {exc}"
        ) from exc
    return model


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise GenerationCheckpointError(
                "CUDA was requested but PyTorch cannot access a CUDA device."
            )
        return torch.device("cuda")
    raise GenerationCheckpointError("device must be 'auto', 'cpu', or 'cuda'.")


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise GenerationCheckpointError(f"{name} must be a string-keyed mapping.")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise GenerationCheckpointError(f"{name} is missing: {', '.join(missing)}.")
    if unknown:
        raise GenerationCheckpointError(f"{name} has unknown keys: {', '.join(unknown)}.")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise GenerationCheckpointError(f"{name} must be a non-empty string.")
    return value


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GenerationCheckpointError(f"{name} must be an integer >= {minimum}.")
    return value


def _number(
    value: object,
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GenerationCheckpointError(f"{name} must be numeric.")
    converted = float(value)
    if not math.isfinite(converted) or converted < minimum:
        raise GenerationCheckpointError(f"{name} must be finite and >= {minimum}.")
    if maximum is not None and converted > maximum:
        raise GenerationCheckpointError(f"{name} must be <= {maximum}.")
    return converted


def _sha256(value: object, name: str) -> str:
    converted = _string(value, name)
    if not _SHA256_PATTERN.fullmatch(converted):
        raise GenerationCheckpointError(
            f"{name} must be 64 lowercase hexadecimal digits."
        )
    return converted


__all__ = [
    "GenerationBundle",
    "GenerationCheckpointError",
    "load_generation_bundle",
]
