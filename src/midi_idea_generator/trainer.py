"""Training orchestration for the small autoregressive GRU."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import pickle
import random
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .dataset import TokenizedSequenceDataset, make_collate_fn
from .model import GRUModel
from .training_config import TrainingConfig
from .training_metrics import (
    METRICS_SCHEMA_VERSION,
    TrainingMetricsAccumulator,
    TrainingMetricsError,
)
from .utils import relative_label, write_json


LOGGER = logging.getLogger(__name__)
CHECKPOINT_SCHEMA_VERSION = 2


class TrainingError(RuntimeError):
    """Raised when a training run cannot be started or completed safely."""


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Token-weighted metrics collected for one completed epoch."""

    epoch: int
    train_loss: float
    validation_loss: float
    train_perplexity: float
    validation_perplexity: float
    train_tokens: int
    validation_tokens: int
    mean_gradient_norm: float
    duration_seconds: float
    train_metrics: dict[str, Any]
    validation_metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TrainingReport:
    """Artifacts and metrics returned by a completed training run."""

    training_run_id: str
    device: str
    num_parameters: int
    start_epoch: int
    completed_epochs: int
    best_epoch: int
    best_validation_loss: float
    stopped_early: bool
    checkpoint_dir: Path
    best_checkpoint: Path
    latest_checkpoint: Path
    training_report_path: Path
    tensorboard_dir: Path
    history: tuple[EpochMetrics, ...]


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto``/``cpu``/``cuda`` and reject unavailable CUDA."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise TrainingError(
                "CUDA was requested but PyTorch cannot access a CUDA device."
            )
        return torch.device("cuda")
    raise TrainingError("device must be 'auto', 'cpu', or 'cuda'.")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TrainingError("Training seed must be a non-negative integer.")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def run_training(
    config: TrainingConfig,
    *,
    epochs_override: int | None = None,
) -> TrainingReport:
    """Train or resume the configured GRU against the Stage 2 manifest."""

    epochs = config.training.epochs if epochs_override is None else epochs_override
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise TrainingError("Epoch count must be a positive integer.")

    device = resolve_device(config.device)
    seed_everything(config.seed)
    train_dataset = TokenizedSequenceDataset(
        config.paths.tokenization_manifest_path,
        "train",
        project_root=config.project_root,
        max_sequence_length=config.data.max_sequence_length,
        verify_hashes=True,
    )
    validation_dataset = TokenizedSequenceDataset(
        config.paths.tokenization_manifest_path,
        "validation",
        project_root=config.project_root,
        max_sequence_length=config.data.max_sequence_length,
        verify_hashes=True,
    )
    _validate_dataset_pair(train_dataset, validation_dataset)
    ignored_target_token_ids = (
        train_dataset.tonic_token_ids | train_dataset.mode_token_ids
    )

    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(config.seed)
    common_loader_kwargs: dict[str, Any] = {
        "batch_size": config.data.batch_size,
        "num_workers": config.data.num_workers,
        "collate_fn": make_collate_fn(train_dataset.pad_token_id),
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if config.data.num_workers > 0:
        common_loader_kwargs["worker_init_fn"] = _seed_worker
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=loader_generator,
        **common_loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **common_loader_kwargs,
    )

    model = GRUModel(
        train_dataset.vocabulary_size,
        train_dataset.pad_token_id,
        embedding_dim=config.model.embedding_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    amp_enabled = _mixed_precision_enabled(config.training.mixed_precision, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    compatibility = _compatibility_snapshot(
        config,
        train_dataset,
        model,
        device=device,
        amp_enabled=amp_enabled,
    )

    if config.paths.resume_from is None:
        training_run_id = _create_training_run_id(compatibility)
        checkpoint_dir = _create_unique_run_directory(
            config.paths.checkpoints_dir, training_run_id
        )
        training_run_id = checkpoint_dir.name
        tensorboard_dir = config.paths.tensorboard_log_dir / training_run_id
        start_epoch = 1
        best_validation_loss = math.inf
        best_epoch = 0
        epochs_without_improvement = 0
        history: list[EpochMetrics] = []
    else:
        resume_path = config.paths.resume_from
        if resume_path.name != "latest.pt":
            raise TrainingError(
                "Training can only resume from a run's latest.pt checkpoint."
            )
        payload = _load_checkpoint(resume_path, device)
        _validate_checkpoint_compatibility(payload, compatibility)
        training_run_id = _checkpoint_run_id(payload)
        checkpoint_dir = resume_path.parent
        if checkpoint_dir.name != training_run_id:
            raise TrainingError(
                "Checkpoint training_run_id must match its parent directory."
            )
        best_checkpoint = checkpoint_dir / "best.pt"
        if not best_checkpoint.is_file():
            raise TrainingError(
                "Checkpoint run is incomplete: its best.pt artifact is missing."
            )
        tensorboard_dir = config.paths.tensorboard_log_dir / training_run_id
        completed_epoch = _checkpoint_int(payload, "epoch", minimum=1)
        best_validation_loss = _checkpoint_float(
            payload, "best_validation_loss", finite=True
        )
        best_epoch = _checkpoint_int(payload, "best_epoch", minimum=1)
        epochs_without_improvement = _checkpoint_int(
            payload, "epochs_without_improvement", minimum=0
        )
        history = _history_from_checkpoint(
            payload.get("history"), completed_epoch=completed_epoch
        )
        _validate_resume_progress(
            history=history,
            completed_epoch=completed_epoch,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            epochs_without_improvement=epochs_without_improvement,
        )
        if (
            epochs_without_improvement
            >= config.training.early_stopping_patience
        ):
            raise TrainingError(
                "Checkpoint already reached the configured early-stopping "
                "patience; start a new experiment to continue with a different "
                "policy."
            )
        try:
            model.load_state_dict(_checkpoint_mapping(payload, "model_state_dict"))
            optimizer.load_state_dict(
                _checkpoint_mapping(payload, "optimizer_state_dict")
            )
            scaler_state = _checkpoint_mapping(payload, "scaler_state_dict")
            scaler.load_state_dict(dict(scaler_state))
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise TrainingError(
                "Checkpoint contains incompatible model or optimizer state."
            ) from exc
        loader_state = payload.get("loader_generator_state")
        if not isinstance(loader_state, Tensor):
            raise TrainingError("Checkpoint has an invalid DataLoader RNG state.")
        try:
            loader_generator.set_state(loader_state.cpu())
        except RuntimeError as exc:
            raise TrainingError(
                "Checkpoint has an invalid DataLoader RNG state."
            ) from exc
        _restore_rng_state(_checkpoint_mapping(payload, "rng_state"))
        start_epoch = completed_epoch + 1
        if start_epoch > epochs:
            raise TrainingError(
                f"Checkpoint already completed epoch {completed_epoch}; "
                f"configured total epochs is {epochs}."
            )

    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoint_dir / "best.pt"
    latest_checkpoint = checkpoint_dir / "latest.pt"
    stopped_early = False
    writer = SummaryWriter(log_dir=tensorboard_dir)
    try:
        LOGGER.info(
            "Training %s on %s: %d train / %d validation sequences, %d parameters",
            training_run_id,
            device,
            len(train_dataset),
            len(validation_dataset),
            model.num_parameters,
        )
        for epoch in range(start_epoch, epochs + 1):
            started = time.monotonic()
            (
                train_loss,
                train_tokens,
                mean_gradient_norm,
                train_diagnostics,
            ) = _train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                train_dataset.pad_token_id,
                config.training.gradient_clip,
                amp_enabled,
                train_dataset.technique_token_ids,
                train_dataset.token_type_by_id,
                ignored_target_token_ids,
            )
            (
                validation_loss,
                validation_tokens,
                validation_diagnostics,
            ) = _evaluate_epoch(
                model,
                validation_loader,
                device,
                train_dataset.pad_token_id,
                amp_enabled,
                train_dataset.technique_token_ids,
                train_dataset.token_type_by_id,
                ignored_target_token_ids,
            )
            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                train_perplexity=_perplexity(train_loss),
                validation_perplexity=_perplexity(validation_loss),
                train_tokens=train_tokens,
                validation_tokens=validation_tokens,
                mean_gradient_norm=mean_gradient_norm,
                duration_seconds=time.monotonic() - started,
                train_metrics=train_diagnostics,
                validation_metrics=validation_diagnostics,
            )
            history.append(metrics)
            _write_tensorboard_metrics(writer, metrics, optimizer)

            improved = (
                validation_loss
                < best_validation_loss - config.training.early_stopping_min_delta
            )
            if improved:
                best_validation_loss = validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            payload = _checkpoint_payload(
                training_run_id=training_run_id,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                loader_generator=loader_generator,
                config=config,
                requested_total_epochs=epochs,
                compatibility=compatibility,
                best_validation_loss=best_validation_loss,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                history=history,
            )
            if epoch % config.training.checkpoint_every_epochs == 0:
                _atomic_torch_save(checkpoint_dir / f"epoch-{epoch:04d}.pt", payload)
            if improved:
                _atomic_torch_save(best_checkpoint, payload)
            # ``latest.pt`` is the commit marker for a fully published epoch.
            # Publishing it last prevents resume from observing metadata that
            # refers to a best or periodic checkpoint which failed to save.
            _atomic_torch_save(latest_checkpoint, payload)

            LOGGER.info(
                "Epoch %d/%d | train %.4f | validation %.4f | "
                "perplexity %.2f | token top-1 %.1f%% | type top-1 %.1f%% | "
                "grad %.3f | %.1fs",
                epoch,
                epochs,
                train_loss,
                validation_loss,
                metrics.validation_perplexity,
                100.0
                * float(validation_diagnostics["total"]["token_top1_accuracy"]),
                100.0
                * float(validation_diagnostics["total"]["type_top1_accuracy"]),
                mean_gradient_norm,
                metrics.duration_seconds,
            )
            writer.flush()
            if (
                epochs_without_improvement
                >= config.training.early_stopping_patience
            ):
                stopped_early = True
                LOGGER.info(
                    "Early stopping after %d epoch(s) without improvement.",
                    epochs_without_improvement,
                )
                break
    finally:
        writer.close()

    if not history or not best_checkpoint.is_file() or not latest_checkpoint.is_file():
        raise TrainingError("Training finished without complete checkpoint artifacts.")
    training_report_path = checkpoint_dir / "training_report.json"
    write_json(
        training_report_path,
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "training_run_id": training_run_id,
            "device": str(device),
            "num_parameters": model.num_parameters,
            "start_epoch": start_epoch,
            "requested_total_epochs": epochs,
            "completed_epoch": history[-1].epoch,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "stopped_early": stopped_early,
            "compatibility": compatibility,
            "configuration": _jsonable(asdict(config)),
            "artifacts": {
                "best_checkpoint": relative_label(
                    best_checkpoint, config.project_root
                ),
                "latest_checkpoint": relative_label(
                    latest_checkpoint, config.project_root
                ),
                "tensorboard_dir": relative_label(
                    tensorboard_dir, config.project_root
                ),
            },
            "history": [asdict(metrics) for metrics in history],
        },
    )
    return TrainingReport(
        training_run_id=training_run_id,
        device=str(device),
        num_parameters=model.num_parameters,
        start_epoch=start_epoch,
        completed_epochs=history[-1].epoch,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        stopped_early=stopped_early,
        checkpoint_dir=checkpoint_dir,
        best_checkpoint=best_checkpoint,
        latest_checkpoint=latest_checkpoint,
        training_report_path=training_report_path,
        tensorboard_dir=tensorboard_dir,
        history=tuple(history),
    )


def _train_epoch(
    model: GRUModel,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    pad_token_id: int,
    gradient_clip: float,
    amp_enabled: bool,
    technique_token_ids: Sequence[int] | frozenset[int],
    token_type_by_id: Sequence[str],
    ignored_target_token_ids: Sequence[int] | frozenset[int] | set[int] = (),
) -> tuple[float, int, float, dict[str, Any]]:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    gradient_norms: list[float] = []
    diagnostics = _metrics_accumulator(
        pad_token_id,
        technique_token_ids,
        token_type_by_id,
        ignored_target_token_ids,
    )
    for batch in loader:
        input_ids, target_ids, unknown_mask = _batch_tensors(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(input_ids)
            loss_sum, num_tokens = token_cross_entropy(
                logits,
                target_ids,
                pad_token_id,
                unknown_technique_decision_mask=unknown_mask,
                technique_token_ids=technique_token_ids,
                ignored_target_token_ids=ignored_target_token_ids,
            )
            loss = loss_sum / num_tokens
        _update_metrics(diagnostics, logits, target_ids, unknown_mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        try:
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=gradient_clip, error_if_nonfinite=True
            )
        except RuntimeError as exc:
            raise TrainingError(f"Gradient norm became non-finite: {exc}") from exc
        scaler.step(optimizer)
        scaler.update()
        count = int(num_tokens.item())
        total_loss += float(loss_sum.detach().item())
        total_tokens += count
        gradient_norms.append(float(gradient_norm.detach().item()))
    if total_tokens == 0 or not gradient_norms:
        raise TrainingError("Training DataLoader produced no target tokens.")
    report = diagnostics.snapshot().to_dict()
    if report["total"]["count"] != total_tokens:
        raise TrainingError("Training diagnostics lost target tokens.")
    return (
        total_loss / total_tokens,
        total_tokens,
        sum(gradient_norms) / len(gradient_norms),
        report,
    )


@torch.no_grad()
def _evaluate_epoch(
    model: GRUModel,
    loader: DataLoader[Any],
    device: torch.device,
    pad_token_id: int,
    amp_enabled: bool,
    technique_token_ids: Sequence[int] | frozenset[int],
    token_type_by_id: Sequence[str],
    ignored_target_token_ids: Sequence[int] | frozenset[int] | set[int] = (),
) -> tuple[float, int, dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    diagnostics = _metrics_accumulator(
        pad_token_id,
        technique_token_ids,
        token_type_by_id,
        ignored_target_token_ids,
    )
    for batch in loader:
        input_ids, target_ids, unknown_mask = _batch_tensors(batch, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(input_ids)
            loss_sum, num_tokens = token_cross_entropy(
                logits,
                target_ids,
                pad_token_id,
                unknown_technique_decision_mask=unknown_mask,
                technique_token_ids=technique_token_ids,
                ignored_target_token_ids=ignored_target_token_ids,
            )
        _update_metrics(diagnostics, logits, target_ids, unknown_mask)
        count = int(num_tokens.item())
        total_loss += float(loss_sum.item())
        total_tokens += count
    if total_tokens == 0:
        raise TrainingError("Validation DataLoader produced no target tokens.")
    report = diagnostics.snapshot().to_dict()
    if report["total"]["count"] != total_tokens:
        raise TrainingError("Validation diagnostics lost target tokens.")
    return total_loss / total_tokens, total_tokens, report


def _metrics_accumulator(
    pad_token_id: int,
    technique_token_ids: Sequence[int] | frozenset[int],
    token_type_by_id: Sequence[str],
    ignored_target_token_ids: Sequence[int] | frozenset[int] | set[int] = (),
) -> TrainingMetricsAccumulator:
    try:
        return TrainingMetricsAccumulator(
            pad_token_id,
            technique_token_ids,
            token_type_by_id,
            ignored_target_token_ids=ignored_target_token_ids,
        )
    except TrainingMetricsError as exc:
        raise TrainingError(f"Invalid training metrics contract: {exc}") from exc


def _update_metrics(
    accumulator: TrainingMetricsAccumulator,
    logits: Tensor,
    targets: Tensor,
    unknown_mask: Tensor,
) -> None:
    try:
        accumulator.update(logits, targets, unknown_mask)
    except TrainingMetricsError as exc:
        raise TrainingError(f"Could not compute training metrics: {exc}") from exc


def token_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    pad_token_id: int,
    *,
    unknown_technique_decision_mask: Tensor | None = None,
    technique_token_ids: Sequence[int] | frozenset[int] = (),
    ignored_target_token_ids: Sequence[int] | frozenset[int] | set[int] = (),
) -> tuple[Tensor, Tensor]:
    """Return the partial-label objective and optimized-target count.

    At an ``UNLABELED`` note's post-``Duration`` decision, the base target is
    known but the presence of a guitar technique is not.  Those positions use
    a softmax over the non-technique vocabulary: the real structural target
    still trains normally, while the six ``Technique`` logits receive exactly
    zero gradient.  Fully labelled and ordinary positions use the complete
    vocabulary. Prompt-only targets such as ``Tonic`` and ``Mode`` can be
    excluded from the objective without removing their input tokens from the
    recurrent context. Their classes remain in every other softmax denominator.
    """

    if logits.ndim != 3 or targets.ndim != 2:
        raise TrainingError("Logits and targets must have shapes (B,T,V) and (B,T).")
    if logits.shape[:2] != targets.shape:
        raise TrainingError("Logit and target batch/time dimensions must match.")
    if targets.dtype != torch.long:
        raise TrainingError("Targets must use torch.long token identifiers.")
    if targets.device != logits.device:
        raise TrainingError("Logits and targets must share a device.")
    vocabulary_size = logits.shape[-1]
    if (
        isinstance(pad_token_id, bool)
        or not isinstance(pad_token_id, int)
        or not 0 <= pad_token_id < vocabulary_size
    ):
        raise TrainingError("pad_token_id must be inside the model vocabulary.")
    ignored_ids = _normalize_ignored_target_ids(
        ignored_target_token_ids,
        vocabulary_size=vocabulary_size,
        pad_token_id=pad_token_id,
    )
    real_mask = targets != pad_token_id
    ignored_mask = torch.zeros_like(targets, dtype=torch.bool)
    if ignored_ids:
        ignored_tensor = torch.tensor(
            ignored_ids, dtype=torch.long, device=targets.device
        )
        ignored_mask = torch.isin(targets, ignored_tensor)
    objective_mask = real_mask & ~ignored_mask
    if unknown_technique_decision_mask is None:
        unknown_mask = torch.zeros_like(targets, dtype=torch.bool)
    else:
        if (
            not isinstance(unknown_technique_decision_mask, Tensor)
            or unknown_technique_decision_mask.dtype != torch.bool
            or unknown_technique_decision_mask.shape != targets.shape
            or unknown_technique_decision_mask.device != targets.device
        ):
            raise TrainingError(
                "unknown_technique_decision_mask must be a boolean tensor "
                "matching targets."
            )
        unknown_mask = unknown_technique_decision_mask
    if torch.any(unknown_mask & ~real_mask):
        raise TrainingError(
            "Unknown technique decisions cannot mark padded target positions."
        )
    if torch.any(unknown_mask & ignored_mask):
        raise TrainingError(
            "Unknown technique decisions cannot mark ignored target positions."
        )

    normalized_technique_ids: list[int] = []
    for index, token_id in enumerate(technique_token_ids):
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < vocabulary_size
        ):
            raise TrainingError(
                f"technique_token_ids[{index}] must be inside the vocabulary."
            )
        normalized_technique_ids.append(token_id)
    if len(set(normalized_technique_ids)) != len(normalized_technique_ids):
        raise TrainingError("technique_token_ids must be distinct.")
    normalized_technique_ids.sort()
    if pad_token_id in normalized_technique_ids:
        raise TrainingError("PAD cannot be a technique token ID.")
    if torch.any(unknown_mask) and not normalized_technique_ids:
        raise TrainingError(
            "Unknown technique decisions require technique_token_ids."
        )

    num_tokens = objective_mask.sum()
    if int(num_tokens.item()) <= 0:
        raise TrainingError(
            "A training batch cannot contain only padding or ignored targets."
        )
    loss_sum = logits.sum() * 0.0
    ordinary_mask = objective_mask & ~unknown_mask
    if torch.any(ordinary_mask):
        loss_sum = loss_sum + F.cross_entropy(
            logits[ordinary_mask],
            targets[ordinary_mask],
            reduction="sum",
        )
    if torch.any(unknown_mask):
        technique_tensor = torch.tensor(
            normalized_technique_ids,
            dtype=torch.long,
            device=targets.device,
        )
        if torch.any(
            (targets[unknown_mask].unsqueeze(1) == technique_tensor).any(dim=1)
        ):
            raise TrainingError(
                "An UNLABELED post-Duration target cannot be a Technique token."
            )
        allowed = torch.ones(
            vocabulary_size, dtype=torch.bool, device=targets.device
        )
        allowed[technique_tensor] = False
        remap = torch.full(
            (vocabulary_size,), -1, dtype=torch.long, device=targets.device
        )
        remap[allowed] = torch.arange(
            int(allowed.sum().item()), dtype=torch.long, device=targets.device
        )
        restricted_targets = remap[targets[unknown_mask]]
        if torch.any(restricted_targets < 0):
            raise TrainingError(
                "Restricted post-Duration target is outside the base vocabulary."
            )
        loss_sum = loss_sum + F.cross_entropy(
            logits[unknown_mask][:, allowed],
            restricted_targets,
            reduction="sum",
        )
    if not torch.isfinite(loss_sum):
        raise TrainingError("Cross-entropy loss became non-finite.")
    return loss_sum, num_tokens


def _normalize_ignored_target_ids(
    value: Sequence[int] | frozenset[int] | set[int],
    *,
    vocabulary_size: int,
    pad_token_id: int,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(
        value, (Sequence, set, frozenset)
    ):
        raise TrainingError("ignored_target_token_ids must be a collection of IDs.")
    normalized: list[int] = []
    for index, token_id in enumerate(value):
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < vocabulary_size
        ):
            raise TrainingError(
                f"ignored_target_token_ids[{index}] must be inside the vocabulary."
            )
        normalized.append(token_id)
    if len(set(normalized)) != len(normalized):
        raise TrainingError("ignored_target_token_ids must be distinct.")
    if pad_token_id in normalized:
        raise TrainingError("PAD cannot be an ignored target token ID.")
    return tuple(sorted(normalized))


def _batch_tensors(
    batch: Mapping[str, Any], device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    input_ids = batch.get("input_ids")
    target_ids = batch.get("target_ids")
    unknown_mask = batch.get("unknown_technique_decision_mask")
    if (
        not isinstance(input_ids, Tensor)
        or not isinstance(target_ids, Tensor)
        or not isinstance(unknown_mask, Tensor)
    ):
        raise TrainingError(
            "Dataset batches must contain input_ids, target_ids, and "
            "unknown_technique_decision_mask tensors."
        )
    if unknown_mask.dtype != torch.bool or unknown_mask.shape != target_ids.shape:
        raise TrainingError(
            "unknown_technique_decision_mask must be boolean and match target_ids."
        )
    return (
        input_ids.to(device, non_blocking=device.type == "cuda"),
        target_ids.to(device, non_blocking=device.type == "cuda"),
        unknown_mask.to(device, non_blocking=device.type == "cuda"),
    )


def _mixed_precision_enabled(mode: str, device: torch.device) -> bool:
    if mode == "auto":
        return device.type == "cuda"
    if mode == "off":
        return False
    if mode == "on":
        if device.type != "cuda":
            raise TrainingError("Mixed precision can only be enabled on CUDA.")
        return True
    raise TrainingError("mixed_precision must be auto, on, or off.")


def _validate_dataset_pair(
    train: TokenizedSequenceDataset, validation: TokenizedSequenceDataset
) -> None:
    if len(train) == 0 or len(validation) == 0:
        raise TrainingError("Training and validation splits must both be non-empty.")
    attributes = (
        "vocabulary_size",
        "pad_token_id",
        "tokenization_run_id",
        "tokenizer_sha256",
        "configuration_sha256",
        "tokenization_manifest_sha256",
        "technique_token_ids",
        "tonic_token_ids",
        "mode_token_ids",
        "token_type_by_id",
    )
    for attribute in attributes:
        if getattr(train, attribute) != getattr(validation, attribute):
            raise TrainingError(
                f"Training and validation datasets disagree on {attribute}."
            )


def _compatibility_snapshot(
    config: TrainingConfig,
    dataset: TokenizedSequenceDataset,
    model: GRUModel,
    *,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, Any]:
    return {
        "tokenization_run_id": dataset.tokenization_run_id,
        "tokenization_manifest_sha256": dataset.tokenization_manifest_sha256,
        "tokenizer_sha256": dataset.tokenizer_sha256,
        "tokenization_configuration_sha256": dataset.configuration_sha256,
        "vocabulary_size": dataset.vocabulary_size,
        "pad_token_id": dataset.pad_token_id,
        "model": asdict(config.model),
        "data": asdict(config.data),
        "optimizer": {
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "gradient_clip": config.training.gradient_clip,
            "mixed_precision": config.training.mixed_precision,
        },
        "seed": config.seed,
        "num_parameters": model.num_parameters,
        "execution": {
            "resolved_device_type": device.type,
            "amp_enabled": amp_enabled,
        },
        "training_implementation_sha256": _training_implementation_sha256(),
        "torch_version": str(torch.__version__),
    }


def _checkpoint_payload(
    *,
    training_run_id: str,
    epoch: int,
    model: GRUModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loader_generator: torch.Generator,
    config: TrainingConfig,
    requested_total_epochs: int,
    compatibility: Mapping[str, Any],
    best_validation_loss: float,
    best_epoch: int,
    epochs_without_improvement: int,
    history: Sequence[EpochMetrics],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "training_run_id": training_run_id,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loader_generator_state": loader_generator.get_state(),
        "rng_state": _capture_rng_state(),
        "configuration": _jsonable(asdict(config)),
        "requested_total_epochs": requested_total_epochs,
        "compatibility": dict(compatibility),
        "best_validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "history": [asdict(metrics) for metrics in history],
        "torch_version": str(torch.__version__),
    }


def _capture_rng_state() -> dict[str, Any]:
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "name": name,
            "keys": keys.tolist(),
            "position": position,
            "has_gauss": has_gauss,
            "cached_gaussian": cached_gaussian,
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(value: Mapping[str, Any]) -> None:
    python_state = value.get("python")
    numpy_state = value.get("numpy")
    torch_cpu = value.get("torch_cpu")
    torch_cuda = value.get("torch_cuda")
    if not isinstance(python_state, tuple):
        raise TrainingError("Checkpoint has an invalid Python RNG state.")
    if not isinstance(numpy_state, Mapping):
        raise TrainingError("Checkpoint has an invalid NumPy RNG state.")
    if not isinstance(torch_cpu, Tensor) or not isinstance(torch_cuda, list):
        raise TrainingError("Checkpoint has an invalid PyTorch RNG state.")
    try:
        random.setstate(python_state)
        np.random.set_state(
            (
                str(numpy_state["name"]),
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        torch.set_rng_state(torch_cpu.cpu())
        if torch.cuda.is_available():
            if not all(isinstance(state, Tensor) for state in torch_cuda):
                raise TypeError("CUDA RNG entries must be tensors")
            torch.cuda.set_rng_state_all([state.cpu() for state in torch_cuda])
    except (IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise TrainingError("Checkpoint contains an invalid RNG state.") from exc


def _load_checkpoint(path: Path, device: torch.device) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError, pickle.UnpicklingError) as exc:
        raise TrainingError(f"Could not load checkpoint '{path}': {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TrainingError("Checkpoint root must be a mapping.")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise TrainingError(
            f"Checkpoint schema must be {CHECKPOINT_SCHEMA_VERSION}."
        )
    return payload


def _validate_checkpoint_compatibility(
    payload: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    actual = _checkpoint_mapping(payload, "compatibility")
    try:
        matches = _canonical_hash(actual) == _canonical_hash(expected)
    except (TypeError, ValueError) as exc:
        raise TrainingError(
            "Checkpoint compatibility metadata is not valid canonical JSON."
        ) from exc
    if not matches:
        raise TrainingError(
            "Checkpoint is incompatible with the current tokenization, model, "
            "data, optimizer, or seed configuration."
        )


def _history_from_checkpoint(
    value: object, *, completed_epoch: int
) -> list[EpochMetrics]:
    if not isinstance(value, list):
        raise TrainingError("Checkpoint history must be a list.")
    history: list[EpochMetrics] = []
    try:
        for expected_epoch, raw in enumerate(value, start=1):
            if not isinstance(raw, Mapping):
                raise TypeError
            metrics = EpochMetrics(**dict(raw))
            if metrics.epoch != expected_epoch:
                raise ValueError
            numeric_values = (
                metrics.train_loss,
                metrics.validation_loss,
                metrics.train_perplexity,
                metrics.validation_perplexity,
                metrics.mean_gradient_norm,
                metrics.duration_seconds,
            )
            if not all(
                math.isfinite(number) and number >= 0 for number in numeric_values
            ):
                raise ValueError
            if metrics.train_tokens <= 0 or metrics.validation_tokens <= 0:
                raise ValueError
            _validate_metrics_report(
                metrics.train_metrics,
                "train_metrics",
                expected_count=metrics.train_tokens,
                expected_objective_nll=metrics.train_loss,
            )
            _validate_metrics_report(
                metrics.validation_metrics,
                "validation_metrics",
                expected_count=metrics.validation_tokens,
                expected_objective_nll=metrics.validation_loss,
            )
            history.append(metrics)
    except (TypeError, ValueError) as exc:
        raise TrainingError("Checkpoint contains invalid epoch history.") from exc
    if not history or len(history) != completed_epoch:
        raise TrainingError(
            "Checkpoint history must contain every epoch through its completed epoch."
        )
    return history


def _validate_metrics_report(
    value: object,
    name: str,
    *,
    expected_count: int,
    expected_objective_nll: float,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    expected_keys = {
        "schema_version",
        "batches",
        "total",
        "post_duration_unknown",
        "by_target_type",
    }
    if set(value) != expected_keys:
        raise ValueError(f"{name} has invalid keys")
    schema_version = value["schema_version"]
    batches = value["batches"]
    if (
        isinstance(schema_version, bool)
        or schema_version != METRICS_SCHEMA_VERSION
        or isinstance(batches, bool)
        or not isinstance(batches, int)
        or batches <= 0
    ):
        raise ValueError(f"{name} has invalid schema or batch count")
    total_count, total_objective = _validate_metric_aggregate(
        value["total"], f"{name}.total"
    )
    post_count, _ = _validate_metric_aggregate(
        value["post_duration_unknown"], f"{name}.post_duration_unknown"
    )
    by_target_type = value["by_target_type"]
    if not isinstance(by_target_type, Mapping) or not by_target_type:
        raise ValueError(f"{name}.by_target_type must be a non-empty mapping")
    type_count = 0
    for token_type, aggregate in by_target_type.items():
        if not isinstance(token_type, str) or not token_type:
            raise ValueError(f"{name}.by_target_type has an invalid token type")
        aggregate_count, _ = _validate_metric_aggregate(
            aggregate, f"{name}.by_target_type.{token_type}"
        )
        type_count += aggregate_count
    if total_count != expected_count or type_count != total_count:
        raise ValueError(f"{name} token counts are inconsistent")
    if post_count > total_count:
        raise ValueError(f"{name} post-Duration count exceeds its total")
    if total_objective is None or not math.isclose(
        total_objective,
        expected_objective_nll,
        rel_tol=1e-4,
        abs_tol=1e-6,
    ):
        raise ValueError(f"{name} objective NLL disagrees with epoch loss")


def _validate_metric_aggregate(
    value: object, name: str
) -> tuple[int, float | None]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    metric_keys = {
        "full_vocab_nll",
        "full_vocab_perplexity",
        "objective_nll",
        "objective_perplexity",
        "token_top1_accuracy",
        "token_top5_accuracy",
        "type_top1_accuracy",
    }
    if set(value) != {"count", *metric_keys}:
        raise ValueError(f"{name} has invalid keys")
    count = value["count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"{name}.count must be a non-negative integer")
    if count == 0:
        if any(value[key] is not None for key in metric_keys):
            raise ValueError(f"{name} must use null metrics when count is zero")
        return count, None
    for key in metric_keys:
        number = value[key]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
        ):
            raise ValueError(f"{name}.{key} must be finite")
        converted = float(number)
        if key.endswith("accuracy"):
            if not 0.0 <= converted <= 1.0:
                raise ValueError(f"{name}.{key} must be between zero and one")
        elif converted < 0.0:
            raise ValueError(f"{name}.{key} must be non-negative")
    return count, float(value["objective_nll"])


def _validate_resume_progress(
    *,
    history: Sequence[EpochMetrics],
    completed_epoch: int,
    best_epoch: int,
    best_validation_loss: float,
    epochs_without_improvement: int,
) -> None:
    """Reject internally inconsistent progress metadata before state restoration."""

    if history[-1].epoch != completed_epoch:
        raise TrainingError("Checkpoint history does not end at its completed epoch.")
    if best_epoch > completed_epoch:
        raise TrainingError("Checkpoint best_epoch exceeds its completed epoch.")
    recorded_best = history[best_epoch - 1].validation_loss
    if recorded_best != best_validation_loss:
        raise TrainingError(
            "Checkpoint best validation loss does not match its recorded best epoch."
        )
    if epochs_without_improvement != completed_epoch - best_epoch:
        raise TrainingError(
            "Checkpoint early-stopping progress is inconsistent with best_epoch."
        )


def _checkpoint_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise TrainingError(f"Checkpoint field '{key}' must be a mapping.")
    return value


def _checkpoint_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise TrainingError(f"Checkpoint field '{key}' must be a non-empty string.")
    return value


def _checkpoint_run_id(payload: Mapping[str, Any]) -> str:
    value = _checkpoint_string(payload, "training_run_id")
    if value in {".", ".."} or Path(value).name != value:
        raise TrainingError("Checkpoint training_run_id is not a safe directory name.")
    if len(value) > 128 or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in value
    ):
        raise TrainingError("Checkpoint training_run_id is not a safe directory name.")
    return value


def _checkpoint_int(
    payload: Mapping[str, Any], key: str, *, minimum: int
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrainingError(
            f"Checkpoint field '{key}' must be an integer >= {minimum}."
        )
    return value


def _checkpoint_float(
    payload: Mapping[str, Any], key: str, *, finite: bool
) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingError(f"Checkpoint field '{key}' must be numeric.")
    converted = float(value)
    if finite and not math.isfinite(converted):
        raise TrainingError(f"Checkpoint field '{key}' must be finite.")
    return converted


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent
        )
        os.close(descriptor)
        descriptor = None
        temporary = Path(name)
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
        temporary = None
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise TrainingError(f"Could not save checkpoint '{destination}': {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _create_training_run_id(compatibility: Mapping[str, Any]) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{_canonical_hash(compatibility)[:10]}"


def _create_unique_run_directory(root: Path, base_name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for suffix in range(1000):
        name = base_name if suffix == 0 else f"{base_name}-{suffix:03d}"
        candidate = root / name
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise TrainingError("Could not allocate a unique checkpoint run directory.")


def _canonical_hash(value: Any) -> str:
    serialized = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _training_implementation_sha256() -> str:
    package_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for filename in (
        "dataset.py",
        "model.py",
        "trainer.py",
        "training_config.py",
        "training_metrics.py",
    ):
        path = package_dir / filename
        digest.update(filename.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _perplexity(loss: float) -> float:
    return math.exp(min(loss, 50.0))


def _write_tensorboard_metrics(
    writer: SummaryWriter,
    metrics: EpochMetrics,
    optimizer: torch.optim.Optimizer,
) -> None:
    writer.add_scalar("loss/train", metrics.train_loss, metrics.epoch)
    writer.add_scalar("loss/validation", metrics.validation_loss, metrics.epoch)
    writer.add_scalar("perplexity/train", metrics.train_perplexity, metrics.epoch)
    writer.add_scalar(
        "perplexity/validation", metrics.validation_perplexity, metrics.epoch
    )
    writer.add_scalar("optimization/gradient_norm", metrics.mean_gradient_norm, metrics.epoch)
    writer.add_scalar("optimization/learning_rate", optimizer.param_groups[0]["lr"], metrics.epoch)
    _write_tensorboard_report(
        writer, "train", metrics.train_metrics, metrics.epoch
    )
    _write_tensorboard_report(
        writer, "validation", metrics.validation_metrics, metrics.epoch
    )


def _write_tensorboard_report(
    writer: SummaryWriter,
    phase: str,
    report: Mapping[str, Any],
    epoch: int,
) -> None:
    scopes: list[tuple[str, Mapping[str, Any]]] = [
        ("total", report["total"]),
        ("post_duration_unknown", report["post_duration_unknown"]),
    ]
    by_target_type = report["by_target_type"]
    scopes.extend(
        (f"by_target_type/{token_type}", aggregate)
        for token_type, aggregate in sorted(by_target_type.items())
    )
    for scope, aggregate in scopes:
        for metric_name, metric_value in aggregate.items():
            if metric_value is not None:
                writer.add_scalar(
                    f"metrics/{phase}/{scope}/{metric_name}",
                    metric_value,
                    epoch,
                )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "EpochMetrics",
    "TrainingError",
    "TrainingReport",
    "resolve_device",
    "run_training",
    "seed_everything",
    "token_cross_entropy",
]
