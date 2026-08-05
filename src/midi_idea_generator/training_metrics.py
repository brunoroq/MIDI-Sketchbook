"""Token-weighted diagnostics for symbolic-music language-model training.

The accumulator in this module is deliberately independent from the trainer.
It accepts already-computed logits and preserves sums across batches so that
an epoch report is not biased by batch size or sequence padding.

For an ``UNLABELED`` note, the token immediately after ``Duration`` has a known
structural target but may have omitted technique tokens before it. At those
positions the objective diagnostics exclude technique logits from the softmax
denominator while retaining the structural target. Full-vocabulary NLL is
reported alongside it as a diagnostic, never as the partial-label objective.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from numbers import Integral
from typing import Any

import torch
from torch import Tensor


METRICS_SCHEMA_VERSION = 1
_TOP_K = 5
_PERPLEXITY_MAX_NLL = 50.0


class TrainingMetricsError(ValueError):
    """Raised when a metrics batch does not satisfy the diagnostics contract."""


@dataclass(frozen=True, slots=True)
class MetricAggregate:
    """Token-weighted metrics for one non-overlapping or overlapping slice."""

    count: int
    full_vocab_nll: float | None
    full_vocab_perplexity: float | None
    objective_nll: float | None
    objective_perplexity: float | None
    token_top1_accuracy: float | None
    token_top5_accuracy: float | None
    type_top1_accuracy: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        """Return a strict JSON-compatible representation."""

        return {
            "count": self.count,
            "full_vocab_nll": self.full_vocab_nll,
            "full_vocab_perplexity": self.full_vocab_perplexity,
            "objective_nll": self.objective_nll,
            "objective_perplexity": self.objective_perplexity,
            "token_top1_accuracy": self.token_top1_accuracy,
            "token_top5_accuracy": self.token_top5_accuracy,
            "type_top1_accuracy": self.type_top1_accuracy,
        }


@dataclass(frozen=True, slots=True)
class TrainingMetricsReport:
    """Immutable, JSONable snapshot accumulated over one or more batches."""

    total: MetricAggregate
    post_duration_unknown: MetricAggregate
    by_target_type: Mapping[str, MetricAggregate]
    batches: int

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic structure suitable for training reports."""

        return {
            "schema_version": METRICS_SCHEMA_VERSION,
            "batches": self.batches,
            "total": self.total.to_dict(),
            "post_duration_unknown": self.post_duration_unknown.to_dict(),
            "by_target_type": {
                token_type: metrics.to_dict()
                for token_type, metrics in sorted(self.by_target_type.items())
            },
        }

    def to_json(self) -> str:
        """Serialize with non-finite values forbidden by construction."""

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )


@dataclass(slots=True)
class _RunningAggregate:
    count: int = 0
    full_vocab_nll_sum: float = 0.0
    objective_nll_sum: float = 0.0
    token_top1_correct: int = 0
    token_top5_correct: int = 0
    type_top1_correct: int = 0

    def add(
        self,
        *,
        full_vocab_nll: Tensor,
        objective_nll: Tensor,
        token_top1_correct: Tensor,
        token_top5_correct: Tensor,
        type_top1_correct: Tensor,
        selected: Tensor,
    ) -> None:
        count = int(selected.sum().item())
        if count == 0:
            return
        self.count += count
        self.full_vocab_nll_sum += float(
            full_vocab_nll[selected].double().sum().item()
        )
        self.objective_nll_sum += float(
            objective_nll[selected].double().sum().item()
        )
        self.token_top1_correct += int(token_top1_correct[selected].sum().item())
        self.token_top5_correct += int(token_top5_correct[selected].sum().item())
        self.type_top1_correct += int(type_top1_correct[selected].sum().item())

    def snapshot(self) -> MetricAggregate:
        if self.count == 0:
            return MetricAggregate(
                count=0,
                full_vocab_nll=None,
                full_vocab_perplexity=None,
                objective_nll=None,
                objective_perplexity=None,
                token_top1_accuracy=None,
                token_top5_accuracy=None,
                type_top1_accuracy=None,
            )
        full_nll = self.full_vocab_nll_sum / self.count
        objective_nll = self.objective_nll_sum / self.count
        return MetricAggregate(
            count=self.count,
            full_vocab_nll=full_nll,
            full_vocab_perplexity=_perplexity(full_nll),
            objective_nll=objective_nll,
            objective_perplexity=_perplexity(objective_nll),
            token_top1_accuracy=self.token_top1_correct / self.count,
            token_top5_accuracy=self.token_top5_correct / self.count,
            type_top1_accuracy=self.type_top1_correct / self.count,
        )


class TrainingMetricsAccumulator:
    """Accumulate loss and accuracy diagnostics across an epoch.

    Accuracy metrics use the same restricted logits as ``objective_nll``:
    technique IDs are unavailable only at positions marked by
    ``post_duration_unknown_mask``. The full-vocabulary NLL remains available
    to reveal calibration differences hidden by the partial-label objective.

    ``token_type_by_id`` can be either a contiguous ``id -> type`` mapping or a
    sequence indexed by token ID. Every vocabulary ID must have one non-empty
    type label.
    """

    def __init__(
        self,
        pad_token_id: int,
        technique_token_ids: Sequence[int] | frozenset[int] | set[int],
        token_type_by_id: Mapping[int, str] | Sequence[str],
    ) -> None:
        self._token_type_by_id = _normalize_token_types(token_type_by_id)
        self._vocabulary_size = len(self._token_type_by_id)
        self._pad_token_id = _require_token_id(
            pad_token_id,
            "pad_token_id",
            vocabulary_size=self._vocabulary_size,
        )
        self._technique_token_ids = _normalize_technique_ids(
            technique_token_ids,
            vocabulary_size=self._vocabulary_size,
            pad_token_id=self._pad_token_id,
            token_types=self._token_type_by_id,
        )
        unique_types = tuple(sorted(set(self._token_type_by_id)))
        type_to_index = {
            token_type: index for index, token_type in enumerate(unique_types)
        }
        self._type_index_by_id = torch.tensor(
            [type_to_index[token_type] for token_type in self._token_type_by_id],
            dtype=torch.long,
        )
        self._target_types = unique_types
        self.reset()

    @property
    def vocabulary_size(self) -> int:
        return self._vocabulary_size

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def technique_token_ids(self) -> tuple[int, ...]:
        return self._technique_token_ids

    @property
    def batches(self) -> int:
        return self._batches

    def reset(self) -> None:
        """Discard all accumulated batches while preserving the contract."""

        self._total = _RunningAggregate()
        self._post_duration_unknown = _RunningAggregate()
        self._by_target_type = {
            token_type: _RunningAggregate() for token_type in self._target_types
        }
        self._batches = 0

    @torch.no_grad()
    def update(
        self,
        logits: Tensor,
        targets: Tensor,
        post_duration_unknown_mask: Tensor,
    ) -> None:
        """Add one ``(batch, time, vocabulary)`` logits batch.

        PAD targets are ignored. The unknown mask must use ``torch.bool`` and
        cannot mark padding or a technique target. Inputs are never mutated and
        no autograd graph is retained.
        """

        _validate_batch_tensors(
            logits,
            targets,
            post_duration_unknown_mask,
            vocabulary_size=self._vocabulary_size,
            pad_token_id=self._pad_token_id,
            technique_token_ids=self._technique_token_ids,
        )
        flat_logits = logits.detach().reshape(-1, self._vocabulary_size)
        flat_targets = targets.detach().reshape(-1)
        flat_unknown = post_duration_unknown_mask.detach().reshape(-1)
        real = flat_targets != self._pad_token_id
        if not bool(real.any()):
            self._batches += 1
            return

        observed_logits = flat_logits[real].float()
        observed_targets = flat_targets[real]
        observed_unknown = flat_unknown[real]
        if not torch.isfinite(observed_logits).all():
            raise TrainingMetricsError("logits contain non-finite values at real targets.")

        full_log_probs = torch.log_softmax(observed_logits, dim=-1)
        full_vocab_nll = -full_log_probs.gather(
            1, observed_targets.unsqueeze(1)
        ).squeeze(1)

        objective_logits = observed_logits.clone()
        if bool(observed_unknown.any()):
            technique_ids = torch.tensor(
                self._technique_token_ids,
                dtype=torch.long,
                device=objective_logits.device,
            )
            unknown_rows = observed_unknown.nonzero(as_tuple=False).squeeze(1)
            objective_logits[
                unknown_rows.unsqueeze(1), technique_ids.unsqueeze(0)
            ] = -torch.inf
        objective_log_probs = torch.log_softmax(objective_logits, dim=-1)
        objective_nll = -objective_log_probs.gather(
            1, observed_targets.unsqueeze(1)
        ).squeeze(1)
        if not torch.isfinite(full_vocab_nll).all() or not torch.isfinite(
            objective_nll
        ).all():
            raise TrainingMetricsError("NLL became non-finite.")

        top1_ids = objective_logits.argmax(dim=-1)
        top_k = min(_TOP_K, self._vocabulary_size)
        top_ids = objective_logits.topk(top_k, dim=-1).indices
        token_top1_correct = top1_ids == observed_targets
        token_top5_correct = (top_ids == observed_targets.unsqueeze(1)).any(dim=1)
        type_index = self._type_index_by_id.to(observed_targets.device)
        target_type_indices = type_index[observed_targets]
        type_top1_correct = type_index[top1_ids] == target_type_indices
        all_selected = torch.ones_like(observed_unknown, dtype=torch.bool)

        values = {
            "full_vocab_nll": full_vocab_nll,
            "objective_nll": objective_nll,
            "token_top1_correct": token_top1_correct,
            "token_top5_correct": token_top5_correct,
            "type_top1_correct": type_top1_correct,
        }
        self._total.add(selected=all_selected, **values)
        self._post_duration_unknown.add(selected=observed_unknown, **values)
        for token_type, aggregate in self._by_target_type.items():
            type_id = self._type_index_by_id[
                self._token_type_by_id.index(token_type)
            ]
            selected = target_type_indices == int(type_id.item())
            aggregate.add(selected=selected, **values)
        self._batches += 1

    def snapshot(self) -> TrainingMetricsReport:
        """Return an immutable report without resetting the accumulator."""

        return TrainingMetricsReport(
            total=self._total.snapshot(),
            post_duration_unknown=self._post_duration_unknown.snapshot(),
            by_target_type={
                token_type: aggregate.snapshot()
                for token_type, aggregate in sorted(self._by_target_type.items())
            },
            batches=self._batches,
        )


def _normalize_token_types(
    value: Mapping[int, str] | Sequence[str],
) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        if not value:
            raise TrainingMetricsError("token_type_by_id cannot be empty.")
        normalized_keys: list[int] = []
        for key in value:
            if isinstance(key, bool) or not isinstance(key, Integral):
                raise TrainingMetricsError(
                    "token_type_by_id mapping keys must be integer token IDs."
                )
            normalized_keys.append(int(key))
        if sorted(normalized_keys) != list(range(len(normalized_keys))):
            raise TrainingMetricsError(
                "token_type_by_id mapping keys must be contiguous from zero."
            )
        ordered = tuple(value[index] for index in range(len(value)))
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
            raise TrainingMetricsError(
                "token_type_by_id must be a non-empty mapping or sequence."
            )
        ordered = tuple(value)
    for index, token_type in enumerate(ordered):
        if not isinstance(token_type, str) or not token_type:
            raise TrainingMetricsError(
                f"token_type_by_id[{index}] must be a non-empty string."
            )
    return ordered


def _normalize_technique_ids(
    value: Sequence[int] | frozenset[int] | set[int],
    *,
    vocabulary_size: int,
    pad_token_id: int,
    token_types: Sequence[str],
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(
        value, (Sequence, set, frozenset)
    ):
        raise TrainingMetricsError("technique_token_ids must be a collection of IDs.")
    normalized = tuple(
        _require_token_id(
            token_id,
            f"technique_token_ids[{index}]",
            vocabulary_size=vocabulary_size,
        )
        for index, token_id in enumerate(value)
    )
    if not normalized:
        raise TrainingMetricsError("technique_token_ids cannot be empty.")
    if len(set(normalized)) != len(normalized):
        raise TrainingMetricsError("technique_token_ids must be distinct.")
    if pad_token_id in normalized:
        raise TrainingMetricsError("PAD cannot be a technique token ID.")
    for token_id in normalized:
        if token_types[token_id] != "Technique":
            raise TrainingMetricsError(
                "Every technique_token_id must have token type 'Technique'."
            )
    return tuple(sorted(normalized))


def _validate_batch_tensors(
    logits: Tensor,
    targets: Tensor,
    unknown_mask: Tensor,
    *,
    vocabulary_size: int,
    pad_token_id: int,
    technique_token_ids: Sequence[int],
) -> None:
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise TrainingMetricsError("logits must have shape (batch, time, vocabulary).")
    if not logits.dtype.is_floating_point:
        raise TrainingMetricsError("logits must use a floating-point dtype.")
    if logits.shape[2] != vocabulary_size:
        raise TrainingMetricsError("logits vocabulary does not match token_type_by_id.")
    if logits.shape[0] < 1 or logits.shape[1] < 1:
        raise TrainingMetricsError("logits cannot have an empty batch or time axis.")
    if not isinstance(targets, Tensor) or targets.ndim != 2:
        raise TrainingMetricsError("targets must have shape (batch, time).")
    if targets.dtype != torch.long:
        raise TrainingMetricsError("targets must use torch.long token IDs.")
    if tuple(targets.shape) != tuple(logits.shape[:2]):
        raise TrainingMetricsError("targets batch/time dimensions must match logits.")
    if not isinstance(unknown_mask, Tensor) or unknown_mask.ndim != 2:
        raise TrainingMetricsError(
            "post_duration_unknown_mask must have shape (batch, time)."
        )
    if unknown_mask.dtype != torch.bool:
        raise TrainingMetricsError("post_duration_unknown_mask must use torch.bool.")
    if tuple(unknown_mask.shape) != tuple(targets.shape):
        raise TrainingMetricsError(
            "post_duration_unknown_mask dimensions must match targets."
        )
    if logits.device != targets.device or logits.device != unknown_mask.device:
        raise TrainingMetricsError("logits, targets, and mask must share a device.")
    minimum, maximum = torch.aminmax(targets)
    if minimum.item() < 0 or maximum.item() >= vocabulary_size:
        raise TrainingMetricsError("targets contain IDs outside the vocabulary.")
    if bool((unknown_mask & (targets == pad_token_id)).any()):
        raise TrainingMetricsError(
            "post_duration_unknown_mask cannot mark a PAD target."
        )
    technique_ids = torch.tensor(
        tuple(technique_token_ids), dtype=torch.long, device=targets.device
    )
    unknown_targets = targets[unknown_mask]
    if unknown_targets.numel() and bool(
        (unknown_targets.unsqueeze(1) == technique_ids.unsqueeze(0)).any()
    ):
        raise TrainingMetricsError(
            "A post-duration unknown position cannot target a technique token."
        )


def _require_token_id(
    value: object,
    name: str,
    *,
    vocabulary_size: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TrainingMetricsError(f"{name} must be an integer token ID.")
    converted = int(value)
    if not 0 <= converted < vocabulary_size:
        raise TrainingMetricsError(
            f"{name} must be in the range [0, {vocabulary_size})."
        )
    return converted


def _perplexity(nll: float) -> float:
    return math.exp(min(nll, _PERPLEXITY_MAX_NLL))


__all__ = [
    "METRICS_SCHEMA_VERSION",
    "MetricAggregate",
    "TrainingMetricsAccumulator",
    "TrainingMetricsError",
    "TrainingMetricsReport",
]
