"""Tests for token-weighted training diagnostics and partial-label metrics."""

from __future__ import annotations

import json
import math
from typing import Callable

import pytest
import torch

from midi_idea_generator.training_metrics import (
    METRICS_SCHEMA_VERSION,
    TrainingMetricsAccumulator,
    TrainingMetricsError,
)


TOKEN_TYPES = (
    "PAD",
    "Pitch",
    "Position",
    "Technique",
    "Technique",
    "Bar",
    "Pitch",
)
TECHNIQUE_IDS = (3, 4)


def _accumulator() -> TrainingMetricsAccumulator:
    return TrainingMetricsAccumulator(
        pad_token_id=0,
        technique_token_ids=TECHNIQUE_IDS,
        token_type_by_id=TOKEN_TYPES,
    )


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # At the unknown first position, Technique_3 dominates the full logits but
    # Position_2 dominates once technique candidates are excluded.
    logits = torch.tensor(
        [
            [
                [-4.0, 1.0, 7.0, 10.0, 8.0, 6.0, 0.0],
                [-4.0, 7.0, 0.0, 1.0, 1.5, 2.0, 8.0],
                [-3.0, 0.0, 1.0, 9.0, 3.0, 2.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([[2, 1, 3, 0]], dtype=torch.long)
    unknown = torch.tensor([[True, False, False, False]], dtype=torch.bool)
    return logits, targets, unknown


def _expected_losses(
    logits: torch.Tensor, targets: torch.Tensor, unknown: torch.Tensor
) -> tuple[float, float, float, float]:
    real = targets != 0
    observed_logits = logits[real]
    observed_targets = targets[real]
    full = torch.nn.functional.cross_entropy(
        observed_logits, observed_targets, reduction="none"
    )
    restricted_logits = observed_logits.clone()
    observed_unknown = unknown[real]
    rows = observed_unknown.nonzero(as_tuple=False).squeeze(1)
    restricted_logits[rows.unsqueeze(1), torch.tensor(TECHNIQUE_IDS).unsqueeze(0)] = (
        -torch.inf
    )
    objective = torch.nn.functional.cross_entropy(
        restricted_logits, observed_targets, reduction="none"
    )
    return (
        float(full.mean()),
        float(objective.mean()),
        float(full[observed_unknown].mean()),
        float(objective[observed_unknown].mean()),
    )


def test_metrics_separate_full_and_restricted_partial_label_objectives() -> None:
    accumulator = _accumulator()
    logits, targets, unknown = _batch()
    expected_full, expected_objective, unknown_full, unknown_objective = (
        _expected_losses(logits, targets, unknown)
    )

    accumulator.update(logits, targets, unknown)
    report = accumulator.snapshot()

    assert report.batches == 1
    assert report.total.count == 3
    assert report.total.full_vocab_nll == pytest.approx(expected_full)
    assert report.total.objective_nll == pytest.approx(expected_objective)
    assert report.total.full_vocab_perplexity == pytest.approx(math.exp(expected_full))
    assert report.total.objective_perplexity == pytest.approx(
        math.exp(expected_objective)
    )
    assert report.total.token_top1_accuracy == pytest.approx(2 / 3)
    assert report.total.token_top5_accuracy == pytest.approx(1.0)
    # The incorrect exact Pitch prediction is another token of the same type.
    assert report.total.type_top1_accuracy == pytest.approx(1.0)

    special = report.post_duration_unknown
    assert special.count == 1
    assert special.full_vocab_nll == pytest.approx(unknown_full)
    assert special.objective_nll == pytest.approx(unknown_objective)
    assert special.full_vocab_nll > special.objective_nll
    assert special.token_top1_accuracy == 1.0
    assert special.token_top5_accuracy == 1.0
    assert special.type_top1_accuracy == 1.0


def test_breakdown_is_by_target_type_and_includes_clear_absences() -> None:
    accumulator = _accumulator()
    accumulator.update(*_batch())

    breakdown = accumulator.snapshot().by_target_type

    assert breakdown["Pitch"].count == 1
    assert breakdown["Pitch"].token_top1_accuracy == 0.0
    assert breakdown["Pitch"].type_top1_accuracy == 1.0
    assert breakdown["Position"].count == 1
    assert breakdown["Position"].token_top1_accuracy == 1.0
    assert breakdown["Technique"].count == 1
    assert breakdown["Technique"].token_top1_accuracy == 1.0
    assert breakdown["PAD"].count == 0
    assert breakdown["PAD"].objective_nll is None
    assert breakdown["PAD"].objective_perplexity is None
    assert breakdown["Bar"].count == 0
    assert breakdown["Bar"].token_top5_accuracy is None


def test_prompt_targets_are_excluded_from_total_and_type_metrics() -> None:
    token_types = (
        "PAD",
        "BOS",
        "Tonic",
        "Mode",
        "Bar",
        "Pitch",
        "Technique",
    )
    accumulator = TrainingMetricsAccumulator(
        pad_token_id=0,
        technique_token_ids=(6,),
        token_type_by_id=token_types,
        ignored_target_token_ids=(2, 3),
    )
    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 2.0, 1.0, 5.0, 0.0, -1.0],
            ]
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([[2, 3, 4]], dtype=torch.long)
    unknown = torch.zeros((1, 3), dtype=torch.bool)

    accumulator.update(logits, targets, unknown)
    report = accumulator.snapshot()

    expected = torch.nn.functional.cross_entropy(
        logits[:, 2, :], torch.tensor([4]), reduction="mean"
    )
    assert accumulator.ignored_target_token_ids == (2, 3)
    assert report.total.count == 1
    assert report.total.objective_nll == pytest.approx(float(expected))
    assert report.by_target_type["Bar"].count == 1
    assert report.by_target_type["Tonic"].count == 0
    assert report.by_target_type["Tonic"].objective_nll is None
    assert report.by_target_type["Mode"].count == 0
    assert report.by_target_type["Mode"].token_top1_accuracy is None


def test_accumulation_across_batches_is_token_weighted() -> None:
    logits, targets, unknown = _batch()
    combined = _accumulator()
    split = _accumulator()

    combined.update(logits, targets, unknown)
    split.update(logits[:, :1], targets[:, :1], unknown[:, :1])
    split.update(logits[:, 1:], targets[:, 1:], unknown[:, 1:])

    combined_report = combined.snapshot()
    split_report = split.snapshot()
    assert combined_report.batches == 1
    assert split_report.batches == 2
    assert split_report.total == combined_report.total
    assert split_report.post_duration_unknown == (
        combined_report.post_duration_unknown
    )
    assert split_report.by_target_type == combined_report.by_target_type


def test_update_never_retains_or_populates_gradients() -> None:
    accumulator = _accumulator()
    logits, targets, unknown = _batch()
    logits.requires_grad_(True)

    accumulator.update(logits, targets, unknown)

    assert logits.grad is None
    assert accumulator.snapshot().total.count == 3


def test_all_padding_batch_is_counted_but_has_absent_metrics() -> None:
    accumulator = _accumulator()
    logits = torch.full((2, 3, len(TOKEN_TYPES)), float("nan"))
    targets = torch.zeros((2, 3), dtype=torch.long)
    unknown = torch.zeros((2, 3), dtype=torch.bool)

    accumulator.update(logits, targets, unknown)
    report = accumulator.snapshot()

    assert report.batches == 1
    assert report.total.count == 0
    assert report.total.full_vocab_nll is None
    assert report.total.full_vocab_perplexity is None
    assert report.total.objective_nll is None
    assert report.total.token_top1_accuracy is None
    assert report.post_duration_unknown.count == 0
    assert report.post_duration_unknown.type_top1_accuracy is None


def test_reset_clears_counts_without_changing_contract() -> None:
    accumulator = _accumulator()
    accumulator.update(*_batch())

    accumulator.reset()
    report = accumulator.snapshot()

    assert report.batches == 0
    assert report.total.count == 0
    assert set(report.by_target_type) == set(TOKEN_TYPES)
    assert accumulator.vocabulary_size == len(TOKEN_TYPES)
    assert accumulator.pad_token_id == 0
    assert accumulator.technique_token_ids == TECHNIQUE_IDS
    assert accumulator.ignored_target_token_ids == ()


def test_report_is_strict_jsonable_and_deterministic() -> None:
    accumulator = _accumulator()
    accumulator.update(*_batch())
    report = accumulator.snapshot()

    encoded = report.to_json()
    decoded = json.loads(encoded)

    assert decoded == report.to_dict()
    assert decoded["schema_version"] == METRICS_SCHEMA_VERSION
    assert list(decoded["by_target_type"]) == sorted(set(TOKEN_TYPES))
    assert "NaN" not in encoded and "Infinity" not in encoded


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: TrainingMetricsAccumulator(True, TECHNIQUE_IDS, TOKEN_TYPES),
            "pad_token_id",
        ),
        (
            lambda: TrainingMetricsAccumulator(7, TECHNIQUE_IDS, TOKEN_TYPES),
            "pad_token_id",
        ),
        (
            lambda: TrainingMetricsAccumulator(0, (), TOKEN_TYPES),
            "cannot be empty",
        ),
        (
            lambda: TrainingMetricsAccumulator(0, (3, 3), TOKEN_TYPES),
            "distinct",
        ),
        (
            lambda: TrainingMetricsAccumulator(0, (0, 3), TOKEN_TYPES),
            "PAD",
        ),
        (
            lambda: TrainingMetricsAccumulator(0, (1,), TOKEN_TYPES),
            "token type 'Technique'",
        ),
        (
            lambda: TrainingMetricsAccumulator(
                0, TECHNIQUE_IDS, TOKEN_TYPES, ignored_target_token_ids=(0,)
            ),
            "PAD",
        ),
        (
            lambda: TrainingMetricsAccumulator(
                0, TECHNIQUE_IDS, TOKEN_TYPES, ignored_target_token_ids=(1, 1)
            ),
            "distinct",
        ),
        (
            lambda: TrainingMetricsAccumulator(
                0, TECHNIQUE_IDS, TOKEN_TYPES, ignored_target_token_ids=(True,)
            ),
            "integer token ID",
        ),
        (
            lambda: TrainingMetricsAccumulator(
                0,
                TECHNIQUE_IDS,
                TOKEN_TYPES,
                ignored_target_token_ids=(len(TOKEN_TYPES),),
            ),
            "range",
        ),
        (
            lambda: TrainingMetricsAccumulator(
                0, TECHNIQUE_IDS, TOKEN_TYPES, ignored_target_token_ids="Tonic"
            ),
            "collection",
        ),
        (
            lambda: TrainingMetricsAccumulator(
                0, (2,), {0: "PAD", 2: "Technique"}
            ),
            "contiguous",
        ),
        (
            lambda: TrainingMetricsAccumulator(0, (3,), ("PAD", "", "X", "Technique")),
            "non-empty string",
        ),
    ],
)
def test_constructor_rejects_invalid_metric_contracts(
    factory: Callable[[], TrainingMetricsAccumulator], message: str
) -> None:
    with pytest.raises(TrainingMetricsError, match=message):
        factory()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("logit_rank", "logits must have shape"),
        ("logit_dtype", "floating-point"),
        ("vocabulary", "vocabulary"),
        ("target_rank", "targets must have shape"),
        ("target_dtype", "torch.long"),
        ("target_shape", "dimensions must match"),
        ("mask_rank", "unknown_mask must have shape"),
        ("mask_dtype", "must use torch.bool"),
        ("mask_shape", "dimensions must match targets"),
        ("target_range", "outside the vocabulary"),
        ("unknown_pad", "cannot mark a PAD"),
        ("unknown_technique", "cannot target a technique"),
        ("nonfinite", "non-finite"),
    ],
)
def test_update_rejects_invalid_batches(mutation: str, message: str) -> None:
    accumulator = _accumulator()
    logits, targets, unknown = _batch()
    if mutation == "logit_rank":
        logits = logits[0]
    elif mutation == "logit_dtype":
        logits = logits.long()
    elif mutation == "vocabulary":
        logits = logits[:, :, :-1]
    elif mutation == "target_rank":
        targets = targets[0]
    elif mutation == "target_dtype":
        targets = targets.float()
    elif mutation == "target_shape":
        targets = targets[:, :-1]
    elif mutation == "mask_rank":
        unknown = unknown[0]
    elif mutation == "mask_dtype":
        unknown = unknown.long()
    elif mutation == "mask_shape":
        unknown = unknown[:, :-1]
    elif mutation == "target_range":
        targets[0, 0] = len(TOKEN_TYPES)
    elif mutation == "unknown_pad":
        unknown[0, -1] = True
    elif mutation == "unknown_technique":
        targets[0, 0] = TECHNIQUE_IDS[0]
    else:
        logits[0, 0, 0] = float("inf")

    with pytest.raises(TrainingMetricsError, match=message):
        accumulator.update(logits, targets, unknown)


def test_update_rejects_unknown_mask_on_ignored_target() -> None:
    accumulator = TrainingMetricsAccumulator(
        0,
        TECHNIQUE_IDS,
        TOKEN_TYPES,
        ignored_target_token_ids=(2,),
    )
    logits, targets, unknown = _batch()

    with pytest.raises(TrainingMetricsError, match="ignored target"):
        accumulator.update(logits, targets, unknown)
