"""Deterministic source-level dataset splitting."""

from __future__ import annotations

import math
import random
from typing import Iterable

from .config import SplitConfig

SPLIT_ORDER = ("train", "validation", "test")


def _target_counts(num_sources: int, config: SplitConfig) -> dict[str, int]:
    ratios = {
        "train": config.train,
        "validation": config.validation,
        "test": config.test,
    }
    positive = [name for name in SPLIT_ORDER if ratios[name] > 0]
    counts = {name: 0 for name in SPLIT_ORDER}
    if num_sources == 0:
        return counts
    if num_sources < len(positive):
        for name in positive[:num_sources]:
            counts[name] = 1
        return counts

    raw = {name: ratios[name] * num_sources for name in SPLIT_ORDER}
    counts = {name: math.floor(raw[name]) for name in SPLIT_ORDER}
    remainder = num_sources - sum(counts.values())
    fractional_order = sorted(
        SPLIT_ORDER,
        key=lambda name: (-(raw[name] - counts[name]), SPLIT_ORDER.index(name)),
    )
    for name in fractional_order[:remainder]:
        counts[name] += 1

    # When enough sources exist, keep every requested evaluation split present.
    for empty_name in positive:
        if counts[empty_name] > 0:
            continue
        donors = [
            name
            for name in positive
            if counts[name] > 1
        ]
        if not donors:
            continue
        donor = max(
            donors,
            key=lambda name: (counts[name] - raw[name], counts[name], -SPLIT_ORDER.index(name)),
        )
        counts[donor] -= 1
        counts[empty_name] += 1
    return counts


def assign_source_splits(
    source_ids: Iterable[str], config: SplitConfig, seed: int
) -> dict[str, str]:
    """Assign each unique source to exactly one split.

    Sorting before the seeded shuffle makes the result independent of input
    discovery order. The function accepts source identifiers, never fragments,
    which makes leakage structurally difficult.
    """

    ordered = sorted(source_ids)
    if len(ordered) != len(set(ordered)):
        raise ValueError("Source identifiers must be unique before splitting")
    random.Random(seed).shuffle(ordered)
    counts = _target_counts(len(ordered), config)
    assignments: dict[str, str] = {}
    offset = 0
    for split in SPLIT_ORDER:
        split_sources = ordered[offset : offset + counts[split]]
        assignments.update({source: split for source in split_sources})
        offset += counts[split]
    if offset != len(ordered):
        raise RuntimeError("Internal split allocation did not consume all sources")
    return assignments
