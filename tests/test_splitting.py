"""Tests for deterministic source-level dataset splitting."""

from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from midi_idea_generator.config import SplitConfig
from midi_idea_generator.splitting import assign_source_splits


def test_source_splits_are_deterministic_and_independent_of_input_order() -> None:
    sources = [f"album/song-{index:02d}.mid" for index in range(20)]
    config = SplitConfig(train=0.8, validation=0.1, test=0.1)

    first = assign_source_splits(sources, config, seed=31415)
    reordered = assign_source_splits(reversed(sources), config, seed=31415)
    another_seed = assign_source_splits(sources, config, seed=27182)

    assert first == reordered
    assert first != another_seed
    assert set(first) == set(sources)
    assert Counter(first.values()) == {
        "train": 16,
        "validation": 2,
        "test": 2,
    }


def test_every_fragment_variant_inherits_exactly_one_source_split() -> None:
    sources = [f"source-{index}.mid" for index in range(12)]
    assignments = assign_source_splits(
        sources,
        SplitConfig(train=0.5, validation=0.25, test=0.25),
        seed=7,
    )
    synthetic_fragments = [
        (source, phrase_index, transpose)
        for source in sources
        for phrase_index in range(3)
        for transpose in (-1, 0, 1)
    ]
    observed_by_source: dict[str, set[str]] = defaultdict(set)
    sources_by_split: dict[str, set[str]] = defaultdict(set)
    for source, _phrase_index, _transpose in synthetic_fragments:
        split = assignments[source]
        observed_by_source[source].add(split)
        sources_by_split[split].add(source)

    assert all(splits == {assignments[source]} for source, splits in observed_by_source.items())
    assert sources_by_split["train"].isdisjoint(sources_by_split["validation"])
    assert sources_by_split["train"].isdisjoint(sources_by_split["test"])
    assert sources_by_split["validation"].isdisjoint(sources_by_split["test"])


def test_small_dataset_keeps_requested_splits_when_enough_sources_exist() -> None:
    assignments = assign_source_splits(
        ["a.mid", "b.mid", "c.mid"],
        SplitConfig(train=0.8, validation=0.1, test=0.1),
        seed=42,
    )

    assert Counter(assignments.values()) == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }


def test_source_splitting_rejects_duplicate_source_identifiers() -> None:
    with pytest.raises(ValueError, match="Source identifiers must be unique"):
        assign_source_splits(
            ["same.mid", "same.mid"],
            SplitConfig(),
            seed=42,
        )
