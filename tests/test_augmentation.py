"""Tests for deterministic, range-safe pitch transposition."""

from __future__ import annotations

import pretty_midi
import pytest

from midi_idea_generator.augmentation import (
    augmentation_offsets,
    transpose_instrument,
    transpose_midi,
)
from midi_idea_generator.config import AugmentationConfig
from midi_idea_generator.midi_io import canonical_pitch_bend_range_controls
from midi_idea_generator.preprocessing import build_single_track_midi


def _snapshot(instrument) -> list[tuple[int, int, float, float]]:
    return [
        (note.pitch, note.velocity, note.start, note.end)
        for note in instrument.notes
    ]


def test_transpose_instrument_changes_only_pitch_and_does_not_mutate_source(
    make_instrument,
) -> None:
    source = make_instrument(
        [(30, 0.0, 0.5, 70), (100, 0.5, 1.25, 105)],
        program=27,
        name="synthetic guitar",
    )
    before = _snapshot(source)

    transposed = transpose_instrument(
        source,
        semitones=5,
        pitch_min=21,
        pitch_max=108,
    )

    assert transposed is not None
    assert _snapshot(source) == before
    assert transposed is not source
    assert transposed.program == source.program
    assert transposed.name == source.name
    assert [note.pitch for note in transposed.notes] == [35, 105]
    assert [note.velocity for note in transposed.notes] == [70, 105]
    assert [(note.start, note.end) for note in transposed.notes] == [
        (0.0, 0.5),
        (0.5, 1.25),
    ]


@pytest.mark.parametrize("semitones", [-10, 9])
def test_transpose_instrument_rejects_entire_out_of_range_variant(
    make_instrument,
    semitones: int,
) -> None:
    source = make_instrument([(30, 0.0, 0.5), (100, 0.5, 1.0)])
    before = _snapshot(source)

    result = transpose_instrument(
        source,
        semitones=semitones,
        pitch_min=21,
        pitch_max=108,
    )

    assert result is None
    assert _snapshot(source) == before


def test_transpose_midi_preserves_tempo_meter_and_note_timing(
    make_instrument,
) -> None:
    source = make_instrument([(60, 0.125, 0.625), (64, 1.0, 1.75)])
    midi = build_single_track_midi(source, tempo_bpm=90.0)

    transposed = transpose_midi(
        midi,
        semitones=-3,
        pitch_min=21,
        pitch_max=108,
    )

    assert transposed is not None
    assert [note.pitch for note in midi.instruments[0].notes] == [60, 64]
    assert [note.pitch for note in transposed.instruments[0].notes] == [57, 61]
    assert [
        (note.start, note.end) for note in transposed.instruments[0].notes
    ] == [(0.125, 0.625), (1.0, 1.75)]
    _, tempi = transposed.get_tempo_changes()
    assert list(tempi) == pytest.approx([90.0])
    assert [
        (change.numerator, change.denominator, change.time)
        for change in transposed.time_signature_changes
    ] == [(4, 4, 0.0)]


def test_transposition_preserves_pitch_bend_curve_and_range(
    make_instrument,
) -> None:
    source = make_instrument([(60, 0.0, 1.0)])
    source.pitch_bends = [
        pretty_midi.PitchBend(pitch=4096, time=0.25),
        pretty_midi.PitchBend(pitch=0, time=0.75),
    ]
    source.control_changes = canonical_pitch_bend_range_controls()
    midi = build_single_track_midi(source, tempo_bpm=120.0)

    transposed = transpose_midi(
        midi,
        semitones=5,
        pitch_min=21,
        pitch_max=108,
    )

    assert transposed is not None
    output = transposed.instruments[0]
    assert [note.pitch for note in output.notes] == [65]
    assert [(bend.pitch, bend.time) for bend in output.pitch_bends] == [
        (4096, 0.25),
        (0, 0.75),
    ]
    assert [
        (change.number, change.value, change.time)
        for change in output.control_changes
    ] == [(101, 0, 0.0), (100, 0, 0.0), (6, 6, 0.0)]
    assert [note.pitch for note in source.notes] == [60]


def test_augmentation_offsets_are_ordered_and_limited_to_configured_splits() -> None:
    enabled = AugmentationConfig(
        enabled=True,
        min_semitones=-2,
        max_semitones=2,
        apply_to_splits=("train",),
    )
    disabled = AugmentationConfig(
        enabled=False,
        min_semitones=-2,
        max_semitones=2,
        apply_to_splits=("train",),
    )
    positive_only = AugmentationConfig(
        enabled=True,
        min_semitones=2,
        max_semitones=3,
        apply_to_splits=("train",),
    )

    assert augmentation_offsets(enabled, "train") == (-2, -1, 0, 1, 2)
    assert augmentation_offsets(enabled, "validation") == (0,)
    assert augmentation_offsets(disabled, "train") == (0,)
    assert augmentation_offsets(positive_only, "train") == (0, 2, 3)
