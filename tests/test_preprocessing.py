"""Unit tests for Stage 1 normalization and phrase extraction."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import mido
import pytest

from midi_idea_generator.config import ProcessingConfig
from midi_idea_generator.midi_io import (
    get_midi_duration_seconds,
    read_midi,
    write_midi,
)
from midi_idea_generator.preprocessing import (
    build_single_track_midi,
    quantization_step_seconds,
    quantize_instrument,
    remove_initial_silence,
    split_instrument_into_phrases,
)


def _note_snapshot(instrument) -> list[tuple[int, int, float, float]]:
    return [
        (note.pitch, note.velocity, note.start, note.end)
        for note in instrument.notes
    ]


def test_remove_initial_silence_shifts_copy_without_mutating_source(
    make_instrument,
) -> None:
    source = make_instrument(
        [(67, 3.0, 3.75, 70), (60, 2.0, 2.25, 100)],
        program=30,
        name="delayed guitar",
    )
    before = _note_snapshot(source)

    shifted = remove_initial_silence(source)

    assert _note_snapshot(source) == before
    assert shifted is not source
    assert all(
        shifted_note is not source_note
        for shifted_note, source_note in zip(shifted.notes, source.notes)
    )
    assert shifted.program == source.program
    assert shifted.name == source.name
    assert shifted.is_drum == source.is_drum
    shifted_by_pitch = {note.pitch: note for note in shifted.notes}
    assert shifted_by_pitch[60].start == pytest.approx(0.0)
    assert shifted_by_pitch[60].end == pytest.approx(0.25)
    assert shifted_by_pitch[67].start == pytest.approx(1.0)
    assert shifted_by_pitch[67].end == pytest.approx(1.75)
    assert shifted_by_pitch[60].velocity == 100


def test_quantization_uses_half_up_rounding_and_one_cell_minimum_duration(
    make_instrument,
) -> None:
    source = make_instrument(
        [
            # At 120 BPM with four subdivisions, one cell is 0.125 seconds.
            # Both onset and 0.1875-second duration lie exactly on half cells.
            (60, 0.0625, 0.25),
            # Just below the onset midpoint, with a duration far below one cell.
            (61, 0.062499, 0.072499),
        ]
    )
    before = _note_snapshot(source)

    quantized = quantize_instrument(
        source,
        tempo_bpm=120.0,
        subdivisions_per_beat=4,
    )

    assert quantization_step_seconds(120.0, 4) == pytest.approx(0.125)
    assert _note_snapshot(source) == before
    notes = {note.pitch: note for note in quantized.notes}
    assert notes[60].start == pytest.approx(0.125)
    assert notes[60].end - notes[60].start == pytest.approx(0.25)
    assert notes[61].start == pytest.approx(0.0)
    assert notes[61].end - notes[61].start == pytest.approx(0.125)


@pytest.mark.parametrize(
    ("tempo", "subdivisions", "message"),
    [
        (0.0, 4, "tempo_bpm must be positive"),
        (float("inf"), 4, "tempo_bpm must be positive"),
        (120.0, 0, "subdivisions_per_beat must be positive"),
    ],
)
def test_quantization_rejects_invalid_grid_parameters(
    tempo: float,
    subdivisions: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        quantization_step_seconds(tempo, subdivisions)


def test_phrase_windows_are_half_open_and_crossing_notes_are_clipped(
    make_instrument,
) -> None:
    source = make_instrument(
        [
            (60, 0.0, 0.5),
            (61, 3.5, 4.5),
            (62, 4.0, 4.25),
            (63, 7.75, 8.25),
            (64, 8.0, 8.5),
        ]
    )
    before = _note_snapshot(source)
    config = ProcessingConfig(
        phrase_bars=2,
        include_partial_final_phrase=True,
        min_notes_per_phrase=1,
    )

    phrases = split_instrument_into_phrases(source, tempo_bpm=120.0, config=config)

    assert _note_snapshot(source) == before
    assert [phrase.phrase_index for phrase in phrases] == [0, 1, 2]
    assert [phrase.num_notes for phrase in phrases] == [2, 2, 1]
    assert all(
        phrase.nominal_duration_seconds == pytest.approx(4.0)
        for phrase in phrases
    )
    first_notes = {note.pitch: note for note in phrases[0].midi.instruments[0].notes}
    second_notes = {note.pitch: note for note in phrases[1].midi.instruments[0].notes}
    third_notes = {note.pitch: note for note in phrases[2].midi.instruments[0].notes}
    assert first_notes[61].start == pytest.approx(3.5)
    assert first_notes[61].end == pytest.approx(4.0)
    # A note exactly at 4.0 belongs only to the second half-open window.
    assert 62 not in first_notes
    assert second_notes[62].start == pytest.approx(0.0)
    assert second_notes[62].end == pytest.approx(0.25)
    assert second_notes[63].start == pytest.approx(3.75)
    assert second_notes[63].end == pytest.approx(4.0)
    assert third_notes[64].start == pytest.approx(0.0)
    assert third_notes[64].end == pytest.approx(0.5)


def test_phrase_partial_window_and_minimum_note_behavior(make_instrument) -> None:
    source = make_instrument(
        [
            (60, 0.0, 0.5),
            (61, 3.5, 4.5),
            (62, 4.0, 4.25),
            (63, 7.75, 8.25),
            (64, 8.0, 8.5),
        ]
    )
    base = ProcessingConfig(
        phrase_bars=2,
        include_partial_final_phrase=True,
        min_notes_per_phrase=1,
    )

    without_partial = split_instrument_into_phrases(
        source,
        tempo_bpm=120.0,
        config=replace(base, include_partial_final_phrase=False),
    )
    requiring_two_notes = split_instrument_into_phrases(
        source,
        tempo_bpm=120.0,
        config=replace(base, min_notes_per_phrase=2),
    )

    assert [phrase.phrase_index for phrase in without_partial] == [0, 1]
    assert all(
        note.pitch != 64
        for phrase in without_partial
        for note in phrase.midi.instruments[0].notes
    )
    assert [phrase.phrase_index for phrase in requiring_two_notes] == [0, 1]


def test_short_material_requires_partial_phrases_to_be_enabled(
    make_instrument,
) -> None:
    source = make_instrument([(60, 0.0, 0.5)])
    config = ProcessingConfig(phrase_bars=2, include_partial_final_phrase=False)

    assert split_instrument_into_phrases(source, 120.0, config) == []
    included = split_instrument_into_phrases(
        source,
        120.0,
        replace(config, include_partial_final_phrase=True),
    )
    assert len(included) == 1


def test_near_boundary_float_is_snapped_to_the_next_phrase(
    make_instrument,
) -> None:
    tempo_bpm = 60_000_000 / 343_400
    window_duration = 2 * 4 * 60.0 / tempo_bpm
    represented_just_below = math.nextafter(window_duration, 0.0)
    source = make_instrument(
        [(60, 0.0, 0.25), (72, represented_just_below, window_duration + 0.25)]
    )
    config = ProcessingConfig(
        phrase_bars=2,
        include_partial_final_phrase=True,
        min_notes_per_phrase=1,
    )

    phrases = split_instrument_into_phrases(source, tempo_bpm, config)

    assert [phrase.phrase_index for phrase in phrases] == [0, 1]
    first_pitches = [note.pitch for note in phrases[0].midi.instruments[0].notes]
    second_notes = phrases[1].midi.instruments[0].notes
    assert first_pitches == [60]
    assert [note.pitch for note in second_notes] == [72]
    assert second_notes[0].start == pytest.approx(0.0)


def test_high_resolution_short_note_survives_validated_round_trip(
    tmp_path: Path,
    make_instrument,
) -> None:
    resolution = 960
    one_tick_seconds = 0.5 / resolution
    instrument = make_instrument([(60, 0.0, one_tick_seconds)])
    midi = build_single_track_midi(
        instrument, tempo_bpm=120.0, resolution=resolution
    )
    path = tmp_path / "one-tick.mid"

    write_midi(midi, path)
    restored = read_midi(path)

    assert restored.resolution == resolution
    assert len(restored.instruments) == 1
    assert len(restored.instruments[0].notes) == 1


def test_complete_phrase_uses_end_of_track_duration_and_preserves_trailing_rest(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "two-bars-with-rest.mid"
    raw = mido.MidiFile(type=0, ticks_per_beat=480)
    raw.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("set_tempo", tempo=500_000, time=0),
                mido.MetaMessage(
                    "time_signature", numerator=4, denominator=4, time=0
                ),
                mido.Message("note_on", note=60, velocity=90, time=0),
                mido.Message("note_off", note=60, velocity=0, time=480),
                mido.MetaMessage("end_of_track", time=3360),
            ]
        )
    )
    raw.save(source_path)
    source = read_midi(source_path)
    config = ProcessingConfig(
        phrase_bars=2,
        include_partial_final_phrase=False,
        min_notes_per_phrase=1,
    )

    phrases = split_instrument_into_phrases(
        source.instruments[0],
        tempo_bpm=120.0,
        config=config,
        resolution=source.resolution,
        source_duration_seconds=get_midi_duration_seconds(source),
    )
    output_path = tmp_path / "preserved-rest.mid"
    write_midi(phrases[0].midi, output_path)
    restored = read_midi(output_path)

    assert len(phrases) == 1
    assert get_midi_duration_seconds(source) == pytest.approx(4.0)
    assert get_midi_duration_seconds(restored) == pytest.approx(4.0)
    assert restored.get_end_time() == pytest.approx(0.5)
