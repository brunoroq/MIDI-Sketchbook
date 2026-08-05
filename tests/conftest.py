"""Shared factories for fully synthetic Stage 1 MIDI tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeAlias

import mido
import pretty_midi
import pytest


NoteSpec: TypeAlias = tuple[int, float, float] | tuple[int, float, float, int]
TickNoteSpec: TypeAlias = tuple[int, int, int] | tuple[int, int, int, int]


def _make_instrument(
    notes: Iterable[NoteSpec],
    *,
    program: int = 0,
    is_drum: bool = False,
    name: str = "synthetic",
) -> pretty_midi.Instrument:
    instrument = pretty_midi.Instrument(
        program=program,
        is_drum=is_drum,
        name=name,
    )
    for note_spec in notes:
        pitch, start, end, *velocity = note_spec
        instrument.notes.append(
            pretty_midi.Note(
                velocity=velocity[0] if velocity else 90,
                pitch=pitch,
                start=start,
                end=end,
            )
        )
    return instrument


def _write_midi_file(
    path: Path,
    instruments: Iterable[pretty_midi.Instrument],
    *,
    tempo_bpm: float = 120.0,
    time_signature: tuple[int, int] | None = (4, 4),
) -> Path:
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm, resolution=480)
    if time_signature is not None:
        numerator, denominator = time_signature
        midi.time_signature_changes.append(
            pretty_midi.TimeSignature(numerator, denominator, 0.0)
        )
    midi.instruments.extend(instruments)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
    if time_signature is None:
        # pretty_midi writes a default 4/4 meta-event even when its in-memory
        # time_signature_changes list is empty. Remove that generated event so
        # inspection sees a genuinely signature-less synthetic SMF.
        raw_midi = mido.MidiFile(path)
        for track in raw_midi.tracks:
            track[:] = [message for message in track if message.type != "time_signature"]
        raw_midi.save(path)
    return path


def _write_pitch_bend_midi_file(
    path: Path,
    *,
    bends: Iterable[tuple[int, int]],
    notes: Iterable[TickNoteSpec] = ((60, 0, 960),),
    range_events: Iterable[tuple[int, int]] = ((0, 6),),
    tempo_bpm: float = 120.0,
    ticks_per_beat: int = 480,
) -> Path:
    """Write an ordered raw-MIDI pitch-wheel fixture with explicit RPN events."""

    raw = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    conductor = mido.MidiTrack(
        [
            mido.MetaMessage(
                "set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0
            ),
            mido.MetaMessage(
                "time_signature", numerator=4, denominator=4, time=0
            ),
            mido.MetaMessage("end_of_track", time=0),
        ]
    )
    raw.tracks.append(conductor)

    absolute_events: list[tuple[int, int, mido.Message]] = [
        (0, 0, mido.Message("program_change", program=29, channel=0, time=0))
    ]
    for tick, semitones in range_events:
        absolute_events.extend(
            [
                (
                    tick,
                    1,
                    mido.Message(
                        "control_change", control=101, value=0, channel=0, time=0
                    ),
                ),
                (
                    tick,
                    2,
                    mido.Message(
                        "control_change", control=100, value=0, channel=0, time=0
                    ),
                ),
                (
                    tick,
                    3,
                    mido.Message(
                        "control_change",
                        control=6,
                        value=semitones,
                        channel=0,
                        time=0,
                    ),
                ),
            ]
        )
    for tick, pitch in bends:
        absolute_events.append(
            (
                tick,
                10,
                mido.Message("pitchwheel", pitch=pitch, channel=0, time=0),
            )
        )
    for note_spec in notes:
        pitch, start, end, *velocity = note_spec
        absolute_events.extend(
            [
                (
                    start,
                    20,
                    mido.Message(
                        "note_on",
                        note=pitch,
                        velocity=velocity[0] if velocity else 90,
                        channel=0,
                        time=0,
                    ),
                ),
                (
                    end,
                    19,
                    mido.Message(
                        "note_off", note=pitch, velocity=0, channel=0, time=0
                    ),
                ),
            ]
        )
    absolute_events.sort(key=lambda event: (event[0], event[1]))
    musical_track = mido.MidiTrack()
    previous_tick = 0
    for tick, _, message in absolute_events:
        musical_track.append(message.copy(time=tick - previous_tick))
        previous_tick = tick
    musical_track.append(mido.MetaMessage("end_of_track", time=0))
    raw.tracks.append(musical_track)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(path)
    return path


@pytest.fixture
def make_instrument() -> Callable[..., pretty_midi.Instrument]:
    """Return a concise factory for detached pretty_midi instruments."""

    return _make_instrument


@pytest.fixture
def write_midi_file() -> Callable[..., Path]:
    """Return a writer that creates original synthetic MIDI files in tmp_path."""

    return _write_midi_file


@pytest.fixture
def write_pitch_bend_midi_file() -> Callable[..., Path]:
    """Return a writer for channel-correct, ordered pitch-bend fixtures."""

    return _write_pitch_bend_midi_file
