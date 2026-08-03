"""Shared factories for fully synthetic Stage 1 MIDI tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeAlias

import mido
import pretty_midi
import pytest


NoteSpec: TypeAlias = tuple[int, float, float] | tuple[int, float, float, int]


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


@pytest.fixture
def make_instrument() -> Callable[..., pretty_midi.Instrument]:
    """Return a concise factory for detached pretty_midi instruments."""

    return _make_instrument


@pytest.fixture
def write_midi_file() -> Callable[..., Path]:
    """Return a writer that creates original synthetic MIDI files in tmp_path."""

    return _write_midi_file
