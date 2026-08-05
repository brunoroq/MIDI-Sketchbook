"""Pitch transposition that preserves relative pitch-wheel expression."""

from __future__ import annotations

import pretty_midi

from .config import AugmentationConfig
from .midi_io import get_constant_tempo, get_midi_duration_seconds
from .preprocessing import build_single_track_midi, clone_instrument, clone_note


def augmentation_offsets(config: AugmentationConfig, split: str) -> tuple[int, ...]:
    """Return deterministic offsets, always including the original at zero."""

    if not config.enabled or split not in config.apply_to_splits:
        return (0,)
    offsets = set(range(config.min_semitones, config.max_semitones + 1))
    offsets.add(0)
    return tuple(sorted(offsets))


def transpose_instrument(
    instrument: pretty_midi.Instrument,
    semitones: int,
    pitch_min: int,
    pitch_max: int,
) -> pretty_midi.Instrument | None:
    """Transpose notes while copying bends/RPN, or reject out-of-range notes."""

    transposed_pitches = [note.pitch + semitones for note in instrument.notes]
    if any(pitch < pitch_min or pitch > pitch_max for pitch in transposed_pitches):
        return None
    notes = (
        clone_note(note, pitch=note.pitch + semitones)
        for note in instrument.notes
    )
    return clone_instrument(instrument, notes)


def transpose_midi(
    midi: pretty_midi.PrettyMIDI,
    semitones: int,
    pitch_min: int,
    pitch_max: int,
) -> pretty_midi.PrettyMIDI | None:
    """Transpose a Stage 1 MIDI without changing timing or pitch-wheel curves."""

    if len(midi.instruments) != 1:
        raise ValueError("Stage-one phrase MIDI must contain exactly one instrument")
    tempo, _, issue = get_constant_tempo(midi, tolerance=0.01)
    if issue or tempo is None:
        raise ValueError(issue or "MIDI tempo could not be determined")
    instrument = transpose_instrument(
        midi.instruments[0], semitones, pitch_min, pitch_max
    )
    if instrument is None:
        return None
    return build_single_track_midi(
        instrument,
        tempo,
        resolution=midi.resolution,
        nominal_duration_seconds=get_midi_duration_seconds(midi),
    )
