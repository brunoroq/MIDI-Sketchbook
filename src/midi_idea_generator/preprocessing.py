"""Pure, small transformations for stage-one MIDI preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import pretty_midi

from .config import ProcessingConfig
from .midi_io import set_midi_duration_seconds

TICKS_PER_BEAT = 480
BEATS_PER_BAR_4_4 = 4


@dataclass(frozen=True, slots=True)
class MidiPhrase:
    """One fixed-window phrase produced from a normalized instrument."""

    phrase_index: int
    midi: pretty_midi.PrettyMIDI
    num_notes: int
    nominal_duration_seconds: float


def clone_note(
    note: pretty_midi.Note,
    *,
    pitch: int | None = None,
    start: float | None = None,
    end: float | None = None,
) -> pretty_midi.Note:
    """Return a detached copy of a note with selected fields replaced."""

    return pretty_midi.Note(
        velocity=int(note.velocity),
        pitch=int(note.pitch if pitch is None else pitch),
        start=float(note.start if start is None else start),
        end=float(note.end if end is None else end),
    )


def clone_instrument(
    instrument: pretty_midi.Instrument,
    notes: Iterable[pretty_midi.Note] | None = None,
) -> pretty_midi.Instrument:
    """Return a clean copy containing notes only, sorted deterministically."""

    cloned = pretty_midi.Instrument(
        program=int(instrument.program),
        is_drum=bool(instrument.is_drum),
        name=instrument.name,
    )
    source_notes = instrument.notes if notes is None else notes
    cloned.notes = sorted(
        (clone_note(note) for note in source_notes),
        key=lambda note: (note.start, note.pitch, note.end, note.velocity),
    )
    return cloned


def remove_initial_silence(
    instrument: pretty_midi.Instrument,
) -> pretty_midi.Instrument:
    """Shift the first note onset to zero without mutating the input."""

    if not instrument.notes:
        return clone_instrument(instrument)
    offset = min(note.start for note in instrument.notes)
    shifted = (
        clone_note(
            note,
            start=max(0.0, note.start - offset),
            end=note.end - offset,
        )
        for note in instrument.notes
    )
    return clone_instrument(instrument, shifted)


def _round_half_up(value: float) -> int:
    """Round a non-negative value to nearest integer with halves upward."""

    return int(math.floor(value + 0.5))


def quantization_step_seconds(tempo_bpm: float, subdivisions_per_beat: int) -> float:
    """Calculate the duration of one quantization cell."""

    if not math.isfinite(tempo_bpm) or tempo_bpm <= 0:
        raise ValueError("tempo_bpm must be positive and finite")
    if subdivisions_per_beat <= 0:
        raise ValueError("subdivisions_per_beat must be positive")
    return 60.0 / tempo_bpm / subdivisions_per_beat


def quantize_instrument(
    instrument: pretty_midi.Instrument,
    tempo_bpm: float,
    subdivisions_per_beat: int,
) -> pretty_midi.Instrument:
    """Quantize onsets and durations, preserving at least one time cell."""

    step = quantization_step_seconds(tempo_bpm, subdivisions_per_beat)
    quantized: list[pretty_midi.Note] = []
    for note in instrument.notes:
        onset_cells = _round_half_up(note.start / step)
        duration_cells = max(1, _round_half_up((note.end - note.start) / step))
        start = onset_cells * step
        quantized.append(clone_note(note, start=start, end=start + duration_cells * step))
    return clone_instrument(instrument, quantized)


def normalize_instrument(
    instrument: pretty_midi.Instrument,
    tempo_bpm: float,
    config: ProcessingConfig,
) -> pretty_midi.Instrument:
    """Remove unsupported events, optionally shift silence, and quantize."""

    normalized = clone_instrument(instrument)
    if config.remove_initial_silence:
        normalized = remove_initial_silence(normalized)
    if config.quantize:
        normalized = quantize_instrument(
            normalized,
            tempo_bpm=tempo_bpm,
            subdivisions_per_beat=config.subdivisions_per_beat,
        )
    return normalized


def build_single_track_midi(
    instrument: pretty_midi.Instrument,
    tempo_bpm: float,
    *,
    numerator: int = 4,
    denominator: int = 4,
    resolution: int = TICKS_PER_BEAT,
    nominal_duration_seconds: float | None = None,
) -> pretty_midi.PrettyMIDI:
    """Build a clean constant-tempo MIDI containing one instrument."""

    if tempo_bpm <= 0 or not math.isfinite(tempo_bpm):
        raise ValueError("tempo_bpm must be positive and finite")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    midi = pretty_midi.PrettyMIDI(
        initial_tempo=float(tempo_bpm), resolution=int(resolution)
    )
    midi.time_signature_changes.append(
        pretty_midi.TimeSignature(numerator, denominator, 0.0)
    )
    midi.instruments.append(clone_instrument(instrument))
    if nominal_duration_seconds is not None:
        set_midi_duration_seconds(midi, nominal_duration_seconds)
    return midi


def _phrase_window_count(
    duration: float, window_duration: float, include_partial: bool
) -> int:
    tolerance = window_duration * 1e-9
    full_windows = int(math.floor((duration + tolerance) / window_duration))
    covered = full_windows * window_duration
    has_partial = duration - covered > tolerance
    if include_partial and has_partial:
        return full_windows + 1
    return full_windows


def _phrase_index(onset: float, window_duration: float) -> int:
    ratio = onset / window_duration
    nearest_boundary = round(ratio)
    if math.isclose(ratio, nearest_boundary, rel_tol=0.0, abs_tol=1e-9):
        return int(nearest_boundary)
    return int(math.floor(ratio))


def split_instrument_into_phrases(
    instrument: pretty_midi.Instrument,
    tempo_bpm: float,
    config: ProcessingConfig,
    *,
    resolution: int = TICKS_PER_BEAT,
    source_duration_seconds: float | None = None,
) -> list[MidiPhrase]:
    """Split notes into fixed 4/4 windows based on their onset.

    Windows are half-open. A note beginning exactly at a boundary belongs to
    the next phrase. A note crossing a boundary remains only in the phrase in
    which it began and is clipped at that phrase's end.
    """

    if not instrument.notes:
        return []
    if tempo_bpm <= 0 or not math.isfinite(tempo_bpm):
        raise ValueError("tempo_bpm must be positive and finite")
    seconds_per_beat = 60.0 / tempo_bpm
    window_duration = (
        config.phrase_bars * BEATS_PER_BAR_4_4 * seconds_per_beat
    )
    duration = max(note.end for note in instrument.notes)
    if source_duration_seconds is not None:
        if source_duration_seconds < 0 or not math.isfinite(source_duration_seconds):
            raise ValueError("source_duration_seconds must be non-negative and finite")
        duration = max(duration, source_duration_seconds)
    window_count = _phrase_window_count(
        duration, window_duration, config.include_partial_final_phrase
    )
    if window_count == 0:
        return []

    grouped: list[list[pretty_midi.Note]] = [[] for _ in range(window_count)]
    for note in instrument.notes:
        phrase_index = _phrase_index(note.start, window_duration)
        if phrase_index < 0 or phrase_index >= window_count:
            continue
        phrase_start = phrase_index * window_duration
        phrase_end = phrase_start + window_duration
        local_start = max(0.0, note.start - phrase_start)
        local_end = min(note.end, phrase_end) - phrase_start
        if local_end <= local_start:
            continue
        grouped[phrase_index].append(
            clone_note(note, start=local_start, end=local_end)
        )

    phrases: list[MidiPhrase] = []
    for phrase_index, notes in enumerate(grouped):
        if len(notes) < config.min_notes_per_phrase:
            continue
        phrase_instrument = clone_instrument(instrument, notes)
        phrases.append(
            MidiPhrase(
                phrase_index=phrase_index,
                midi=build_single_track_midi(
                    phrase_instrument,
                    tempo_bpm,
                    resolution=resolution,
                    nominal_duration_seconds=window_duration,
                ),
                num_notes=len(notes),
                nominal_duration_seconds=window_duration,
            )
        )
    return phrases
