"""Pure, small transformations for stage-one MIDI preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import pretty_midi

from .config import ProcessingConfig
from .midi_io import (
    PITCH_BEND_MAX,
    PITCH_BEND_MIN,
    canonical_pitch_bend_range_controls,
    exact_note_identity,
    set_midi_duration_seconds,
)

TICKS_PER_BEAT = 480
BEATS_PER_BAR_4_4 = 4


@dataclass(frozen=True, slots=True)
class MidiPhrase:
    """One fixed-window phrase produced from a normalized instrument."""

    phrase_index: int
    midi: pretty_midi.PrettyMIDI
    num_notes: int
    nominal_duration_seconds: float
    num_pitch_bend_events: int
    num_expressive_pitch_bend_events: int
    pitch_bend_range_semitones: int | None
    synthetic_initial_pitch_bend: bool
    synthetic_final_pitch_bend_reset: bool


class QuantizationCollisionError(ValueError):
    """Raised when distinct source notes become one ambiguous grid event."""


class PitchBendNormalizationError(ValueError):
    """Raised when a source curve cannot be represented at the canonical range."""


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


def clone_pitch_bend(
    bend: pretty_midi.PitchBend,
    *,
    pitch: int | None = None,
    time: float | None = None,
) -> pretty_midi.PitchBend:
    """Return a detached pitch-wheel event."""

    return pretty_midi.PitchBend(
        pitch=int(bend.pitch if pitch is None else pitch),
        time=float(bend.time if time is None else time),
    )


def clone_control_change(
    change: pretty_midi.ControlChange,
    *,
    time: float | None = None,
) -> pretty_midi.ControlChange:
    """Return a detached controller event."""

    return pretty_midi.ControlChange(
        number=int(change.number),
        value=int(change.value),
        time=float(change.time if time is None else time),
    )


def clone_instrument(
    instrument: pretty_midi.Instrument,
    notes: Iterable[pretty_midi.Note] | None = None,
    pitch_bends: Iterable[pretty_midi.PitchBend] | None = None,
    control_changes: Iterable[pretty_midi.ControlChange] | None = None,
) -> pretty_midi.Instrument:
    """Return a detached copy of every supported Stage 1 event."""

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
    source_bends = instrument.pitch_bends if pitch_bends is None else pitch_bends
    cloned.pitch_bends = sorted(
        (clone_pitch_bend(bend) for bend in source_bends),
        key=lambda bend: bend.time,
    )
    source_controls = (
        instrument.control_changes if control_changes is None else control_changes
    )
    cloned.control_changes = sorted(
        (clone_control_change(change) for change in source_controls),
        key=lambda change: change.time,
    )
    return cloned


def _seconds_per_tick(tempo_bpm: float, resolution: int) -> float:
    if not math.isfinite(tempo_bpm) or tempo_bpm <= 0:
        raise ValueError("tempo_bpm must be positive and finite")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    return 60.0 / tempo_bpm / resolution


def normalize_pitch_bends(
    instrument: pretty_midi.Instrument,
    *,
    tempo_bpm: float,
    resolution: int,
    source_range_semitones: float | None,
    canonical_range_semitones: int = 6,
) -> pretty_midi.Instrument:
    """Normalize a curve to +/-6 semitones and one wheel event per tick."""

    expressive = any(bend.pitch != 0 for bend in instrument.pitch_bends)
    if not expressive:
        # Neutral exporter bookkeeping carries no musical state.
        return clone_instrument(
            instrument,
            pitch_bends=[],
            control_changes=[],
        )
    if source_range_semitones is None or source_range_semitones <= 0:
        raise PitchBendNormalizationError(
            "Expressive pitch bends require a positive explicit source range"
        )
    if canonical_range_semitones != 6:
        raise PitchBendNormalizationError(
            "Stage 1 supports only a canonical +/-6-semitone range"
        )

    seconds_per_tick = _seconds_per_tick(tempo_bpm, resolution)
    by_tick: dict[int, pretty_midi.PitchBend] = {}
    for bend in instrument.pitch_bends:
        semitones = float(bend.pitch) * source_range_semitones / 8192.0
        if abs(semitones) > canonical_range_semitones + 1e-9:
            raise PitchBendNormalizationError(
                "Pitch-bend excursion exceeds the canonical +/-6-semitone range"
            )
        normalized_pitch = int(
            round(semitones * 8192.0 / canonical_range_semitones)
        )
        # +8192 denotes the exact positive endpoint mathematically, while the
        # MIDI wire format's positive endpoint is +8191.
        if normalized_pitch == 8192:
            normalized_pitch = PITCH_BEND_MAX
        if not PITCH_BEND_MIN <= normalized_pitch <= PITCH_BEND_MAX:
            raise PitchBendNormalizationError(
                "Normalized pitch bend would exceed the MIDI wheel range"
            )
        tick = _round_half_up(float(bend.time) / seconds_per_tick)
        # Source order is stable; the final event is the effective state when
        # exporters emit several wheel messages at the same tick.
        by_tick[tick] = pretty_midi.PitchBend(
            pitch=normalized_pitch,
            time=tick * seconds_per_tick,
        )

    return clone_instrument(
        instrument,
        pitch_bends=[by_tick[tick] for tick in sorted(by_tick)],
        control_changes=canonical_pitch_bend_range_controls(
            canonical_range_semitones
        ),
    )


def deduplicate_exact_notes(
    instrument: pretty_midi.Instrument,
) -> pretty_midi.Instrument:
    """Collapse only note events already identical in the source MIDI."""

    unique_notes: dict[tuple[int, int, float, float], pretty_midi.Note] = {}
    for note in instrument.notes:
        unique_notes.setdefault(exact_note_identity(note), note)
    return clone_instrument(instrument, unique_notes.values())


def remove_initial_silence(
    instrument: pretty_midi.Instrument,
    *,
    tempo_bpm: float = 120.0,
    resolution: int = TICKS_PER_BEAT,
) -> pretty_midi.Instrument:
    """Shift notes and their effective pitch-wheel state to the first onset."""

    if not instrument.notes:
        return clone_instrument(instrument)
    offset = min(note.start for note in instrument.notes)
    seconds_per_tick = _seconds_per_tick(tempo_bpm, resolution)
    offset_tick = _round_half_up(offset / seconds_per_tick)
    shifted = (
        clone_note(
            note,
            start=max(0.0, note.start - offset),
            end=note.end - offset,
        )
        for note in instrument.notes
    )
    bend_by_tick: dict[int, int] = {}
    effective_pitch = 0
    for bend in instrument.pitch_bends:
        source_tick = _round_half_up(bend.time / seconds_per_tick)
        if source_tick <= offset_tick:
            effective_pitch = int(bend.pitch)
        if source_tick >= offset_tick:
            bend_by_tick[source_tick - offset_tick] = int(bend.pitch)
    if 0 not in bend_by_tick and effective_pitch != 0:
        bend_by_tick[0] = effective_pitch
    shifted_bends = [
        pretty_midi.PitchBend(
            pitch=bend_by_tick[tick], time=tick * seconds_per_tick
        )
        for tick in sorted(bend_by_tick)
    ]
    return clone_instrument(
        instrument,
        shifted,
        pitch_bends=shifted_bends,
    )


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
    identities: set[tuple[int, int, float, float]] = set()
    for note in instrument.notes:
        onset_cells = _round_half_up(note.start / step)
        duration_cells = max(1, _round_half_up((note.end - note.start) / step))
        start = onset_cells * step
        quantized_note = clone_note(
            note, start=start, end=start + duration_cells * step
        )
        identity = exact_note_identity(quantized_note)
        if identity in identities:
            raise QuantizationCollisionError(
                "Quantization mapped distinct notes to the same pitch, velocity, "
                "onset, and duration"
            )
        identities.add(identity)
        quantized.append(quantized_note)
    return clone_instrument(instrument, quantized)


def normalize_instrument(
    instrument: pretty_midi.Instrument,
    tempo_bpm: float,
    config: ProcessingConfig,
    *,
    resolution: int = TICKS_PER_BEAT,
    source_pitch_bend_range_semitones: float | None = None,
    canonical_pitch_bend_range_semitones: int = 6,
) -> pretty_midi.Instrument:
    """Normalize notes and pitch-wheel curves without mutating the source."""

    normalized = deduplicate_exact_notes(instrument)
    normalized = normalize_pitch_bends(
        normalized,
        tempo_bpm=tempo_bpm,
        resolution=resolution,
        source_range_semitones=source_pitch_bend_range_semitones,
        canonical_range_semitones=canonical_pitch_bend_range_semitones,
    )
    if config.remove_initial_silence:
        normalized = remove_initial_silence(
            normalized,
            tempo_bpm=tempo_bpm,
            resolution=resolution,
        )
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


def _slice_pitch_bends(
    instrument: pretty_midi.Instrument,
    *,
    phrase_start_tick: int,
    phrase_end_tick: int,
    seconds_per_tick: float,
    canonical_range_semitones: int,
) -> tuple[
    list[pretty_midi.PitchBend],
    list[pretty_midi.ControlChange],
    bool,
    bool,
]:
    """Create an autonomous half-open pitch-wheel window."""

    source_by_tick: dict[int, int] = {}
    for bend in instrument.pitch_bends:
        tick = _round_half_up(bend.time / seconds_per_tick)
        source_by_tick[tick] = int(bend.pitch)

    effective_at_start = 0
    for tick in sorted(source_by_tick):
        pitch = source_by_tick[tick]
        if tick <= phrase_start_tick:
            effective_at_start = pitch
        else:
            break

    local_by_tick: dict[int, int] = {
        tick - phrase_start_tick: pitch
        for tick, pitch in source_by_tick.items()
        if phrase_start_tick <= tick < phrase_end_tick
    }
    synthetic_initial = False
    if 0 not in local_by_tick and effective_at_start != 0:
        local_by_tick[0] = effective_at_start
        synthetic_initial = True

    effective_at_end = effective_at_start
    for local_tick in sorted(local_by_tick):
        effective_at_end = local_by_tick[local_tick]
    synthetic_final_reset = False
    if effective_at_end != 0:
        local_by_tick[phrase_end_tick - phrase_start_tick] = 0
        synthetic_final_reset = True

    if not any(pitch != 0 for pitch in local_by_tick.values()):
        return [], [], False, False
    bends = [
        pretty_midi.PitchBend(
            pitch=local_by_tick[tick], time=tick * seconds_per_tick
        )
        for tick in sorted(local_by_tick)
    ]
    controls = canonical_pitch_bend_range_controls(
        canonical_range_semitones
    )
    return bends, controls, synthetic_initial, synthetic_final_reset


def split_instrument_into_phrases(
    instrument: pretty_midi.Instrument,
    tempo_bpm: float,
    config: ProcessingConfig,
    *,
    resolution: int = TICKS_PER_BEAT,
    source_duration_seconds: float | None = None,
    canonical_pitch_bend_range_semitones: int = 6,
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
    seconds_per_tick = _seconds_per_tick(tempo_bpm, resolution)
    window_duration = (
        config.phrase_bars * BEATS_PER_BAR_4_4 * seconds_per_beat
    )
    window_ticks = config.phrase_bars * BEATS_PER_BAR_4_4 * resolution
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
        phrase_start_tick = phrase_index * window_ticks
        phrase_end_tick = phrase_start_tick + window_ticks
        (
            phrase_bends,
            phrase_controls,
            synthetic_initial,
            synthetic_final_reset,
        ) = _slice_pitch_bends(
            instrument,
            phrase_start_tick=phrase_start_tick,
            phrase_end_tick=phrase_end_tick,
            seconds_per_tick=seconds_per_tick,
            canonical_range_semitones=canonical_pitch_bend_range_semitones,
        )
        phrase_instrument = clone_instrument(
            instrument,
            notes,
            pitch_bends=phrase_bends,
            control_changes=phrase_controls,
        )
        expressive_bends = sum(
            bend.pitch != 0 for bend in phrase_instrument.pitch_bends
        )
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
                num_pitch_bend_events=len(phrase_instrument.pitch_bends),
                num_expressive_pitch_bend_events=expressive_bends,
                pitch_bend_range_semitones=(
                    canonical_pitch_bend_range_semitones
                    if expressive_bends
                    else None
                ),
                synthetic_initial_pitch_bend=synthetic_initial,
                synthetic_final_pitch_bend_reset=synthetic_final_reset,
            )
        )
    return phrases
