"""Project validated guitar-technique sidecars through Stage 1 transforms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import pretty_midi

from .config import ProcessingConfig
from .midi_io import exact_note_identity
from .preprocessing import (
    MidiPhrase,
    _phrase_index,
    _round_half_up,
    clone_note,
    quantization_step_seconds,
)
from .techniques import NoteRef, Technique, TechniqueSidecar, TechniqueType


TECHNIQUE_COVERAGE_COMPLETE = "COMPLETE"
TECHNIQUE_COVERAGE_UNLABELED = "UNLABELED"

FRAGMENT_TECHNIQUE_TYPES: tuple[str, ...] = (
    "DEAD_NOTE",
    "PALM_MUTE_ON",
    "PALM_MUTE_OFF",
    "SLIDE_UP",
    "SLIDE_DOWN",
    "VIBRATO",
)
_FRAGMENT_TECHNIQUE_ORDER = {
    technique: index for index, technique in enumerate(FRAGMENT_TECHNIQUE_TYPES)
}


class TechniqueProjectionError(ValueError):
    """Raised when annotations cannot follow Stage 1 transforms unambiguously."""


@dataclass(frozen=True, slots=True)
class FragmentTechnique:
    """One Stage 2 technique token attached to a canonical fragment note."""

    type: str
    note_index: int

    def __post_init__(self) -> None:
        if self.type not in _FRAGMENT_TECHNIQUE_ORDER:
            raise TechniqueProjectionError(
                f"Unsupported fragment technique type: {self.type!r}."
            )
        if isinstance(self.note_index, bool) or not isinstance(self.note_index, int):
            raise TechniqueProjectionError("note_index must be an integer.")
        if self.note_index < 0:
            raise TechniqueProjectionError("note_index cannot be negative.")

    def as_dict(self) -> dict[str, object]:
        """Return the strict JSON representation consumed by Stage 2."""

        return {"type": self.type, "note_index": self.note_index}


def project_phrase_techniques(
    *,
    source_midi: pretty_midi.PrettyMIDI,
    instrument_index: int,
    sidecar: TechniqueSidecar | None,
    normalized_instrument: pretty_midi.Instrument,
    phrase: MidiPhrase,
    tempo_bpm: float,
    processing: ProcessingConfig,
    semitones: int,
    pitch_min: int,
    pitch_max: int,
) -> tuple[FragmentTechnique, ...] | None:
    """Project source-note semantics onto one transposed Stage 1 phrase.

    ``None`` means that a slide target would leave the configured pitch range,
    so the complete transposed variant must be skipped.  An absent sidecar is
    intentionally different from a complete sidecar with no annotations, but
    both produce an empty tuple here; their coverage is stored separately.
    """

    if sidecar is None:
        return ()
    if not 0 <= instrument_index < len(source_midi.instruments):
        raise TechniqueProjectionError("instrument_index is outside source MIDI.")
    if sidecar.instrument_index != instrument_index:
        raise TechniqueProjectionError(
            "Technique sidecar instrument does not match the selected source track."
        )
    if not math.isfinite(tempo_bpm) or tempo_bpm <= 0:
        raise TechniqueProjectionError("tempo_bpm must be positive and finite.")

    annotations = sidecar.annotations_by_note
    projected = _normalized_note_techniques(
        source_midi=source_midi,
        instrument_index=instrument_index,
        annotations=annotations,
        normalized_instrument=normalized_instrument,
        tempo_bpm=tempo_bpm,
        processing=processing,
    )
    phrase_notes = _phrase_note_techniques(
        normalized_instrument=normalized_instrument,
        normalized_annotations=projected,
        phrase=phrase,
        tempo_bpm=tempo_bpm,
        processing=processing,
    )

    canonical_notes = sorted(
        phrase.midi.instruments[0].notes,
        key=lambda note: (note.start, note.pitch, note.end, note.velocity),
    )
    by_identity = {
        exact_note_identity(note): techniques for note, techniques in phrase_notes
    }
    if len(by_identity) != len(phrase_notes):
        raise TechniqueProjectionError(
            "Phrase clipping created ambiguous duplicate note identities."
        )
    if set(by_identity) != {exact_note_identity(note) for note in canonical_notes}:
        raise TechniqueProjectionError(
            "Technique projection no longer matches the normalized phrase notes."
        )

    result: list[FragmentTechnique] = []
    palm_muted = False
    for note_index, note in enumerate(canonical_notes):
        note_techniques = by_identity[exact_note_identity(note)]
        kinds = {technique.type for technique in note_techniques}
        note_is_muted = TechniqueType.PALM_MUTE in kinds
        if note_is_muted != palm_muted:
            result.append(
                FragmentTechnique(
                    "PALM_MUTE_ON" if note_is_muted else "PALM_MUTE_OFF",
                    note_index,
                )
            )
            palm_muted = note_is_muted

        for technique in note_techniques:
            if technique.type == TechniqueType.PALM_MUTE:
                continue
            if technique.type in {TechniqueType.SLIDE_UP, TechniqueType.SLIDE_DOWN}:
                assert technique.target_pitch is not None
                transposed_target = technique.target_pitch + semitones
                if not pitch_min <= transposed_target <= pitch_max:
                    return None
            result.append(FragmentTechnique(technique.type.value, note_index))

    result.sort(
        key=lambda item: (item.note_index, _FRAGMENT_TECHNIQUE_ORDER[item.type])
    )
    return tuple(result)


def source_technique_counts(
    sidecar: TechniqueSidecar | None,
) -> dict[str, int]:
    """Count canonical per-note memberships in a validated source sidecar."""

    counts = {technique.value: 0 for technique in TechniqueType}
    if sidecar is None:
        return counts
    for entry in sidecar.note_techniques:
        for technique in entry.techniques:
            counts[technique.type.value] += 1
    return counts


def _normalized_note_techniques(
    *,
    source_midi: pretty_midi.PrettyMIDI,
    instrument_index: int,
    annotations: Mapping[NoteRef, tuple[Technique, ...]],
    normalized_instrument: pretty_midi.Instrument,
    tempo_bpm: float,
    processing: ProcessingConfig,
) -> dict[tuple[int, int, float, float], tuple[Technique, ...]]:
    source_instrument = source_midi.instruments[instrument_index]
    if not source_instrument.notes:
        raise TechniqueProjectionError("Selected source instrument has no notes.")
    offset = (
        min(note.start for note in source_instrument.notes)
        if processing.remove_initial_silence
        else 0.0
    )
    step = (
        quantization_step_seconds(tempo_bpm, processing.subdivisions_per_beat)
        if processing.quantize
        else None
    )

    unique_source: dict[
        tuple[int, int, float, float], pretty_midi.Note
    ] = {}
    for note in source_instrument.notes:
        unique_source.setdefault(exact_note_identity(note), note)

    projected: dict[
        tuple[int, int, float, float], tuple[Technique, ...]
    ] = {}
    for source_note in unique_source.values():
        transformed = clone_note(
            source_note,
            start=max(0.0, source_note.start - offset),
            end=source_note.end - offset,
        )
        if step is not None:
            onset_cells = _round_half_up(transformed.start / step)
            duration_cells = max(
                1, _round_half_up((transformed.end - transformed.start) / step)
            )
            start = onset_cells * step
            transformed = clone_note(
                transformed,
                start=start,
                end=start + duration_cells * step,
            )
        identity = exact_note_identity(transformed)
        if identity in projected:
            raise TechniqueProjectionError(
                "Preprocessing created an ambiguous annotated note identity."
            )
        note_ref = NoteRef(
            onset_tick=int(source_midi.time_to_tick(source_note.start)),
            end_tick=int(source_midi.time_to_tick(source_note.end)),
            pitch=int(source_note.pitch),
            velocity=int(source_note.velocity),
        )
        projected[identity] = annotations.get(note_ref, ())

    normalized_identities = {
        exact_note_identity(note) for note in normalized_instrument.notes
    }
    if len(normalized_identities) != len(normalized_instrument.notes):
        raise TechniqueProjectionError(
            "Normalized instrument contains ambiguous duplicate note identities."
        )
    if set(projected) != normalized_identities:
        raise TechniqueProjectionError(
            "Technique projection does not match normalized source notes."
        )
    return projected


def _phrase_note_techniques(
    *,
    normalized_instrument: pretty_midi.Instrument,
    normalized_annotations: Mapping[
        tuple[int, int, float, float], tuple[Technique, ...]
    ],
    phrase: MidiPhrase,
    tempo_bpm: float,
    processing: ProcessingConfig,
) -> list[tuple[pretty_midi.Note, tuple[Technique, ...]]]:
    window_duration = processing.phrase_bars * 4 * 60.0 / tempo_bpm
    phrase_start = phrase.phrase_index * window_duration
    phrase_end = phrase_start + window_duration
    projected: list[tuple[pretty_midi.Note, tuple[Technique, ...]]] = []
    for note in normalized_instrument.notes:
        if _phrase_index(note.start, window_duration) != phrase.phrase_index:
            continue
        local_start = max(0.0, note.start - phrase_start)
        local_end = min(note.end, phrase_end) - phrase_start
        if local_end <= local_start:
            continue
        local_note = clone_note(note, start=local_start, end=local_end)
        projected.append(
            (local_note, normalized_annotations[exact_note_identity(note)])
        )
    return projected


__all__ = [
    "FRAGMENT_TECHNIQUE_TYPES",
    "FragmentTechnique",
    "TECHNIQUE_COVERAGE_COMPLETE",
    "TECHNIQUE_COVERAGE_UNLABELED",
    "TechniqueProjectionError",
    "project_phrase_techniques",
    "source_technique_counts",
]
