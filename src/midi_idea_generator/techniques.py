"""Strict optional sidecars for symbolic guitar techniques.

The sidecar is deliberately independent from the MIDI preprocessing and
tokenization pipelines.  A file named ``riff.mid.techniques.json`` describes
only the selected ``pretty_midi`` instrument in ``riff.mid``.  Musical times
are absolute integer ticks in the source MIDI timebase; seconds are never used
as annotation identities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Never

import pretty_midi

from .midi_io import get_midi_duration_seconds


SIDECAR_SCHEMA_VERSION = 1
SIDECAR_SUFFIX = ".techniques.json"
MAX_SIDECAR_SIZE_BYTES = 1_048_576
MAX_ANNOTATIONS = 10_000
MAX_SLIDE_SEMITONES = 24

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ROOT_KEYS = {
    "schema_version",
    "source_midi",
    "source_sha256",
    "ticks_per_quarter",
    "instrument_index",
    "coverage",
    "note_techniques",
    "palm_mute_ranges",
}


class TechniqueSidecarError(ValueError):
    """Raised when a present guitar-technique sidecar is invalid."""


class TechniqueType(str, Enum):
    """Canonical domain techniques exposed to later pipeline stages."""

    DEAD_NOTE = "DEAD_NOTE"
    PALM_MUTE = "PALM_MUTE"
    SLIDE_UP = "SLIDE_UP"
    SLIDE_DOWN = "SLIDE_DOWN"
    VIBRATO = "VIBRATO"


_TECHNIQUE_ORDER = {
    technique: index for index, technique in enumerate(TechniqueType)
}


@dataclass(frozen=True, slots=True)
class NoteRef:
    """Exact identity of one canonical note in source-MIDI ticks."""

    onset_tick: int
    end_tick: int
    pitch: int
    velocity: int

    def __post_init__(self) -> None:
        _bounded_int(self.onset_tick, "NoteRef.onset_tick", minimum=0)
        _bounded_int(self.end_tick, "NoteRef.end_tick", minimum=1)
        if self.end_tick <= self.onset_tick:
            raise TechniqueSidecarError(
                "NoteRef.end_tick must be greater than onset_tick."
            )
        _bounded_int(self.pitch, "NoteRef.pitch", minimum=0, maximum=127)
        _bounded_int(self.velocity, "NoteRef.velocity", minimum=1, maximum=127)


@dataclass(frozen=True, slots=True)
class Technique:
    """One normalized technique attached to a note."""

    type: TechniqueType
    target_pitch: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.type, TechniqueType):
            raise TechniqueSidecarError("Technique.type must be a TechniqueType.")
        is_slide = self.type in {
            TechniqueType.SLIDE_UP,
            TechniqueType.SLIDE_DOWN,
        }
        if is_slide:
            if self.target_pitch is None:
                raise TechniqueSidecarError(
                    "Slide techniques require target_pitch."
                )
            _bounded_int(
                self.target_pitch,
                "Technique.target_pitch",
                minimum=0,
                maximum=127,
            )
        elif self.target_pitch is not None:
            raise TechniqueSidecarError(
                "Only slide techniques may define target_pitch."
            )


@dataclass(frozen=True, slots=True)
class NoteTechniques:
    """Canonical, ordered techniques attached to one exact note."""

    note: NoteRef
    techniques: tuple[Technique, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.note, NoteRef):
            raise TechniqueSidecarError("NoteTechniques.note must be a NoteRef.")
        if not isinstance(self.techniques, tuple) or not self.techniques:
            raise TechniqueSidecarError(
                "NoteTechniques.techniques must be a non-empty tuple."
            )
        if not all(isinstance(item, Technique) for item in self.techniques):
            raise TechniqueSidecarError(
                "NoteTechniques.techniques must contain Technique values."
            )
        kinds = tuple(technique.type for technique in self.techniques)
        if len(set(kinds)) != len(kinds):
            raise TechniqueSidecarError(
                "A note cannot contain duplicate canonical techniques."
            )
        expected = tuple(
            sorted(self.techniques, key=lambda item: _TECHNIQUE_ORDER[item.type])
        )
        if self.techniques != expected:
            raise TechniqueSidecarError(
                "NoteTechniques.techniques must use canonical order."
            )
        if TechniqueType.DEAD_NOTE in kinds and (
            TechniqueType.SLIDE_UP in kinds
            or TechniqueType.SLIDE_DOWN in kinds
            or TechniqueType.VIBRATO in kinds
        ):
            raise TechniqueSidecarError(
                "DEAD_NOTE cannot coexist with SLIDE or VIBRATO."
            )
        if (
            TechniqueType.SLIDE_UP in kinds
            and TechniqueType.SLIDE_DOWN in kinds
        ):
            raise TechniqueSidecarError("A note may contain only one SLIDE.")
        for technique in self.techniques:
            if technique.type == TechniqueType.SLIDE_UP:
                assert technique.target_pitch is not None
                interval = technique.target_pitch - self.note.pitch
                if not 1 <= interval <= MAX_SLIDE_SEMITONES:
                    raise TechniqueSidecarError(
                        "SLIDE_UP target must be 1-24 semitones above the note."
                    )
            elif technique.type == TechniqueType.SLIDE_DOWN:
                assert technique.target_pitch is not None
                interval = self.note.pitch - technique.target_pitch
                if not 1 <= interval <= MAX_SLIDE_SEMITONES:
                    raise TechniqueSidecarError(
                        "SLIDE_DOWN target must be 1-24 semitones below the note."
                    )


@dataclass(frozen=True, slots=True)
class PalmMuteRange:
    """Half-open source-tick interval whose note attacks are palm-muted."""

    start_tick: int
    end_tick: int

    def __post_init__(self) -> None:
        _bounded_int(self.start_tick, "PalmMuteRange.start_tick", minimum=0)
        _bounded_int(self.end_tick, "PalmMuteRange.end_tick", minimum=1)
        if self.end_tick <= self.start_tick:
            raise TechniqueSidecarError(
                "PalmMuteRange.end_tick must be greater than start_tick."
            )

    def contains_onset(self, onset_tick: int) -> bool:
        """Return whether an onset belongs to this half-open interval."""

        return self.start_tick <= onset_tick < self.end_tick


@dataclass(frozen=True, slots=True)
class TechniqueSidecar:
    """One fully validated and canonicalized optional sidecar."""

    path: Path
    fingerprint: str
    size_bytes: int
    source_midi: str
    source_sha256: str
    ticks_per_quarter: int
    instrument_index: int
    coverage: str
    note_techniques: tuple[NoteTechniques, ...]
    palm_mute_ranges: tuple[PalmMuteRange, ...]

    @property
    def sha256(self) -> str:
        """Alias clarifying that ``fingerprint`` is a SHA-256 digest."""

        return self.fingerprint

    @property
    def annotations_by_note(
        self,
    ) -> Mapping[NoteRef, tuple[Technique, ...]]:
        """Return an immutable lookup of canonical techniques by exact note."""

        return MappingProxyType(
            {entry.note: entry.techniques for entry in self.note_techniques}
        )

    def techniques_for(self, note: NoteRef) -> tuple[Technique, ...]:
        """Return canonical techniques for ``note``, or an empty tuple."""

        for entry in self.note_techniques:
            if entry.note == note:
                return entry.techniques
        return ()


def sidecar_path_for(midi_path: str | Path) -> Path:
    """Return the fixed sibling sidecar path for a MIDI path."""

    path = Path(midi_path).expanduser()
    if not path.name:
        raise TechniqueSidecarError("midi_path must name a MIDI file.")
    return path.with_name(f"{path.name}{SIDECAR_SUFFIX}")


def load_technique_sidecar(
    midi_path: str | Path,
    *,
    source_sha256: str,
    midi: pretty_midi.PrettyMIDI,
    instrument_index: int,
) -> TechniqueSidecar | None:
    """Load and validate an optional sidecar against its exact source MIDI.

    A missing sidecar is a supported legacy case and returns ``None``.  Any
    present but malformed, stale, ambiguous, or unsafe sidecar raises
    :class:`TechniqueSidecarError`; annotations are never silently discarded.
    Palm-mute ranges are retained and also expanded to ``PALM_MUTE`` membership
    in the returned canonical note annotations.
    """

    source_path = Path(midi_path).expanduser()
    sidecar_path = sidecar_path_for(source_path)
    if sidecar_path.is_symlink():
        raise TechniqueSidecarError(
            f"Technique sidecar cannot be a symlink: {sidecar_path}"
        )
    if not sidecar_path.exists():
        return None
    if not sidecar_path.is_file():
        raise TechniqueSidecarError(
            f"Technique sidecar is not a regular file: {sidecar_path}"
        )

    expected_source_sha = _require_sha256(source_sha256, "source_sha256")
    source_bytes = _read_regular_source(source_path)
    actual_source_sha = hashlib.sha256(source_bytes).hexdigest()
    if actual_source_sha != expected_source_sha:
        raise TechniqueSidecarError(
            "source_sha256 does not match the current MIDI file."
        )

    raw = _read_sidecar(sidecar_path)
    payload = _load_strict_json(raw)
    root = _require_mapping(payload, "sidecar")
    _require_exact_keys(root, _ROOT_KEYS, "sidecar")

    schema_version = _require_int(root["schema_version"], "schema_version")
    if schema_version != SIDECAR_SCHEMA_VERSION:
        raise TechniqueSidecarError(
            f"schema_version must be {SIDECAR_SCHEMA_VERSION}."
        )
    source_midi = _require_string(root["source_midi"], "source_midi")
    if source_midi != source_path.name or Path(source_midi).name != source_midi:
        raise TechniqueSidecarError(
            "source_midi must equal the exact sibling MIDI filename."
        )
    declared_source_sha = _require_sha256(
        root["source_sha256"], "sidecar.source_sha256"
    )
    if declared_source_sha != expected_source_sha:
        raise TechniqueSidecarError(
            "Sidecar source_sha256 does not match the current MIDI file."
        )
    ticks_per_quarter = _require_positive_int(
        root["ticks_per_quarter"], "ticks_per_quarter"
    )
    midi_resolution = _require_positive_int(
        getattr(midi, "resolution", None), "midi.resolution"
    )
    if ticks_per_quarter != midi_resolution:
        raise TechniqueSidecarError(
            "ticks_per_quarter does not match the MIDI resolution."
        )
    selected_index = _require_nonnegative_int(
        instrument_index, "instrument_index argument"
    )
    declared_index = _require_nonnegative_int(
        root["instrument_index"], "sidecar.instrument_index"
    )
    if declared_index != selected_index:
        raise TechniqueSidecarError(
            "Sidecar instrument_index does not match the selected instrument."
        )
    if selected_index >= len(midi.instruments):
        raise TechniqueSidecarError(
            "instrument_index is outside the MIDI instrument list."
        )
    coverage = _require_string(root["coverage"], "coverage")
    if coverage != "COMPLETE":
        raise TechniqueSidecarError("coverage must be 'COMPLETE'.")

    available_notes = _canonical_notes(midi, selected_index)
    structural_end_tick = int(
        midi.time_to_tick(get_midi_duration_seconds(midi))
    )
    if structural_end_tick < 0:
        raise TechniqueSidecarError("MIDI structural duration is invalid.")

    parsed_entries, raw_technique_count = _parse_note_techniques(
        root["note_techniques"], available_notes
    )
    palm_ranges = _parse_palm_mute_ranges(
        root["palm_mute_ranges"],
        available_notes,
        structural_end_tick,
    )
    if raw_technique_count + len(palm_ranges) > MAX_ANNOTATIONS:
        raise TechniqueSidecarError(
            f"Sidecar exceeds the maximum of {MAX_ANNOTATIONS} annotations."
        )

    expanded: dict[NoteRef, list[Technique]] = {
        entry.note: list(entry.techniques) for entry in parsed_entries
    }
    for note in available_notes:
        if any(interval.contains_onset(note.onset_tick) for interval in palm_ranges):
            expanded.setdefault(note, []).append(
                Technique(TechniqueType.PALM_MUTE)
            )
    canonical_entries = tuple(
        NoteTechniques(
            note=note,
            techniques=tuple(
                sorted(
                    techniques,
                    key=lambda item: _TECHNIQUE_ORDER[item.type],
                )
            ),
        )
        for note, techniques in sorted(
            expanded.items(), key=lambda item: _note_sort_key(item[0])
        )
    )
    canonical_annotation_count = sum(
        len(entry.techniques) for entry in canonical_entries
    )
    if canonical_annotation_count > MAX_ANNOTATIONS:
        raise TechniqueSidecarError(
            "Expanded note techniques exceed the maximum of "
            f"{MAX_ANNOTATIONS} annotations."
        )

    _verify_unchanged(source_path, source_bytes, "MIDI source")
    _verify_unchanged(sidecar_path, raw, "technique sidecar")
    fingerprint = hashlib.sha256(raw).hexdigest()
    return TechniqueSidecar(
        path=sidecar_path.resolve(),
        fingerprint=fingerprint,
        size_bytes=len(raw),
        source_midi=source_midi,
        source_sha256=declared_source_sha,
        ticks_per_quarter=ticks_per_quarter,
        instrument_index=declared_index,
        coverage=coverage,
        note_techniques=canonical_entries,
        palm_mute_ranges=palm_ranges,
    )


def _canonical_notes(
    midi: pretty_midi.PrettyMIDI, instrument_index: int
) -> tuple[NoteRef, ...]:
    notes: set[NoteRef] = set()
    for index, note in enumerate(midi.instruments[instrument_index].notes):
        try:
            onset_tick = int(midi.time_to_tick(float(note.start)))
            end_tick = int(midi.time_to_tick(float(note.end)))
            reference = NoteRef(
                onset_tick=onset_tick,
                end_tick=end_tick,
                pitch=int(note.pitch),
                velocity=int(note.velocity),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise TechniqueSidecarError(
                f"Selected instrument note {index} has invalid fields: {exc}"
            ) from exc
        notes.add(reference)
    return tuple(sorted(notes, key=_note_sort_key))


def _parse_note_techniques(
    value: object,
    available_notes: Sequence[NoteRef],
) -> tuple[tuple[NoteTechniques, ...], int]:
    raw_entries = _require_list(value, "note_techniques")
    if len(raw_entries) > MAX_ANNOTATIONS:
        raise TechniqueSidecarError(
            f"note_techniques exceeds the maximum of {MAX_ANNOTATIONS} entries."
        )
    available = set(available_notes)
    seen_notes: set[NoteRef] = set()
    entries: list[NoteTechniques] = []
    technique_count = 0
    for index, raw_entry in enumerate(raw_entries):
        name = f"note_techniques[{index}]"
        entry = _require_mapping(raw_entry, name)
        _require_exact_keys(entry, {"note", "techniques"}, name)
        note = _parse_note_ref(entry["note"], f"{name}.note")
        if note not in available:
            raise TechniqueSidecarError(
                f"{name}.note does not match a note in the selected instrument."
            )
        if note in seen_notes:
            raise TechniqueSidecarError(
                f"Duplicate note_techniques entry for {note}."
            )
        seen_notes.add(note)

        raw_techniques = _require_list(entry["techniques"], f"{name}.techniques")
        if not raw_techniques:
            raise TechniqueSidecarError(f"{name}.techniques cannot be empty.")
        technique_count += len(raw_techniques)
        if technique_count > MAX_ANNOTATIONS:
            raise TechniqueSidecarError(
                f"Sidecar exceeds the maximum of {MAX_ANNOTATIONS} annotations."
            )
        techniques = [
            _parse_technique(raw, note, f"{name}.techniques[{technique_index}]")
            for technique_index, raw in enumerate(raw_techniques)
        ]
        kinds = [technique.type for technique in techniques]
        if len(set(kinds)) != len(kinds):
            raise TechniqueSidecarError(f"{name} contains duplicate techniques.")
        if TechniqueType.SLIDE_UP in kinds and TechniqueType.SLIDE_DOWN in kinds:
            raise TechniqueSidecarError(f"{name} may contain only one SLIDE.")
        if TechniqueType.DEAD_NOTE in kinds and (
            TechniqueType.SLIDE_UP in kinds
            or TechniqueType.SLIDE_DOWN in kinds
            or TechniqueType.VIBRATO in kinds
        ):
            raise TechniqueSidecarError(
                f"{name}: DEAD_NOTE cannot coexist with SLIDE or VIBRATO."
            )
        ordered = tuple(
            sorted(techniques, key=lambda item: _TECHNIQUE_ORDER[item.type])
        )
        entries.append(NoteTechniques(note=note, techniques=ordered))
    return (
        tuple(sorted(entries, key=lambda item: _note_sort_key(item.note))),
        technique_count,
    )


def _parse_note_ref(value: object, name: str) -> NoteRef:
    note = _require_mapping(value, name)
    _require_exact_keys(
        note,
        {"onset_tick", "end_tick", "pitch", "velocity"},
        name,
    )
    return NoteRef(
        onset_tick=_require_nonnegative_int(note["onset_tick"], f"{name}.onset_tick"),
        end_tick=_require_positive_int(note["end_tick"], f"{name}.end_tick"),
        pitch=_bounded_int(note["pitch"], f"{name}.pitch", minimum=0, maximum=127),
        velocity=_bounded_int(
            note["velocity"], f"{name}.velocity", minimum=1, maximum=127
        ),
    )


def _parse_technique(value: object, note: NoteRef, name: str) -> Technique:
    technique = _require_mapping(value, name)
    raw_type = _require_string(technique.get("type"), f"{name}.type")
    if raw_type in {"DEAD_NOTE", "VIBRATO"}:
        _require_exact_keys(technique, {"type"}, name)
        kind = (
            TechniqueType.DEAD_NOTE
            if raw_type == "DEAD_NOTE"
            else TechniqueType.VIBRATO
        )
        return Technique(kind)
    if raw_type != "SLIDE":
        raise TechniqueSidecarError(
            f"{name}.type must be DEAD_NOTE, SLIDE, or VIBRATO."
        )
    _require_exact_keys(
        technique, {"type", "direction", "target_pitch"}, name
    )
    direction = _require_string(technique["direction"], f"{name}.direction")
    if direction not in {"UP", "DOWN"}:
        raise TechniqueSidecarError(f"{name}.direction must be UP or DOWN.")
    target = _bounded_int(
        technique["target_pitch"],
        f"{name}.target_pitch",
        minimum=0,
        maximum=127,
    )
    delta = target - note.pitch
    if direction == "UP":
        if not 1 <= delta <= MAX_SLIDE_SEMITONES:
            raise TechniqueSidecarError(
                f"{name} UP target must be 1-{MAX_SLIDE_SEMITONES} "
                "semitones above the note."
            )
        return Technique(TechniqueType.SLIDE_UP, target)
    if not 1 <= -delta <= MAX_SLIDE_SEMITONES:
        raise TechniqueSidecarError(
            f"{name} DOWN target must be 1-{MAX_SLIDE_SEMITONES} "
            "semitones below the note."
        )
    return Technique(TechniqueType.SLIDE_DOWN, target)


def _parse_palm_mute_ranges(
    value: object,
    available_notes: Sequence[NoteRef],
    structural_end_tick: int,
) -> tuple[PalmMuteRange, ...]:
    raw_ranges = _require_list(value, "palm_mute_ranges")
    if len(raw_ranges) > MAX_ANNOTATIONS:
        raise TechniqueSidecarError(
            f"palm_mute_ranges exceeds the maximum of {MAX_ANNOTATIONS} entries."
        )
    ranges: list[PalmMuteRange] = []
    for index, raw_range in enumerate(raw_ranges):
        name = f"palm_mute_ranges[{index}]"
        interval = _require_mapping(raw_range, name)
        _require_exact_keys(interval, {"start_tick", "end_tick"}, name)
        parsed = PalmMuteRange(
            start_tick=_require_nonnegative_int(
                interval["start_tick"], f"{name}.start_tick"
            ),
            end_tick=_require_positive_int(
                interval["end_tick"], f"{name}.end_tick"
            ),
        )
        if parsed.end_tick > structural_end_tick:
            raise TechniqueSidecarError(
                f"{name}.end_tick exceeds the MIDI structural duration."
            )
        if not any(parsed.contains_onset(note.onset_tick) for note in available_notes):
            raise TechniqueSidecarError(
                f"{name} does not affect any note onset."
            )
        ranges.append(parsed)
    ranges.sort(key=lambda item: (item.start_tick, item.end_tick))
    for previous, current in zip(ranges, ranges[1:], strict=False):
        if current.start_tick < previous.end_tick:
            raise TechniqueSidecarError("Palm-mute ranges cannot overlap.")
        if current.start_tick == previous.end_tick:
            raise TechniqueSidecarError(
                "Adjacent palm-mute ranges must be merged."
            )
    return tuple(ranges)


def _read_regular_source(path: Path) -> bytes:
    if path.is_symlink():
        raise TechniqueSidecarError(f"MIDI source cannot be a symlink: {path}")
    if not path.is_file():
        raise TechniqueSidecarError(f"MIDI source is not a regular file: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise TechniqueSidecarError(f"Could not read MIDI source '{path}': {exc}") from exc


def _read_sidecar(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise TechniqueSidecarError(
            f"Technique sidecar is not a safe regular file: {path}"
        )
    try:
        declared_size = path.stat().st_size
        if declared_size > MAX_SIDECAR_SIZE_BYTES:
            raise TechniqueSidecarError(
                f"Technique sidecar exceeds {MAX_SIDECAR_SIZE_BYTES} bytes."
            )
        raw = path.read_bytes()
    except OSError as exc:
        raise TechniqueSidecarError(
            f"Could not read technique sidecar '{path}': {exc}"
        ) from exc
    if len(raw) > MAX_SIDECAR_SIZE_BYTES:
        raise TechniqueSidecarError(
            f"Technique sidecar exceeds {MAX_SIDECAR_SIZE_BYTES} bytes."
        )
    return raw


def _load_strict_json(raw: bytes) -> object:
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        TechniqueSidecarError,
    ) as exc:
        if isinstance(exc, TechniqueSidecarError):
            raise
        raise TechniqueSidecarError(
            f"Technique sidecar is not valid UTF-8 JSON: {exc}"
        ) from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TechniqueSidecarError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Never:
    raise TechniqueSidecarError(f"Non-finite JSON number is not allowed: {value}")


def _verify_unchanged(path: Path, expected: bytes, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise TechniqueSidecarError(f"{label} changed while loading.")
    try:
        current = path.read_bytes()
    except OSError as exc:
        raise TechniqueSidecarError(f"{label} changed while loading: {exc}") from exc
    if current != expected:
        raise TechniqueSidecarError(f"{label} changed while loading.")


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TechniqueSidecarError(f"{name} must be an object.")
    return value


def _require_list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TechniqueSidecarError(f"{name} must be an array.")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], required: set[str], name: str
) -> None:
    actual = set(value)
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing:
        raise TechniqueSidecarError(
            f"{name} is missing field(s): {', '.join(missing)}."
        )
    if unknown:
        raise TechniqueSidecarError(
            f"{name} has unknown field(s): {', '.join(unknown)}."
        )


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TechniqueSidecarError(f"{name} must be a non-empty string.")
    return value


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TechniqueSidecarError(f"{name} must be an integer.")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    converted = _require_int(value, name)
    if converted < 0:
        raise TechniqueSidecarError(f"{name} must be non-negative.")
    return converted


def _require_positive_int(value: object, name: str) -> int:
    converted = _require_int(value, name)
    if converted <= 0:
        raise TechniqueSidecarError(f"{name} must be positive.")
    return converted


def _bounded_int(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    converted = _require_int(value, name)
    if converted < minimum or (maximum is not None and converted > maximum):
        upper = f" and {maximum}" if maximum is not None else ""
        raise TechniqueSidecarError(
            f"{name} must be between {minimum}{upper}."
        )
    return converted


def _require_sha256(value: object, name: str) -> str:
    converted = _require_string(value, name)
    if not _SHA256_PATTERN.fullmatch(converted):
        raise TechniqueSidecarError(
            f"{name} must be 64 lowercase hexadecimal characters."
        )
    return converted


def _note_sort_key(note: NoteRef) -> tuple[int, int, int, int]:
    return note.onset_tick, note.pitch, note.end_tick, note.velocity


__all__ = [
    "MAX_ANNOTATIONS",
    "MAX_SIDECAR_SIZE_BYTES",
    "MAX_SLIDE_SEMITONES",
    "NoteRef",
    "NoteTechniques",
    "PalmMuteRange",
    "SIDECAR_SCHEMA_VERSION",
    "SIDECAR_SUFFIX",
    "Technique",
    "TechniqueSidecar",
    "TechniqueSidecarError",
    "TechniqueType",
    "load_technique_sidecar",
    "sidecar_path_for",
]
