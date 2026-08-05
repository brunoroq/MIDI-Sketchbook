"""MIDI discovery, parsing, inspection, and writing boundaries."""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import asdict, dataclass
import logging
import math
import os
from pathlib import Path
import tempfile
from typing import Any
import warnings

import mido
import numpy as np
import pretty_midi

from .config import TrackSelectionConfig, ValidationConfig

LOGGER = logging.getLogger(__name__)
MIDI_EXTENSIONS = {".mid", ".midi"}
_DURATION_ATTRIBUTE = "_midi_idea_duration_seconds"
_GUITAR_PRO_PITCH_BEND_RANGE_RPN = ((101, 0), (100, 0), (6, 6))
_PITCH_BEND_RPN_CONTROLLERS = {6, 38, 100, 101}
_EVENT_TIME_TOLERANCE_SECONDS = 1e-9
PITCH_BEND_MIN = -8192
PITCH_BEND_MAX = 8191


class MidiReadError(ValueError):
    """Raised when a path cannot be parsed as a MIDI file."""

    def __init__(self, message: str, manifest_reason: str) -> None:
        super().__init__(message)
        self.manifest_reason = manifest_reason


class UnsupportedMidiError(ValueError):
    """Raised for a readable SMF outside the stage-one contract."""

    def __init__(self, message: str, manifest_reason: str) -> None:
        super().__init__(message)
        self.manifest_reason = manifest_reason


class MidiWriteError(OSError):
    """Raised when a processed MIDI cannot be written."""

    def __init__(self, message: str, manifest_reason: str) -> None:
        super().__init__(message)
        self.manifest_reason = manifest_reason


def set_midi_duration_seconds(
    midi: pretty_midi.PrettyMIDI, duration_seconds: float
) -> None:
    """Attach a structural duration, including trailing silence, to a MIDI."""

    if not np.isfinite(duration_seconds) or duration_seconds < 0:
        raise ValueError("MIDI duration must be non-negative and finite")
    setattr(midi, _DURATION_ATTRIBUTE, float(duration_seconds))


def get_midi_duration_seconds(midi: pretty_midi.PrettyMIDI) -> float:
    """Return structural duration when known, otherwise the final event time."""

    stored = getattr(midi, _DURATION_ATTRIBUTE, None)
    return float(midi.get_end_time() if stored is None else stored)


def exact_note_identity(note: pretty_midi.Note) -> tuple[int, int, float, float]:
    """Return the fields that make two note events exactly redundant."""

    return (int(note.pitch), int(note.velocity), float(note.start), float(note.end))


@dataclass(frozen=True, slots=True)
class TrackInspection:
    """Validation metadata for one ``pretty_midi`` instrument."""

    track_number: int
    name: str
    program: int
    is_drum: bool
    num_notes: int
    raw_note_events: int
    duplicate_notes_collapsed: int
    min_pitch: int | None
    max_pitch: int | None
    has_pitch_bends: bool
    num_pitch_bend_events: int
    num_expressive_pitch_bend_events: int
    source_pitch_bend_range_semitones: float | None
    valid: bool
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        result = asdict(self)
        result["instrument_index"] = self.track_number
        result["issues"] = list(self.issues)
        return result


@dataclass(frozen=True, slots=True)
class MidiInspection:
    """Inspection result for one source file, including parse failures."""

    source_file: Path
    readable: bool
    compatible: bool
    duration_seconds: float | None
    resolution: int | None
    tempo_bpm: float | None
    tempo_change_count: int | None
    time_signatures: tuple[str, ...]
    tracks: tuple[TrackInspection, ...]
    selected_track: int | None
    issues: tuple[str, ...]

    @property
    def discard_reason(self) -> str | None:
        """Return a stable, human-readable discard reason when incompatible."""

        return None if self.compatible else "; ".join(self.issues)

    def to_dict(self, *, source_label: str | None = None) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "source_file": source_label or str(self.source_file),
            "readable": self.readable,
            "compatible": self.compatible,
            "duration_seconds": self.duration_seconds,
            "resolution": self.resolution,
            "tempo_bpm": self.tempo_bpm,
            "tempo_change_count": self.tempo_change_count,
            "time_signatures": list(self.time_signatures),
            "tracks": [track.to_dict() for track in self.tracks],
            "selected_track": self.selected_track,
            "selected_instrument_index": self.selected_track,
            "issues": list(self.issues),
            "discard_reason": self.discard_reason,
        }


def discover_midi_files(input_dir: str | Path) -> list[Path]:
    """Return MIDI files below ``input_dir`` in deterministic path order."""

    directory = Path(input_dir).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"MIDI input directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"MIDI input path is not a directory: {directory}")
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() in MIDI_EXTENSIONS
        ),
        key=lambda path: (path.as_posix().casefold(), path.as_posix()),
    )


def _raise_overlapping_notes(midi_path: Path, track_number: int) -> None:
    raise UnsupportedMidiError(
        f"Unsupported MIDI '{midi_path}': overlapping note-on "
        f"events in raw track {track_number}",
        "Overlapping note-on events for the same pitch/channel",
    )


def _validate_note_lifetimes(raw_midi: mido.MidiFile, midi_path: Path) -> None:
    """Allow exact unison duplicates while rejecting ambiguous overlaps."""

    for track_number, track in enumerate(raw_midi.tracks):
        absolute_tick = 0
        active_notes: dict[tuple[int, int], list[tuple[int, int]]] = {}
        duplicate_counts: dict[tuple[int, int, int, int], int] = defaultdict(int)
        duplicate_ends: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)

        for message in track:
            absolute_tick += int(message.time)
            if message.type == "note_on" and message.velocity > 0:
                key = (message.channel, message.note)
                active = active_notes.setdefault(key, [])
                if active and any(
                    start != absolute_tick or velocity != message.velocity
                    for start, velocity in active
                ):
                    _raise_overlapping_notes(midi_path, track_number)
                active.append((absolute_tick, message.velocity))
                group = (*key, absolute_tick, message.velocity)
                duplicate_counts[group] += 1
            elif message.type == "note_off" or (
                message.type == "note_on" and message.velocity == 0
            ):
                key = (message.channel, message.note)
                active = active_notes.get(key)
                if active:
                    start, velocity = active.pop(0)
                    duplicate_ends[(*key, start, velocity)].append(absolute_tick)
                    if not active:
                        del active_notes[key]

        if active_notes:
            raise UnsupportedMidiError(
                f"Unsupported MIDI '{midi_path}': dangling note-on events "
                f"in raw track {track_number}",
                "Dangling note-on events without matching note-off",
            )

        for group, count in duplicate_counts.items():
            if count <= 1:
                continue
            end_ticks = duplicate_ends[group]
            if len(end_ticks) != count or len(set(end_ticks)) != 1:
                _raise_overlapping_notes(midi_path, track_number)


def detect_pitch_bend_range(
    control_changes: list[pretty_midi.ControlChange],
) -> tuple[float | None, str | None]:
    """Return one explicit constant Pitch Bend Sensitivity RPN.

    Stage 1 deliberately accepts no other controllers. Ordering is significant:
    Data Entry must follow an RPN 0,0 selection. Repeated declarations are
    accepted only when they describe the same range.
    """

    if not control_changes:
        return None, None
    ordered = sorted(
        enumerate(control_changes),
        key=lambda item: (float(item[1].time), item[0]),
    )
    rpn_msb: int | None = None
    rpn_lsb: int | None = None
    ranges: list[float] = []
    latest_range_index: int | None = None
    for _, change in ordered:
        if (
            not np.isfinite(change.time)
            or change.time < -_EVENT_TIME_TOLERANCE_SECONDS
        ):
            return None, "Pitch-bend RPN contains invalid timing"
        number = int(change.number)
        value = int(change.value)
        if number not in _PITCH_BEND_RPN_CONTROLLERS:
            return None, "Track contains unsupported MIDI control changes"
        if number == 101:
            rpn_msb = value
            latest_range_index = None
        elif number == 100:
            rpn_lsb = value
            latest_range_index = None
        elif number == 6:
            if (rpn_msb, rpn_lsb) != (0, 0):
                return None, "Pitch-bend Data Entry appears before RPN 0,0 selection"
            if value <= 0:
                return None, "Pitch-bend range must be positive"
            ranges.append(float(value))
            latest_range_index = len(ranges) - 1
        else:  # CC38: optional cents for the latest Data Entry MSB.
            if (rpn_msb, rpn_lsb) != (0, 0) or latest_range_index is None:
                return None, "Pitch-bend fine range appears without Data Entry MSB"
            coarse = math.floor(ranges[latest_range_index])
            ranges[latest_range_index] = coarse + value / 100.0

    if not ranges:
        return None, "Pitch-bend RPN selection has no range Data Entry"
    first = ranges[0]
    if any(not math.isclose(value, first, abs_tol=1e-9) for value in ranges[1:]):
        return None, "Pitch-bend range changes within the track"
    return first, None


def canonical_pitch_bend_range_controls(
    semitones: int = 6,
) -> list[pretty_midi.ControlChange]:
    """Create the ordered time-zero RPN for a canonical integer range."""

    if semitones != 6:
        raise ValueError("Stage 1 supports only a canonical +/-6-semitone range")
    return [
        pretty_midi.ControlChange(number=number, value=value, time=0.0)
        for number, value in _GUITAR_PRO_PITCH_BEND_RANGE_RPN
    ]


def read_midi(path: str | Path) -> pretty_midi.PrettyMIDI:
    """Parse one MIDI file, wrapping parser failures with path context."""

    midi_path = Path(path).expanduser().resolve()
    if not midi_path.is_file():
        raise MidiReadError(
            f"MIDI file does not exist: {midi_path}",
            "MIDI file does not exist",
        )
    try:
        raw_midi = mido.MidiFile(filename=str(midi_path), clip=False)
    except Exception as exc:
        error_type = type(exc).__name__
        raise MidiReadError(
            f"Could not parse '{midi_path}' ({error_type}): {exc}",
            f"Could not parse MIDI ({error_type})",
        ) from exc
    if raw_midi.type == 2:
        raise UnsupportedMidiError(
            f"Unsupported MIDI '{midi_path}': asynchronous SMF type 2",
            "Unsupported asynchronous Standard MIDI File type 2",
        )
    if raw_midi.ticks_per_beat <= 0:
        raise UnsupportedMidiError(
            f"Unsupported MIDI '{midi_path}': division must use positive PPQ",
            "MIDI timing division must use a positive pulses-per-quarter value",
        )
    if raw_midi.type == 0 and len(raw_midi.tracks) != 1:
        raise UnsupportedMidiError(
            f"Unsupported MIDI '{midi_path}': SMF type 0 contains "
            f"{len(raw_midi.tracks)} tracks",
            "Standard MIDI File type 0 must contain exactly one track",
        )
    if raw_midi.type == 1:
        misplaced = sorted(
            {
                message.type
                for track in raw_midi.tracks[1:]
                for message in track
                if message.is_meta
                and message.type in {"set_tempo", "time_signature"}
            }
        )
        if misplaced:
            events = ", ".join(misplaced)
            raise UnsupportedMidiError(
                f"Unsupported MIDI '{midi_path}': global metadata {events} "
                "appears outside track 0",
                f"Global metadata appears outside track 0: {events}",
            )
    _validate_note_lifetimes(raw_midi, midi_path)
    duration_midi = copy.deepcopy(raw_midi)
    for track in duration_midi.tracks:
        for message in track:
            # pretty_midi writes a one-tick EOT guard. Treat that as exporter
            # bookkeeping while preserving meaningful trailing rests.
            if message.type == "end_of_track" and message.time <= 1:
                message.time = 0
    raw_duration_seconds = float(duration_midi.length)
    try:
        with warnings.catch_warnings():
            # Type-1 Guitar Pro exports place key signatures on the musical
            # track. Tempo and meter placement were already validated above;
            # key signatures are intentionally unused by this note-only stage.
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"Tempo, Key or Time signature change events found on "
                    r"non-zero tracks\..*"
                ),
                category=RuntimeWarning,
                module=r"pretty_midi\.pretty_midi",
            )
            midi = pretty_midi.PrettyMIDI(mido_object=raw_midi)
    except Exception as exc:
        # pretty_midi can surface several parser exception types. This is the
        # deliberate file-format fault boundary; the cause is retained in logs.
        error_type = type(exc).__name__
        raise MidiReadError(
            f"Could not parse '{midi_path}' ({error_type}): {exc}",
            f"Could not parse MIDI ({error_type})",
        ) from exc
    set_midi_duration_seconds(midi, raw_duration_seconds)
    return midi


def _canonical_midi(midi: pretty_midi.PrettyMIDI) -> tuple[Any, ...]:
    tempo_times, tempi = midi.get_tempo_changes()
    tempo_map = tuple(
        (
            int(midi.time_to_tick(float(time))),
            int(round(60_000_000 / float(tempo))),
        )
        for time, tempo in zip(tempo_times, tempi, strict=True)
    )
    signatures = tuple(
        sorted(
            (
                int(midi.time_to_tick(float(change.time))),
                int(change.numerator),
                int(change.denominator),
            )
            for change in midi.time_signature_changes
        )
    )
    instruments = tuple(
        sorted(
            (
                int(instrument.program),
                bool(instrument.is_drum),
                tuple(
                    sorted(
                        (
                            int(note.pitch),
                            int(note.velocity),
                            int(midi.time_to_tick(float(note.start))),
                            int(midi.time_to_tick(float(note.end))),
                        )
                        for note in instrument.notes
                    )
                ),
                tuple(
                    (
                        int(midi.time_to_tick(float(bend.time))),
                        int(bend.pitch),
                    )
                    for bend in instrument.pitch_bends
                ),
                tuple(
                    (
                        int(midi.time_to_tick(float(change.time))),
                        int(change.number),
                        int(change.value),
                    )
                    for change in instrument.control_changes
                ),
            )
            for instrument in midi.instruments
        )
    )
    structural_end_tick = int(
        midi.time_to_tick(get_midi_duration_seconds(midi))
    )
    return tempo_map, signatures, instruments, structural_end_tick


def _round_trip_matches(
    expected: pretty_midi.PrettyMIDI,
    actual: pretty_midi.PrettyMIDI,
) -> bool:
    expected_tempi, expected_signatures, expected_instruments, expected_end = (
        _canonical_midi(expected)
    )
    actual_tempi, actual_signatures, actual_instruments, actual_end = (
        _canonical_midi(actual)
    )
    if len(expected_tempi) != len(actual_tempi):
        return False
    tempo_matches = all(
        expected_tick == actual_tick
        and abs(expected_microseconds - actual_microseconds) <= 1
        for (expected_tick, expected_microseconds), (
            actual_tick,
            actual_microseconds,
        ) in zip(expected_tempi, actual_tempi, strict=True)
    )
    return (
        tempo_matches
        and expected_signatures == actual_signatures
        and expected_instruments == actual_instruments
        and expected_end == actual_end
    )


def _write_structural_end_time(
    path: Path, midi: pretty_midi.PrettyMIDI
) -> None:
    target_tick = int(midi.time_to_tick(get_midi_duration_seconds(midi)))
    raw_midi = mido.MidiFile(filename=str(path), clip=False)
    conductor_track = raw_midi.tracks[0]
    messages = [
        message for message in conductor_track if message.type != "end_of_track"
    ]
    current_tick = sum(int(message.time) for message in messages)
    messages.append(
        mido.MetaMessage(
            "end_of_track",
            time=max(0, target_tick - current_tick),
        )
    )
    conductor_track[:] = messages
    raw_midi.save(path)


def _order_channel_events(path: Path) -> None:
    """Repair pretty_midi's same-tick ordering for pitch-bend RPN events."""

    raw_midi = mido.MidiFile(filename=str(path), clip=False)

    def priority(message: mido.Message | mido.MetaMessage) -> int:
        if message.type == "track_name":
            return 0
        if message.type == "program_change":
            return 10
        if message.type == "control_change":
            controller_priority = {101: 20, 100: 21, 6: 22, 38: 23}
            return controller_priority.get(message.control, 40)
        if message.type == "pitchwheel":
            return 30
        return 50

    for track in raw_midi.tracks:
        absolute_tick = 0
        events: list[tuple[int, int, mido.Message | mido.MetaMessage]] = []
        for event_index, message in enumerate(track):
            absolute_tick += int(message.time)
            events.append((absolute_tick, event_index, message.copy(time=0)))
        events.sort(key=lambda item: (item[0], priority(item[2]), item[1]))
        previous_tick = 0
        rebuilt = mido.MidiTrack()
        for event_tick, _, message in events:
            rebuilt.append(message.copy(time=event_tick - previous_tick))
            previous_tick = event_tick
        track[:] = rebuilt
    raw_midi.save(path)


def write_midi(midi: pretty_midi.PrettyMIDI, path: str | Path) -> Path:
    """Validate and atomically write a MIDI file."""

    output_path = Path(path).expanduser().resolve()
    temporary_path: Path | None = None
    try:
        for instrument in midi.instruments:
            expressive = [bend for bend in instrument.pitch_bends if bend.pitch != 0]
            bend_range, range_issue = detect_pitch_bend_range(
                instrument.control_changes
            )
            if expressive and (range_issue is not None or bend_range != 6.0):
                detail = range_issue or f"range is {bend_range}, expected 6"
                raise ValueError(
                    "Expressive pitch bends require a constant explicit "
                    f"+/-6-semitone RPN ({detail})"
                )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}-",
            suffix=".mid",
            dir=output_path.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        midi.write(str(temporary_path))
        _order_channel_events(temporary_path)
        _write_structural_end_time(temporary_path, midi)
        reparsed = read_midi(temporary_path)
        if not _round_trip_matches(midi, reparsed):
            raise ValueError("MIDI round-trip changed musical events")
        os.replace(temporary_path, output_path)
        temporary_path = None
    except (MidiReadError, UnsupportedMidiError) as exc:
        raise MidiWriteError(
            f"Could not validate written MIDI '{output_path}': {exc}",
            f"Written MIDI failed validation: {exc.manifest_reason}",
        ) from exc
    except ValueError as exc:
        raise MidiWriteError(
            f"Could not write MIDI '{output_path}': {exc}",
            f"MIDI write validation failed: {exc}",
        ) from exc
    except OSError as exc:
        raise MidiWriteError(
            f"Could not write MIDI '{output_path}': {exc}",
            f"Could not write MIDI ({type(exc).__name__})",
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path


def get_constant_tempo(
    midi: pretty_midi.PrettyMIDI, tolerance: float
) -> tuple[float | None, int, str | None]:
    """Return the initial BPM and an issue when the tempo is not constant."""

    _, tempi = midi.get_tempo_changes()
    if len(tempi) == 0:
        return 120.0, 0, None
    first = float(tempi[0])
    if not np.all(np.isfinite(tempi)) or first <= 0:
        return None, len(tempi), "Tempo contains a non-positive or non-finite value"
    if not np.allclose(tempi, first, rtol=0.0, atol=tolerance):
        values = ", ".join(f"{float(value):.3f}" for value in tempi[:5])
        suffix = "..." if len(tempi) > 5 else ""
        return first, len(tempi), f"Tempo is not constant ({values}{suffix} BPM)"
    return first, len(tempi), None


def _time_signature_issues(
    midi: pretty_midi.PrettyMIDI, config: ValidationConfig
) -> tuple[tuple[str, ...], list[str]]:
    changes = midi.time_signature_changes
    if not changes:
        if config.allow_missing_time_signature:
            return ("4/4 (implicit)",), []
        return (), ["Time signature is missing"]
    signatures = tuple(f"{change.numerator}/{change.denominator}" for change in changes)
    allowed = config.allowed_time_signature
    unsupported = sorted(
        {
            (change.numerator, change.denominator)
            for change in changes
            if (change.numerator, change.denominator) != allowed
        }
    )
    issues: list[str] = []
    if unsupported:
        formatted = ", ".join(
            f"{numerator}/{denominator}" for numerator, denominator in unsupported
        )
        issues.append(f"Unsupported time signature(s): {formatted}; expected 4/4")
    return signatures, issues


def inspect_track(
    instrument: pretty_midi.Instrument,
    track_number: int,
    config: ValidationConfig,
) -> TrackInspection:
    """Inspect an instrument track against stage-one constraints."""

    notes = instrument.notes
    unique_note_count = len({exact_note_identity(note) for note in notes})
    duplicate_note_count = len(notes) - unique_note_count
    pitches = [int(note.pitch) for note in notes]
    issues: list[str] = []
    if config.exclude_drums and instrument.is_drum:
        issues.append("Drum tracks are excluded")
    if unique_note_count < config.min_notes_per_track:
        issues.append(
            f"Track has {unique_note_count} unique notes; "
            f"minimum is {config.min_notes_per_track}"
        )
    if pitches:
        minimum = min(pitches)
        maximum = max(pitches)
        if minimum < config.pitch_min or maximum > config.pitch_max:
            issues.append(
                f"Pitch range {minimum}-{maximum} is outside "
                f"{config.pitch_min}-{config.pitch_max}"
            )
    else:
        minimum = maximum = None
    expressive_pitch_bends = [
        bend for bend in instrument.pitch_bends if bend.pitch != 0
    ]
    bend_range, bend_range_issue = detect_pitch_bend_range(
        instrument.control_changes
    )
    if config.reject_pitch_bends and expressive_pitch_bends:
        issues.append("Track contains pitch bends")
    if bend_range_issue is not None:
        issues.append(bend_range_issue)
    if (
        expressive_pitch_bends
        and config.require_explicit_pitch_bend_range
        and bend_range is None
    ):
        issues.append("Expressive pitch bends require an explicit range RPN")
    if expressive_pitch_bends and bend_range is not None:
        range_setup_times = [
            float(change.time)
            for change in instrument.control_changes
            if change.number == 6
        ]
        first_expressive_time = min(
            float(bend.time) for bend in expressive_pitch_bends
        )
        if (
            not range_setup_times
            or min(range_setup_times)
            > first_expressive_time + _EVENT_TIME_TOLERANCE_SECONDS
        ):
            issues.append("Pitch-bend range is declared after the expressive curve")
    if expressive_pitch_bends and bend_range is not None:
        canonical_range = config.canonical_pitch_bend_range_semitones
        if any(
            abs(int(bend.pitch) * bend_range / 8192.0)
            > canonical_range + 1e-9
            for bend in expressive_pitch_bends
        ):
            issues.append(
                "Pitch-bend excursion exceeds the canonical "
                f"+/-{canonical_range}-semitone range"
            )
    invalid_bends = sum(
        1
        for bend in instrument.pitch_bends
        if bend.pitch < PITCH_BEND_MIN
        or bend.pitch > PITCH_BEND_MAX
        or bend.time < 0
        or not np.isfinite(bend.time)
    )
    if invalid_bends:
        issues.append(
            f"Track contains {invalid_bends} pitch-bend event(s) with invalid values"
        )
    invalid_notes = sum(
        1
        for note in notes
        if note.start < 0
        or note.end <= note.start
        or not np.isfinite((note.start, note.end)).all()
    )
    if invalid_notes:
        issues.append(f"Track contains {invalid_notes} note(s) with invalid timing")
    return TrackInspection(
        track_number=track_number,
        name=str(instrument.name),
        program=int(instrument.program),
        is_drum=bool(instrument.is_drum),
        num_notes=unique_note_count,
        raw_note_events=len(notes),
        duplicate_notes_collapsed=duplicate_note_count,
        min_pitch=minimum,
        max_pitch=maximum,
        has_pitch_bends=bool(expressive_pitch_bends),
        num_pitch_bend_events=len(instrument.pitch_bends),
        num_expressive_pitch_bend_events=len(expressive_pitch_bends),
        source_pitch_bend_range_semitones=bend_range,
        valid=not issues,
        issues=tuple(issues),
    )


def _select_track(
    tracks: tuple[TrackInspection, ...], config: TrackSelectionConfig
) -> tuple[int | None, str | None]:
    if config.mode == "index":
        assert config.track_index is not None  # Guaranteed by configuration validation.
        if config.track_index >= len(tracks):
            return None, (
                f"Requested instrument index {config.track_index}, but file has {len(tracks)}"
            )
        selected = tracks[config.track_index]
        if not selected.valid:
            details = ", ".join(selected.issues)
            return None, f"Requested instrument index {config.track_index} is invalid: {details}"
        return config.track_index, None

    candidates = [track for track in tracks if track.valid]
    if not candidates:
        if not tracks:
            return None, "No instrumental tracks found"
        details = "; ".join(
            f"instrument {track.track_number}: {', '.join(track.issues)}"
            for track in tracks
        )
        return None, f"No valid instrumental track found ({details})"
    selected = max(candidates, key=lambda track: (track.num_notes, -track.track_number))
    return selected.track_number, None


def inspect_midi(
    path: str | Path,
    validation: ValidationConfig,
    track_selection: TrackSelectionConfig | None = None,
) -> MidiInspection:
    """Inspect one MIDI without allowing a corrupt file to abort a batch."""

    source_path = Path(path).expanduser().resolve()
    selection = track_selection or TrackSelectionConfig()
    try:
        midi = read_midi(source_path)
    except UnsupportedMidiError as exc:
        LOGGER.warning("%s", exc)
        return MidiInspection(
            source_file=source_path,
            readable=True,
            compatible=False,
            duration_seconds=None,
            resolution=None,
            tempo_bpm=None,
            tempo_change_count=None,
            time_signatures=(),
            tracks=(),
            selected_track=None,
            issues=(exc.manifest_reason,),
        )
    except MidiReadError as exc:
        LOGGER.warning("%s", exc)
        return MidiInspection(
            source_file=source_path,
            readable=False,
            compatible=False,
            duration_seconds=None,
            resolution=None,
            tempo_bpm=None,
            tempo_change_count=None,
            time_signatures=(),
            tracks=(),
            selected_track=None,
            issues=(exc.manifest_reason,),
        )

    issues: list[str] = []
    tempo, tempo_change_count, tempo_issue = get_constant_tempo(
        midi, validation.tempo_tolerance
    )
    if tempo_issue:
        issues.append(tempo_issue)
    if midi.lyrics:
        issues.append("Lyrics are unsupported in the stage-one MVP")
    signatures, signature_issues = _time_signature_issues(midi, validation)
    issues.extend(signature_issues)
    tracks = tuple(
        inspect_track(instrument, index, validation)
        for index, instrument in enumerate(midi.instruments)
    )
    selected_track, selection_issue = _select_track(tracks, selection)
    if selection_issue:
        issues.append(selection_issue)
    compatible = not issues and selected_track is not None and tempo is not None
    return MidiInspection(
        source_file=source_path,
        readable=True,
        compatible=compatible,
        duration_seconds=get_midi_duration_seconds(midi),
        resolution=int(midi.resolution),
        tempo_bpm=tempo,
        tempo_change_count=tempo_change_count,
        time_signatures=signatures,
        tracks=tracks,
        selected_track=selected_track,
        issues=tuple(issues),
    )
