"""Strict tonic/mode labels and deterministic Stage 1 inference.

A sibling ``riff.mid.tonality.json`` file is an authoritative manual label.
When it is absent, Stage 1 can infer one label from either the normalized
source instrument or each untransposed phrase.  Tonality is kept outside MIDI
key-signature events because those events cannot represent the supported modes
and are frequently exporter defaults rather than annotations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Never

import pretty_midi


SIDECAR_SCHEMA_VERSION = 1
SIDECAR_SUFFIX = ".tonality.json"
MAX_SIDECAR_SIZE_BYTES = 65_536

TONIC_NAMES = (
    "C",
    "C_SHARP",
    "D",
    "D_SHARP",
    "E",
    "F",
    "F_SHARP",
    "G",
    "G_SHARP",
    "A",
    "A_SHARP",
    "B",
    "UNKNOWN",
)
MODE_NAMES = (
    "MAJOR",
    "MINOR",
    "DORIAN",
    "PHRYGIAN",
    "LYDIAN",
    "MIXOLYDIAN",
    "LOCRIAN",
    "HARMONIC_MINOR",
    "PHRYGIAN_DOMINANT",
    "BLUES",
    "UNKNOWN",
)
TONALITY_METHODS = ("MANUAL", "AUTO_SOURCE", "AUTO_FRAGMENT", "UNKNOWN")

_ROOT_KEYS = {
    "schema_version",
    "source_midi",
    "source_sha256",
    "instrument_index",
    "tonic",
    "mode",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PITCH_CLASS_BY_TONIC = {
    tonic: pitch_class for pitch_class, tonic in enumerate(TONIC_NAMES[:-1])
}
_TONIC_ALIASES = {
    "C#": "C_SHARP",
    "DB": "C_SHARP",
    "D#": "D_SHARP",
    "EB": "D_SHARP",
    "F#": "F_SHARP",
    "GB": "F_SHARP",
    "G#": "G_SHARP",
    "AB": "G_SHARP",
    "A#": "A_SHARP",
    "BB": "A_SHARP",
}
_MODE_ALIASES = {
    "IONIAN": "MAJOR",
    "AEOLIAN": "MINOR",
    "NATURAL_MINOR": "MINOR",
}
_MODE_INTERVALS: dict[str, frozenset[int]] = {
    "MAJOR": frozenset({0, 2, 4, 5, 7, 9, 11}),
    "MINOR": frozenset({0, 2, 3, 5, 7, 8, 10}),
    "DORIAN": frozenset({0, 2, 3, 5, 7, 9, 10}),
    "PHRYGIAN": frozenset({0, 1, 3, 5, 7, 8, 10}),
    "LYDIAN": frozenset({0, 2, 4, 6, 7, 9, 11}),
    "MIXOLYDIAN": frozenset({0, 2, 4, 5, 7, 9, 10}),
    "LOCRIAN": frozenset({0, 1, 3, 5, 6, 8, 10}),
    "HARMONIC_MINOR": frozenset({0, 2, 3, 5, 7, 8, 11}),
    "PHRYGIAN_DOMINANT": frozenset({0, 1, 4, 5, 7, 8, 10}),
    "BLUES": frozenset({0, 3, 5, 6, 7, 10}),
}


class TonalityError(ValueError):
    """Raised when a tonic/mode value violates the canonical contract."""


class TonalitySidecarError(TonalityError):
    """Raised when a present manual tonality sidecar is invalid or stale."""


def normalize_tonic(value: object) -> str:
    """Return a canonical tonic name, accepting common sharp/flat aliases."""

    if not isinstance(value, str) or not value.strip():
        raise TonalityError("tonic must be a non-empty string")
    candidate = value.strip().upper().replace("-", "_").replace(" ", "_")
    candidate = _TONIC_ALIASES.get(candidate, candidate)
    if candidate not in TONIC_NAMES:
        raise TonalityError(
            "tonic must be one of: " + ", ".join(TONIC_NAMES)
        )
    return candidate


def normalize_mode(value: object) -> str:
    """Return a canonical mode name, accepting unambiguous common aliases."""

    if not isinstance(value, str) or not value.strip():
        raise TonalityError("mode must be a non-empty string")
    candidate = value.strip().upper().replace("-", "_").replace(" ", "_")
    candidate = _MODE_ALIASES.get(candidate, candidate)
    if candidate not in MODE_NAMES:
        raise TonalityError("mode must be one of: " + ", ".join(MODE_NAMES))
    return candidate


def transpose_tonic(tonic: object, semitones: int) -> str:
    """Transpose a canonical tonic modulo twelve; UNKNOWN remains UNKNOWN."""

    normalized = normalize_tonic(tonic)
    if isinstance(semitones, bool) or not isinstance(semitones, int):
        raise TonalityError("semitones must be an integer")
    if normalized == "UNKNOWN":
        return normalized
    pitch_class = (_PITCH_CLASS_BY_TONIC[normalized] + semitones) % 12
    return TONIC_NAMES[pitch_class]


def _confidence(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TonalityError(f"{name} must be a number in [0, 1] or null")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise TonalityError(f"{name} must be finite and inside [0, 1]")
    return converted


@dataclass(frozen=True, slots=True)
class Tonality:
    """One canonical tonic/mode pair plus its provenance."""

    tonic: str
    mode: str
    method: str = "MANUAL"
    tonic_confidence: float | None = None
    mode_confidence: float | None = None

    def __post_init__(self) -> None:
        tonic = normalize_tonic(self.tonic)
        mode = normalize_mode(self.mode)
        if tonic == "UNKNOWN" and mode != "UNKNOWN":
            raise TonalityError("UNKNOWN tonic requires UNKNOWN mode")
        if not isinstance(self.method, str) or self.method not in TONALITY_METHODS:
            raise TonalityError(
                "method must be one of: " + ", ".join(TONALITY_METHODS)
            )
        tonic_confidence = _confidence(
            self.tonic_confidence, "tonic_confidence"
        )
        mode_confidence = _confidence(self.mode_confidence, "mode_confidence")
        if self.method in {"AUTO_SOURCE", "AUTO_FRAGMENT"}:
            if tonic == "UNKNOWN" or mode == "UNKNOWN":
                raise TonalityError("Automatic tonality must select tonic and mode")
            if tonic_confidence is None or mode_confidence is None:
                raise TonalityError("Automatic tonality requires both confidences")
        elif tonic_confidence is not None or mode_confidence is not None:
            raise TonalityError(
                "Manual and unknown tonality must not declare confidence values"
            )
        if self.method == "UNKNOWN" and (tonic, mode) != ("UNKNOWN", "UNKNOWN"):
            raise TonalityError("UNKNOWN method requires UNKNOWN tonic and mode")
        object.__setattr__(self, "tonic", tonic)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "tonic_confidence", tonic_confidence)
        object.__setattr__(self, "mode_confidence", mode_confidence)

    @classmethod
    def unknown(cls) -> Tonality:
        """Return the canonical deliberately-unresolved label."""

        return cls("UNKNOWN", "UNKNOWN", method="UNKNOWN")

    def transposed(self, semitones: int) -> Tonality:
        """Return this label with only its tonic transposed."""

        return Tonality(
            tonic=transpose_tonic(self.tonic, semitones),
            mode=self.mode,
            method=self.method,
            tonic_confidence=self.tonic_confidence,
            mode_confidence=self.mode_confidence,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the stable Stage 1 manifest representation."""

        return {
            "tonic": self.tonic,
            "mode": self.mode,
            "method": self.method,
            "tonic_confidence": self.tonic_confidence,
            "mode_confidence": self.mode_confidence,
        }


@dataclass(frozen=True, slots=True)
class TonalitySidecar:
    """A fully validated manual label bound to exact MIDI bytes."""

    path: Path
    fingerprint: str
    size_bytes: int
    source_midi: str
    source_sha256: str
    instrument_index: int
    tonality: Tonality

    @property
    def sha256(self) -> str:
        return self.fingerprint


def sidecar_path_for(midi_path: str | Path) -> Path:
    """Return the fixed sibling tonality-sidecar path for a MIDI path."""

    path = Path(midi_path).expanduser()
    if not path.name:
        raise TonalitySidecarError("midi_path must name a MIDI file")
    return path.with_name(f"{path.name}{SIDECAR_SUFFIX}")


def load_tonality_sidecar(
    midi_path: str | Path,
    *,
    source_sha256: str,
    midi: pretty_midi.PrettyMIDI,
    instrument_index: int,
) -> TonalitySidecar | None:
    """Load a strict optional sidecar, returning ``None`` only when absent."""

    source_path = Path(midi_path).expanduser()
    sidecar_path = sidecar_path_for(source_path)
    if sidecar_path.is_symlink():
        raise TonalitySidecarError(
            f"Tonality sidecar cannot be a symlink: {sidecar_path}"
        )
    if not sidecar_path.exists():
        return None
    if not sidecar_path.is_file():
        raise TonalitySidecarError(
            f"Tonality sidecar is not a regular file: {sidecar_path}"
        )

    expected_sha = _require_sha256(source_sha256, "source_sha256")
    source_bytes = _read_regular_file(source_path, "MIDI source", None)
    if hashlib.sha256(source_bytes).hexdigest() != expected_sha:
        raise TonalitySidecarError(
            "source_sha256 does not match the current MIDI file"
        )
    raw = _read_regular_file(
        sidecar_path, "tonality sidecar", MAX_SIDECAR_SIZE_BYTES
    )
    payload = _load_strict_json(raw)
    root = _require_mapping(payload, "sidecar")
    if set(root) != _ROOT_KEYS:
        missing = sorted(_ROOT_KEYS - set(root))
        extra = sorted(set(root) - _ROOT_KEYS)
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unknown: " + ", ".join(extra))
        raise TonalitySidecarError(
            "sidecar must contain exactly the required keys ("
            + "; ".join(details)
            + ")"
        )
    schema_version = _require_int(root["schema_version"], "schema_version")
    if schema_version != SIDECAR_SCHEMA_VERSION:
        raise TonalitySidecarError(
            f"schema_version must be {SIDECAR_SCHEMA_VERSION}"
        )
    source_midi = _require_string(root["source_midi"], "source_midi")
    if source_midi != source_path.name or Path(source_midi).name != source_midi:
        raise TonalitySidecarError(
            "source_midi must equal the exact sibling MIDI filename"
        )
    declared_sha = _require_sha256(
        root["source_sha256"], "sidecar.source_sha256"
    )
    if declared_sha != expected_sha:
        raise TonalitySidecarError(
            "Sidecar source_sha256 does not match the current MIDI file"
        )
    selected_index = _require_nonnegative_int(
        instrument_index, "instrument_index argument"
    )
    declared_index = _require_nonnegative_int(
        root["instrument_index"], "sidecar.instrument_index"
    )
    if declared_index != selected_index:
        raise TonalitySidecarError(
            "Sidecar instrument_index does not match the selected instrument"
        )
    if selected_index >= len(midi.instruments):
        raise TonalitySidecarError(
            "instrument_index is outside the MIDI instrument list"
        )
    raw_tonic = _require_string(root["tonic"], "tonic")
    raw_mode = _require_string(root["mode"], "mode")
    if raw_tonic not in TONIC_NAMES:
        raise TonalitySidecarError(
            "tonic must use a canonical value: " + ", ".join(TONIC_NAMES)
        )
    if raw_mode not in MODE_NAMES:
        raise TonalitySidecarError(
            "mode must use a canonical value: " + ", ".join(MODE_NAMES)
        )
    try:
        tonality = Tonality(raw_tonic, raw_mode, method="MANUAL")
    except TonalityError as exc:
        raise TonalitySidecarError(str(exc)) from exc

    _verify_unchanged(source_path, source_bytes, "MIDI source")
    _verify_unchanged(sidecar_path, raw, "tonality sidecar")
    return TonalitySidecar(
        path=sidecar_path.resolve(),
        fingerprint=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        source_midi=source_midi,
        source_sha256=declared_sha,
        instrument_index=declared_index,
        tonality=tonality,
    )


def infer_tonality(
    instrument: pretty_midi.Instrument,
    *,
    method: str = "AUTO_SOURCE",
) -> Tonality:
    """Infer a deterministic best tonic/mode candidate from note pitches.

    Every non-empty instrument receives the highest-scoring supported mode.
    Confidence records the normalized margin over the runner-up; it does not
    suppress a label.  This makes augmentation deterministic and avoids
    treating ambiguous material as missing data.
    """

    if not isinstance(instrument, pretty_midi.Instrument):
        raise TonalityError("instrument must be a pretty_midi.Instrument")
    if method not in {"AUTO_SOURCE", "AUTO_FRAGMENT"}:
        raise TonalityError("inference method must be AUTO_SOURCE or AUTO_FRAGMENT")
    notes = sorted(
        instrument.notes,
        key=lambda note: (note.start, note.pitch, note.end, note.velocity),
    )
    if not notes:
        return Tonality.unknown()

    positive_durations = [max(float(note.end - note.start), 1e-9) for note in notes]
    ordered_durations = sorted(positive_durations)
    typical_duration = ordered_durations[len(ordered_durations) // 2]
    pitch_weights = [0.0] * 12
    for note, duration in zip(notes, positive_durations, strict=True):
        duration_ratio = min(duration / typical_duration, 2.0)
        pitch_weights[int(note.pitch) % 12] += 1.0 + 0.25 * duration_ratio
    total_pitch_weight = sum(pitch_weights)

    onset_groups: dict[float, list[pretty_midi.Note]] = {}
    for note in notes:
        onset_groups.setdefault(float(note.start), []).append(note)
    bass_weights = [0.0] * 12
    for group in onset_groups.values():
        bass_weights[min(group, key=lambda note: note.pitch).pitch % 12] += 1.0
    total_bass_weight = sum(bass_weights)
    first_group = onset_groups[min(onset_groups)]
    final_group = onset_groups[max(onset_groups)]
    first_pc = min(first_group, key=lambda note: note.pitch).pitch % 12
    final_pc = min(final_group, key=lambda note: note.pitch).pitch % 12
    present_pitch_classes = {
        pitch_class for pitch_class, weight in enumerate(pitch_weights) if weight > 0
    }

    scores: dict[tuple[int, str], float] = {}
    for tonic_pc in range(12):
        for mode in MODE_NAMES[:-1]:
            intervals = _MODE_INTERVALS[mode]
            scale = {(tonic_pc + interval) % 12 for interval in intervals}
            in_scale = sum(pitch_weights[pitch_class] for pitch_class in scale)
            coverage = in_scale / total_pitch_weight
            outside = 1.0 - coverage
            tonic_presence = pitch_weights[tonic_pc] / total_pitch_weight
            fifth_presence = pitch_weights[(tonic_pc + 7) % 12] / total_pitch_weight
            bass_presence = bass_weights[tonic_pc] / total_bass_weight
            first_bonus = 1.0 if first_pc == tonic_pc else 0.0
            final_bonus = 1.0 if final_pc == tonic_pc else 0.0
            completeness = len(present_pitch_classes & scale) / len(scale)
            scores[(tonic_pc, mode)] = (
                5.0 * coverage
                - 3.0 * outside
                + 1.5 * tonic_presence
                + 1.0 * bass_presence
                + 0.35 * fifth_presence
                + 0.5 * first_bonus
                + 0.75 * final_bonus
                + 0.25 * completeness
            )

    mode_order = {mode: index for index, mode in enumerate(MODE_NAMES[:-1])}
    ranked = sorted(
        scores,
        key=lambda candidate: (
            -scores[candidate],
            candidate[0],
            mode_order[candidate[1]],
        ),
    )
    best_tonic_pc, best_mode = ranked[0]
    tonic_best_scores = [
        max(scores[(tonic_pc, mode)] for mode in MODE_NAMES[:-1])
        for tonic_pc in range(12)
    ]
    tonic_runner_up = sorted(tonic_best_scores, reverse=True)[1]
    tonic_confidence = _margin_confidence(
        tonic_best_scores[best_tonic_pc], tonic_runner_up
    )
    mode_scores = sorted(
        (scores[(best_tonic_pc, mode)] for mode in MODE_NAMES[:-1]),
        reverse=True,
    )
    mode_confidence = _margin_confidence(mode_scores[0], mode_scores[1])
    return Tonality(
        tonic=TONIC_NAMES[best_tonic_pc],
        mode=best_mode,
        method=method,
        tonic_confidence=tonic_confidence,
        mode_confidence=mode_confidence,
    )


def _margin_confidence(best: float, runner_up: float) -> float:
    margin = max(0.0, float(best) - float(runner_up))
    return margin / (1.0 + margin)


def _read_regular_file(path: Path, label: str, maximum_size: int | None) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise TonalitySidecarError(f"{label} is not a safe regular file: {path}")
    try:
        if maximum_size is not None and path.stat().st_size > maximum_size:
            raise TonalitySidecarError(
                f"{label} exceeds {maximum_size} bytes"
            )
        raw = path.read_bytes()
    except OSError as exc:
        raise TonalitySidecarError(f"Could not read {label} '{path}': {exc}") from exc
    if maximum_size is not None and len(raw) > maximum_size:
        raise TonalitySidecarError(f"{label} exceeds {maximum_size} bytes")
    return raw


def _verify_unchanged(path: Path, expected: bytes, label: str) -> None:
    current = _read_regular_file(
        path,
        label,
        MAX_SIDECAR_SIZE_BYTES if label == "tonality sidecar" else None,
    )
    if current != expected:
        raise TonalitySidecarError(f"{label} changed while it was being validated")


def _reject_json_constant(value: str) -> Never:
    raise TonalitySidecarError(f"JSON constant {value} is not permitted")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TonalitySidecarError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_strict_json(raw: bytes) -> object:
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TonalitySidecarError(f"Tonality sidecar is not valid UTF-8 JSON: {exc}") from exc


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TonalitySidecarError(f"{name} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise TonalitySidecarError(f"{name} keys must be strings")
    return value


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TonalitySidecarError(f"{name} must be a non-empty string")
    return value


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TonalitySidecarError(f"{name} must be an integer")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    converted = _require_int(value, name)
    if converted < 0:
        raise TonalitySidecarError(f"{name} cannot be negative")
    return converted


def _require_sha256(value: object, name: str) -> str:
    converted = _require_string(value, name)
    if not _SHA256_PATTERN.fullmatch(converted):
        raise TonalitySidecarError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    return converted
