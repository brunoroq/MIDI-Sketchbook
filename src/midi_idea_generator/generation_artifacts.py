"""Validated artifact export for decoded symbolic guitar generations.

This module deliberately starts after sampling.  It receives the
``DecodedMidi`` already validated by :mod:`midi_idea_generator.tokenizer` and
writes four human-facing artifacts:

* a Standard MIDI file;
* an exact token/provenance JSON document;
* a generated-technique sidecar; and
* an optional headless piano-roll PNG.

The generated-technique suffix is intentionally different from the ingestible
``.mid.techniques.json`` source-sidecar suffix.  GuitarREMI v1 retains slide
direction, but not the source sidecar's exact target pitch, and represents palm
mute as note-indexed state transitions.  Pretending this reduced information
were a lossless source annotation would be unsafe.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import logging
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from symusic import ControlChange

from .midi_io import detect_pitch_bend_range, read_midi
from .tokenizer import (
    TECHNIQUE_TYPES,
    DecodedMidi,
    TechniqueAnnotation,
)
from .utils import write_json


GENERATION_ARTIFACT_SCHEMA_VERSION = 1
GENERATED_TECHNIQUE_SCHEMA_VERSION = 1
PITCH_BEND_SENSITIVITY_SEMITONES = 6

_TECHNIQUE_ORDER = {
    technique_type: index
    for index, technique_type in enumerate(TECHNIQUE_TYPES)
}
_RPN_CONTROLS = ((101, 0), (100, 0), (6, 6))
_SAFE_STEM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class GenerationArtifactError(RuntimeError):
    """Raised when decoded output cannot be published without data loss."""


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    """One published artifact and its exact content identity."""

    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class GenerationArtifacts:
    """Files produced for one decoded generated sequence."""

    midi: ArtifactFile
    tokens: ArtifactFile
    techniques: ArtifactFile
    visualization: ArtifactFile | None
    num_notes: int
    num_pitch_bends: int
    num_techniques: int


def write_generation_artifacts(
    decoded: DecodedMidi,
    token_ids: Sequence[int],
    tokens: Sequence[str],
    provenance: Mapping[str, Any],
    output_stem: str | Path,
    *,
    program: int,
    visualization_enabled: bool = True,
    dpi: int = 150,
) -> GenerationArtifacts:
    """Validate and publish one decoded generation.

    ``output_stem`` is a filename stem, not a directory or a path with a MIDI
    extension.  For ``outputs/generated/sample-001`` this function writes:

    ``sample-001.mid``
        MIDI notes, tempo, program and pitch-wheel events.  A constant
        time-zero RPN declaring +/-6 semitones is added whenever pitch-bend
        events exist.
    ``sample-001.tokens.json``
        Exact token identifiers, rendered token strings, provenance and hashes
        of the companion artifacts.
    ``sample-001.techniques.generated.json``
        Exact ``{type, note_index}`` annotations plus their reduced symbolic
        semantics.  Its suffix cannot be mistaken for a source annotation.
    ``sample-001.piano-roll.png``
        Optional headless visualization of notes, bends and technique labels.

    Existing targets are never overwritten.  All data is rendered and checked
    in a sibling staging directory before publication, and handled failures
    remove any newly published member of the artifact set.
    """

    ids = _validate_token_ids(token_ids)
    rendered_tokens = _validate_tokens(tokens, expected_length=len(ids))
    normalized_provenance = _normalize_json_mapping(provenance, "provenance")
    resolved_program = _require_program(program)
    if not isinstance(visualization_enabled, bool):
        raise GenerationArtifactError("visualization_enabled must be a boolean.")
    resolved_dpi = _require_dpi(dpi)

    score, techniques = _validated_decoded(decoded, resolved_program)
    track = score.tracks[0]
    num_notes = len(track.notes)
    num_pitch_bends = len(track.pitch_bends)

    paths = _artifact_paths(output_stem, visualization_enabled)
    paths["midi"].parent.mkdir(parents=True, exist_ok=True)
    _reject_existing_targets(paths)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{paths['midi'].stem}-",
            dir=paths["midi"].parent,
        )
    )
    staged = {
        kind: staging_dir / path.name
        for kind, path in paths.items()
    }
    published: list[Path] = []
    try:
        rendered_score = copy.copy(score)
        rendered_track = rendered_score.tracks[0]
        if rendered_track.controls:
            raise GenerationArtifactError(
                "Decoded generations cannot contain pre-existing control changes."
            )
        if rendered_track.pitch_bends:
            rendered_track.controls.extend(
                ControlChange(0, number, value)
                for number, value in _RPN_CONTROLS
            )

        _write_bytes(staged["midi"], rendered_score.dumps_midi())
        _validate_midi_round_trip(
            score,
            staged["midi"],
            program=resolved_program,
            expect_rpn=bool(rendered_track.pitch_bends),
        )
        midi_fingerprint = _fingerprint(staged["midi"], paths["midi"])

        technique_payload = _technique_payload(
            midi=midi_fingerprint,
            score=score,
            program=resolved_program,
            techniques=techniques,
        )
        write_json(staged["techniques"], technique_payload)
        technique_fingerprint = _fingerprint(
            staged["techniques"], paths["techniques"]
        )

        visualization_fingerprint: ArtifactFile | None = None
        if visualization_enabled:
            _render_piano_roll(
                score,
                techniques,
                staged["visualization"],
                program=resolved_program,
                dpi=resolved_dpi,
            )
            visualization_fingerprint = _fingerprint(
                staged["visualization"], paths["visualization"]
            )

        token_payload = {
            "schema_version": GENERATION_ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "generated_token_sequence",
            "ids": list(ids),
            "tokens": list(rendered_tokens),
            "programs": [[resolved_program, False]],
            "provenance": normalized_provenance,
            "summary": {
                "num_tokens": len(ids),
                "num_notes": num_notes,
                "num_pitch_bends": num_pitch_bends,
                "num_techniques": len(techniques),
            },
            "artifacts": {
                "midi": _artifact_reference(midi_fingerprint),
                "techniques": _artifact_reference(technique_fingerprint),
                "visualization": (
                    _artifact_reference(visualization_fingerprint)
                    if visualization_fingerprint is not None
                    else None
                ),
            },
        }
        write_json(staged["tokens"], token_payload)
        token_fingerprint = _fingerprint(staged["tokens"], paths["tokens"])

        for kind in ("midi", "techniques", "visualization", "tokens"):
            if kind not in staged:
                continue
            os.replace(staged[kind], paths[kind])
            published.append(paths[kind])

        return GenerationArtifacts(
            midi=midi_fingerprint,
            tokens=token_fingerprint,
            techniques=technique_fingerprint,
            visualization=visualization_fingerprint,
            num_notes=num_notes,
            num_pitch_bends=num_pitch_bends,
            num_techniques=len(techniques),
        )
    except GenerationArtifactError:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        for path in published:
            path.unlink(missing_ok=True)
        raise GenerationArtifactError(
            f"Could not write generation artifacts for '{paths['midi'].stem}': "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _validated_decoded(
    decoded: DecodedMidi, program: int
) -> tuple[Any, tuple[TechniqueAnnotation, ...]]:
    if not isinstance(decoded, DecodedMidi):
        raise GenerationArtifactError("decoded must be a DecodedMidi value.")
    score = decoded.score
    if str(getattr(score, "ttype", "")) != "symusic.core.Tick":
        raise GenerationArtifactError("Decoded score must use integer tick timing.")
    tpq = getattr(score, "tpq", None)
    if isinstance(tpq, bool) or not isinstance(tpq, int) or tpq <= 0:
        raise GenerationArtifactError("Decoded score must have a positive TPQ.")
    if len(score.tracks) != 1:
        raise GenerationArtifactError("Decoded score must contain exactly one track.")
    track = score.tracks[0]
    if bool(track.is_drum):
        raise GenerationArtifactError("Generated guitar MIDI cannot be a drum track.")
    if int(track.program) != program:
        raise GenerationArtifactError(
            f"Requested program {program} does not match decoded program "
            f"{int(track.program)}."
        )
    if not track.notes:
        raise GenerationArtifactError("Decoded generation contains no notes.")

    note_count = len(track.notes)
    techniques: list[TechniqueAnnotation] = []
    seen: set[tuple[int, str]] = set()
    for index, annotation in enumerate(decoded.techniques):
        if not isinstance(annotation, TechniqueAnnotation):
            raise GenerationArtifactError(
                f"decoded.techniques[{index}] must be a TechniqueAnnotation."
            )
        if annotation.type not in _TECHNIQUE_ORDER:
            raise GenerationArtifactError(
                f"Unknown generated technique type: {annotation.type!r}."
            )
        if (
            isinstance(annotation.note_index, bool)
            or not isinstance(annotation.note_index, int)
            or not 0 <= annotation.note_index < note_count
        ):
            raise GenerationArtifactError(
                f"Technique note_index {annotation.note_index!r} is outside the "
                f"{note_count}-note score."
            )
        identity = (annotation.note_index, annotation.type)
        if identity in seen:
            raise GenerationArtifactError(
                f"Duplicate generated technique {annotation.type} for note "
                f"{annotation.note_index}."
            )
        seen.add(identity)
        techniques.append(annotation)
    canonical = tuple(
        sorted(
            techniques,
            key=lambda item: (item.note_index, _TECHNIQUE_ORDER[item.type]),
        )
    )
    if tuple(techniques) != canonical:
        raise GenerationArtifactError(
            "Generated techniques must use canonical note-index/type order."
        )
    return score, canonical


def _validate_midi_round_trip(
    score: Any,
    path: Path,
    *,
    program: int,
    expect_rpn: bool,
) -> None:
    midi = read_midi(path)
    if int(midi.resolution) != int(score.tpq):
        raise GenerationArtifactError("MIDI export changed ticks_per_quarter.")
    if len(midi.instruments) != 1:
        raise GenerationArtifactError("MIDI export changed the one-track contract.")
    instrument = midi.instruments[0]
    if instrument.is_drum or int(instrument.program) != program:
        raise GenerationArtifactError("MIDI export changed the guitar program.")

    track = score.tracks[0]
    expected_notes = tuple(
        sorted(
            (
                int(note.time),
                int(note.end),
                int(note.pitch),
                int(note.velocity),
            )
            for note in track.notes
        )
    )
    actual_notes = tuple(
        sorted(
            (
                int(midi.time_to_tick(float(note.start))),
                int(midi.time_to_tick(float(note.end))),
                int(note.pitch),
                int(note.velocity),
            )
            for note in instrument.notes
        )
    )
    if actual_notes != expected_notes:
        raise GenerationArtifactError("MIDI export changed note events.")

    expected_bends = tuple(
        sorted((int(bend.time), int(bend.value)) for bend in track.pitch_bends)
    )
    actual_bends = tuple(
        sorted(
            (
                int(midi.time_to_tick(float(bend.time))),
                int(bend.pitch),
            )
            for bend in instrument.pitch_bends
        )
    )
    if actual_bends != expected_bends:
        raise GenerationArtifactError("MIDI export changed pitch-bend events.")

    controls = tuple(
        (
            int(midi.time_to_tick(float(change.time))),
            int(change.number),
            int(change.value),
        )
        for change in instrument.control_changes
    )
    expected_controls = (
        tuple((0, number, value) for number, value in _RPN_CONTROLS)
        if expect_rpn
        else ()
    )
    if controls != expected_controls:
        raise GenerationArtifactError(
            "MIDI export did not preserve the canonical pitch-bend RPN order."
        )
    bend_range, issue = detect_pitch_bend_range(instrument.control_changes)
    if expect_rpn and (issue is not None or bend_range != 6.0):
        raise GenerationArtifactError(
            "MIDI export does not declare a constant +/-6-semitone bend range."
        )
    if not expect_rpn and (issue is not None or bend_range is not None):
        raise GenerationArtifactError(
            "MIDI without pitch bends unexpectedly contains bend-range controls."
        )


def _technique_payload(
    *,
    midi: ArtifactFile,
    score: Any,
    program: int,
    techniques: Sequence[TechniqueAnnotation],
) -> dict[str, Any]:
    return {
        "schema_version": GENERATED_TECHNIQUE_SCHEMA_VERSION,
        "artifact_type": "generated_guitar_techniques",
        "midi_file": midi.path.name,
        "midi_sha256": midi.sha256,
        "midi_size_bytes": midi.size_bytes,
        "ticks_per_quarter": int(score.tpq),
        "instrument_index": 0,
        "program": program,
        "coverage": "COMPLETE",
        "representation": "GuitarREMI_directional_v1",
        "note_index_order": "onset_tick,pitch,end_tick,velocity",
        "slide_semantics": (
            "SLIDE_UP and SLIDE_DOWN preserve direction only; target pitch is "
            "not encoded by GuitarREMI v1."
        ),
        "palm_mute_semantics": (
            "PALM_MUTE_ON and PALM_MUTE_OFF change state at the indexed "
            "canonical note."
        ),
        "techniques": [
            {"type": annotation.type, "note_index": annotation.note_index}
            for annotation in techniques
        ],
    }


def _render_piano_roll(
    score: Any,
    techniques: Sequence[TechniqueAnnotation],
    path: Path,
    *,
    program: int,
    dpi: int,
) -> None:
    FigureCanvasAgg, Figure, Rectangle = _load_matplotlib()

    track = score.tracks[0]
    tpq = int(score.tpq)
    notes = sorted(
        track.notes,
        key=lambda note: (
            int(note.time),
            int(note.pitch),
            int(note.end),
            int(note.velocity),
        ),
    )
    duration_beats = max(float(score.end()) / tpq, 0.25)
    has_bends = bool(track.pitch_bends)
    figure = Figure(
        figsize=(12.0, 7.0 if has_bends else 5.5),
        constrained_layout=True,
    )
    FigureCanvasAgg(figure)
    if has_bends:
        grid = figure.add_gridspec(2, 1, height_ratios=(4, 1), hspace=0.08)
        note_axis = figure.add_subplot(grid[0])
        bend_axis = figure.add_subplot(grid[1], sharex=note_axis)
    else:
        note_axis = figure.add_subplot(1, 1, 1)
        bend_axis = None

    for note in notes:
        start = int(note.time) / tpq
        duration = max(int(note.duration) / tpq, 1.0 / tpq)
        velocity = int(note.velocity)
        color_value = 0.25 + 0.65 * velocity / 127.0
        note_axis.add_patch(
            Rectangle(
                (start, int(note.pitch) - 0.4),
                duration,
                0.8,
                facecolor=(0.12, 0.35, color_value),
                edgecolor=(0.04, 0.12, 0.25),
                linewidth=0.6,
            )
        )

    labels_by_note: dict[int, list[str]] = {}
    for annotation in techniques:
        labels_by_note.setdefault(annotation.note_index, []).append(annotation.type)
    for note_index, labels in labels_by_note.items():
        note = notes[note_index]
        note_axis.text(
            int(note.time) / tpq,
            int(note.pitch) + 0.5,
            " / ".join(labels),
            fontsize=6.5,
            rotation=25,
            ha="left",
            va="bottom",
            color="#6a1b9a",
            clip_on=True,
        )

    pitches = [int(note.pitch) for note in notes]
    note_axis.set_ylim(max(0, min(pitches) - 2), min(127, max(pitches) + 3))
    note_axis.set_xlim(0.0, duration_beats)
    note_axis.set_ylabel("MIDI pitch")
    note_axis.set_title(
        f"Generated guitar piano roll | program {program} | TPQ {tpq}"
    )
    note_axis.grid(axis="y", color="#dddddd", linewidth=0.4)
    for bar_start in range(0, math.ceil(duration_beats) + 1, 4):
        note_axis.axvline(bar_start, color="#aaaaaa", linewidth=0.6, zorder=0)

    if bend_axis is not None:
        ordered_bends = sorted(
            track.pitch_bends, key=lambda bend: (int(bend.time), int(bend.value))
        )
        bend_times = [int(bend.time) / tpq for bend in ordered_bends]
        bend_values = [
            int(bend.value) * PITCH_BEND_SENSITIVITY_SEMITONES / 8192.0
            for bend in ordered_bends
        ]
        if bend_times[-1] < duration_beats:
            bend_times.append(duration_beats)
            bend_values.append(bend_values[-1])
        bend_axis.step(
            bend_times,
            bend_values,
            where="post",
            color="#d84315",
            linewidth=1.1,
        )
        bend_axis.axhline(0.0, color="#777777", linewidth=0.6)
        bend_axis.set_ylim(-6.25, 6.25)
        bend_axis.set_ylabel("Bend\n(semitones)")
        bend_axis.set_xlabel("Beats")
        bend_axis.grid(color="#e0e0e0", linewidth=0.4)
    else:
        note_axis.set_xlabel("Beats")

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="png",
        dpi=dpi,
        metadata={"Software": "MIDI-Sketchbook"},
    )
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _load_matplotlib() -> tuple[Any, Any, Any]:
    """Import the headless renderer without depending on a writable home."""

    original_config = os.environ.get("MPLCONFIGDIR")
    temporary_config: Path | None = None
    if original_config is None:
        temporary_config = Path(
            tempfile.mkdtemp(prefix="midi-sketchbook-matplotlib-")
        )
        os.environ["MPLCONFIGDIR"] = str(temporary_config)
    try:
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Rectangle
    except ModuleNotFoundError as exc:
        raise GenerationArtifactError(
            "Piano-roll visualization requires the declared matplotlib dependency."
        ) from exc
    finally:
        if temporary_config is not None:
            os.environ.pop("MPLCONFIGDIR", None)
            shutil.rmtree(temporary_config, ignore_errors=True)
    return FigureCanvasAgg, Figure, Rectangle


def _artifact_paths(
    output_stem: str | Path, visualization_enabled: bool
) -> dict[str, Path]:
    raw = Path(output_stem).expanduser()
    if raw.name in {"", ".", ".."} or not _SAFE_STEM.fullmatch(raw.name):
        raise GenerationArtifactError(
            "output_stem must end in a safe filename stem of at most 128 characters."
        )
    if raw.suffix.lower() in {".mid", ".midi", ".json", ".png"}:
        raise GenerationArtifactError(
            "output_stem must not include .mid, .midi, .json, or .png."
        )
    parent = raw.parent.resolve()
    base = raw.name
    paths = {
        "midi": parent / f"{base}.mid",
        "tokens": parent / f"{base}.tokens.json",
        "techniques": parent / f"{base}.techniques.generated.json",
    }
    if visualization_enabled:
        paths["visualization"] = parent / f"{base}.piano-roll.png"
    return paths


def _reject_existing_targets(paths: Mapping[str, Path]) -> None:
    existing = [path for path in paths.values() if path.exists() or path.is_symlink()]
    if existing:
        rendered = ", ".join(path.name for path in existing)
        raise GenerationArtifactError(
            f"Generation artifacts already exist and will not be overwritten: {rendered}."
        )


def _validate_token_ids(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise GenerationArtifactError("token_ids must be a sequence of integers.")
    ids: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GenerationArtifactError(
                f"token_ids[{index}] must be a nonnegative integer."
            )
        ids.append(value)
    if not ids:
        raise GenerationArtifactError("token_ids cannot be empty.")
    return tuple(ids)


def _validate_tokens(values: Sequence[str], *, expected_length: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise GenerationArtifactError("tokens must be a sequence of strings.")
    tokens: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value or "\x00" in value:
            raise GenerationArtifactError(
                f"tokens[{index}] must be a non-empty string."
            )
        tokens.append(value)
    if len(tokens) != expected_length:
        raise GenerationArtifactError(
            "tokens and token_ids must have exactly the same length."
        )
    return tuple(tokens)


def _normalize_json_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GenerationArtifactError(f"{name} must be a JSON object.")
    normalized = _normalize_json_value(value, name)
    assert isinstance(normalized, dict)
    return normalized


def _normalize_json_value(value: object, name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GenerationArtifactError(f"{name} cannot contain NaN or infinity.")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or "\x00" in key:
                raise GenerationArtifactError(
                    f"All keys in {name} must be non-empty strings."
                )
            normalized[key] = _normalize_json_value(item, f"{name}.{key}")
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _normalize_json_value(item, f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise GenerationArtifactError(
        f"{name} contains a non-JSON value of type {type(value).__name__}."
    )


def _require_program(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 127:
        raise GenerationArtifactError("program must be an integer from 0 to 127.")
    return value


def _require_dpi(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 72 <= value <= 600:
        raise GenerationArtifactError("dpi must be an integer from 72 to 600.")
    return value


def _write_bytes(path: Path, payload: bytes) -> None:
    if not isinstance(payload, bytes) or not payload:
        raise GenerationArtifactError("MIDI serializer returned no bytes.")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fingerprint(path: Path, final_path: Path) -> ArtifactFile:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    if size <= 0:
        raise GenerationArtifactError(f"Generated artifact is empty: {final_path.name}.")
    return ArtifactFile(path=final_path, sha256=digest.hexdigest(), size_bytes=size)


def _artifact_reference(artifact: ArtifactFile) -> dict[str, Any]:
    return {
        "file": artifact.path.name,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
    }


__all__ = [
    "ArtifactFile",
    "GENERATION_ARTIFACT_SCHEMA_VERSION",
    "GENERATED_TECHNIQUE_SCHEMA_VERSION",
    "GenerationArtifactError",
    "GenerationArtifacts",
    "PITCH_BEND_SENSITIVITY_SEMITONES",
    "write_generation_artifacts",
]
