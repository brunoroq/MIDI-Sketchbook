"""Tests for the optional, strict guitar-technique sidecar boundary."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable

import pytest

import midi_idea_generator.techniques as techniques_module
from midi_idea_generator.midi_io import read_midi
from midi_idea_generator.techniques import (
    MAX_SIDECAR_SIZE_BYTES,
    NoteRef,
    TechniqueSidecarError,
    TechniqueType,
    load_technique_sidecar,
    sidecar_path_for,
)


def _source_project(
    tmp_path: Path,
    make_instrument: Callable[..., Any],
    write_midi_file: Callable[..., Path],
) -> tuple[Path, Any, str, tuple[NoteRef, ...], dict[str, Any]]:
    source_path = write_midi_file(
        tmp_path / "nested" / "riff.mid",
        [
            make_instrument(
                [
                    (60, 0.0, 0.25, 91),
                    (64, 0.5, 0.75, 92),
                    (67, 1.0, 1.25, 93),
                ],
                program=30,
            )
        ],
        tempo_bpm=120.0,
    )
    midi = read_midi(source_path)
    digest = sha256(source_path.read_bytes()).hexdigest()
    notes = tuple(
        sorted(
            (
                NoteRef(
                    onset_tick=int(midi.time_to_tick(note.start)),
                    end_tick=int(midi.time_to_tick(note.end)),
                    pitch=note.pitch,
                    velocity=note.velocity,
                )
                for note in midi.instruments[0].notes
            ),
            key=lambda note: (note.onset_tick, note.pitch),
        )
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "source_midi": source_path.name,
        "source_sha256": digest,
        "ticks_per_quarter": midi.resolution,
        "instrument_index": 0,
        "coverage": "COMPLETE",
        "note_techniques": [],
        "palm_mute_ranges": [],
    }
    return source_path, midi, digest, notes, payload


def _note_payload(note: NoteRef) -> dict[str, int]:
    return {
        "onset_tick": note.onset_tick,
        "end_tick": note.end_tick,
        "pitch": note.pitch,
        "velocity": note.velocity,
    }


def _write_sidecar(source_path: Path, payload: dict[str, Any]) -> Path:
    path = sidecar_path_for(source_path)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _load(source_path: Path, midi: Any, digest: str):
    return load_technique_sidecar(
        source_path,
        source_sha256=digest,
        midi=midi,
        instrument_index=0,
    )


def test_sidecar_path_uses_full_midi_filename_and_fixed_suffix() -> None:
    assert sidecar_path_for("riff.mid") == Path("riff.mid.techniques.json")
    assert sidecar_path_for("parts/riff.midi") == Path(
        "parts/riff.midi.techniques.json"
    )


def test_missing_sidecar_is_a_supported_legacy_case(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    source, midi, digest, _, _ = _source_project(
        tmp_path, make_instrument, write_midi_file
    )

    assert _load(source, midi, digest) is None


def test_loads_all_techniques_and_expands_palm_mute_by_note_onset(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    source, midi, digest, notes, payload = _source_project(
        tmp_path, make_instrument, write_midi_file
    )
    payload["note_techniques"] = [
        {
            "note": _note_payload(notes[1]),
            # Deliberately non-canonical input order.
            "techniques": [
                {"type": "VIBRATO"},
                {"type": "SLIDE", "direction": "UP", "target_pitch": 67},
            ],
        },
        {
            "note": _note_payload(notes[0]),
            "techniques": [{"type": "DEAD_NOTE"}],
        },
    ]
    payload["palm_mute_ranges"] = [
        {"start_tick": notes[1].onset_tick, "end_tick": notes[2].end_tick}
    ]
    sidecar_path = _write_sidecar(source, payload)

    sidecar = _load(source, midi, digest)

    assert sidecar is not None
    assert sidecar.path == sidecar_path.resolve()
    assert sidecar.fingerprint == sha256(sidecar_path.read_bytes()).hexdigest()
    assert sidecar.sha256 == sidecar.fingerprint
    assert sidecar.size_bytes == len(sidecar_path.read_bytes())
    assert sidecar.source_midi == "riff.mid"
    assert sidecar.source_sha256 == digest
    assert sidecar.ticks_per_quarter == 480
    assert sidecar.instrument_index == 0
    assert sidecar.coverage == "COMPLETE"
    assert [entry.note for entry in sidecar.note_techniques] == list(notes)
    assert [
        technique.type for technique in sidecar.techniques_for(notes[0])
    ] == [TechniqueType.DEAD_NOTE]
    second = sidecar.techniques_for(notes[1])
    assert [technique.type for technique in second] == [
        TechniqueType.PALM_MUTE,
        TechniqueType.SLIDE_UP,
        TechniqueType.VIBRATO,
    ]
    assert second[1].target_pitch == 67
    assert [
        technique.type for technique in sidecar.techniques_for(notes[2])
    ] == [TechniqueType.PALM_MUTE]


def test_annotations_lookup_is_immutable(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    source, midi, digest, notes, payload = _source_project(
        tmp_path, make_instrument, write_midi_file
    )
    payload["note_techniques"] = [
        {
            "note": _note_payload(notes[0]),
            "techniques": [{"type": "VIBRATO"}],
        }
    ]
    _write_sidecar(source, payload)
    sidecar = _load(source, midi, digest)
    assert sidecar is not None

    with pytest.raises(TypeError):
        sidecar.annotations_by_note[notes[0]] = ()  # type: ignore[index]


def test_empty_complete_sidecar_is_valid(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    source, midi, digest, _, payload = _source_project(
        tmp_path, make_instrument, write_midi_file
    )
    _write_sidecar(source, payload)

    sidecar = _load(source, midi, digest)

    assert sidecar is not None
    assert sidecar.note_techniques == ()
    assert sidecar.palm_mute_ranges == ()
    assert dict(sidecar.annotations_by_note) == {}


def test_slide_down_becomes_final_domain_type(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    source, midi, digest, notes, payload = _source_project(
        tmp_path, make_instrument, write_midi_file
    )
    payload["note_techniques"] = [
        {
            "note": _note_payload(notes[2]),
            "techniques": [
                {"type": "SLIDE", "direction": "DOWN", "target_pitch": 64}
            ],
        }
    ]
    _write_sidecar(source, payload)

    sidecar = _load(source, midi, digest)

    assert sidecar is not None
    assert sidecar.techniques_for(notes[2])[0].type == TechniqueType.SLIDE_DOWN
    assert sidecar.techniques_for(notes[2])[0].target_pitch == 64


def test_palm_mute_range_is_start_inclusive_and_end_exclusive(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    source, midi, digest, notes, payload = _source_project(
        tmp_path, make_instrument, write_midi_file
    )
    payload["palm_mute_ranges"] = [
        {"start_tick": notes[0].onset_tick, "end_tick": notes[1].onset_tick}
    ]
    _write_sidecar(source, payload)

    sidecar = _load(source, midi, digest)

    assert sidecar is not None
    assert sidecar.techniques_for(notes[0])[0].type == TechniqueType.PALM_MUTE
    assert sidecar.techniques_for(notes[1]) == ()


def test_canonicalizes_note_and_nonadjacent_range_order(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    source, midi, digest, notes, payload = _source_project(
        tmp_path, make_instrument, write_midi_file
    )
    payload["note_techniques"] = [
        {
            "note": _note_payload(notes[2]),
            "techniques": [{"type": "VIBRATO"}],
        },
        {
            "note": _note_payload(notes[0]),
            "techniques": [{"type": "VIBRATO"}],
        },
    ]
    payload["palm_mute_ranges"] = [
        {"start_tick": notes[2].onset_tick, "end_tick": notes[2].end_tick},
        {"start_tick": notes[0].onset_tick, "end_tick": notes[0].end_tick},
    ]
    _write_sidecar(source, payload)

    sidecar = _load(source, midi, digest)

    assert sidecar is not None
    assert [entry.note for entry in sidecar.note_techniques] == [notes[0], notes[2]]
    assert [interval.start_tick for interval in sidecar.palm_mute_ranges] == [
        notes[0].onset_tick,
        notes[2].onset_tick,
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.pop("coverage"), "missing field"),
        (lambda value: value.__setitem__("unexpected", 1), "unknown field"),
        (lambda value: value.__setitem__("schema_version", 2), "schema_version"),
        (lambda value: value.__setitem__("source_midi", "other.mid"), "source_midi"),
        (lambda value: value.__setitem__("coverage", "PARTIAL"), "coverage"),
        (lambda value: value.__setitem__("ticks_per_quarter", True), "integer"),
        (lambda value: value.__setitem__("note_techniques", {}), "array"),
        (lambda value: value.__setitem__("palm_mute_ranges", {}), "array"),
    ],
)
def test_rejects_invalid_root_contract(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
    mutate: Callable[[dict[str, Any]], Any],
    message: str,
) -> None:
    source, midi, digest, _, payload = _source_project(
        tmp_path, make_instrument, write_midi_file
    )
    mutate(payload)
    _write_sidecar(source, payload)

    with pytest.raises(TechniqueSidecarError, match=message):
        _load(source, midi, digest)


def test_rejects_duplicate_json_keys(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    source, midi, digest, _, payload = _source_project(
        tmp_path, make_instrument, write_midi_file
    )
    serialized = json.dumps(payload).replace(
        '"schema_version": 1',
        '"schema_version": 1, "schema_version": 1',
        1,
    )
    sidecar_path_for(source).write_text(serialized, encoding="utf-8")

    with pytest.raises(TechniqueSidecarError, match="Duplicate JSON key"):
        _load(source, midi, digest)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"\xff\xfe", "UTF-8 JSON"),
        (b'{"value": NaN}', "Non-finite"),
        (b"[]", "must be an object"),
    ],
)
def test_rejects_non_strict_json(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
    raw: bytes,
    message: str,
) -> None:
    source, midi, digest, _, _ = _source_project(
        tmp_path, make_instrument, write_midi_file
    )
    sidecar_path_for(source).write_bytes(raw)

    with pytest.raises(TechniqueSidecarError, match=message):
        _load(source, midi, digest)


def test_rejects_sidecar_symlink_even_when_broken(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    source, midi, digest, _, _ = _source_project(
        tmp_path, make_instrument, write_midi_file
    )
    sidecar_path_for(source).symlink_to(tmp_path / "missing.json")

    with pytest.raises(TechniqueSidecarError, match="cannot be a symlink"):
        _load(source, midi, digest)


def test_rejects_midi_source_symlink(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    source, midi, digest, _, payload = _source_project(
        tmp_path, make_instrument, write_midi_file
    )
    linked_source = source.with_name("linked.mid")
    linked_source.symlink_to(source)
    payload["source_midi"] = linked_source.name
    _write_sidecar(linked_source, payload)

    with pytest.raises(TechniqueSidecarError, match="MIDI source cannot be a symlink"):
        load_technique_sidecar(
            linked_source,
            source_sha256=digest,
            midi=midi,
            instrument_index=0,
        )


def test_rejects_oversized_sidecar_before_parsing(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    source, midi, digest, _, payload = _source_project(
        tmp_path, make_instrument, write_midi_file
    )
    serialized = json.dumps(payload).encode("utf-8")
    raw = serialized + b" " * (MAX_SIDECAR_SIZE_BYTES + 1 - len(serialized))
    sidecar_path_for(source).write_bytes(raw)

    with pytest.raises(TechniqueSidecarError, match="exceeds"):
        _load(source, midi, digest)


def test_rejects_annotation_count_over_limit(
    tmp_path: Path, make_instrument, write_midi_file, monkeypatch
) -> None:
    source, midi, digest, notes, payload = _source_project(
        tmp_path, make_instrument, write_midi_file
    )
    payload["note_techniques"] = [
        {
            "note": _note_payload(notes[0]),
            "techniques": [{"type": "VIBRATO"}, {"type": "DEAD_NOTE"}],
        }
    ]
    _write_sidecar(source, payload)
    monkeypatch.setattr(techniques_module, "MAX_ANNOTATIONS", 1)

    with pytest.raises(TechniqueSidecarError, match="maximum"):
        _load(source, midi, digest)


def test_rejects_supplied_or_declared_source_hash_mismatch(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    source, midi, digest, _, payload = _source_project(
        tmp_path, make_instrument, write_midi_file
    )
    _write_sidecar(source, payload)
    different = "0" * 64 if digest != "0" * 64 else "1" * 64

    with pytest.raises(TechniqueSidecarError, match="current MIDI file"):
        load_technique_sidecar(
            source,
            source_sha256=different,
            midi=midi,
            instrument_index=0,
        )

    payload["source_sha256"] = different
    _write_sidecar(source, payload)
    with pytest.raises(TechniqueSidecarError, match="Sidecar source_sha256"):
        _load(source, midi, digest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ticks_per_quarter", 960, "MIDI resolution"),
        ("instrument_index", 1, "selected instrument"),
    ],
)
def test_rejects_midi_binding_mismatch(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
    field: str,
    value: int,
    message: str,
) -> None:
    source, midi, digest, _, payload = _source_project(
        tmp_path, make_instrument, write_midi_file
    )
    payload[field] = value
    _write_sidecar(source, payload)

    with pytest.raises(TechniqueSidecarError, match=message):
        _load(source, midi, digest)


def test_rejects_selected_instrument_outside_midi(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    source, midi, digest, _, payload = _source_project(
        tmp_path, make_instrument, write_midi_file
    )
    payload["instrument_index"] = 1
    _write_sidecar(source, payload)

    with pytest.raises(TechniqueSidecarError, match="outside"):
        load_technique_sidecar(
            source,
            source_sha256=digest,
            midi=midi,
            instrument_index=1,
        )


def test_rejects_note_reference_that_does_not_match_selected_instrument(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    source, midi, digest, notes, payload = _source_project(
        tmp_path, make_instrument, write_midi_file
    )
    wrong_note = _note_payload(notes[0])
    wrong_note["end_tick"] += 1
    payload["note_techniques"] = [
        {"note": wrong_note, "techniques": [{"type": "VIBRATO"}]}
    ]
    _write_sidecar(source, payload)

    with pytest.raises(TechniqueSidecarError, match="does not match"):
        _load(source, midi, digest)


def test_rejects_duplicate_note_entries(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    source, midi, digest, notes, payload = _source_project(
        tmp_path, make_instrument, write_midi_file
    )
    entry = {
        "note": _note_payload(notes[0]),
        "techniques": [{"type": "VIBRATO"}],
    }
    payload["note_techniques"] = [entry, deepcopy(entry)]
    _write_sidecar(source, payload)

    with pytest.raises(TechniqueSidecarError, match="Duplicate note_techniques"):
        _load(source, midi, digest)


@pytest.mark.parametrize(
    ("techniques", "message"),
    [
        ([], "cannot be empty"),
        ([{"type": "PALM_MUTE"}], "must be DEAD_NOTE"),
        ([{"type": "VIBRATO", "amount": 2}], "unknown field"),
        ([{"type": "VIBRATO"}, {"type": "VIBRATO"}], "duplicate"),
        ([{"type": "DEAD_NOTE"}, {"type": "VIBRATO"}], "cannot coexist"),
        (
            [
                {"type": "SLIDE", "direction": "UP", "target_pitch": 58},
            ],
            "above",
        ),
        (
            [
                {"type": "SLIDE", "direction": "SIDEWAYS", "target_pitch": 65},
            ],
            "UP or DOWN",
        ),
        (
            [
                {"type": "SLIDE", "direction": "UP", "target_pitch": 85},
            ],
            "1-24",
        ),
    ],
)
def test_rejects_invalid_note_techniques(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
    techniques: list[dict[str, Any]],
    message: str,
) -> None:
    source, midi, digest, notes, payload = _source_project(
        tmp_path, make_instrument, write_midi_file
    )
    payload["note_techniques"] = [
        {"note": _note_payload(notes[0]), "techniques": techniques}
    ]
    _write_sidecar(source, payload)

    with pytest.raises(TechniqueSidecarError, match=message):
        _load(source, midi, digest)


@pytest.mark.parametrize(
    ("ranges", "message"),
    [
        ([{"start_tick": 10, "end_tick": 10}], "greater"),
        ([{"start_tick": -1, "end_tick": 10}], "non-negative"),
        ([{"start_tick": 0, "end_tick": 1201}], "structural duration"),
        ([{"start_tick": 241, "end_tick": 479}], "does not affect"),
        (
            [
                {"start_tick": 0, "end_tick": 600},
                {"start_tick": 480, "end_tick": 1200},
            ],
            "overlap",
        ),
        (
            [
                {"start_tick": 0, "end_tick": 480},
                {"start_tick": 480, "end_tick": 1200},
            ],
            "must be merged",
        ),
        ([{"start_tick": 0, "end_tick": 240, "extra": 1}], "unknown field"),
    ],
)
def test_rejects_invalid_palm_mute_ranges(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
    ranges: list[dict[str, Any]],
    message: str,
) -> None:
    source, midi, digest, _, payload = _source_project(
        tmp_path, make_instrument, write_midi_file
    )
    payload["palm_mute_ranges"] = ranges
    _write_sidecar(source, payload)

    with pytest.raises(TechniqueSidecarError, match=message):
        _load(source, midi, digest)


def test_exact_duplicate_source_notes_have_one_canonical_annotation(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    source = write_midi_file(
        tmp_path / "duplicate.mid",
        [make_instrument([(60, 0.0, 0.25, 90), (60, 0.0, 0.25, 90)])],
    )
    midi = read_midi(source)
    digest = sha256(source.read_bytes()).hexdigest()
    note = midi.instruments[0].notes[0]
    reference = NoteRef(
        int(midi.time_to_tick(note.start)),
        int(midi.time_to_tick(note.end)),
        note.pitch,
        note.velocity,
    )
    payload = {
        "schema_version": 1,
        "source_midi": source.name,
        "source_sha256": digest,
        "ticks_per_quarter": midi.resolution,
        "instrument_index": 0,
        "coverage": "COMPLETE",
        "note_techniques": [
            {
                "note": _note_payload(reference),
                "techniques": [{"type": "DEAD_NOTE"}],
            }
        ],
        "palm_mute_ranges": [],
    }
    _write_sidecar(source, payload)

    sidecar = _load(source, midi, digest)

    assert sidecar is not None
    assert len(sidecar.note_techniques) == 1
    assert sidecar.techniques_for(reference)[0].type == TechniqueType.DEAD_NOTE
