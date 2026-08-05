"""Unit tests for strict tonality labels, inference, and sidecars."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pretty_midi
import pytest

from midi_idea_generator.tonality import (
    MODE_NAMES,
    TONIC_NAMES,
    Tonality,
    TonalityError,
    TonalitySidecarError,
    infer_tonality,
    load_tonality_sidecar,
    normalize_mode,
    normalize_tonic,
    sidecar_path_for,
    transpose_tonic,
)


def _sidecar_payload(midi_path: Path, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "source_midi": midi_path.name,
        "source_sha256": hashlib.sha256(midi_path.read_bytes()).hexdigest(),
        "instrument_index": 0,
        "tonic": "E",
        "mode": "PHRYGIAN",
    }
    payload.update(overrides)
    return payload


def _write_sidecar(midi_path: Path, payload: object) -> Path:
    path = sidecar_path_for(midi_path)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_public_names_normalizers_and_transposition_are_canonical() -> None:
    assert TONIC_NAMES == (
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
    assert "BLUES" in MODE_NAMES
    assert normalize_tonic("f#") == "F_SHARP"
    assert normalize_tonic("Eb") == "D_SHARP"
    assert normalize_mode("natural minor") == "MINOR"
    assert transpose_tonic("B", 1) == "C"
    assert transpose_tonic("C", -1) == "B"
    assert transpose_tonic("UNKNOWN", 11) == "UNKNOWN"


def test_tonality_rejects_incoherent_unknown_and_invalid_confidence() -> None:
    with pytest.raises(TonalityError, match="UNKNOWN tonic requires UNKNOWN mode"):
        Tonality("UNKNOWN", "MINOR")
    with pytest.raises(TonalityError, match="requires both confidences"):
        Tonality("E", "MINOR", method="AUTO_SOURCE")
    with pytest.raises(TonalityError, match=r"inside \[0, 1\]"):
        Tonality(
            "E",
            "MINOR",
            method="AUTO_SOURCE",
            tonic_confidence=2.0,
            mode_confidence=0.5,
        )


def test_inference_selects_e_phrygian_and_is_transposition_equivariant(
    make_instrument,
) -> None:
    pitches = [52, 53, 55, 57, 59, 60, 62, 64]
    source = make_instrument(
        [(pitch, index * 0.25, index * 0.25 + 0.2) for index, pitch in enumerate(pitches)]
    )
    shifted = make_instrument(
        [
            (pitch + 3, index * 0.25, index * 0.25 + 0.2)
            for index, pitch in enumerate(pitches)
        ]
    )

    inferred = infer_tonality(source, method="AUTO_SOURCE")
    transposed_inference = infer_tonality(shifted, method="AUTO_SOURCE")

    assert inferred.tonic == "E"
    assert inferred.mode == "PHRYGIAN"
    assert inferred.method == "AUTO_SOURCE"
    assert 0.0 <= inferred.tonic_confidence <= 1.0
    assert 0.0 <= inferred.mode_confidence <= 1.0
    assert transposed_inference.tonic == transpose_tonic(inferred.tonic, 3)
    assert transposed_inference.mode == inferred.mode


def test_empty_instrument_is_the_only_inference_case_left_unknown() -> None:
    inferred = infer_tonality(pretty_midi.Instrument(program=0))
    assert inferred == Tonality.unknown()


def test_valid_sidecar_is_bound_to_exact_source_and_returns_manual_label(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    midi_path = write_midi_file(
        tmp_path / "riff.mid",
        [make_instrument([(52, 0.0, 0.5), (53, 0.5, 1.0)])],
    )
    payload = _sidecar_payload(midi_path)
    sidecar_path = _write_sidecar(midi_path, payload)
    midi = pretty_midi.PrettyMIDI(str(midi_path))

    sidecar = load_tonality_sidecar(
        midi_path,
        source_sha256=str(payload["source_sha256"]),
        midi=midi,
        instrument_index=0,
    )

    assert sidecar is not None
    assert sidecar.path == sidecar_path.resolve()
    assert sidecar.tonality == Tonality("E", "PHRYGIAN", method="MANUAL")
    assert sidecar.sha256 == hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    assert sidecar.size_bytes == sidecar_path.stat().st_size


def test_missing_sidecar_is_supported(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    midi_path = write_midi_file(
        tmp_path / "riff.mid", [make_instrument([(60, 0.0, 0.5)])]
    )
    digest = hashlib.sha256(midi_path.read_bytes()).hexdigest()
    assert (
        load_tonality_sidecar(
            midi_path,
            source_sha256=digest,
            midi=pretty_midi.PrettyMIDI(str(midi_path)),
            instrument_index=0,
        )
        is None
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"source_midi": "other.mid"}, "source_midi"),
        ({"source_sha256": "0" * 64}, "source_sha256"),
        ({"instrument_index": 1}, "instrument_index"),
        ({"tonic": "F#"}, "canonical"),
        ({"mode": "IONIAN"}, "canonical"),
        ({"tonic": "UNKNOWN", "mode": "MINOR"}, "UNKNOWN tonic"),
        ({"extra": True}, "exactly the required keys"),
    ],
)
def test_present_invalid_sidecar_is_never_silently_ignored(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
    overrides: dict[str, object],
    message: str,
) -> None:
    midi_path = write_midi_file(
        tmp_path / "riff.mid", [make_instrument([(60, 0.0, 0.5)])]
    )
    payload = _sidecar_payload(midi_path, **overrides)
    _write_sidecar(midi_path, payload)

    with pytest.raises(TonalitySidecarError, match=message):
        load_tonality_sidecar(
            midi_path,
            source_sha256=hashlib.sha256(midi_path.read_bytes()).hexdigest(),
            midi=pretty_midi.PrettyMIDI(str(midi_path)),
            instrument_index=0,
        )


def test_sidecar_symlink_is_rejected(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    midi_path = write_midi_file(
        tmp_path / "riff.mid", [make_instrument([(60, 0.0, 0.5)])]
    )
    real = tmp_path / "label.json"
    real.write_text(json.dumps(_sidecar_payload(midi_path)), encoding="utf-8")
    sidecar_path_for(midi_path).symlink_to(real.name)

    with pytest.raises(TonalitySidecarError, match="symlink"):
        load_tonality_sidecar(
            midi_path,
            source_sha256=hashlib.sha256(midi_path.read_bytes()).hexdigest(),
            midi=pretty_midi.PrettyMIDI(str(midi_path)),
            instrument_index=0,
        )
