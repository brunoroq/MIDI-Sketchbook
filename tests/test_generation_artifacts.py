"""Tests for validated MIDI, metadata, sidecar, and piano-roll export."""

from __future__ import annotations

from hashlib import sha256
import importlib
from importlib.util import find_spec
import json
from pathlib import Path

import mido
import pretty_midi
import pytest

from midi_idea_generator.generation_artifacts import (
    GenerationArtifactError,
    write_generation_artifacts,
)
from midi_idea_generator.midi_io import detect_pitch_bend_range
from midi_idea_generator.techniques import sidecar_path_for
from midi_idea_generator.tokenization_config import RemiTokenizerConfig
from midi_idea_generator.tokenizer import (
    TechniqueAnnotation,
    build_tokenizer,
    decode_symbolic_token_ids,
    encode_midi,
)


_HAS_MATPLOTLIB = find_spec("matplotlib") is not None


def _fingerprint(path: Path) -> tuple[str, int]:
    raw = path.read_bytes()
    return sha256(raw).hexdigest(), len(raw)


def _decoded_fixture(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
    *,
    bends: bool,
    techniques: tuple[TechniqueAnnotation, ...],
):
    tokenizer = build_tokenizer(RemiTokenizerConfig())
    instrument = make_instrument(
        [(60, 0.0, 0.5), (64, 0.5, 1.0), (67, 1.0, 1.5)],
        program=30,
        name="generated guitar",
    )
    if bends:
        instrument.pitch_bends.extend(
            [
                pretty_midi.PitchBend(pitch=0, time=0.0),
                pretty_midi.PitchBend(pitch=4096, time=0.25),
                pretty_midi.PitchBend(pitch=0, time=0.5),
            ]
        )
    source = write_midi_file(tmp_path / "source.mid", [instrument])
    encoded = encode_midi(tokenizer, source, techniques=techniques)
    decoded = decode_symbolic_token_ids(
        tokenizer, encoded.ids, encoded.programs
    )
    token_strings = tuple(tokenizer[token_id] for token_id in encoded.ids)
    return tokenizer, encoded, decoded, token_strings


def test_writes_exact_midi_tokens_and_generated_technique_sidecar(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    annotations = (
        TechniqueAnnotation("DEAD_NOTE", 0),
        TechniqueAnnotation("PALM_MUTE_ON", 0),
        TechniqueAnnotation("SLIDE_UP", 1),
        TechniqueAnnotation("VIBRATO", 1),
        TechniqueAnnotation("PALM_MUTE_OFF", 2),
        TechniqueAnnotation("SLIDE_DOWN", 2),
    )
    tokenizer, encoded, decoded, token_strings = _decoded_fixture(
        tmp_path,
        make_instrument,
        write_midi_file,
        bends=True,
        techniques=annotations,
    )
    output_stem = tmp_path / "generated" / "sample-001"

    report = write_generation_artifacts(
        decoded,
        encoded.ids,
        token_strings,
        {
            "checkpoint_sha256": "a" * 64,
            "seed": 42,
            "sampling": {"temperature": 0.9, "top_k": 20},
        },
        output_stem,
        program=30,
        visualization_enabled=False,
        dpi=100,
    )

    assert report.midi.path == output_stem.with_suffix(".mid")
    assert report.tokens.path == Path(f"{output_stem}.tokens.json")
    assert report.techniques.path == Path(
        f"{output_stem}.techniques.generated.json"
    )
    assert report.visualization is None
    assert report.num_notes == 3
    assert report.num_pitch_bends == 3
    assert report.num_techniques == 6
    for artifact in (
        report.midi,
        report.tokens,
        report.techniques,
    ):
        assert artifact is not None
        assert artifact.path.is_file()
        assert (artifact.sha256, artifact.size_bytes) == _fingerprint(artifact.path)

    midi = pretty_midi.PrettyMIDI(str(report.midi.path))
    assert midi.resolution == 24
    assert len(midi.instruments) == 1
    instrument = midi.instruments[0]
    assert instrument.program == 30
    assert instrument.is_drum is False
    assert [bend.pitch for bend in instrument.pitch_bends] == [0, 4095, 0]
    assert [(change.number, change.value) for change in instrument.control_changes] == [
        (101, 0),
        (100, 0),
        (6, 6),
    ]
    assert detect_pitch_bend_range(instrument.control_changes) == (6.0, None)

    raw_midi = mido.MidiFile(report.midi.path, clip=False)
    tick_zero_channel_events: list[tuple[str, int | None, int | None]] = []
    for track in raw_midi.tracks:
        tick = 0
        for message in track:
            tick += int(message.time)
            if tick != 0:
                continue
            if message.type == "program_change":
                tick_zero_channel_events.append(("program", message.program, None))
            elif message.type == "control_change":
                tick_zero_channel_events.append(
                    ("control", message.control, message.value)
                )
    assert tick_zero_channel_events[:4] == [
        ("program", 30, None),
        ("control", 101, 0),
        ("control", 100, 0),
        ("control", 6, 6),
    ]

    reencoded = encode_midi(
        tokenizer, report.midi.path, techniques=decoded.techniques
    )
    assert reencoded.ids == encoded.ids
    assert reencoded.programs == encoded.programs
    assert reencoded.techniques == annotations

    sidecar = json.loads(report.techniques.path.read_text(encoding="utf-8"))
    assert set(sidecar) == {
        "schema_version",
        "artifact_type",
        "midi_file",
        "midi_sha256",
        "midi_size_bytes",
        "ticks_per_quarter",
        "instrument_index",
        "program",
        "coverage",
        "representation",
        "note_index_order",
        "slide_semantics",
        "palm_mute_semantics",
        "techniques",
    }
    assert sidecar["schema_version"] == 1
    assert sidecar["artifact_type"] == "generated_guitar_techniques"
    assert sidecar["midi_file"] == "sample-001.mid"
    assert sidecar["midi_sha256"] == report.midi.sha256
    assert sidecar["midi_size_bytes"] == report.midi.size_bytes
    assert sidecar["ticks_per_quarter"] == 24
    assert sidecar["program"] == 30
    assert sidecar["coverage"] == "COMPLETE"
    assert sidecar["techniques"] == [
        {"type": annotation.type, "note_index": annotation.note_index}
        for annotation in annotations
    ]
    assert "direction only" in sidecar["slide_semantics"]
    assert "target pitch is not encoded" in sidecar["slide_semantics"]
    assert report.techniques.path != sidecar_path_for(report.midi.path)
    assert not sidecar_path_for(report.midi.path).exists()

    token_payload = json.loads(report.tokens.path.read_text(encoding="utf-8"))
    assert set(token_payload) == {
        "schema_version",
        "artifact_type",
        "ids",
        "tokens",
        "programs",
        "provenance",
        "summary",
        "artifacts",
    }
    assert token_payload["ids"] == list(encoded.ids)
    assert token_payload["tokens"] == list(token_strings)
    assert token_payload["programs"] == [[30, False]]
    assert token_payload["provenance"]["seed"] == 42
    assert token_payload["summary"] == {
        "num_tokens": len(encoded.ids),
        "num_notes": 3,
        "num_pitch_bends": 3,
        "num_techniques": 6,
    }
    assert token_payload["artifacts"]["midi"] == {
        "file": report.midi.path.name,
        "sha256": report.midi.sha256,
        "size_bytes": report.midi.size_bytes,
    }
    assert token_payload["artifacts"]["techniques"]["sha256"] == (
        report.techniques.sha256
    )
    assert token_payload["artifacts"]["visualization"] is None


@pytest.mark.skipif(
    not _HAS_MATPLOTLIB,
    reason="matplotlib is declared by the project but absent from this test venv",
)
def test_headless_visualization_contains_notes_bends_and_technique_labels(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    import matplotlib.image as mpimg

    _, encoded, decoded, token_strings = _decoded_fixture(
        tmp_path,
        make_instrument,
        write_midi_file,
        bends=True,
        techniques=(TechniqueAnnotation("VIBRATO", 1),),
    )

    report = write_generation_artifacts(
        decoded,
        encoded.ids,
        token_strings,
        {"seed": 42},
        tmp_path / "visualized",
        program=30,
        visualization_enabled=True,
        dpi=100,
    )

    assert report.visualization is not None
    assert report.visualization.path.name == "visualized.piano-roll.png"
    assert report.visualization.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    image = mpimg.imread(report.visualization.path)
    assert image.ndim == 3
    assert image.shape[0] > 100
    assert image.shape[1] > image.shape[0]
    assert float(image.var()) > 0.0
    token_payload = json.loads(report.tokens.path.read_text(encoding="utf-8"))
    assert token_payload["artifacts"]["visualization"] == {
        "file": report.visualization.path.name,
        "sha256": report.visualization.sha256,
        "size_bytes": report.visualization.size_bytes,
    }


def test_without_bends_or_visualization_writes_no_rpn_and_complete_empty_sidecar(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    _, encoded, decoded, token_strings = _decoded_fixture(
        tmp_path,
        make_instrument,
        write_midi_file,
        bends=False,
        techniques=(),
    )

    report = write_generation_artifacts(
        decoded,
        encoded.ids,
        token_strings,
        {},
        tmp_path / "plain",
        program=30,
        visualization_enabled=False,
    )

    assert report.visualization is None
    assert not (tmp_path / "plain.piano-roll.png").exists()
    midi = pretty_midi.PrettyMIDI(str(report.midi.path))
    assert midi.instruments[0].pitch_bends == []
    assert midi.instruments[0].control_changes == []
    sidecar = json.loads(report.techniques.path.read_text(encoding="utf-8"))
    assert sidecar["coverage"] == "COMPLETE"
    assert sidecar["techniques"] == []
    token_payload = json.loads(report.tokens.path.read_text(encoding="utf-8"))
    assert token_payload["artifacts"]["visualization"] is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("program", "does not match decoded program"),
        ("token_length", "exactly the same length"),
        ("provenance", "NaN or infinity"),
        ("extension", "must not include"),
    ],
)
def test_rejects_invalid_export_contract_before_writing(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
    mutation: str,
    message: str,
) -> None:
    _, encoded, decoded, token_strings = _decoded_fixture(
        tmp_path,
        make_instrument,
        write_midi_file,
        bends=False,
        techniques=(),
    )
    program = 30
    tokens = token_strings
    provenance = {}
    output_stem = tmp_path / "invalid"
    if mutation == "program":
        program = 29
    elif mutation == "token_length":
        tokens = token_strings[:-1]
    elif mutation == "provenance":
        provenance = {"temperature": float("nan")}
    elif mutation == "extension":
        output_stem = tmp_path / "invalid.mid"
    else:  # pragma: no cover - protects the parametrization itself.
        raise AssertionError(mutation)

    with pytest.raises(GenerationArtifactError, match=message):
        write_generation_artifacts(
            decoded,
            encoded.ids,
            tokens,
            provenance,
            output_stem,
            program=program,
        )

    assert not list(tmp_path.glob("invalid*"))


def test_failure_rolls_back_staged_and_published_artifacts(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, encoded, decoded, token_strings = _decoded_fixture(
        tmp_path,
        make_instrument,
        write_midi_file,
        bends=True,
        techniques=(TechniqueAnnotation("VIBRATO", 1),),
    )
    module = importlib.import_module(
        "midi_idea_generator.generation_artifacts"
    )

    def fail_visualization(*args, **kwargs):
        raise RuntimeError("render failed deliberately")

    monkeypatch.setattr(module, "_render_piano_roll", fail_visualization)
    output_stem = tmp_path / "rollback"

    with pytest.raises(GenerationArtifactError, match="render failed deliberately"):
        write_generation_artifacts(
            decoded,
            encoded.ids,
            token_strings,
            {"seed": 7},
            output_stem,
            program=30,
        )

    assert not list(tmp_path.glob("rollback*"))
    assert not list(tmp_path.glob(".rollback-*"))


def test_existing_artifact_set_is_never_overwritten(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    _, encoded, decoded, token_strings = _decoded_fixture(
        tmp_path,
        make_instrument,
        write_midi_file,
        bends=False,
        techniques=(),
    )
    output_stem = tmp_path / "immutable"
    first = write_generation_artifacts(
        decoded,
        encoded.ids,
        token_strings,
        {},
        output_stem,
        program=30,
        visualization_enabled=False,
    )
    snapshot = {
        path: path.read_bytes()
        for path in (first.midi.path, first.tokens.path, first.techniques.path)
    }

    with pytest.raises(GenerationArtifactError, match="will not be overwritten"):
        write_generation_artifacts(
            decoded,
            encoded.ids,
            token_strings,
            {},
            output_stem,
            program=30,
            visualization_enabled=False,
        )

    assert {path: path.read_bytes() for path in snapshot} == snapshot
