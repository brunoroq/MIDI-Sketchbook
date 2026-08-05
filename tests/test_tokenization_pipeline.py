"""End-to-end tests for the immutable Stage 2 tokenization pipeline."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import importlib
import json
from pathlib import Path
from typing import Any

import pretty_midi
import pytest
import yaml

from midi_idea_generator.tokenization_config import (
    TokenizationConfig,
    load_tokenization_config,
)
from midi_idea_generator.tokenization_pipeline import (
    TokenizationPipelineError,
    run_tokenization,
)


_STAGE_ONE_RUN_ID = "0123456789abcdefabcd"
_STAGE_ONE_CONFIGURATION = {"fixture": "stage-one-schema-three"}
_STAGE_ONE_CONFIG_SHA256 = sha256(
    json.dumps(
        _STAGE_ONE_CONFIGURATION,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
).hexdigest()


@dataclass(slots=True)
class _SyntheticProject:
    root: Path
    config_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    midi_paths: dict[str, Path]

    def config(self) -> TokenizationConfig:
        return load_tokenization_config(self.config_path)

    def write_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _source_digest(source_file: str) -> str:
    return sha256(source_file.encode("utf-8")).hexdigest()


def _file_fingerprint(path: Path) -> tuple[str, int]:
    raw = path.read_bytes()
    return sha256(raw).hexdigest(), len(raw)


def _refresh_declared_output(
    project: _SyntheticProject, *, fragment_index: int = 0
) -> None:
    fragment = project.manifest["fragments"][fragment_index]
    path = project.root / fragment["output_file"]
    digest, size = _file_fingerprint(path)
    fragment["output_sha256"] = digest
    fragment["output_size_bytes"] = size


def _write_tokenization_config(
    project_root: Path,
    *,
    filename: str = "tokenize.yaml",
    tokenizer_overrides: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "paths": {
            "preprocessing_manifest_path": "data/splits/manifest.json",
            "tokenized_dir": "data/tokenized",
            "manifest_path": "data/tokenized/manifest.json",
        },
        "tokenizer": tokenizer_overrides or {},
    }
    path = project_root / "configs" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _make_synthetic_project(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
    *,
    splits: tuple[str, ...] = ("train",),
) -> _SyntheticProject:
    project_root = tmp_path / "synthetic-stage-two-project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='synthetic-stage-two-test'\n",
        encoding="utf-8",
    )
    processed_run_dir = (
        project_root / "data" / "processed" / "runs" / _STAGE_ONE_RUN_ID
    )

    sources: list[dict[str, Any]] = []
    fragments: list[dict[str, Any]] = []
    midi_paths: dict[str, Path] = {}
    for index, split in enumerate(splits):
        source_file = f"data/raw/{split}-source.mid"
        source_sha256 = _source_digest(source_file)
        output_path = processed_run_dir / split / f"{split}-riff.mid"
        notes = [
            (60 + index, 0.0, 0.5),
            (64 + index, 0.5, 1.0),
            (67 + index, 1.0, 1.5),
        ]
        write_midi_file(
            output_path,
            [make_instrument(notes, program=30, name=f"{split} guitar")],
            tempo_bpm=120.0 + index,
        )
        output_sha256, output_size_bytes = _file_fingerprint(output_path)
        output_file = output_path.relative_to(project_root).as_posix()
        sources.append(
            {
                "source_file": source_file,
                "source_sha256": source_sha256,
                "compatible": True,
                "split": split,
                "track_number": 0,
                "instrument_index": 0,
            }
        )
        fragments.append(
            {
                "source_file": source_file,
                "split": split,
                "track_number": 0,
                "instrument_index": 0,
                "phrase_index": 0,
                "transpose_semitones": 0,
                "output_file": output_file,
                "num_notes": len(notes),
                "num_pitch_bend_events": 0,
                "num_expressive_pitch_bend_events": 0,
                "pitch_bend_range_semitones": None,
                "synthetic_initial_pitch_bend": False,
                "synthetic_final_pitch_bend_reset": False,
                "output_sha256": output_sha256,
                "output_size_bytes": output_size_bytes,
                "technique_coverage": "UNLABELED",
                "techniques": [],
                "nominal_duration_seconds": 4.0,
                "actual_note_duration_seconds": 1.5,
            }
        )
        midi_paths[split] = output_path

    manifest: dict[str, Any] = {
        "schema_version": 3,
        "run_id": _STAGE_ONE_RUN_ID,
        "processed_run_dir": processed_run_dir.relative_to(project_root).as_posix(),
        "configuration_sha256": _STAGE_ONE_CONFIG_SHA256,
        "configuration": _STAGE_ONE_CONFIGURATION,
        "tool_versions": {"fixture": "1"},
        "random_seed": 99,
        "split_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "summary": {"generated_fragments": len(fragments)},
        "sources": sources,
        "fragments": fragments,
    }
    manifest_path = project_root / "data" / "splits" / "manifest.json"
    project = _SyntheticProject(
        root=project_root,
        config_path=_write_tokenization_config(project_root),
        manifest_path=manifest_path,
        manifest=manifest,
        midi_paths=midi_paths,
    )
    project.write_manifest()
    return project


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _assert_no_partial_publication(project_root: Path) -> None:
    tokenized_dir = project_root / "data" / "tokenized"
    assert not (tokenized_dir / "manifest.json").exists()
    runs_dir = tokenized_dir / "runs"
    assert not runs_dir.exists() or not any(runs_dir.iterdir())
    assert not tokenized_dir.exists() or not list(tokenized_dir.glob(".stage-*"))


def test_pipeline_publishes_all_splits_with_bounded_special_tokens(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    project = _make_synthetic_project(
        tmp_path,
        make_instrument,
        write_midi_file,
        splits=("train", "validation", "test"),
    )

    report = run_tokenization(project.config())

    assert report.num_sequences == 3
    assert report.reused_run is False
    assert report.manifest_path == (project.root / "data/tokenized/manifest.json")
    assert report.tokenized_run_dir.name == report.tokenization_run_id
    assert (report.tokenized_run_dir / "tokenizer.json").is_file()
    assert (report.tokenized_run_dir / "manifest.json").read_bytes() == (
        report.manifest_path.read_bytes()
    )

    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["preprocessing"]["schema_version"] == 3
    assert manifest["preprocessing"]["run_id"] == _STAGE_ONE_RUN_ID
    assert manifest["tokenizer"]["type"] == "GuitarREMI"
    assert manifest["tokenizer"]["pitch_bend_sensitivity_semitones"] == 6
    assert set(manifest["tokenizer"]["technique_token_ids"]) == {
        "DEAD_NOTE",
        "PALM_MUTE_ON",
        "PALM_MUTE_OFF",
        "SLIDE_UP",
        "SLIDE_DOWN",
        "VIBRATO",
    }
    assert len(set(manifest["tokenizer"]["technique_token_ids"].values())) == 6
    assert manifest["summary"]["sequences"] == 3
    assert manifest["summary"]["techniques"] == {
        "total_tokens": 0,
        "by_type": {
            "DEAD_NOTE": 0,
            "PALM_MUTE_ON": 0,
            "PALM_MUTE_OFF": 0,
            "SLIDE_UP": 0,
            "SLIDE_DOWN": 0,
            "VIBRATO": 0,
        },
        "coverage": {"complete_sequences": 0, "unlabeled_sequences": 3},
    }
    assert manifest["summary"]["pitch_bends"] == {
        "total_tokens": 0,
        "sequences_with_pitch_bends": 0,
    }
    assert {
        split: manifest["summary"]["by_split"][split]["sequences"]
        for split in ("train", "validation", "test")
    } == {"train": 1, "validation": 1, "test": 1}

    special = manifest["tokenizer"]["special_token_ids"]
    for record in manifest["sequences"]:
        sequence_path = project.root / record["sequence_file"]
        assert sequence_path.parent == report.tokenized_run_dir / record["split"]
        payload = json.loads(sequence_path.read_text(encoding="utf-8"))
        ids = payload["ids"]
        assert payload["schema_version"] == 2
        assert payload["sequence_id"] == record["sequence_id"]
        assert payload["programs"] == [[30, False]]
        assert payload["technique_coverage"] == "UNLABELED"
        assert payload["techniques"] == []
        assert ids[0] == special["bos"]
        assert ids[-1] == special["eos"]
        assert special["pad"] not in ids
        assert ids[1:-1]
        assert record["num_tokens"] == len(ids)
        assert record["num_musical_tokens"] == len(ids) - 2
        assert record["technique_coverage"] == "UNLABELED"
        assert record["techniques"] == []
        assert record["num_technique_tokens"] == 0
        assert record["num_pitch_bend_tokens"] == 0
        assert record["round_trip_ok"] is True
        assert record["token_error_ratio"] == 0.0


def test_pipeline_ignores_processed_midi_not_declared_as_a_fragment(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    project = _make_synthetic_project(tmp_path, make_instrument, write_midi_file)
    unlisted = write_midi_file(
        project.midi_paths["train"].parent / "unlisted.mid",
        [make_instrument([(72, 0.0, 0.5)], program=30)],
    )

    report = run_tokenization(project.config())
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))

    assert unlisted.is_file()
    assert report.num_sequences == 1
    assert len(manifest["sequences"]) == 1
    assert manifest["sequences"][0]["processed_midi"] == (
        project.manifest["fragments"][0]["output_file"]
    )
    assert "unlisted" not in "\n".join(_snapshot(report.tokenized_run_dir))


def test_pipeline_serializes_guitar_techniques_and_native_pitch_bends(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    project = _make_synthetic_project(tmp_path, make_instrument, write_midi_file)
    midi = pretty_midi.PrettyMIDI(str(project.midi_paths["train"]))
    midi.instruments[0].pitch_bends.extend(
        [
            pretty_midi.PitchBend(pitch=0, time=0.0),
            pretty_midi.PitchBend(pitch=4096, time=0.25),
            pretty_midi.PitchBend(pitch=8191, time=0.75),
            pretty_midi.PitchBend(pitch=0, time=1.25),
        ]
    )
    midi.write(str(project.midi_paths["train"]))
    fragment = project.manifest["fragments"][0]
    fragment.update(
        {
            "num_pitch_bend_events": 4,
            "num_expressive_pitch_bend_events": 2,
            "pitch_bend_range_semitones": 6,
            "technique_coverage": "COMPLETE",
            "techniques": [
                {"type": "DEAD_NOTE", "note_index": 0},
                {"type": "PALM_MUTE_ON", "note_index": 0},
                {"type": "SLIDE_UP", "note_index": 1},
                {"type": "VIBRATO", "note_index": 1},
                {"type": "PALM_MUTE_OFF", "note_index": 2},
                {"type": "SLIDE_DOWN", "note_index": 2},
            ],
        }
    )
    _refresh_declared_output(project)
    project.write_manifest()

    report = run_tokenization(project.config())
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    record = manifest["sequences"][0]
    payload = json.loads(
        (project.root / record["sequence_file"]).read_text(encoding="utf-8")
    )

    assert set(payload) == {
        "schema_version",
        "sequence_id",
        "ids",
        "programs",
        "technique_coverage",
        "techniques",
    }
    assert payload["technique_coverage"] == "COMPLETE"
    assert payload["techniques"] == fragment["techniques"]
    assert record["technique_coverage"] == "COMPLETE"
    assert record["techniques"] == fragment["techniques"]
    assert record["num_technique_tokens"] == 6
    assert record["num_pitch_bend_tokens"] == 4
    technique_ids = set(manifest["tokenizer"]["technique_token_ids"].values())
    assert sum(token_id in technique_ids for token_id in payload["ids"]) == 6
    assert manifest["summary"]["techniques"] == {
        "total_tokens": 6,
        "by_type": {
            "DEAD_NOTE": 1,
            "PALM_MUTE_ON": 1,
            "PALM_MUTE_OFF": 1,
            "SLIDE_UP": 1,
            "SLIDE_DOWN": 1,
            "VIBRATO": 1,
        },
        "coverage": {"complete_sequences": 1, "unlabeled_sequences": 0},
    }
    assert manifest["summary"]["pitch_bends"] == {
        "total_tokens": 4,
        "sequences_with_pitch_bends": 1,
    }


def test_symbolic_annotations_change_sequence_identity_without_changing_midi(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    project = _make_synthetic_project(tmp_path, make_instrument, write_midi_file)
    first = run_tokenization(project.config())
    first_manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))

    project.manifest["fragments"][0].update(
        {
            "technique_coverage": "COMPLETE",
            "techniques": [{"type": "VIBRATO", "note_index": 1}],
        }
    )
    project.write_manifest()
    second = run_tokenization(project.config())
    second_manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))

    assert second.tokenization_run_id != first.tokenization_run_id
    assert (
        second_manifest["sequences"][0]["sequence_id"]
        != first_manifest["sequences"][0]["sequence_id"]
    )
    assert (
        second_manifest["sequences"][0]["processed_midi_sha256"]
        == first_manifest["sequences"][0]["processed_midi_sha256"]
    )


def test_identical_rerun_reuses_a_byte_identical_immutable_run(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    project = _make_synthetic_project(tmp_path, make_instrument, write_midi_file)
    config = project.config()

    first = run_tokenization(config)
    first_snapshot = _snapshot(first.tokenized_run_dir)
    first_manifest = first.manifest_path.read_bytes()
    second = run_tokenization(config)

    assert first.reused_run is False
    assert second.reused_run is True
    assert second.tokenization_run_id == first.tokenization_run_id
    assert second.tokenized_run_dir == first.tokenized_run_dir
    assert _snapshot(second.tokenized_run_dir) == first_snapshot
    assert second.manifest_path.read_bytes() == first_manifest
    assert [path.name for path in second.tokenized_run_dir.parent.iterdir()] == [
        first.tokenization_run_id
    ]


def test_changed_midi_and_changed_config_each_create_a_distinct_run(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    project = _make_synthetic_project(tmp_path, make_instrument, write_midi_file)
    base_config = project.config()
    first = run_tokenization(base_config)
    first_snapshot = _snapshot(first.tokenized_run_dir)

    write_midi_file(
        project.midi_paths["train"],
        [
            make_instrument(
                [(61, 0.0, 0.5), (65, 0.5, 1.0), (68, 1.0, 1.5)],
                program=30,
            )
        ],
    )
    _refresh_declared_output(project)
    project.write_manifest()
    changed_input = run_tokenization(base_config)

    changed_config_path = _write_tokenization_config(
        project.root,
        filename="tokenize-31-tempos.yaml",
        tokenizer_overrides={"num_tempos": 31},
    )
    changed_config = run_tokenization(load_tokenization_config(changed_config_path))

    assert len(
        {
            first.tokenization_run_id,
            changed_input.tokenization_run_id,
            changed_config.tokenization_run_id,
        }
    ) == 3
    assert _snapshot(first.tokenized_run_dir) == first_snapshot
    assert len(list(first.tokenized_run_dir.parent.iterdir())) == 3


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("schema", "schema_version 3"),
        ("duplicate", "Duplicate fragment output_file"),
        ("traversal", "cannot contain '..' traversal"),
        ("source_split", "split does not match its original source split"),
        ("path_split", "not in the declared split directory"),
        ("corrupt_midi", "Stage 2 rejected 1 declared fragment"),
    ],
)
def test_invalid_stage_one_inputs_abort_without_partial_publication(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
    case: str,
    message: str,
) -> None:
    project = _make_synthetic_project(tmp_path, make_instrument, write_midi_file)

    if case == "schema":
        project.manifest["schema_version"] = 2
    elif case == "duplicate":
        project.manifest["fragments"].append(
            deepcopy(project.manifest["fragments"][0])
        )
        project.manifest["summary"]["generated_fragments"] = 2
    elif case == "traversal":
        project.manifest["fragments"][0]["output_file"] = "../escape.mid"
    elif case == "source_split":
        project.manifest["fragments"][0]["split"] = "validation"
    elif case == "path_split":
        project.manifest["fragments"][0]["split"] = "validation"
        project.manifest["sources"][0]["split"] = "validation"
    elif case == "corrupt_midi":
        project.midi_paths["train"].write_bytes(b"deliberately corrupt MIDI")
        _refresh_declared_output(project)
    else:  # pragma: no cover - protects the parametrization itself.
        raise AssertionError(f"Unhandled test case: {case}")
    project.write_manifest()

    with pytest.raises(TokenizationPipelineError, match=message):
        run_tokenization(project.config())

    _assert_no_partial_publication(project.root)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unknown", r"Unknown field\(s\).*future_field"),
        ("missing", r"Missing field\(s\).*techniques"),
        ("fingerprint", "do not match the declared fragment file"),
        ("coverage", "must be UNLABELED or COMPLETE"),
        ("unlabeled_annotations", "must be empty.*UNLABELED"),
        ("unordered", "must use canonical"),
        ("duplicate_technique", "Duplicate technique"),
        ("note_index", "outside the 3-note fragment"),
        ("bend_count", "cannot exceed num_pitch_bend_events"),
        ("bend_range", "must be 6 when expressive pitch bends are present"),
    ],
)
def test_fragment_schema_three_contract_is_strict(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
    case: str,
    message: str,
) -> None:
    project = _make_synthetic_project(tmp_path, make_instrument, write_midi_file)
    fragment = project.manifest["fragments"][0]

    if case == "unknown":
        fragment["future_field"] = True
    elif case == "missing":
        fragment.pop("techniques")
    elif case == "fingerprint":
        fragment["output_sha256"] = "0" * 64
    elif case == "coverage":
        fragment["technique_coverage"] = "PARTIAL"
    elif case == "unlabeled_annotations":
        fragment["techniques"] = [{"type": "VIBRATO", "note_index": 0}]
    elif case == "unordered":
        fragment.update(
            {
                "technique_coverage": "COMPLETE",
                "techniques": [
                    {"type": "VIBRATO", "note_index": 1},
                    {"type": "DEAD_NOTE", "note_index": 0},
                ],
            }
        )
    elif case == "duplicate_technique":
        fragment.update(
            {
                "technique_coverage": "COMPLETE",
                "techniques": [
                    {"type": "VIBRATO", "note_index": 0},
                    {"type": "VIBRATO", "note_index": 0},
                ],
            }
        )
    elif case == "note_index":
        fragment.update(
            {
                "technique_coverage": "COMPLETE",
                "techniques": [{"type": "VIBRATO", "note_index": 3}],
            }
        )
    elif case == "bend_count":
        fragment["num_expressive_pitch_bend_events"] = 1
    elif case == "bend_range":
        fragment.update(
            {
                "num_pitch_bend_events": 1,
                "num_expressive_pitch_bend_events": 1,
                "pitch_bend_range_semitones": None,
            }
        )
    else:  # pragma: no cover - protects the parametrization itself.
        raise AssertionError(f"Unhandled test case: {case}")
    project.write_manifest()

    with pytest.raises(TokenizationPipelineError, match=message):
        run_tokenization(project.config())

    _assert_no_partial_publication(project.root)


def test_failed_new_attempt_preserves_previous_authoritative_run(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    project = _make_synthetic_project(tmp_path, make_instrument, write_midi_file)
    config = project.config()
    successful = run_tokenization(config)
    previous_manifest = successful.manifest_path.read_bytes()
    previous_snapshot = _snapshot(successful.tokenized_run_dir)

    project.midi_paths["train"].write_bytes(b"corrupt replacement")
    _refresh_declared_output(project)
    project.write_manifest()
    with pytest.raises(TokenizationPipelineError, match="rejected 1 declared fragment"):
        run_tokenization(config)

    assert successful.manifest_path.read_bytes() == previous_manifest
    assert _snapshot(successful.tokenized_run_dir) == previous_snapshot
    assert not list((project.root / "data/tokenized").glob(".stage-*"))


def test_fragment_changed_during_encoding_is_detected(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _make_synthetic_project(tmp_path, make_instrument, write_midi_file)
    pipeline = importlib.import_module("midi_idea_generator.tokenization_pipeline")
    real_encode = pipeline.encode_midi

    def mutating_encode(tokenizer, path, *, techniques):
        result = real_encode(tokenizer, path, techniques=techniques)
        Path(path).write_bytes(Path(path).read_bytes() + b"changed")
        return result

    monkeypatch.setattr(pipeline, "encode_midi", mutating_encode)

    with pytest.raises(TokenizationPipelineError, match="changed while it was tokenized"):
        run_tokenization(project.config())

    _assert_no_partial_publication(project.root)


def test_cli_main_reports_success_and_configuration_errors(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = _make_synthetic_project(tmp_path, make_instrument, write_midi_file)
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    tokenize_midis = importlib.import_module("tokenize_midis")

    assert tokenize_midis.main(["--config", str(project.config_path)]) == 0
    success = capsys.readouterr()
    assert "Tokenization complete: 1 sequence(s)" in success.out
    assert "Created immutable tokenized run" in success.out
    assert success.err == ""

    assert tokenize_midis.main(["--config", str(project.root / "missing.yaml")]) == 2
    failure = capsys.readouterr()
    assert failure.out == ""
    assert "ERROR | Could not read configuration" in failure.err
