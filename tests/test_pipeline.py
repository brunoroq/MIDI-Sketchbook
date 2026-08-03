"""End-to-end Stage 1 preprocessing tests using only temporary MIDI data."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

from midi_idea_generator.config import SplitConfig, load_preprocess_config
from midi_idea_generator.midi_io import read_midi
from midi_idea_generator.pipeline import run_preprocessing


def _write_pipeline_config(project_root: Path) -> Path:
    payload = {
        "random_seed": 99,
        "paths": {
            "input_dir": "data/raw",
            "processed_dir": "data/processed",
            "manifest_path": "data/splits/manifest.json",
        },
        "validation": {
            "pitch_min": 21,
            "pitch_max": 108,
            "allowed_time_signature": [4, 4],
            "allow_missing_time_signature": True,
            "reject_pitch_bends": True,
            "exclude_drums": True,
            "min_notes_per_track": 1,
            "tempo_tolerance": 0.01,
        },
        "track_selection": {"mode": "most_notes", "track_index": None},
        "preprocessing": {
            "phrase_bars": 2,
            "remove_initial_silence": True,
            "quantize": True,
            "subdivisions_per_beat": 4,
            "include_partial_final_phrase": True,
            "min_notes_per_phrase": 1,
        },
        "augmentation": {
            "enabled": True,
            "min_semitones": -1,
            "max_semitones": 1,
            "apply_to_splits": ["train"],
        },
        "splits": {"train": 1.0, "validation": 0.0, "test": 0.0},
    }
    config_path = project_root / "configs" / "preprocess.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def test_pipeline_continues_past_corrupt_midi_and_writes_complete_manifest(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    project_root = tmp_path / "synthetic-project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='synthetic-stage-one-test'\n",
        encoding="utf-8",
    )
    raw_dir = project_root / "data" / "raw"
    valid_path = write_midi_file(
        raw_dir / "valid.mid",
        [make_instrument([(60, 1.03, 1.21), (64, 1.27, 1.51)])],
        tempo_bpm=120.0,
    )
    corrupt_path = raw_dir / "broken.midi"
    corrupt_path.write_bytes(b"deliberately corrupt MIDI fixture")
    config = load_preprocess_config(_write_pipeline_config(project_root))

    report = run_preprocessing(config)

    assert report.discovered_files == 2
    assert report.compatible_sources == 1
    assert report.discarded_sources == 1
    assert report.generated_fragments == 3
    assert report.manifest_path == (
        project_root / "data/splits/manifest.json"
    ).resolve()
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["random_seed"] == 99
    assert manifest["summary"] == {
        "compatible_sources": 1,
        "discarded_sources": 1,
        "discovered_files": 2,
        "generated_fragments": 3,
    }

    sources = {
        Path(source["source_file"]).name: source for source in manifest["sources"]
    }
    valid = sources[valid_path.name]
    corrupt = sources[corrupt_path.name]
    assert valid["compatible"] is True
    assert valid["discard_reason"] is None
    assert valid["split"] == "train"
    assert valid["track_number"] == 0
    assert valid["tempo_bpm"] == pytest.approx(120.0)
    assert valid["time_signature"] == "4/4"
    assert valid["num_notes"] == 2
    assert valid["num_base_fragments"] == 1
    assert valid["num_fragments_generated"] == 3
    assert corrupt["compatible"] is False
    assert corrupt["split"] is None
    assert corrupt["track_number"] is None
    assert corrupt["num_base_fragments"] == 0
    assert corrupt["num_fragments_generated"] == 0
    assert "Could not parse" in corrupt["discard_reason"]

    fragments = manifest["fragments"]
    assert {fragment["source_file"] for fragment in fragments} == {
        valid["source_file"]
    }
    assert {fragment["split"] for fragment in fragments} == {"train"}
    assert {fragment["phrase_index"] for fragment in fragments} == {0}
    assert {fragment["transpose_semitones"] for fragment in fragments} == {
        -1,
        0,
        1,
    }
    assert {fragment["num_notes"] for fragment in fragments} == {2}
    assert all(
        fragment["nominal_duration_seconds"] == pytest.approx(4.0)
        for fragment in fragments
    )

    pitches_by_offset: dict[int, list[int]] = {}
    for fragment in fragments:
        output_path = project_root / fragment["output_file"]
        assert output_path.is_file()
        generated = read_midi(output_path)
        assert len(generated.instruments) == 1
        pitches_by_offset[fragment["transpose_semitones"]] = [
            note.pitch for note in generated.instruments[0].notes
        ]
    assert pitches_by_offset == {
        -1: [59, 63],
        0: [60, 64],
        1: [61, 65],
    }
    assert len(list((project_root / "data/processed").rglob("*.mid"))) == 3


def test_pipeline_reuses_identical_run_and_isolates_changed_configuration(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    project_root = tmp_path / "repeatable-project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='repeatable-stage-one-test'\n",
        encoding="utf-8",
    )
    write_midi_file(
        project_root / "data/raw/source.mid",
        [make_instrument([(60, 0.0, 0.5), (64, 0.5, 1.0)])],
    )
    config = load_preprocess_config(_write_pipeline_config(project_root))

    first = run_preprocessing(config)
    second = run_preprocessing(config)

    assert first.run_id == second.run_id
    assert first.processed_run_dir == second.processed_run_dir
    runs_dir = project_root / "data/processed/runs"
    assert [path.name for path in runs_dir.iterdir()] == [first.run_id]

    changed = replace(
        config,
        augmentation=replace(
            config.augmentation,
            min_semitones=0,
            max_semitones=0,
        ),
    )
    third = run_preprocessing(changed)
    manifest = json.loads(third.manifest_path.read_text(encoding="utf-8"))

    assert third.run_id != first.run_id
    assert len(list(runs_dir.iterdir())) == 2
    assert manifest["run_id"] == third.run_id
    assert manifest["configuration_sha256"]
    assert manifest["configuration"]["augmentation"]["min_semitones"] == 0
    assert manifest["sources"][0]["source_sha256"]
    assert len(manifest["sources"][0]["source_sha256"]) == 64
    assert manifest["sources"][0]["resolution"] == 480
    assert len(manifest["fragments"]) == 1
    current_outputs = {
        (project_root / fragment["output_file"]).resolve()
        for fragment in manifest["fragments"]
    }
    assert current_outputs == set(third.processed_run_dir.rglob("*.mid"))
    assert all(path.is_relative_to(third.processed_run_dir) for path in current_outputs)


def test_identical_source_files_are_grouped_into_the_same_split(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    project_root = tmp_path / "duplicate-project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='duplicate-stage-one-test'\n", encoding="utf-8"
    )
    raw = project_root / "data/raw"
    original = write_midi_file(
        raw / "song.mid", [make_instrument([(60, 0.0, 0.5)])]
    )
    duplicate = raw / "renamed-copy.mid"
    duplicate.write_bytes(original.read_bytes())
    write_midi_file(raw / "other-a.mid", [make_instrument([(64, 0.0, 0.5)])])
    write_midi_file(raw / "other-b.mid", [make_instrument([(67, 0.0, 0.5)])])
    config = replace(
        load_preprocess_config(_write_pipeline_config(project_root)),
        splits=SplitConfig(train=0.34, validation=0.33, test=0.33),
    )

    report = run_preprocessing(config)
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    sources = {
        Path(source["source_file"]).name: source for source in manifest["sources"]
    }

    assert sources["song.mid"]["source_sha256"] == sources["renamed-copy.mid"][
        "source_sha256"
    ]
    assert sources["song.mid"]["split"] == sources["renamed-copy.mid"]["split"]
    assert sources["song.mid"]["content_group_size"] == 2
    assert sources["renamed-copy.mid"]["content_group_size"] == 2


def test_write_failures_are_path_free_and_manifest_is_reproducible(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    project_root = tmp_path / "write-failure-project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='write-failure-test'\n", encoding="utf-8"
    )
    write_midi_file(
        project_root / "data/raw/quantized-overlap.mid",
        [make_instrument([(60, 0.1, 0.29), (60, 0.29, 0.4)])],
    )
    base = load_preprocess_config(_write_pipeline_config(project_root))
    config = replace(
        base,
        preprocessing=replace(
            base.preprocessing,
            remove_initial_silence=False,
        ),
    )

    first = run_preprocessing(config)
    first_text = first.manifest_path.read_text(encoding="utf-8")
    second = run_preprocessing(config)
    second_text = second.manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(second_text)

    assert first.run_id == second.run_id
    assert first_text == second_text
    assert ".stage-" not in second_text
    assert str(tmp_path) not in second_text
    assert manifest["summary"]["generated_fragments"] == 0
    assert manifest["sources"][0]["write_errors"]
    assert all(
        "reason" in failure
        for failure in manifest["sources"][0]["write_errors"]
    )
