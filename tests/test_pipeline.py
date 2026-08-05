"""End-to-end Stage 1 preprocessing tests using only temporary MIDI data."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from midi_idea_generator.config import SplitConfig, load_preprocess_config
from midi_idea_generator.midi_io import read_midi
from midi_idea_generator.pipeline import run_preprocessing
from midi_idea_generator.techniques import TechniqueSidecarError
from midi_idea_generator.tonality import TonalitySidecarError


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
            "reject_pitch_bends": False,
            "canonical_pitch_bend_range_semitones": 6,
            "require_explicit_pitch_bend_range": True,
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
        "tonality": {"missing_sidecar_policy": "infer_source"},
        "splits": {"train": 1.0, "validation": 0.0, "test": 0.0},
    }
    config_path = project_root / "configs" / "preprocess.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def _write_tonality_sidecar(
    midi_path: Path,
    *,
    tonic: str,
    mode: str,
) -> Path:
    path = midi_path.with_name(f"{midi_path.name}.tonality.json")
    payload = {
        "schema_version": 1,
        "source_midi": midi_path.name,
        "source_sha256": hashlib.sha256(midi_path.read_bytes()).hexdigest(),
        "instrument_index": 0,
        "tonic": tonic,
        "mode": mode,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


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
    assert manifest["schema_version"] == 4
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
    assert valid["raw_note_events"] == 2
    assert valid["duplicate_notes_collapsed"] == 0
    assert valid["num_pitch_bend_events"] == 0
    assert valid["num_expressive_pitch_bend_events"] == 0
    assert valid["source_pitch_bend_range_semitones"] is None
    assert valid["tonality"]["method"] == "AUTO_SOURCE"
    assert valid["tonality"]["tonic"] != "UNKNOWN"
    assert valid["tonality"]["mode"] != "UNKNOWN"
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
    assert {fragment["num_pitch_bend_events"] for fragment in fragments} == {0}
    assert {fragment["tonality"]["method"] for fragment in fragments} == {
        "AUTO_SOURCE"
    }
    assert all(len(fragment["output_sha256"]) == 64 for fragment in fragments)
    assert all(fragment["output_size_bytes"] > 0 for fragment in fragments)
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


def test_manual_tonality_sidecar_is_hashed_and_transposed_with_each_variant(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    project_root = tmp_path / "manual-tonality-project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='manual-tonality-test'\n", encoding="utf-8"
    )
    midi_path = write_midi_file(
        project_root / "data/raw/riff.mid",
        [make_instrument([(52, 0.0, 0.4), (53, 0.5, 0.9), (55, 1.0, 1.4)])],
    )
    sidecar_path = _write_tonality_sidecar(
        midi_path, tonic="E", mode="PHRYGIAN"
    )
    config = load_preprocess_config(_write_pipeline_config(project_root))

    first = run_preprocessing(config)
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    source = manifest["sources"][0]
    by_offset = {
        fragment["transpose_semitones"]: fragment["tonality"]
        for fragment in manifest["fragments"]
    }

    assert source["tonality"] == {
        "tonic": "E",
        "mode": "PHRYGIAN",
        "method": "MANUAL",
        "tonic_confidence": None,
        "mode_confidence": None,
    }
    assert source["tonality_sidecar"].endswith("riff.mid.tonality.json")
    assert source["tonality_sidecar_sha256"] == hashlib.sha256(
        sidecar_path.read_bytes()
    ).hexdigest()
    assert by_offset[-1]["tonic"] == "D_SHARP"
    assert by_offset[0]["tonic"] == "E"
    assert by_offset[1]["tonic"] == "F"
    assert {value["mode"] for value in by_offset.values()} == {"PHRYGIAN"}
    assert {value["method"] for value in by_offset.values()} == {"MANUAL"}

    _write_tonality_sidecar(midi_path, tonic="E", mode="MINOR")
    second = run_preprocessing(config)
    assert second.run_id != first.run_id


def test_infer_fragment_labels_each_untransposed_phrase_independently(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    project_root = tmp_path / "fragment-tonality-project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='fragment-tonality-test'\n", encoding="utf-8"
    )
    e_phrygian = [52, 53, 55, 57, 59, 60, 62, 64]
    c_major = [48, 50, 52, 53, 55, 57, 59, 60]
    notes = [
        (pitch, index * 0.4, index * 0.4 + 0.25)
        for index, pitch in enumerate(e_phrygian)
    ] + [
        (pitch, 4.0 + index * 0.4, 4.0 + index * 0.4 + 0.25)
        for index, pitch in enumerate(c_major)
    ]
    write_midi_file(
        project_root / "data/raw/two-centres.mid",
        [make_instrument(notes)],
    )
    base = load_preprocess_config(_write_pipeline_config(project_root))
    config = replace(
        base,
        augmentation=replace(
            base.augmentation,
            enabled=False,
            min_semitones=0,
            max_semitones=0,
        ),
        tonality=replace(
            base.tonality,
            missing_sidecar_policy="infer_fragment",
        ),
    )

    report = run_preprocessing(config)
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    by_phrase = {
        fragment["phrase_index"]: fragment["tonality"]
        for fragment in manifest["fragments"]
    }

    assert manifest["sources"][0]["tonality"]["method"] == "AUTO_SOURCE"
    assert by_phrase[0]["method"] == "AUTO_FRAGMENT"
    assert by_phrase[0]["tonic"] == "E"
    assert by_phrase[0]["mode"] == "PHRYGIAN"
    assert by_phrase[1]["method"] == "AUTO_FRAGMENT"
    assert by_phrase[1]["tonic"] == "C"
    assert by_phrase[1]["mode"] == "MAJOR"


def test_unknown_policy_emits_explicit_unknown_labels_for_all_variants(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    project_root = tmp_path / "unknown-tonality-project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='unknown-tonality-test'\n", encoding="utf-8"
    )
    write_midi_file(
        project_root / "data/raw/riff.mid",
        [make_instrument([(52, 0.0, 0.5), (59, 0.5, 1.0)])],
    )
    base = load_preprocess_config(_write_pipeline_config(project_root))
    config = replace(
        base,
        tonality=replace(base.tonality, missing_sidecar_policy="unknown"),
    )

    report = run_preprocessing(config)
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    expected = {
        "tonic": "UNKNOWN",
        "mode": "UNKNOWN",
        "method": "UNKNOWN",
        "tonic_confidence": None,
        "mode_confidence": None,
    }
    assert manifest["sources"][0]["tonality"] == expected
    assert all(fragment["tonality"] == expected for fragment in manifest["fragments"])


def test_pipeline_rejects_orphan_tonality_sidecar(tmp_path: Path) -> None:
    project_root = tmp_path / "orphan-tonality-project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='orphan-tonality-test'\n", encoding="utf-8"
    )
    raw = project_root / "data/raw"
    raw.mkdir(parents=True)
    (raw / "ghost.mid.tonality.json").write_text("{}\n", encoding="utf-8")
    config = load_preprocess_config(_write_pipeline_config(project_root))

    with pytest.raises(TonalitySidecarError, match="Orphan tonality sidecar"):
        run_preprocessing(config)


def test_pipeline_projects_complete_technique_sidecar_and_hashes_semantics(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    project_root = tmp_path / "technique-project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='stage-one-technique-test'\n", encoding="utf-8"
    )
    midi_path = write_midi_file(
        project_root / "data/raw/riff.mid",
        [make_instrument([(60, 0.0, 0.5), (64, 0.5, 1.0)])],
    )
    source = read_midi(midi_path)
    notes = source.instruments[0].notes
    source_sha = hashlib.sha256(midi_path.read_bytes()).hexdigest()
    sidecar_path = midi_path.with_name("riff.mid.techniques.json")

    def write_sidecar(*, include_vibrato: bool) -> None:
        slide_techniques: list[dict[str, object]] = [
            {"type": "SLIDE", "direction": "UP", "target_pitch": 67}
        ]
        if include_vibrato:
            slide_techniques.append({"type": "VIBRATO"})
        payload = {
            "schema_version": 1,
            "source_midi": midi_path.name,
            "source_sha256": source_sha,
            "ticks_per_quarter": source.resolution,
            "instrument_index": 0,
            "coverage": "COMPLETE",
            "note_techniques": [
                {
                    "note": {
                        "onset_tick": int(source.time_to_tick(notes[1].start)),
                        "end_tick": int(source.time_to_tick(notes[1].end)),
                        "pitch": notes[1].pitch,
                        "velocity": notes[1].velocity,
                    },
                    "techniques": slide_techniques,
                }
            ],
            "palm_mute_ranges": [{"start_tick": 0, "end_tick": 480}],
        }
        sidecar_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    write_sidecar(include_vibrato=True)
    base = load_preprocess_config(_write_pipeline_config(project_root))
    config = replace(
        base,
        augmentation=replace(
            base.augmentation,
            enabled=False,
            min_semitones=0,
            max_semitones=0,
        ),
    )

    first = run_preprocessing(config)
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    source_record = manifest["sources"][0]
    fragment = manifest["fragments"][0]

    assert source_record["technique_coverage"] == "COMPLETE"
    assert source_record["technique_sidecar"].endswith(
        "riff.mid.techniques.json"
    )
    assert source_record["technique_sidecar_sha256"] == hashlib.sha256(
        sidecar_path.read_bytes()
    ).hexdigest()
    assert source_record["technique_counts"] == {
        "DEAD_NOTE": 0,
        "PALM_MUTE": 1,
        "SLIDE_DOWN": 0,
        "SLIDE_UP": 1,
        "VIBRATO": 1,
    }
    assert fragment["technique_coverage"] == "COMPLETE"
    assert fragment["techniques"] == [
        {"type": "PALM_MUTE_ON", "note_index": 0},
        {"type": "PALM_MUTE_OFF", "note_index": 1},
        {"type": "SLIDE_UP", "note_index": 1},
        {"type": "VIBRATO", "note_index": 1},
    ]

    write_sidecar(include_vibrato=False)
    second = run_preprocessing(config)
    assert second.run_id != first.run_id


def test_pipeline_rejects_orphan_technique_sidecar(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "orphan-sidecar-project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='orphan-sidecar-test'\n", encoding="utf-8"
    )
    raw_dir = project_root / "data/raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "ghost.mid.techniques.json").write_text("{}\n", encoding="utf-8")
    config = load_preprocess_config(_write_pipeline_config(project_root))

    with pytest.raises(TechniqueSidecarError, match="Orphan technique sidecar"):
        run_preprocessing(config)


def test_orphan_sidecar_cannot_hide_behind_an_ignored_midi_symlink(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    project_root = tmp_path / "symlink-sidecar-project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='symlink-sidecar-test'\n", encoding="utf-8"
    )
    raw_dir = project_root / "data/raw"
    real_midi = write_midi_file(
        raw_dir / "real.mid", [make_instrument([(60, 0.0, 0.5)])]
    )
    (raw_dir / "alias.mid").symlink_to(real_midi.name)
    (raw_dir / "alias.mid.techniques.json").write_text("{}\n", encoding="utf-8")
    config = load_preprocess_config(_write_pipeline_config(project_root))

    with pytest.raises(TechniqueSidecarError, match="Orphan technique sidecar"):
        run_preprocessing(config)


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


def test_quantization_collision_discards_only_the_affected_source(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    project_root = tmp_path / "quantization-collision-project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='quantization-collision-test'\n", encoding="utf-8"
    )
    raw_dir = project_root / "data/raw"
    write_midi_file(
        raw_dir / "colliding.mid",
        [make_instrument([(60, 0.01, 0.03), (60, 0.04, 0.06)])],
    )
    write_midi_file(
        raw_dir / "valid.mid",
        [make_instrument([(64, 0.0, 0.25), (67, 0.25, 0.5)])],
    )
    base = load_preprocess_config(_write_pipeline_config(project_root))
    config = replace(
        base,
        preprocessing=replace(base.preprocessing, remove_initial_silence=False),
    )

    report = run_preprocessing(config)
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    sources = {
        Path(source["source_file"]).name: source for source in manifest["sources"]
    }

    assert report.compatible_sources == 1
    assert report.discarded_sources == 1
    assert report.generated_fragments == 3
    assert sources["colliding.mid"]["compatible"] is False
    assert "Quantization mapped distinct notes" in sources["colliding.mid"][
        "discard_reason"
    ]
    assert sources["valid.mid"]["compatible"] is True


def test_pipeline_preserves_pitch_bends_across_fragments_and_transposition(
    tmp_path: Path,
    write_pitch_bend_midi_file,
) -> None:
    project_root = tmp_path / "pitch-bend-project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text(
        "[project]\nname='pitch-bend-stage-one-test'\n", encoding="utf-8"
    )
    write_pitch_bend_midi_file(
        project_root / "data/raw/bent-riff.mid",
        notes=[(60, 0, 480), (64, 3840, 4320)],
        # 2048 at a +/-12 source range is three semitones. Stage 1 must
        # normalize it to 4096 at the canonical +/-6 range.
        bends=[(3360, 2048), (4320, 0)],
        range_events=[(0, 12)],
    )
    config = load_preprocess_config(_write_pipeline_config(project_root))

    report = run_preprocessing(config)
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))

    assert report.compatible_sources == 1
    assert report.generated_fragments == 6
    source = manifest["sources"][0]
    assert source["num_pitch_bend_events"] == 2
    assert source["num_expressive_pitch_bend_events"] == 1
    assert source["source_pitch_bend_range_semitones"] == 12.0
    assert source["canonical_pitch_bend_range_semitones"] == 6

    bend_snapshots: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for fragment in manifest["fragments"]:
        output_path = project_root / fragment["output_file"]
        assert hashlib.sha256(output_path.read_bytes()).hexdigest() == fragment[
            "output_sha256"
        ]
        assert output_path.stat().st_size == fragment["output_size_bytes"]
        output = read_midi(output_path)
        bend_snapshots[
            (fragment["phrase_index"], fragment["transpose_semitones"])
        ] = [
            (bend.pitch, bend.time)
            for bend in output.instruments[0].pitch_bends
        ]
        assert fragment["num_pitch_bend_events"] == 2
        assert fragment["num_expressive_pitch_bend_events"] == 1
        assert fragment["pitch_bend_range_semitones"] == 6
        assert fragment["actual_note_duration_seconds"] == pytest.approx(0.5)

    for semitones in (-1, 0, 1):
        assert bend_snapshots[(0, semitones)] == [
            (4096, pytest.approx(3.5)),
            (0, pytest.approx(4.0)),
        ]
        assert bend_snapshots[(1, semitones)] == [
            (4096, pytest.approx(0.0)),
            (0, pytest.approx(0.5)),
        ]
    first_phrase = [
        fragment
        for fragment in manifest["fragments"]
        if fragment["phrase_index"] == 0
    ]
    second_phrase = [
        fragment
        for fragment in manifest["fragments"]
        if fragment["phrase_index"] == 1
    ]
    assert all(
        fragment["synthetic_final_pitch_bend_reset"]
        for fragment in first_phrase
    )
    assert all(
        fragment["synthetic_initial_pitch_bend"]
        for fragment in second_phrase
    )
