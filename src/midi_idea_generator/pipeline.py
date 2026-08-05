"""Orchestration for inspection and stage-one preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import logging
from pathlib import Path
import platform
import re
import shutil
import tempfile
from typing import Any

from . import __version__
from .augmentation import augmentation_offsets, transpose_midi
from .config import PreprocessConfig
from .midi_io import (
    MidiInspection,
    MidiReadError,
    MidiWriteError,
    UnsupportedMidiError,
    discover_midi_files,
    get_midi_duration_seconds,
    inspect_midi,
    read_midi,
    write_midi,
)
from .preprocessing import (
    PitchBendNormalizationError,
    QuantizationCollisionError,
    normalize_instrument,
    split_instrument_into_phrases,
)
from .splitting import assign_source_splits
from .technique_processing import (
    TECHNIQUE_COVERAGE_COMPLETE,
    TECHNIQUE_COVERAGE_UNLABELED,
    TechniqueProjectionError,
    project_phrase_techniques,
    source_technique_counts,
)
from .techniques import (
    SIDECAR_SUFFIX,
    TechniqueSidecar,
    TechniqueSidecarError,
    load_technique_sidecar,
    sidecar_path_for,
)
from .tonality import (
    SIDECAR_SUFFIX as TONALITY_SIDECAR_SUFFIX,
    Tonality,
    TonalitySidecar,
    TonalitySidecarError,
    infer_tonality,
    load_tonality_sidecar,
    sidecar_path_for as tonality_sidecar_path_for,
)
from .utils import relative_label, write_json

LOGGER = logging.getLogger(__name__)
PIPELINE_SCHEMA_VERSION = 4


@dataclass(frozen=True, slots=True)
class SourceFingerprint:
    """Stable source identity used for provenance and race detection."""

    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class TechniqueSidecarInput:
    """Captured optional-sidecar state used for reproducibility checks."""

    path: Path
    present: bool
    fingerprint: SourceFingerprint | None


@dataclass(frozen=True, slots=True)
class TonalitySidecarInput:
    """Captured optional tonality-sidecar state for reproducibility checks."""

    path: Path
    present: bool
    fingerprint: SourceFingerprint | None


@dataclass(frozen=True, slots=True)
class PreprocessReport:
    """Summary returned after a full preprocessing run."""

    discovered_files: int
    compatible_sources: int
    discarded_sources: int
    generated_fragments: int
    run_id: str
    processed_run_dir: Path
    manifest_path: Path


def inspect_directory(
    input_dir: Path,
    config: PreprocessConfig,
) -> list[MidiInspection]:
    """Inspect every discovered MIDI and continue past corrupt files."""

    files = discover_midi_files(input_dir)
    return [
        inspect_midi(path, config.validation, config.track_selection)
        for path in files
    ]


def _safe_source_stem(source_label: str) -> str:
    stem = Path(source_label).stem
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-_").lower()
    slug = slug[:48] or "midi"
    digest = hashlib.sha256(source_label.encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{digest}"


def _transpose_tag(semitones: int) -> str:
    if semitones < 0:
        return f"m{abs(semitones):02d}"
    if semitones > 0:
        return f"p{semitones:02d}"
    return "orig"


def _source_record(
    inspection: MidiInspection,
    source_label: str,
    split: str | None,
    fingerprint: SourceFingerprint | None,
    content_group_size: int,
    canonical_pitch_bend_range_semitones: int,
    sidecar_input: TechniqueSidecarInput,
    sidecar_label: str,
    tonality_sidecar_input: TonalitySidecarInput,
    tonality_sidecar_label: str,
) -> dict[str, Any]:
    track = (
        inspection.tracks[inspection.selected_track]
        if inspection.selected_track is not None
        else None
    )
    return {
        "source_file": source_label,
        "source_sha256": fingerprint.sha256 if fingerprint else None,
        "source_size_bytes": fingerprint.size_bytes if fingerprint else None,
        "content_group_size": content_group_size,
        "split": split,
        # This is a zero-based pretty_midi instrument index, not a raw MIDI
        # track number. It is named track_number to match the public manifest.
        "track_number": inspection.selected_track,
        "instrument_index": inspection.selected_track,
        "tempo_bpm": inspection.tempo_bpm,
        "time_signature": (
            inspection.time_signatures[0] if inspection.time_signatures else None
        ),
        "duration_seconds": inspection.duration_seconds,
        "resolution": inspection.resolution,
        "num_notes": track.num_notes if track else None,
        "raw_note_events": track.raw_note_events if track else None,
        "duplicate_notes_collapsed": (
            track.duplicate_notes_collapsed if track else None
        ),
        "num_pitch_bend_events": track.num_pitch_bend_events if track else None,
        "num_expressive_pitch_bend_events": (
            track.num_expressive_pitch_bend_events if track else None
        ),
        "source_pitch_bend_range_semitones": (
            track.source_pitch_bend_range_semitones if track else None
        ),
        "canonical_pitch_bend_range_semitones": (
            canonical_pitch_bend_range_semitones
            if track and track.has_pitch_bends
            else None
        ),
        "technique_sidecar": sidecar_label if sidecar_input.present else None,
        "technique_sidecar_sha256": (
            sidecar_input.fingerprint.sha256
            if sidecar_input.fingerprint is not None
            else None
        ),
        "technique_sidecar_size_bytes": (
            sidecar_input.fingerprint.size_bytes
            if sidecar_input.fingerprint is not None
            else None
        ),
        "technique_coverage": (
            None if sidecar_input.present else TECHNIQUE_COVERAGE_UNLABELED
        ),
        "technique_note_count": 0,
        "technique_annotation_count": 0,
        "technique_counts": source_technique_counts(None),
        "palm_mute_range_count": 0,
        "tonality_sidecar": (
            tonality_sidecar_label if tonality_sidecar_input.present else None
        ),
        "tonality_sidecar_sha256": (
            tonality_sidecar_input.fingerprint.sha256
            if tonality_sidecar_input.fingerprint is not None
            else None
        ),
        "tonality_sidecar_size_bytes": (
            tonality_sidecar_input.fingerprint.size_bytes
            if tonality_sidecar_input.fingerprint is not None
            else None
        ),
        "tonality": Tonality.unknown().as_dict(),
        "num_base_fragments": 0,
        "num_fragments_generated": 0,
        "compatible": inspection.compatible,
        "discard_reason": inspection.discard_reason,
        "inspection_issues": list(inspection.issues),
    }


def _mark_discarded(record: dict[str, Any], reason: str) -> None:
    record["compatible"] = False
    record["discard_reason"] = reason
    record["inspection_issues"] = [*record["inspection_issues"], reason]


def _source_label(path: Path, config: PreprocessConfig) -> str:
    relative_source = path.resolve().relative_to(config.paths.input_dir)
    try:
        input_prefix = config.paths.input_dir.relative_to(config.project_root)
    except ValueError:
        return relative_source.as_posix()
    return (input_prefix / relative_source).as_posix()


def _fingerprint_file(path: Path) -> SourceFingerprint:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return SourceFingerprint(sha256=digest.hexdigest(), size_bytes=size)


def _try_fingerprint(path: Path) -> SourceFingerprint | None:
    try:
        return _fingerprint_file(path)
    except OSError as exc:
        LOGGER.warning("Could not fingerprint source %s: %s", path, exc)
        return None


def _capture_sidecar_input(midi_path: Path) -> TechniqueSidecarInput:
    path = sidecar_path_for(midi_path)
    present = path.exists() or path.is_symlink()
    return TechniqueSidecarInput(
        path=path,
        present=present,
        fingerprint=_try_fingerprint(path) if present else None,
    )


def _capture_tonality_sidecar_input(midi_path: Path) -> TonalitySidecarInput:
    path = tonality_sidecar_path_for(midi_path)
    present = path.exists() or path.is_symlink()
    return TonalitySidecarInput(
        path=path,
        present=present,
        fingerprint=_try_fingerprint(path) if present else None,
    )


def _validate_no_orphan_sidecars(input_dir: Path, midi_files: list[Path]) -> None:
    known_midis = set(midi_files)
    for path in sorted(input_dir.rglob(f"*{SIDECAR_SUFFIX}")):
        if not (path.is_file() or path.is_symlink()):
            continue
        midi_name = path.name[: -len(SIDECAR_SUFFIX)]
        expected_midi = path.with_name(midi_name)
        if (
            expected_midi not in known_midis
            or expected_midi.is_symlink()
            or not expected_midi.is_file()
        ):
            raise TechniqueSidecarError(
                "Orphan technique sidecar has no discovered sibling MIDI: "
                f"{path}"
            )
    for path in sorted(input_dir.rglob(f"*{TONALITY_SIDECAR_SUFFIX}")):
        if not (path.is_file() or path.is_symlink()):
            continue
        midi_name = path.name[: -len(TONALITY_SIDECAR_SUFFIX)]
        expected_midi = path.with_name(midi_name)
        if (
            expected_midi not in known_midis
            or expected_midi.is_symlink()
            or not expected_midi.is_file()
        ):
            raise TonalitySidecarError(
                "Orphan tonality sidecar has no discovered sibling MIDI: "
                f"{path}"
            )


def _populate_sidecar_record(
    record: dict[str, Any], sidecar: TechniqueSidecar | None
) -> None:
    if sidecar is None:
        record["technique_coverage"] = TECHNIQUE_COVERAGE_UNLABELED
        return
    counts = source_technique_counts(sidecar)
    record["technique_coverage"] = TECHNIQUE_COVERAGE_COMPLETE
    record["technique_note_count"] = len(sidecar.note_techniques)
    record["technique_annotation_count"] = sum(counts.values())
    record["technique_counts"] = counts
    record["palm_mute_range_count"] = len(sidecar.palm_mute_ranges)


def _resolve_source_tonality(
    sidecar: TonalitySidecar | None,
    normalized_instrument: Any,
    missing_sidecar_policy: str,
) -> Tonality:
    if sidecar is not None:
        return sidecar.tonality
    if missing_sidecar_policy == "unknown":
        return Tonality.unknown()
    # A source estimate remains useful provenance even when fragments will be
    # conditioned by their own local estimates.
    return infer_tonality(normalized_instrument, method="AUTO_SOURCE")


def _resolve_phrase_tonality(
    *,
    sidecar: TonalitySidecar | None,
    source_tonality: Tonality,
    phrase: Any,
    missing_sidecar_policy: str,
) -> Tonality:
    if sidecar is not None or missing_sidecar_policy != "infer_fragment":
        return source_tonality
    return infer_tonality(
        phrase.midi.instruments[0],
        method="AUTO_FRAGMENT",
    )


def _verify_captured_inputs(
    fingerprints: dict[Path, SourceFingerprint | None],
    sidecar_inputs: dict[Path, TechniqueSidecarInput],
    tonality_sidecar_inputs: dict[Path, TonalitySidecarInput],
) -> None:
    changed_sources = [
        path
        for path, expected in fingerprints.items()
        if _try_fingerprint(path) != expected
    ]
    changed_sidecars = [
        path
        for path, expected in sidecar_inputs.items()
        if _capture_sidecar_input(path) != expected
    ]
    changed_tonality_sidecars = [
        path
        for path, expected in tonality_sidecar_inputs.items()
        if _capture_tonality_sidecar_input(path) != expected
    ]
    if changed_sources or changed_sidecars or changed_tonality_sidecars:
        labels = [
            path.name
            for path in (
                *changed_sources,
                *changed_sidecars,
                *changed_tonality_sidecars,
            )
        ][:5]
        raise RuntimeError(
            "Raw MIDI or annotation sidecar changed before publication: "
            + ", ".join(labels)
        )


def _config_snapshot(config: PreprocessConfig) -> dict[str, Any]:
    return {
        "random_seed": config.random_seed,
        "validation": asdict(config.validation),
        "track_selection": asdict(config.track_selection),
        "preprocessing": asdict(config.preprocessing),
        "augmentation": asdict(config.augmentation),
        "tonality": asdict(config.tonality),
        "splits": asdict(config.splits),
    }


def _implementation_sha256() -> str:
    package_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_dir.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _tool_versions() -> dict[str, str]:
    versions = {
        "implementation_sha256": _implementation_sha256(),
        "midi_idea_generator": __version__,
        "python": platform.python_version(),
    }
    for distribution in ("mido", "numpy", "pretty-midi", "PyYAML"):
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _canonical_hash(payload: Any) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _make_run_id(
    config_sha256: str,
    labels: dict[Path, str],
    fingerprints: dict[Path, SourceFingerprint | None],
    sidecar_inputs: dict[Path, TechniqueSidecarInput],
    tonality_sidecar_inputs: dict[Path, TonalitySidecarInput],
    versions: dict[str, str],
) -> str:
    sources = [
        {
            "source_file": labels[path],
            "sha256": fingerprint.sha256 if fingerprint else None,
            "technique_sidecar": {
                "present": sidecar_inputs[path].present,
                "sha256": (
                    sidecar_inputs[path].fingerprint.sha256
                    if sidecar_inputs[path].fingerprint is not None
                    else None
                ),
                "size_bytes": (
                    sidecar_inputs[path].fingerprint.size_bytes
                    if sidecar_inputs[path].fingerprint is not None
                    else None
                ),
            },
            "tonality_sidecar": {
                "present": tonality_sidecar_inputs[path].present,
                "sha256": (
                    tonality_sidecar_inputs[path].fingerprint.sha256
                    if tonality_sidecar_inputs[path].fingerprint is not None
                    else None
                ),
                "size_bytes": (
                    tonality_sidecar_inputs[path].fingerprint.size_bytes
                    if tonality_sidecar_inputs[path].fingerprint is not None
                    else None
                ),
            },
        }
        for path, fingerprint in sorted(
            fingerprints.items(), key=lambda item: labels[item[0]]
        )
    ]
    return _canonical_hash(
        {
            "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
            "config_sha256": config_sha256,
            "sources": sources,
            "tool_versions": versions,
        }
    )[:20]


def _create_staging_directory(processed_dir: Path) -> Path:
    if processed_dir.exists() and not processed_dir.is_dir():
        raise NotADirectoryError(
            f"Processed MIDI path exists but is not a directory: {processed_dir}"
        )
    processed_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=".stage-", dir=processed_dir
        )
    )


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _tree_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    for path in files:
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _publish_run(staging_dir: Path, run_dir: Path) -> bool:
    """Publish or safely reuse an identical immutable run.

    Returns ``True`` only when this call created the final run directory.
    """

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        if not run_dir.is_dir() or _tree_digest(run_dir) != _tree_digest(staging_dir):
            raise FileExistsError(
                "A processed run with this reproducibility fingerprint exists "
                f"but has different contents: {run_dir}"
            )
        return False
    staging_dir.rename(run_dir)
    return True


def _process_sources(
    config: PreprocessConfig,
    inspections: list[MidiInspection],
    labels: dict[Path, str],
    assignments: dict[str, str],
    fingerprints: dict[Path, SourceFingerprint | None],
    sidecar_inputs: dict[Path, TechniqueSidecarInput],
    tonality_sidecar_inputs: dict[Path, TonalitySidecarInput],
    content_group_sizes: dict[Path, int],
    changed_sources: set[Path],
    staging_dir: Path,
    final_run_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_records: list[dict[str, Any]] = []
    fragment_records: list[dict[str, Any]] = []

    for inspection in inspections:
        source_label = labels[inspection.source_file]
        split = assignments.get(source_label)
        expected_fingerprint = fingerprints[inspection.source_file]
        expected_sidecar = sidecar_inputs[inspection.source_file]
        expected_tonality_sidecar = tonality_sidecar_inputs[
            inspection.source_file
        ]
        record = _source_record(
            inspection,
            source_label,
            split,
            expected_fingerprint,
            content_group_sizes.get(inspection.source_file, 1),
            config.validation.canonical_pitch_bend_range_semitones,
            expected_sidecar,
            f"{source_label}{SIDECAR_SUFFIX}",
            expected_tonality_sidecar,
            f"{source_label}{TONALITY_SIDECAR_SUFFIX}",
        )
        source_records.append(record)
        if inspection.source_file in changed_sources:
            reason = "Source file changed while it was being inspected"
            _mark_discarded(record, reason)
            LOGGER.warning("Discarded %s: %s", source_label, reason)
            continue
        if not inspection.compatible or split is None:
            LOGGER.warning("Discarded %s: %s", source_label, record["discard_reason"])
            continue

        assert inspection.selected_track is not None
        assert inspection.tempo_bpm is not None
        if _try_fingerprint(inspection.source_file) != expected_fingerprint:
            reason = "Source file changed before preprocessing"
            _mark_discarded(record, reason)
            LOGGER.warning("Discarded %s: %s", source_label, reason)
            continue

        try:
            midi = read_midi(inspection.source_file)
        except (MidiReadError, UnsupportedMidiError) as exc:
            _mark_discarded(record, exc.manifest_reason)
            LOGGER.warning("Discarded %s after re-read: %s", source_label, exc)
            continue
        if _try_fingerprint(inspection.source_file) != expected_fingerprint:
            reason = "Source file changed during preprocessing"
            _mark_discarded(record, reason)
            LOGGER.warning("Discarded %s: %s", source_label, reason)
            continue
        try:
            assert expected_fingerprint is not None
            sidecar = load_technique_sidecar(
                inspection.source_file,
                source_sha256=expected_fingerprint.sha256,
                midi=midi,
                instrument_index=inspection.selected_track,
            )
        except TechniqueSidecarError as exc:
            reason = f"Invalid technique sidecar: {exc}"
            _mark_discarded(record, reason)
            LOGGER.warning("Discarded %s: %s", source_label, reason)
            continue
        loaded_sidecar_fingerprint = (
            SourceFingerprint(sidecar.sha256, sidecar.size_bytes)
            if sidecar is not None
            else None
        )
        if (
            expected_sidecar.present != (sidecar is not None)
            or expected_sidecar.fingerprint != loaded_sidecar_fingerprint
        ):
            reason = "Technique sidecar changed while it was being loaded"
            _mark_discarded(record, reason)
            LOGGER.warning("Discarded %s: %s", source_label, reason)
            continue
        current_sidecar = _capture_sidecar_input(inspection.source_file)
        if current_sidecar != expected_sidecar:
            reason = "Technique sidecar changed during preprocessing"
            _mark_discarded(record, reason)
            LOGGER.warning("Discarded %s: %s", source_label, reason)
            continue
        _populate_sidecar_record(record, sidecar)

        try:
            tonality_sidecar = load_tonality_sidecar(
                inspection.source_file,
                source_sha256=expected_fingerprint.sha256,
                midi=midi,
                instrument_index=inspection.selected_track,
            )
        except TonalitySidecarError as exc:
            reason = f"Invalid tonality sidecar: {exc}"
            _mark_discarded(record, reason)
            LOGGER.warning("Discarded %s: %s", source_label, reason)
            continue
        loaded_tonality_fingerprint = (
            SourceFingerprint(tonality_sidecar.sha256, tonality_sidecar.size_bytes)
            if tonality_sidecar is not None
            else None
        )
        if (
            expected_tonality_sidecar.present != (tonality_sidecar is not None)
            or expected_tonality_sidecar.fingerprint
            != loaded_tonality_fingerprint
        ):
            reason = "Tonality sidecar changed while it was being loaded"
            _mark_discarded(record, reason)
            LOGGER.warning("Discarded %s: %s", source_label, reason)
            continue
        if (
            _capture_tonality_sidecar_input(inspection.source_file)
            != expected_tonality_sidecar
        ):
            reason = "Tonality sidecar changed during preprocessing"
            _mark_discarded(record, reason)
            LOGGER.warning("Discarded %s: %s", source_label, reason)
            continue

        instrument = midi.instruments[inspection.selected_track]
        initial_silence = (
            min(note.start for note in instrument.notes)
            if config.preprocessing.remove_initial_silence
            else 0.0
        )
        normalized_source_duration = max(
            0.0, get_midi_duration_seconds(midi) - initial_silence
        )
        try:
            normalized = normalize_instrument(
                instrument,
                inspection.tempo_bpm,
                config.preprocessing,
                resolution=midi.resolution,
                source_pitch_bend_range_semitones=(
                    inspection.tracks[inspection.selected_track]
                    .source_pitch_bend_range_semitones
                ),
                canonical_pitch_bend_range_semitones=(
                    config.validation.canonical_pitch_bend_range_semitones
                ),
            )
        except (QuantizationCollisionError, PitchBendNormalizationError) as exc:
            reason = str(exc)
            _mark_discarded(record, reason)
            LOGGER.warning("Discarded %s: %s", source_label, reason)
            continue
        source_tonality = _resolve_source_tonality(
            tonality_sidecar,
            normalized,
            config.tonality.missing_sidecar_policy,
        )
        record["tonality"] = source_tonality.as_dict()
        phrases = split_instrument_into_phrases(
            normalized,
            inspection.tempo_bpm,
            config.preprocessing,
            resolution=midi.resolution,
            source_duration_seconds=normalized_source_duration,
            canonical_pitch_bend_range_semitones=(
                config.validation.canonical_pitch_bend_range_semitones
            ),
        )
        record["num_base_fragments"] = len(phrases)
        if not phrases:
            reason = (
                "No phrase met the configured length and minimum-note constraints"
            )
            _mark_discarded(record, reason)
            LOGGER.warning("Discarded %s: %s", source_label, reason)
            continue

        source_stem = _safe_source_stem(source_label)
        offsets = augmentation_offsets(config.augmentation, split)
        write_failures: list[dict[str, Any]] = []
        variants: list[tuple[Any, int, Any, tuple[Any, ...], Tonality]] = []
        try:
            for phrase in phrases:
                phrase_tonality = _resolve_phrase_tonality(
                    sidecar=tonality_sidecar,
                    source_tonality=source_tonality,
                    phrase=phrase,
                    missing_sidecar_policy=(
                        config.tonality.missing_sidecar_policy
                    ),
                )
                for semitones in offsets:
                    techniques = project_phrase_techniques(
                        source_midi=midi,
                        instrument_index=inspection.selected_track,
                        sidecar=sidecar,
                        normalized_instrument=normalized,
                        phrase=phrase,
                        tempo_bpm=inspection.tempo_bpm,
                        processing=config.preprocessing,
                        semitones=semitones,
                        pitch_min=config.validation.pitch_min,
                        pitch_max=config.validation.pitch_max,
                    )
                    if techniques is None:
                        LOGGER.debug(
                            "Skipped out-of-range slide target at %+d for %s "
                            "phrase %d",
                            semitones,
                            source_label,
                            phrase.phrase_index,
                        )
                        continue
                    transposed = transpose_midi(
                        phrase.midi,
                        semitones=semitones,
                        pitch_min=config.validation.pitch_min,
                        pitch_max=config.validation.pitch_max,
                    )
                    if transposed is None:
                        LOGGER.debug(
                            "Skipped out-of-range transposition %+d for %s phrase %d",
                            semitones,
                            source_label,
                            phrase.phrase_index,
                        )
                        continue
                    variants.append(
                        (
                            phrase,
                            semitones,
                            transposed,
                            techniques,
                            phrase_tonality.transposed(semitones),
                        )
                    )
        except TechniqueProjectionError as exc:
            reason = f"Could not project guitar techniques: {exc}"
            _mark_discarded(record, reason)
            LOGGER.warning("Discarded %s: %s", source_label, reason)
            continue
        for phrase, semitones, transposed, techniques, tonality in variants:
            filename = (
                f"{source_stem}_phrase-{phrase.phrase_index:04d}_"
                f"{_transpose_tag(semitones)}.mid"
            )
            staged_output_path = staging_dir / split / filename
            final_output_path = final_run_dir / split / filename
            try:
                write_midi(transposed, staged_output_path)
            except MidiWriteError as exc:
                write_failures.append(
                    {
                        "phrase_index": phrase.phrase_index,
                        "transpose_semitones": semitones,
                        "reason": exc.manifest_reason,
                    }
                )
                LOGGER.error("%s", exc)
                continue
            output_label = relative_label(final_output_path, config.project_root)
            num_notes = len(transposed.instruments[0].notes)
            output_fingerprint = _fingerprint_file(staged_output_path)
            output_instrument = transposed.instruments[0]
            expressive_bends = sum(
                bend.pitch != 0 for bend in output_instrument.pitch_bends
            )
            actual_note_duration = max(note.end for note in output_instrument.notes)
            fragment_records.append(
                {
                    "source_file": source_label,
                    "split": split,
                    "track_number": inspection.selected_track,
                    "instrument_index": inspection.selected_track,
                    "phrase_index": phrase.phrase_index,
                    "transpose_semitones": semitones,
                    "output_file": output_label,
                    "num_notes": num_notes,
                    "num_pitch_bend_events": len(output_instrument.pitch_bends),
                    "num_expressive_pitch_bend_events": expressive_bends,
                    "pitch_bend_range_semitones": (
                        config.validation.canonical_pitch_bend_range_semitones
                        if expressive_bends
                        else None
                    ),
                    "synthetic_initial_pitch_bend": (
                        phrase.synthetic_initial_pitch_bend
                    ),
                    "synthetic_final_pitch_bend_reset": (
                        phrase.synthetic_final_pitch_bend_reset
                    ),
                    "technique_coverage": record["technique_coverage"],
                    "techniques": [technique.as_dict() for technique in techniques],
                    "tonality": tonality.as_dict(),
                    "output_sha256": output_fingerprint.sha256,
                    "output_size_bytes": output_fingerprint.size_bytes,
                    "nominal_duration_seconds": phrase.nominal_duration_seconds,
                    "actual_note_duration_seconds": float(actual_note_duration),
                }
            )
            record["num_fragments_generated"] += 1
        if write_failures:
            record["write_errors"] = write_failures
        if record["num_fragments_generated"] == 0:
            _mark_discarded(record, "All phrase variants failed validation or writing")
        else:
            LOGGER.info(
                "Processed %s -> %d phrase file(s) in %s",
                source_label,
                record["num_fragments_generated"],
                split,
            )
    return source_records, fragment_records


def run_preprocessing(config: PreprocessConfig) -> PreprocessReport:
    """Run validation, source-level splitting, phrase creation, and augmentation."""

    files = discover_midi_files(config.paths.input_dir)
    _validate_no_orphan_sidecars(config.paths.input_dir, files)
    fingerprints = {path: _try_fingerprint(path) for path in files}
    sidecar_inputs = {path: _capture_sidecar_input(path) for path in files}
    tonality_sidecar_inputs = {
        path: _capture_tonality_sidecar_input(path) for path in files
    }
    inspections = [
        inspect_midi(path, config.validation, config.track_selection)
        for path in files
    ]
    fingerprints_after_inspection = {
        path: _try_fingerprint(path) for path in files
    }
    sidecars_after_inspection = {
        path: _capture_sidecar_input(path) for path in files
    }
    tonality_sidecars_after_inspection = {
        path: _capture_tonality_sidecar_input(path) for path in files
    }
    changed_sources = {
        path
        for path in files
        if fingerprints[path] is None
        or fingerprints_after_inspection[path] is None
        or fingerprints[path] != fingerprints_after_inspection[path]
        or sidecar_inputs[path] != sidecars_after_inspection[path]
        or tonality_sidecar_inputs[path]
        != tonality_sidecars_after_inspection[path]
    }
    labels = {
        inspection.source_file: _source_label(inspection.source_file, config)
        for inspection in inspections
    }
    content_groups: dict[str, list[Path]] = {}
    for inspection in inspections:
        fingerprint = fingerprints[inspection.source_file]
        if (
            inspection.compatible
            and inspection.source_file not in changed_sources
            and fingerprint is not None
        ):
            content_groups.setdefault(fingerprint.sha256, []).append(
                inspection.source_file
            )
    group_assignments = assign_source_splits(
        content_groups, config.splits, config.random_seed
    )
    assignments = {
        labels[path]: group_assignments[group_sha256]
        for group_sha256, paths in content_groups.items()
        for path in paths
    }
    content_group_sizes = {
        path: len(paths)
        for paths in content_groups.values()
        for path in paths
    }
    configuration = _config_snapshot(config)
    configuration_sha256 = _canonical_hash(configuration)
    tool_versions = _tool_versions()
    run_id = _make_run_id(
        configuration_sha256,
        labels,
        fingerprints,
        sidecar_inputs,
        tonality_sidecar_inputs,
        tool_versions,
    )
    final_run_dir = config.paths.processed_dir / "runs" / run_id
    staging_dir = _create_staging_directory(config.paths.processed_dir)
    published_new_run = False
    try:
        source_records, fragment_records = _process_sources(
            config,
            inspections,
            labels,
            assignments,
            fingerprints,
            sidecar_inputs,
            tonality_sidecar_inputs,
            content_group_sizes,
            changed_sources,
            staging_dir,
            final_run_dir,
        )
        compatible_count = sum(record["compatible"] for record in source_records)
        current_files = discover_midi_files(config.paths.input_dir)
        if current_files != files:
            raise RuntimeError("Raw MIDI inventory changed before publication.")
        _validate_no_orphan_sidecars(config.paths.input_dir, current_files)
        _verify_captured_inputs(
            fingerprints,
            sidecar_inputs,
            tonality_sidecar_inputs,
        )
        manifest = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "run_id": run_id,
            "processed_run_dir": relative_label(
                final_run_dir, config.project_root
            ),
            "configuration_sha256": configuration_sha256,
            "configuration": configuration,
            "tool_versions": tool_versions,
            "random_seed": config.random_seed,
            "split_ratios": {
                "train": config.splits.train,
                "validation": config.splits.validation,
                "test": config.splits.test,
            },
            "summary": {
                "discovered_files": len(inspections),
                "compatible_sources": compatible_count,
                "discarded_sources": len(inspections) - compatible_count,
                "generated_fragments": len(fragment_records),
            },
            "sources": source_records,
            "fragments": fragment_records,
        }
        # A manifest references exactly one immutable run. Older runs can
        # remain on disk without being mixed into the current dataset.
        published_new_run = _publish_run(staging_dir, final_run_dir)
        try:
            write_json(config.paths.manifest_path, manifest)
        except BaseException:
            if published_new_run:
                _remove_tree(final_run_dir)
            raise
    finally:
        _remove_tree(staging_dir)
    return PreprocessReport(
        discovered_files=len(inspections),
        compatible_sources=compatible_count,
        discarded_sources=len(inspections) - compatible_count,
        generated_fragments=len(fragment_records),
        run_id=run_id,
        processed_run_dir=final_run_dir,
        manifest_path=config.paths.manifest_path,
    )
