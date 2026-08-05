"""Deterministic Stage 2 orchestration from processed MIDI to REMI tokens."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import re
import shutil
import tempfile
from typing import Any, Mapping, Never, Sequence

from . import __version__
from .tokenization_config import (
    PITCH_BEND_SENSITIVITY_SEMITONES,
    TokenizationConfig,
)
from .tonality import (
    MODE_NAMES,
    TONALITY_METHODS,
    TONIC_NAMES,
    Tonality,
    TonalityError,
)
from .tokenizer import (
    CONDITIONING_SCHEMA_VERSION,
    TECHNIQUE_TYPES,
    EncodedMidi,
    TechniqueAnnotation,
    TokenizationError,
    build_tokenizer,
    encode_midi,
    get_mode_token_ids,
    get_special_token_ids,
    get_technique_token_ids,
    get_tonic_token_ids,
    load_tokenizer,
    save_tokenizer,
)
from .utils import relative_label, write_json


TOKENIZATION_SCHEMA_VERSION = 3
SEQUENCE_SCHEMA_VERSION = 3
PREPROCESSING_SCHEMA_VERSION = 4
_SPLITS = ("train", "validation", "test")
_TECHNIQUE_COVERAGE = ("UNLABELED", "COMPLETE")
_TECHNIQUE_ORDER = {
    technique_type: index for index, technique_type in enumerate(TECHNIQUE_TYPES)
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{20}$")


class TokenizationPipelineError(RuntimeError):
    """Raised when Stage 2 cannot safely publish a complete tokenized run."""


@dataclass(frozen=True, slots=True)
class FileFingerprint:
    """Stable identity of one input or output file."""

    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class FragmentSpec:
    """Validated Stage 1 fragment consumed by Stage 2."""

    source_file: str
    source_sha256: str
    split: str
    track_number: int
    instrument_index: int
    phrase_index: int
    transpose_semitones: int
    output_file: str
    path: Path
    num_notes: int
    num_pitch_bend_events: int
    num_expressive_pitch_bend_events: int
    pitch_bend_range_semitones: int | None
    synthetic_initial_pitch_bend: bool
    synthetic_final_pitch_bend_reset: bool
    output_sha256: str
    output_size_bytes: int
    technique_coverage: str
    techniques: tuple[TechniqueAnnotation, ...]
    nominal_duration_seconds: float
    actual_note_duration_seconds: float
    tonality: Tonality


@dataclass(frozen=True, slots=True)
class PreprocessingManifest:
    """Validated authoritative boundary produced by Stage 1."""

    path: Path
    fingerprint: FileFingerprint
    run_id: str
    processed_run_dir: Path
    configuration_sha256: str
    fragments: tuple[FragmentSpec, ...]


@dataclass(frozen=True, slots=True)
class TokenizationReport:
    """Summary returned by a successful Stage 2 run."""

    tokenization_run_id: str
    tokenized_run_dir: Path
    manifest_path: Path
    num_sequences: int
    vocabulary_size: int
    min_tokens: int
    max_tokens: int
    reused_run: bool


def run_tokenization(config: TokenizationConfig) -> TokenizationReport:
    """Tokenize every fragment declared by the current Stage 1 manifest.

    Processed directories are never globbed.  A malformed, missing, changed,
    or incompatible declared fragment aborts publication of the whole run so
    that downstream training can never observe a partial dataset.
    """

    preprocessing = _load_preprocessing_manifest(
        config.paths.preprocessing_manifest_path, config.project_root
    )
    input_fingerprints = {
        fragment.path: FileFingerprint(
            sha256=fragment.output_sha256,
            size_bytes=fragment.output_size_bytes,
        )
        for fragment in preprocessing.fragments
    }
    configuration = _configuration_snapshot(config)
    configuration_sha256 = _canonical_hash(configuration)
    tool_versions = _tool_versions()
    run_id = _make_run_id(
        preprocessing,
        configuration_sha256,
        tool_versions,
        preprocessing.fragments,
        input_fingerprints,
    )
    final_run_dir = config.paths.tokenized_dir / "runs" / run_id
    staging_dir = _create_staging_directory(config.paths.tokenized_dir)
    published_new_run = False

    try:
        tokenizer = build_tokenizer(config.tokenizer)
        staged_tokenizer_path = staging_dir / "tokenizer.json"
        save_tokenizer(
            tokenizer,
            staged_tokenizer_path,
            additional_attributes={
                "stage": 2,
                "tokenization_schema_version": TOKENIZATION_SCHEMA_VERSION,
                "tokenization_run_id": run_id,
                "configuration_sha256": configuration_sha256,
            },
        )
        restored = load_tokenizer(staged_tokenizer_path)
        if restored.vocab != tokenizer.vocab:
            raise TokenizationPipelineError(
                "Reloading tokenizer.json changed the REMI vocabulary."
            )
        if get_special_token_ids(restored) != get_special_token_ids(tokenizer):
            raise TokenizationPipelineError(
                "Reloading tokenizer.json changed the special-token IDs."
            )
        if get_technique_token_ids(restored) != get_technique_token_ids(tokenizer):
            raise TokenizationPipelineError(
                "Reloading tokenizer.json changed the technique-token IDs."
            )
        if get_tonic_token_ids(restored) != get_tonic_token_ids(tokenizer):
            raise TokenizationPipelineError(
                "Reloading tokenizer.json changed the tonic-token IDs."
            )
        if get_mode_token_ids(restored) != get_mode_token_ids(tokenizer):
            raise TokenizationPipelineError(
                "Reloading tokenizer.json changed the mode-token IDs."
            )
        # Encode with the reloaded artifact itself.  This makes the tokenizer
        # published for Stage 3 the exact implementation exercised here.
        tokenizer = restored

        sequence_records, failures = _tokenize_fragments(
            tokenizer,
            preprocessing.fragments,
            input_fingerprints,
            staging_dir,
            final_run_dir,
            config.project_root,
        )
        if failures:
            details = "\n".join(f"- {failure}" for failure in failures)
            raise TokenizationPipelineError(
                f"Stage 2 rejected {len(failures)} declared fragment(s):\n{details}"
            )

        _verify_unchanged_inputs(preprocessing, input_fingerprints)
        tokenizer_fingerprint = _fingerprint_file(staged_tokenizer_path)
        special_ids = get_special_token_ids(tokenizer)
        technique_ids = get_technique_token_ids(tokenizer)
        tonic_ids = get_tonic_token_ids(tokenizer)
        mode_ids = get_mode_token_ids(tokenizer)
        manifest = {
            "schema_version": TOKENIZATION_SCHEMA_VERSION,
            "tokenization_run_id": run_id,
            "tokenized_run_dir": relative_label(final_run_dir, config.project_root),
            "preprocessing": {
                "manifest_path": relative_label(
                    preprocessing.path, config.project_root
                ),
                "manifest_sha256": preprocessing.fingerprint.sha256,
                "schema_version": PREPROCESSING_SCHEMA_VERSION,
                "run_id": preprocessing.run_id,
                "configuration_sha256": preprocessing.configuration_sha256,
            },
            "configuration_sha256": configuration_sha256,
            "configuration": configuration,
            "tool_versions": tool_versions,
            "tokenizer": {
                "type": "ConditionedGuitarREMI",
                "path": relative_label(
                    final_run_dir / "tokenizer.json", config.project_root
                ),
                "sha256": tokenizer_fingerprint.sha256,
                "size_bytes": tokenizer_fingerprint.size_bytes,
                "vocabulary_sha256": _canonical_hash(tokenizer.vocab),
                "vocabulary_size": len(tokenizer.vocab),
                "special_token_ids": {
                    "pad": special_ids.pad,
                    "bos": special_ids.bos,
                    "eos": special_ids.eos,
                },
                "technique_token_ids": technique_ids,
                "conditioning_schema_version": CONDITIONING_SCHEMA_VERSION,
                "tonic_token_ids": tonic_ids,
                "mode_token_ids": mode_ids,
                "pitch_bend_sensitivity_semitones": (
                    PITCH_BEND_SENSITIVITY_SEMITONES
                ),
            },
            "summary": _summarize_sequences(sequence_records),
            "sequences": sequence_records,
        }
        write_json(staging_dir / "manifest.json", manifest)
        published_new_run = _publish_run(staging_dir, final_run_dir)
        try:
            write_json(config.paths.manifest_path, manifest)
        except BaseException:
            if published_new_run:
                _remove_tree(final_run_dir)
            raise
    except TokenizationError as exc:
        raise TokenizationPipelineError(str(exc)) from exc
    finally:
        _remove_tree(staging_dir)

    summary = manifest["summary"]
    return TokenizationReport(
        tokenization_run_id=run_id,
        tokenized_run_dir=final_run_dir,
        manifest_path=config.paths.manifest_path,
        num_sequences=summary["sequences"],
        vocabulary_size=len(tokenizer.vocab),
        min_tokens=summary["length"]["min_tokens"],
        max_tokens=summary["length"]["max_tokens"],
        reused_run=not published_new_run,
    )


def _load_preprocessing_manifest(path: Path, project_root: Path) -> PreprocessingManifest:
    manifest_path = path.expanduser().resolve()
    try:
        raw_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise TokenizationPipelineError(
            f"Could not read Stage 1 manifest '{manifest_path}': {exc}"
        ) from exc
    fingerprint = FileFingerprint(
        sha256=hashlib.sha256(raw_bytes).hexdigest(), size_bytes=len(raw_bytes)
    )
    try:
        payload = json.loads(
            raw_bytes,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise TokenizationPipelineError(
            f"Stage 1 manifest is not valid UTF-8 JSON: {exc}"
        ) from exc
    root = _require_mapping(payload, "Stage 1 manifest")
    allowed_root = {
        "schema_version",
        "run_id",
        "processed_run_dir",
        "configuration_sha256",
        "configuration",
        "tool_versions",
        "random_seed",
        "split_ratios",
        "summary",
        "sources",
        "fragments",
    }
    _reject_unknown_keys(root, allowed_root, "Stage 1 manifest")
    if (
        _require_int(root.get("schema_version"), "schema_version")
        != PREPROCESSING_SCHEMA_VERSION
    ):
        raise TokenizationPipelineError(
            "Stage 2 requires a Stage 1 manifest with schema_version "
            f"{PREPROCESSING_SCHEMA_VERSION}."
        )
    run_id = _require_string(root.get("run_id"), "run_id")
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise TokenizationPipelineError("Stage 1 run_id must be 20 lowercase hex digits.")
    configuration_sha256 = _require_sha256(
        root.get("configuration_sha256"), "configuration_sha256"
    )
    preprocessing_configuration = _require_mapping(
        root.get("configuration"), "configuration"
    )
    if _canonical_hash(preprocessing_configuration) != configuration_sha256:
        raise TokenizationPipelineError(
            "Stage 1 configuration_sha256 does not match configuration."
        )
    _require_mapping(root.get("tool_versions"), "tool_versions")
    _require_mapping(root.get("split_ratios"), "split_ratios")

    run_label = _require_string(root.get("processed_run_dir"), "processed_run_dir")
    run_candidate = _manifest_path_candidate(
        run_label, project_root, "processed_run_dir"
    )
    if run_candidate.is_symlink():
        raise TokenizationPipelineError("processed_run_dir cannot be a symlink.")
    processed_run_dir = _resolve_manifest_path(
        run_label, project_root, "processed_run_dir"
    )
    if not processed_run_dir.is_dir():
        raise TokenizationPipelineError(
            f"Declared processed_run_dir does not exist: {processed_run_dir}"
        )
    if processed_run_dir.name != run_id or processed_run_dir.parent.name != "runs":
        raise TokenizationPipelineError(
            "processed_run_dir must identify data/processed/runs/<run_id>."
        )

    source_by_name = _validate_sources(root.get("sources"))
    raw_fragments = root.get("fragments")
    if not isinstance(raw_fragments, list) or not raw_fragments:
        raise TokenizationPipelineError(
            "Stage 1 manifest 'fragments' must be a non-empty list."
        )
    fragments: list[FragmentSpec] = []
    seen_outputs: set[Path] = set()
    for index, raw_fragment in enumerate(raw_fragments):
        fragment = _validate_fragment(
            raw_fragment,
            index,
            source_by_name,
            processed_run_dir,
            project_root,
        )
        if fragment.path in seen_outputs:
            raise TokenizationPipelineError(
                f"Duplicate fragment output_file: {fragment.output_file}"
            )
        seen_outputs.add(fragment.path)
        fragments.append(fragment)

    summary = _require_mapping(root.get("summary"), "summary")
    generated = _require_int(
        summary.get("generated_fragments"), "summary.generated_fragments"
    )
    if generated != len(fragments):
        raise TokenizationPipelineError(
            "summary.generated_fragments does not match fragments length."
        )
    ordered = tuple(
        sorted(
            fragments,
            key=lambda item: (_SPLITS.index(item.split), item.output_file),
        )
    )
    return PreprocessingManifest(
        path=manifest_path,
        fingerprint=fingerprint,
        run_id=run_id,
        processed_run_dir=processed_run_dir,
        configuration_sha256=configuration_sha256,
        fragments=ordered,
    )


def _validate_sources(value: object) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise TokenizationPipelineError(
            "Stage 1 manifest 'sources' must be a non-empty list."
        )
    sources: dict[str, Mapping[str, Any]] = {}
    content_splits: dict[str, str] = {}
    for index, raw_source in enumerate(value):
        source = _require_mapping(raw_source, f"sources[{index}]")
        source_file = _require_string(
            source.get("source_file"), f"sources[{index}].source_file"
        )
        if source_file in sources:
            raise TokenizationPipelineError(
                f"Duplicate source_file in Stage 1 manifest: {source_file}"
            )
        compatible = _require_bool(
            source.get("compatible"), f"sources[{index}].compatible"
        )
        split = source.get("split")
        if compatible:
            split = _require_split(split, f"sources[{index}].split")
            source_sha256 = _require_sha256(
                source.get("source_sha256"), f"sources[{index}].source_sha256"
            )
            track_number = _require_nonnegative_int(
                source.get("track_number"), f"sources[{index}].track_number"
            )
            instrument_index = _require_nonnegative_int(
                source.get("instrument_index"),
                f"sources[{index}].instrument_index",
            )
            if track_number != instrument_index:
                raise TokenizationPipelineError(
                    f"sources[{index}] track_number and instrument_index must match."
                )
            previous_split = content_splits.setdefault(source_sha256, split)
            if previous_split != split:
                raise TokenizationPipelineError(
                    "Identical Stage 1 source content appears in multiple splits."
                )
        elif split is not None:
            raise TokenizationPipelineError(
                f"Incompatible source '{source_file}' must not have a split."
            )
        sources[source_file] = source
    return sources


def _validate_fragment(
    value: object,
    index: int,
    sources: Mapping[str, Mapping[str, Any]],
    processed_run_dir: Path,
    project_root: Path,
) -> FragmentSpec:
    name = f"fragments[{index}]"
    fragment = _require_mapping(value, name)
    allowed = {
        "source_file",
        "split",
        "track_number",
        "instrument_index",
        "phrase_index",
        "transpose_semitones",
        "output_file",
        "num_notes",
        "num_pitch_bend_events",
        "num_expressive_pitch_bend_events",
        "pitch_bend_range_semitones",
        "synthetic_initial_pitch_bend",
        "synthetic_final_pitch_bend_reset",
        "output_sha256",
        "output_size_bytes",
        "technique_coverage",
        "techniques",
        "nominal_duration_seconds",
        "actual_note_duration_seconds",
        "tonality",
    }
    _reject_unknown_keys(fragment, allowed, name)
    missing = sorted(allowed - set(fragment))
    if missing:
        raise TokenizationPipelineError(
            f"Missing field(s) in {name}: {', '.join(missing)}."
        )
    source_file = _require_string(fragment["source_file"], f"{name}.source_file")
    source = sources.get(source_file)
    if source is None or source.get("compatible") is not True:
        raise TokenizationPipelineError(
            f"{name} references an absent or incompatible source: {source_file}"
        )
    split = _require_split(fragment["split"], f"{name}.split")
    if source.get("split") != split:
        raise TokenizationPipelineError(
            f"{name} split does not match its original source split."
        )
    source_sha256 = _require_sha256(
        source.get("source_sha256"), f"source '{source_file}'.source_sha256"
    )
    track_number = _require_nonnegative_int(
        fragment["track_number"], f"{name}.track_number"
    )
    instrument_index = _require_nonnegative_int(
        fragment["instrument_index"], f"{name}.instrument_index"
    )
    if track_number != instrument_index:
        raise TokenizationPipelineError(
            f"{name} track_number and instrument_index must match."
        )
    if (
        source.get("track_number") != track_number
        or source.get("instrument_index") != instrument_index
    ):
        raise TokenizationPipelineError(
            f"{name} instrument index does not match its original source."
        )
    phrase_index = _require_nonnegative_int(
        fragment["phrase_index"], f"{name}.phrase_index"
    )
    transpose = _require_int(
        fragment["transpose_semitones"], f"{name}.transpose_semitones"
    )
    num_notes = _require_int(fragment["num_notes"], f"{name}.num_notes")
    if num_notes <= 0:
        raise TokenizationPipelineError(f"{name}.num_notes must be positive.")
    num_pitch_bend_events = _require_nonnegative_int(
        fragment["num_pitch_bend_events"], f"{name}.num_pitch_bend_events"
    )
    num_expressive_pitch_bend_events = _require_nonnegative_int(
        fragment["num_expressive_pitch_bend_events"],
        f"{name}.num_expressive_pitch_bend_events",
    )
    if num_expressive_pitch_bend_events > num_pitch_bend_events:
        raise TokenizationPipelineError(
            f"{name}.num_expressive_pitch_bend_events cannot exceed "
            "num_pitch_bend_events."
        )
    raw_bend_range = fragment["pitch_bend_range_semitones"]
    if raw_bend_range is None:
        pitch_bend_range_semitones = None
    else:
        pitch_bend_range_semitones = _require_positive_int(
            raw_bend_range, f"{name}.pitch_bend_range_semitones"
        )
    if num_expressive_pitch_bend_events:
        if pitch_bend_range_semitones != PITCH_BEND_SENSITIVITY_SEMITONES:
            raise TokenizationPipelineError(
                f"{name}.pitch_bend_range_semitones must be "
                f"{PITCH_BEND_SENSITIVITY_SEMITONES} when expressive pitch "
                "bends are present."
            )
    elif pitch_bend_range_semitones is not None:
        raise TokenizationPipelineError(
            f"{name}.pitch_bend_range_semitones must be null when no "
            "expressive pitch bends are present."
        )
    synthetic_initial_pitch_bend = _require_bool(
        fragment["synthetic_initial_pitch_bend"],
        f"{name}.synthetic_initial_pitch_bend",
    )
    synthetic_final_pitch_bend_reset = _require_bool(
        fragment["synthetic_final_pitch_bend_reset"],
        f"{name}.synthetic_final_pitch_bend_reset",
    )
    output_sha256 = _require_sha256(
        fragment["output_sha256"], f"{name}.output_sha256"
    )
    output_size_bytes = _require_positive_int(
        fragment["output_size_bytes"], f"{name}.output_size_bytes"
    )
    technique_coverage = _require_technique_coverage(
        fragment["technique_coverage"], f"{name}.technique_coverage"
    )
    techniques = _require_techniques(
        fragment["techniques"],
        f"{name}.techniques",
        num_notes=num_notes,
        coverage=technique_coverage,
    )
    nominal_duration = _require_positive_number(
        fragment["nominal_duration_seconds"], f"{name}.nominal_duration_seconds"
    )
    actual_duration = _require_positive_number(
        fragment["actual_note_duration_seconds"],
        f"{name}.actual_note_duration_seconds",
    )
    tonality = _require_tonality(fragment["tonality"], f"{name}.tonality")
    output_file = _require_string(fragment["output_file"], f"{name}.output_file")
    output_candidate = _manifest_path_candidate(
        output_file, project_root, f"{name}.output_file"
    )
    output_path = _resolve_manifest_path(
        output_file, project_root, f"{name}.output_file"
    )
    if not output_path.is_relative_to(processed_run_dir):
        raise TokenizationPipelineError(
            f"{name}.output_file escapes the declared processed_run_dir."
        )
    if output_path.parent != processed_run_dir / split:
        raise TokenizationPipelineError(
            f"{name}.output_file is not in the declared split directory."
        )
    if output_path.suffix.lower() not in {".mid", ".midi"}:
        raise TokenizationPipelineError(f"{name}.output_file must be a MIDI file.")
    if not output_candidate.is_relative_to(processed_run_dir):
        raise TokenizationPipelineError(
            f"{name}.output_file must lexically remain inside processed_run_dir."
        )
    path_components: list[Path] = []
    current = output_candidate
    while current != processed_run_dir:
        path_components.append(current)
        if current == current.parent:
            break
        current = current.parent
    if any(component.is_symlink() for component in path_components):
        raise TokenizationPipelineError(f"{name}.output_file cannot use symlinks.")
    if not output_path.is_file():
        raise TokenizationPipelineError(
            f"Declared fragment does not exist or is not a regular file: {output_path}"
        )
    actual_fingerprint = _fingerprint_file(output_path)
    declared_fingerprint = FileFingerprint(
        sha256=output_sha256, size_bytes=output_size_bytes
    )
    if actual_fingerprint != declared_fingerprint:
        raise TokenizationPipelineError(
            f"{name} output_sha256/output_size_bytes do not match the "
            "declared fragment file."
        )
    return FragmentSpec(
        source_file=source_file,
        source_sha256=source_sha256,
        split=split,
        track_number=track_number,
        instrument_index=instrument_index,
        phrase_index=phrase_index,
        transpose_semitones=transpose,
        output_file=output_file,
        path=output_path,
        num_notes=num_notes,
        num_pitch_bend_events=num_pitch_bend_events,
        num_expressive_pitch_bend_events=num_expressive_pitch_bend_events,
        pitch_bend_range_semitones=pitch_bend_range_semitones,
        synthetic_initial_pitch_bend=synthetic_initial_pitch_bend,
        synthetic_final_pitch_bend_reset=synthetic_final_pitch_bend_reset,
        output_sha256=output_sha256,
        output_size_bytes=output_size_bytes,
        technique_coverage=technique_coverage,
        techniques=techniques,
        nominal_duration_seconds=nominal_duration,
        actual_note_duration_seconds=actual_duration,
        tonality=tonality,
    )


def _tokenize_fragments(
    tokenizer: Any,
    fragments: Sequence[FragmentSpec],
    expected_fingerprints: Mapping[Path, FileFingerprint],
    staging_dir: Path,
    final_run_dir: Path,
    project_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    seen_sequence_ids: set[str] = set()
    for fragment in fragments:
        expected = expected_fingerprints[fragment.path]
        try:
            if _fingerprint_file(fragment.path) != expected:
                raise TokenizationPipelineError(
                    "input changed before it was tokenized"
                )
            encoded = encode_midi(
                tokenizer,
                fragment.path,
                techniques=fragment.techniques,
                tonic=fragment.tonality.tonic,
                mode=fragment.tonality.mode,
            )
            if _fingerprint_file(fragment.path) != expected:
                raise TokenizationPipelineError(
                    "input changed while it was tokenized"
                )
            if encoded.num_notes != fragment.num_notes:
                raise TokenizationPipelineError(
                    f"manifest declares {fragment.num_notes} notes but MIDI has "
                    f"{encoded.num_notes}"
                )
            sequence_id = _sequence_id(fragment, expected)
            if sequence_id in seen_sequence_ids:
                raise TokenizationPipelineError(
                    f"duplicate derived sequence_id '{sequence_id}'"
                )
            seen_sequence_ids.add(sequence_id)
            staged_sequence_path = staging_dir / fragment.split / f"{sequence_id}.json"
            final_sequence_path = final_run_dir / fragment.split / f"{sequence_id}.json"
            sequence_payload = _sequence_payload(sequence_id, encoded, fragment)
            write_json(staged_sequence_path, sequence_payload)
            output_fingerprint = _fingerprint_file(staged_sequence_path)
            records.append(
                {
                    "sequence_id": sequence_id,
                    "split": fragment.split,
                    "source_file": fragment.source_file,
                    "source_sha256": fragment.source_sha256,
                    "track_number": fragment.track_number,
                    "instrument_index": fragment.instrument_index,
                    "phrase_index": fragment.phrase_index,
                    "transpose_semitones": fragment.transpose_semitones,
                    "processed_midi": fragment.output_file,
                    "processed_midi_sha256": expected.sha256,
                    "processed_midi_size_bytes": expected.size_bytes,
                    "sequence_file": relative_label(
                        final_sequence_path, project_root
                    ),
                    "sequence_sha256": output_fingerprint.sha256,
                    "sequence_size_bytes": output_fingerprint.size_bytes,
                    "programs": [list(program) for program in encoded.programs],
                    "num_notes": encoded.num_notes,
                    "num_tokens": encoded.num_tokens,
                    "num_musical_tokens": encoded.num_musical_tokens,
                    "technique_coverage": fragment.technique_coverage,
                    "techniques": _techniques_payload(encoded.techniques),
                    "num_technique_tokens": len(encoded.techniques),
                    "num_pitch_bend_tokens": encoded.num_pitch_bends,
                    "nominal_duration_seconds": fragment.nominal_duration_seconds,
                    "actual_note_duration_seconds": fragment.actual_note_duration_seconds,
                    "token_error_ratio": encoded.token_error_ratio,
                    "round_trip_ok": encoded.round_trip_ok,
                    "tonality": fragment.tonality.as_dict(),
                }
            )
        except (OSError, TokenizationError, TokenizationPipelineError) as exc:
            failures.append(f"{fragment.output_file}: {exc}")
    return records, failures


def _sequence_payload(
    sequence_id: str, encoded: EncodedMidi, fragment: FragmentSpec
) -> dict[str, Any]:
    return {
        "schema_version": SEQUENCE_SCHEMA_VERSION,
        "sequence_id": sequence_id,
        "ids": list(encoded.ids),
        "programs": [list(program) for program in encoded.programs],
        "technique_coverage": fragment.technique_coverage,
        "techniques": _techniques_payload(encoded.techniques),
        "tonality": fragment.tonality.as_dict(),
    }


def _sequence_id(fragment: FragmentSpec, fingerprint: FileFingerprint) -> str:
    stem = Path(fragment.output_file).stem
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-_").lower()
    slug = slug[:64] or "sequence"
    digest = _canonical_hash(
        {
            "output_file": fragment.output_file,
            "sha256": fingerprint.sha256,
            "technique_coverage": fragment.technique_coverage,
            "techniques": _techniques_payload(fragment.techniques),
            "tonality": fragment.tonality.as_dict(),
        }
    )[:12]
    return f"{slug}-{digest}"


def _techniques_payload(
    techniques: Sequence[TechniqueAnnotation],
) -> list[dict[str, Any]]:
    return [
        {"type": technique.type, "note_index": technique.note_index}
        for technique in techniques
    ]


def _configuration_snapshot(config: TokenizationConfig) -> dict[str, Any]:
    return {"tokenizer": asdict(config.tokenizer)}


def _tool_versions() -> dict[str, str]:
    versions = {
        "implementation_sha256": _implementation_sha256(),
        "midi_idea_generator": __version__,
        "python": platform.python_version(),
    }
    for distribution in ("miditok", "symusic", "numpy"):
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _implementation_sha256() -> str:
    package_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for filename in (
        "tonality.py",
        "tokenization_config.py",
        "tokenizer.py",
        "tokenization_pipeline.py",
        "utils.py",
    ):
        path = package_dir / filename
        digest.update(filename.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _make_run_id(
    manifest: PreprocessingManifest,
    configuration_sha256: str,
    tool_versions: Mapping[str, str],
    fragments: Sequence[FragmentSpec],
    fingerprints: Mapping[Path, FileFingerprint],
) -> str:
    return _canonical_hash(
        {
            "schema_version": TOKENIZATION_SCHEMA_VERSION,
            "preprocessing_manifest_sha256": manifest.fingerprint.sha256,
            "preprocessing_run_id": manifest.run_id,
            "configuration_sha256": configuration_sha256,
            "tool_versions": tool_versions,
            "fragments": [
                {
                    "output_file": fragment.output_file,
                    "sha256": fingerprints[fragment.path].sha256,
                }
                for fragment in fragments
            ],
        }
    )[:20]


def _summarize_sequences(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise TokenizationPipelineError("Stage 2 produced no token sequences.")

    def group_summary(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        lengths = [int(record["num_tokens"]) for record in group]
        musical = [int(record["num_musical_tokens"]) for record in group]
        return {
            "sequences": len(group),
            "total_tokens": sum(lengths),
            "total_musical_tokens": sum(musical),
            "min_tokens": min(lengths) if lengths else None,
            "max_tokens": max(lengths) if lengths else None,
            "mean_tokens": (sum(lengths) / len(lengths)) if lengths else None,
        }

    all_summary = group_summary(records)
    technique_counts = {technique_type: 0 for technique_type in TECHNIQUE_TYPES}
    for record in records:
        for technique in record["techniques"]:
            technique_counts[str(technique["type"])] += 1
    complete_sequences = sum(
        record["technique_coverage"] == "COMPLETE" for record in records
    )
    unlabeled_sequences = sum(
        record["technique_coverage"] == "UNLABELED" for record in records
    )
    total_pitch_bends = sum(
        int(record["num_pitch_bend_tokens"]) for record in records
    )
    return {
        "sequences": all_summary["sequences"],
        "total_tokens": all_summary["total_tokens"],
        "total_musical_tokens": all_summary["total_musical_tokens"],
        "length": {
            "min_tokens": all_summary["min_tokens"],
            "max_tokens": all_summary["max_tokens"],
            "mean_tokens": all_summary["mean_tokens"],
        },
        "by_split": {
            split: group_summary(
                [record for record in records if record["split"] == split]
            )
            for split in _SPLITS
        },
        "techniques": {
            "total_tokens": sum(technique_counts.values()),
            "by_type": technique_counts,
            "coverage": {
                "complete_sequences": complete_sequences,
                "unlabeled_sequences": unlabeled_sequences,
            },
        },
        "pitch_bends": {
            "total_tokens": total_pitch_bends,
            "sequences_with_pitch_bends": sum(
                int(record["num_pitch_bend_tokens"]) > 0 for record in records
            ),
        },
        "tonality": {
            "by_tonic": {
                tonic: sum(
                    record["tonality"]["tonic"] == tonic for record in records
                )
                for tonic in TONIC_NAMES
            },
            "by_mode": {
                mode: sum(
                    record["tonality"]["mode"] == mode for record in records
                )
                for mode in MODE_NAMES
            },
            "by_method": {
                method: sum(
                    record["tonality"]["method"] == method for record in records
                )
                for method in TONALITY_METHODS
            },
        },
    }


def _verify_unchanged_inputs(
    manifest: PreprocessingManifest,
    expected: Mapping[Path, FileFingerprint],
) -> None:
    current_manifest = _fingerprint_file(manifest.path)
    if current_manifest != manifest.fingerprint:
        raise TokenizationPipelineError(
            "Stage 1 manifest changed while Stage 2 was running."
        )
    changed = [
        path
        for path, fingerprint in expected.items()
        if _fingerprint_file(path) != fingerprint
    ]
    if changed:
        labels = ", ".join(path.name for path in changed[:5])
        raise TokenizationPipelineError(
            f"Processed MIDI changed before publication: {labels}."
        )


def _fingerprint_file(path: Path) -> FileFingerprint:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise TokenizationPipelineError(f"Could not fingerprint '{path}': {exc}") from exc
    return FileFingerprint(sha256=digest.hexdigest(), size_bytes=size)


def _canonical_hash(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _create_staging_directory(tokenized_dir: Path) -> Path:
    if tokenized_dir.exists() and not tokenized_dir.is_dir():
        raise TokenizationPipelineError(
            f"Tokenized output path exists but is not a directory: {tokenized_dir}"
        )
    tokenized_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=".stage-", dir=tokenized_dir))


def _publish_run(staging_dir: Path, run_dir: Path) -> bool:
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        if not run_dir.is_dir() or _tree_digest(run_dir) != _tree_digest(staging_dir):
            raise TokenizationPipelineError(
                "A tokenized run with this reproducibility fingerprint exists "
                f"but has different contents: {run_dir}"
            )
        return False
    staging_dir.rename(run_dir)
    return True


def _tree_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _resolve_manifest_path(value: str, project_root: Path, name: str) -> Path:
    candidate = _manifest_path_candidate(value, project_root, name)
    try:
        return candidate.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise TokenizationPipelineError(f"Invalid path in {name}: {exc}") from exc


def _manifest_path_candidate(value: str, project_root: Path, name: str) -> Path:
    """Return an absolute lexical path without erasing symlink information."""

    raw_path = Path(value).expanduser()
    if ".." in raw_path.parts:
        raise TokenizationPipelineError(f"{name} cannot contain '..' traversal.")
    try:
        return raw_path if raw_path.is_absolute() else project_root / raw_path
    except (OSError, RuntimeError, ValueError) as exc:
        raise TokenizationPipelineError(f"Invalid path in {name}: {exc}") from exc


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TokenizationPipelineError(f"{name} must be a JSON object.")
    return value


def _reject_json_constant(value: str) -> Never:
    raise ValueError(f"non-standard numeric constant {value}")


def _reject_unknown_keys(
    value: Mapping[str, Any], allowed: set[str], name: str
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise TokenizationPipelineError(f"All keys in {name} must be strings.")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise TokenizationPipelineError(
            f"Unknown field(s) in {name}: {', '.join(unknown)}."
        )


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise TokenizationPipelineError(f"{name} must be a non-empty string.")
    return value


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TokenizationPipelineError(f"{name} must be a boolean.")
    return value


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TokenizationPipelineError(f"{name} must be an integer.")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    converted = _require_int(value, name)
    if converted < 0:
        raise TokenizationPipelineError(f"{name} cannot be negative.")
    return converted


def _require_positive_int(value: object, name: str) -> int:
    converted = _require_int(value, name)
    if converted <= 0:
        raise TokenizationPipelineError(f"{name} must be positive.")
    return converted


def _require_positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TokenizationPipelineError(f"{name} must be a number.")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise TokenizationPipelineError(f"{name} must be finite and positive.")
    return converted


def _require_split(value: object, name: str) -> str:
    split = _require_string(value, name)
    if split not in _SPLITS:
        raise TokenizationPipelineError(
            f"{name} must be train, validation, or test."
        )
    return split


def _require_tonality(value: object, name: str) -> Tonality:
    mapping = _require_mapping(value, name)
    required = {
        "tonic",
        "mode",
        "method",
        "tonic_confidence",
        "mode_confidence",
    }
    _reject_unknown_keys(mapping, required, name)
    missing = sorted(required - set(mapping))
    if missing:
        raise TokenizationPipelineError(
            f"Missing field(s) in {name}: {', '.join(missing)}."
        )
    try:
        tonality = Tonality(
            tonic=mapping["tonic"],
            mode=mapping["mode"],
            method=mapping["method"],
            tonic_confidence=mapping["tonic_confidence"],
            mode_confidence=mapping["mode_confidence"],
        )
    except (TonalityError, TypeError, ValueError) as exc:
        raise TokenizationPipelineError(f"Invalid {name}: {exc}") from exc
    if tonality.as_dict() != dict(mapping):
        raise TokenizationPipelineError(
            f"{name} must already use canonical tonality values."
        )
    return tonality


def _require_technique_coverage(value: object, name: str) -> str:
    coverage = _require_string(value, name)
    if coverage not in _TECHNIQUE_COVERAGE:
        raise TokenizationPipelineError(
            f"{name} must be UNLABELED or COMPLETE."
        )
    return coverage


def _require_techniques(
    value: object,
    name: str,
    *,
    num_notes: int,
    coverage: str,
) -> tuple[TechniqueAnnotation, ...]:
    if not isinstance(value, list):
        raise TokenizationPipelineError(f"{name} must be a JSON list.")

    techniques: list[TechniqueAnnotation] = []
    identities: set[tuple[int, str]] = set()
    for index, raw_technique in enumerate(value):
        item_name = f"{name}[{index}]"
        technique = _require_mapping(raw_technique, item_name)
        allowed = {"type", "note_index"}
        _reject_unknown_keys(technique, allowed, item_name)
        missing = sorted(allowed - set(technique))
        if missing:
            raise TokenizationPipelineError(
                f"Missing field(s) in {item_name}: {', '.join(missing)}."
            )
        technique_type = _require_string(
            technique["type"], f"{item_name}.type"
        )
        if technique_type not in _TECHNIQUE_ORDER:
            raise TokenizationPipelineError(
                f"{item_name}.type must be one of {', '.join(TECHNIQUE_TYPES)}."
            )
        note_index = _require_nonnegative_int(
            technique["note_index"], f"{item_name}.note_index"
        )
        if note_index >= num_notes:
            raise TokenizationPipelineError(
                f"{item_name}.note_index {note_index} is outside the "
                f"{num_notes}-note fragment."
            )
        identity = (note_index, technique_type)
        if identity in identities:
            raise TokenizationPipelineError(
                f"Duplicate technique {technique_type} for note {note_index} "
                f"in {name}."
            )
        identities.add(identity)
        techniques.append(
            TechniqueAnnotation(type=technique_type, note_index=note_index)
        )

    canonical = tuple(
        sorted(
            techniques,
            key=lambda item: (item.note_index, _TECHNIQUE_ORDER[item.type]),
        )
    )
    if tuple(techniques) != canonical:
        raise TokenizationPipelineError(
            f"{name} must use canonical note-index and technique-type order."
        )
    if coverage == "UNLABELED" and canonical:
        raise TokenizationPipelineError(
            f"{name} must be empty when technique_coverage is UNLABELED."
        )
    return canonical


def _require_sha256(value: object, name: str) -> str:
    sha256 = _require_string(value, name)
    if not _SHA256_PATTERN.fullmatch(sha256):
        raise TokenizationPipelineError(
            f"{name} must be 64 lowercase hexadecimal characters."
        )
    return sha256
