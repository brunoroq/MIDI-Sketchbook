"""Strict Stage 3 dataset boundary for immutable Stage 2 token sequences.

The authoritative Stage 2 manifest is the only source of sequence paths.  This
module deliberately never scans token directories: every tokenizer and
sequence artifact is path-checked, fingerprinted, and validated before a
dataset can be used for training.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path, PurePosixPath
import re
from typing import Any


TOKENIZATION_MANIFEST_SCHEMA_VERSION = 2
TOKEN_SEQUENCE_SCHEMA_VERSION = 2
PREPROCESSING_MANIFEST_SCHEMA_VERSION = 3
DEFAULT_MAX_SEQUENCE_LENGTH = 512
SPLITS = ("train", "validation", "test")
TECHNIQUE_TYPES = (
    "DEAD_NOTE",
    "PALM_MUTE_ON",
    "PALM_MUTE_OFF",
    "SLIDE_UP",
    "SLIDE_DOWN",
    "VIBRATO",
)
TECHNIQUE_COVERAGE = ("UNLABELED", "COMPLETE")
PITCH_BEND_SENSITIVITY_SEMITONES = 6

_TECHNIQUE_ORDER = {
    technique_type: index
    for index, technique_type in enumerate(TECHNIQUE_TYPES)
}

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{20}$")
_SEQUENCE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")


class DatasetContractError(ValueError):
    """Raised when Stage 2 artifacts cannot be consumed safely for training."""


@dataclass(frozen=True, slots=True)
class _FilePayload:
    path: Path
    raw: bytes
    sha256: str
    size_bytes: int
    json: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _Sequence:
    sequence_id: str
    split: str
    ids: tuple[int, ...]
    technique_coverage: str


@dataclass(frozen=True, slots=True)
class _Corpus:
    manifest_path: Path
    manifest_sha256: str
    tokenization_run_id: str
    configuration_sha256: str
    tokenizer_path: Path
    tokenizer_sha256: str
    vocabulary_size: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    duration_token_ids: frozenset[int]
    sequences: tuple[_Sequence, ...]


class TokenizedSequenceDataset:
    """Map-style next-token dataset for one immutable Stage 2 split.

    ``max_sequence_length`` applies to the complete stored sequence, including
    BOS and EOS.  Overlong sequences raise :class:`DatasetContractError`;
    nothing is silently truncated.  Items are plain Python values so this
    class remains inspectable without importing PyTorch.  Use
    :func:`make_collate_fn` with ``torch.utils.data.DataLoader`` to create
    dynamically padded tensors.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        *,
        project_root: str | Path | None = None,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        verify_hashes: bool = True,
    ) -> None:
        if split not in SPLITS:
            raise DatasetContractError(
                f"split must be one of {', '.join(SPLITS)}; got {split!r}."
            )
        if (
            isinstance(max_sequence_length, bool)
            or not isinstance(max_sequence_length, int)
            or max_sequence_length < 3
        ):
            raise DatasetContractError(
                "max_sequence_length must be an integer of at least 3."
            )
        if not isinstance(verify_hashes, bool):
            raise DatasetContractError("verify_hashes must be true or false.")

        corpus = _load_corpus(
            manifest_path,
            project_root=project_root,
            verify_hashes=verify_hashes,
        )
        selected = tuple(sequence for sequence in corpus.sequences if sequence.split == split)
        overlong = [
            sequence
            for sequence in selected
            if len(sequence.ids) > max_sequence_length
        ]
        if overlong:
            first = overlong[0]
            raise DatasetContractError(
                f"Split '{split}' contains {len(overlong)} sequence(s) longer than "
                f"max_sequence_length={max_sequence_length}; '{first.sequence_id}' "
                f"has {len(first.ids)} tokens. Increase the configured limit or "
                "implement an explicit windowing policy."
            )

        self._corpus = corpus
        self._sequences = selected
        self._split = split
        self._max_sequence_length = max_sequence_length

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, index: int) -> dict[str, object]:
        sequence = self._sequences[index]
        input_ids = sequence.ids[:-1]
        target_ids = sequence.ids[1:]
        if sequence.technique_coverage == "COMPLETE":
            loss_mask = (True,) * len(input_ids)
        else:
            loss_mask = tuple(
                token_id not in self._corpus.duration_token_ids
                for token_id in input_ids
            )
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
            "length": len(input_ids),
            "sequence_id": sequence.sequence_id,
            "split": sequence.split,
        }

    @property
    def split(self) -> str:
        return self._split

    @property
    def max_sequence_length(self) -> int:
        return self._max_sequence_length

    @property
    def manifest_path(self) -> Path:
        return self._corpus.manifest_path

    @property
    def tokenization_manifest_sha256(self) -> str:
        return self._corpus.manifest_sha256

    @property
    def tokenization_run_id(self) -> str:
        return self._corpus.tokenization_run_id

    @property
    def configuration_sha256(self) -> str:
        return self._corpus.configuration_sha256

    @property
    def tokenizer_path(self) -> Path:
        return self._corpus.tokenizer_path

    @property
    def tokenizer_sha256(self) -> str:
        return self._corpus.tokenizer_sha256

    @property
    def vocabulary_size(self) -> int:
        return self._corpus.vocabulary_size

    @property
    def pad_token_id(self) -> int:
        return self._corpus.pad_token_id

    @property
    def bos_token_id(self) -> int:
        return self._corpus.bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._corpus.eos_token_id


@dataclass(frozen=True, slots=True)
class TokenBatchCollator:
    """Pickle-safe dynamic-padding callable for a PyTorch DataLoader."""

    pad_token_id: int

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.pad_token_id, "pad_token_id")

    def __call__(self, samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
        return collate_token_sequences(samples, self.pad_token_id)


def make_collate_fn(
    pad_token_id: int,
) -> Callable[[Sequence[Mapping[str, object]]], dict[str, object]]:
    """Return a worker-safe collator configured with the manifest PAD id."""

    return TokenBatchCollator(pad_token_id)


def collate_token_sequences(
    samples: Sequence[Mapping[str, object]], pad_token_id: int
) -> dict[str, object]:
    """Dynamically pad one batch and return next-token training tensors.

    Both inputs and targets use PAD in padded positions.  Targets also use PAD
    at real positions explicitly excluded by each sample's ``loss_mask``.  The
    trainer must pass the same id as ``ignore_index`` to cross entropy.  The
    boolean ``attention_mask`` and integer ``lengths`` continue to identify all
    real positions, independently of whether they contribute to loss.
    """

    pad_id = _require_nonnegative_int(pad_token_id, "pad_token_id")
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise DatasetContractError("samples must be a sequence of dataset items.")
    if not samples:
        raise DatasetContractError("Cannot collate an empty batch.")

    normalized: list[
        tuple[tuple[int, ...], tuple[int, ...], tuple[bool, ...], int, str, str]
    ] = []
    for index, sample in enumerate(samples):
        name = f"samples[{index}]"
        if not isinstance(sample, Mapping):
            raise DatasetContractError(f"{name} must be a mapping.")
        required = {
            "input_ids",
            "target_ids",
            "loss_mask",
            "length",
            "sequence_id",
            "split",
        }
        missing = sorted(required - set(sample))
        if missing:
            raise DatasetContractError(
                f"{name} is missing field(s): {', '.join(missing)}."
            )
        input_ids = _normalize_ids(sample["input_ids"], f"{name}.input_ids")
        target_ids = _normalize_ids(sample["target_ids"], f"{name}.target_ids")
        loss_mask = _normalize_loss_mask(sample["loss_mask"], f"{name}.loss_mask")
        length = _require_positive_int(sample["length"], f"{name}.length")
        sequence_id = _require_string(sample["sequence_id"], f"{name}.sequence_id")
        split = _require_split(sample["split"], f"{name}.split")
        if (
            len(input_ids) != len(target_ids)
            or len(input_ids) != len(loss_mask)
            or len(input_ids) != length
        ):
            raise DatasetContractError(
                f"{name} input, target, loss-mask, and declared lengths must match."
            )
        if pad_id in input_ids or pad_id in target_ids:
            raise DatasetContractError(
                f"{name} contains PAD before batch collation."
            )
        normalized.append(
            (input_ids, target_ids, loss_mask, length, sequence_id, split)
        )

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard.
        raise DatasetContractError(
            "PyTorch is required to collate training batches; install the "
            "project dependencies first."
        ) from exc

    batch_size = len(normalized)
    width = max(item[3] for item in normalized)
    inputs = torch.full((batch_size, width), pad_id, dtype=torch.long)
    targets = torch.full((batch_size, width), pad_id, dtype=torch.long)
    loss_mask = torch.zeros((batch_size, width), dtype=torch.bool)
    lengths = torch.tensor([item[3] for item in normalized], dtype=torch.long)
    for row, (input_ids, target_ids, sample_loss_mask, length, _, _) in enumerate(
        normalized
    ):
        inputs[row, :length] = torch.tensor(input_ids, dtype=torch.long)
        targets[row, :length] = torch.tensor(target_ids, dtype=torch.long)
        loss_mask[row, :length] = torch.tensor(sample_loss_mask, dtype=torch.bool)
        targets[row, :length].masked_fill_(~loss_mask[row, :length], pad_id)
    attention_mask = (
        torch.arange(width, dtype=torch.long).unsqueeze(0) < lengths.unsqueeze(1)
    )
    return {
        "input_ids": inputs,
        "target_ids": targets,
        "loss_mask": loss_mask,
        "lengths": lengths,
        "attention_mask": attention_mask,
        "sequence_ids": [item[4] for item in normalized],
        "splits": [item[5] for item in normalized],
    }


def _load_corpus(
    manifest_path: str | Path,
    *,
    project_root: str | Path | None,
    verify_hashes: bool,
) -> _Corpus:
    unresolved_manifest = Path(manifest_path).expanduser()
    if unresolved_manifest.is_symlink():
        raise DatasetContractError("The authoritative Stage 2 manifest cannot be a symlink.")
    resolved_manifest = unresolved_manifest.resolve()
    root = _resolve_project_root(resolved_manifest, project_root)
    if not resolved_manifest.is_relative_to(root):
        raise DatasetContractError("Stage 2 manifest must be inside project_root.")
    manifest = _read_json_file(resolved_manifest, "Stage 2 manifest")
    payload = manifest.json
    required_root = {
        "schema_version",
        "tokenization_run_id",
        "tokenized_run_dir",
        "preprocessing",
        "configuration_sha256",
        "configuration",
        "tool_versions",
        "tokenizer",
        "summary",
        "sequences",
    }
    _require_exact_keys(payload, required_root, "Stage 2 manifest")
    if (
        _require_int(payload["schema_version"], "schema_version")
        != TOKENIZATION_MANIFEST_SCHEMA_VERSION
    ):
        raise DatasetContractError(
            "Stage 3 requires a Stage 2 manifest with schema_version "
            f"{TOKENIZATION_MANIFEST_SCHEMA_VERSION}."
        )
    run_id = _require_string(payload["tokenization_run_id"], "tokenization_run_id")
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise DatasetContractError(
            "tokenization_run_id must be 20 lowercase hexadecimal digits."
        )
    configuration = _require_mapping(payload["configuration"], "configuration")
    configuration_sha256 = _require_sha256(
        payload["configuration_sha256"], "configuration_sha256"
    )
    if _canonical_hash(configuration) != configuration_sha256:
        raise DatasetContractError(
            "configuration_sha256 does not match the embedded configuration."
        )
    _validate_preprocessing(payload["preprocessing"], root)
    _validate_tool_versions(payload["tool_versions"])

    run_dir = _resolve_declared_path(
        payload["tokenized_run_dir"], root, "tokenized_run_dir"
    )
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise DatasetContractError(
            f"Declared tokenized_run_dir is not a regular directory: {run_dir}"
        )
    expected_runs_dir = resolved_manifest.parent / "runs"
    if run_dir.parent != expected_runs_dir or run_dir.name != run_id:
        raise DatasetContractError(
            "tokenized_run_dir must be the authoritative manifest's "
            "runs/<tokenization_run_id> directory."
        )

    tokenizer_info = _validate_tokenizer(
        payload["tokenizer"],
        root=root,
        run_dir=run_dir,
        run_id=run_id,
        configuration_sha256=configuration_sha256,
        verify_hashes=verify_hashes,
    )
    raw_sequences = payload["sequences"]
    if not isinstance(raw_sequences, list) or not raw_sequences:
        raise DatasetContractError("Stage 2 manifest 'sequences' must be non-empty.")
    sequences, records = _validate_sequences(
        raw_sequences,
        root=root,
        run_dir=run_dir,
        preprocessing=payload["preprocessing"],
        tokenizer_info=tokenizer_info,
        verify_hashes=verify_hashes,
    )
    _validate_summary(payload["summary"], records)
    if resolved_manifest.read_bytes() != manifest.raw:
        raise DatasetContractError("Stage 2 manifest changed while it was loaded.")

    return _Corpus(
        manifest_path=resolved_manifest,
        manifest_sha256=manifest.sha256,
        tokenization_run_id=run_id,
        configuration_sha256=configuration_sha256,
        tokenizer_path=tokenizer_info["path"],
        tokenizer_sha256=tokenizer_info["sha256"],
        vocabulary_size=tokenizer_info["vocabulary_size"],
        pad_token_id=tokenizer_info["pad"],
        bos_token_id=tokenizer_info["bos"],
        eos_token_id=tokenizer_info["eos"],
        duration_token_ids=tokenizer_info["duration_token_ids"],
        sequences=sequences,
    )


def _validate_preprocessing(value: object, root: Path) -> Mapping[str, Any]:
    preprocessing = _require_mapping(value, "preprocessing")
    required = {
        "manifest_path",
        "manifest_sha256",
        "schema_version",
        "run_id",
        "configuration_sha256",
    }
    _require_exact_keys(preprocessing, required, "preprocessing")
    _declared_relative_path(preprocessing["manifest_path"], "preprocessing.manifest_path")
    _resolve_declared_path(preprocessing["manifest_path"], root, "preprocessing.manifest_path")
    _require_sha256(preprocessing["manifest_sha256"], "preprocessing.manifest_sha256")
    if (
        _require_int(
            preprocessing["schema_version"], "preprocessing.schema_version"
        )
        != PREPROCESSING_MANIFEST_SCHEMA_VERSION
    ):
        raise DatasetContractError(
            "preprocessing.schema_version must be "
            f"{PREPROCESSING_MANIFEST_SCHEMA_VERSION}."
        )
    preprocessing_run_id = _require_string(preprocessing["run_id"], "preprocessing.run_id")
    if not _RUN_ID_PATTERN.fullmatch(preprocessing_run_id):
        raise DatasetContractError(
            "preprocessing.run_id must be 20 lowercase hexadecimal digits."
        )
    _require_sha256(
        preprocessing["configuration_sha256"],
        "preprocessing.configuration_sha256",
    )
    return preprocessing


def _validate_tool_versions(value: object) -> None:
    versions = _require_mapping(value, "tool_versions")
    if not versions:
        raise DatasetContractError("tool_versions must not be empty.")
    for key, version in versions.items():
        if not isinstance(key, str) or not key:
            raise DatasetContractError("tool_versions keys must be non-empty strings.")
        _require_string(version, f"tool_versions.{key}")
    implementation = versions.get("implementation_sha256")
    if implementation is not None:
        _require_sha256(implementation, "tool_versions.implementation_sha256")


def _validate_tokenizer(
    value: object,
    *,
    root: Path,
    run_dir: Path,
    run_id: str,
    configuration_sha256: str,
    verify_hashes: bool,
) -> dict[str, Any]:
    tokenizer = _require_mapping(value, "tokenizer")
    required = {
        "type",
        "path",
        "sha256",
        "size_bytes",
        "vocabulary_sha256",
        "vocabulary_size",
        "special_token_ids",
        "technique_token_ids",
        "pitch_bend_sensitivity_semitones",
    }
    _require_exact_keys(tokenizer, required, "tokenizer")
    if tokenizer["type"] != "GuitarREMI":
        raise DatasetContractError(
            "Stage 3 requires tokenizer.type 'GuitarREMI'."
        )
    tokenizer_path = _resolve_declared_path(tokenizer["path"], root, "tokenizer.path")
    if tokenizer_path != run_dir / "tokenizer.json":
        raise DatasetContractError("tokenizer.path must be <tokenized_run_dir>/tokenizer.json.")
    _reject_symlink_path(tokenizer_path, root, "tokenizer.path")
    if not tokenizer_path.is_file():
        raise DatasetContractError(f"Declared tokenizer does not exist: {tokenizer_path}")
    declared_sha256 = _require_sha256(tokenizer["sha256"], "tokenizer.sha256")
    declared_size = _require_positive_int(tokenizer["size_bytes"], "tokenizer.size_bytes")
    vocabulary_sha256 = _require_sha256(
        tokenizer["vocabulary_sha256"], "tokenizer.vocabulary_sha256"
    )
    vocabulary_size = _require_positive_int(
        tokenizer["vocabulary_size"], "tokenizer.vocabulary_size"
    )
    special = _require_mapping(tokenizer["special_token_ids"], "tokenizer.special_token_ids")
    _require_exact_keys(special, {"pad", "bos", "eos"}, "tokenizer.special_token_ids")
    pad = _require_nonnegative_int(special["pad"], "tokenizer.special_token_ids.pad")
    bos = _require_nonnegative_int(special["bos"], "tokenizer.special_token_ids.bos")
    eos = _require_nonnegative_int(special["eos"], "tokenizer.special_token_ids.eos")
    if len({pad, bos, eos}) != 3:
        raise DatasetContractError("PAD, BOS, and EOS token IDs must be distinct.")
    technique_token_ids = _normalize_technique_token_ids(
        tokenizer["technique_token_ids"], "tokenizer.technique_token_ids"
    )
    if set(technique_token_ids.values()).intersection({pad, bos, eos}):
        raise DatasetContractError(
            "Technique token IDs must be distinct from PAD, BOS, and EOS."
        )
    pitch_bend_sensitivity = _require_positive_int(
        tokenizer["pitch_bend_sensitivity_semitones"],
        "tokenizer.pitch_bend_sensitivity_semitones",
    )
    if pitch_bend_sensitivity != PITCH_BEND_SENSITIVITY_SEMITONES:
        raise DatasetContractError(
            "tokenizer.pitch_bend_sensitivity_semitones must be "
            f"{PITCH_BEND_SENSITIVITY_SEMITONES}."
        )

    tokenizer_file = _read_json_file(tokenizer_path, "tokenizer")
    if verify_hashes and (
        tokenizer_file.sha256 != declared_sha256
        or tokenizer_file.size_bytes != declared_size
    ):
        raise DatasetContractError("Tokenizer hash or size does not match the manifest.")
    embedded = tokenizer_file.json
    if embedded.get("tokenization") != "GuitarREMI":
        raise DatasetContractError(
            "tokenizer.json does not describe a GuitarREMI tokenizer."
        )
    if (
        embedded.get("stage") != 2
        or embedded.get("tokenization_schema_version")
        != TOKENIZATION_MANIFEST_SCHEMA_VERSION
    ):
        raise DatasetContractError(
            "tokenizer.json is not a Stage 2 schema "
            f"{TOKENIZATION_MANIFEST_SCHEMA_VERSION} artifact."
        )
    if embedded.get("tokenization_run_id") != run_id:
        raise DatasetContractError("tokenizer.json run ID does not match the manifest.")
    if embedded.get("configuration_sha256") != configuration_sha256:
        raise DatasetContractError(
            "tokenizer.json configuration hash does not match the manifest."
        )

    try:
        from .tokenizer import (
            get_special_token_ids,
            get_technique_token_ids,
            load_tokenizer,
        )

        restored = load_tokenizer(tokenizer_path)
        actual_special = get_special_token_ids(restored)
        actual_technique_ids = get_technique_token_ids(restored)
        actual_vocabulary = restored.vocab
    except Exception as exc:
        raise DatasetContractError(f"Could not validate tokenizer.json: {exc}") from exc
    if not isinstance(actual_vocabulary, Mapping):
        raise DatasetContractError("Tokenizer must contain exactly one vocabulary.")
    if len(actual_vocabulary) != vocabulary_size:
        raise DatasetContractError("Tokenizer vocabulary size does not match the manifest.")
    if _canonical_hash(actual_vocabulary) != vocabulary_sha256:
        raise DatasetContractError("Tokenizer vocabulary hash does not match the manifest.")
    if (actual_special.pad, actual_special.bos, actual_special.eos) != (pad, bos, eos):
        raise DatasetContractError("Tokenizer special-token IDs do not match the manifest.")
    if actual_technique_ids != technique_token_ids:
        raise DatasetContractError(
            "Tokenizer technique-token IDs do not match the manifest."
        )
    actual_pitch_bend_sensitivity = getattr(
        restored, "pitch_bend_sensitivity_semitones", None
    )
    if actual_pitch_bend_sensitivity != pitch_bend_sensitivity:
        raise DatasetContractError(
            "Tokenizer pitch-bend sensitivity does not match the manifest."
        )
    try:
        tokenizer_unchanged = tokenizer_path.read_bytes() == tokenizer_file.raw
    except OSError as exc:
        raise DatasetContractError(
            f"Could not recheck tokenizer '{tokenizer_path}': {exc}"
        ) from exc
    if not tokenizer_unchanged:
        raise DatasetContractError("Tokenizer changed while it was being validated.")
    vocabulary_by_id: dict[int, str] = {}
    for token, token_id in actual_vocabulary.items():
        if not isinstance(token, str) or not token:
            raise DatasetContractError(
                "Tokenizer vocabulary keys must be non-empty strings."
            )
        normalized_id = _require_nonnegative_int(
            token_id, f"tokenizer vocabulary token {token!r}"
        )
        if normalized_id in vocabulary_by_id:
            raise DatasetContractError("Tokenizer vocabulary IDs must be unique.")
        vocabulary_by_id[normalized_id] = token
    vocabulary_ids = frozenset(vocabulary_by_id)
    if len(vocabulary_ids) != vocabulary_size:
        raise DatasetContractError("Tokenizer vocabulary IDs must be unique.")
    pitch_bend_ids = frozenset(
        token_id
        for token_id, token in vocabulary_by_id.items()
        if token.startswith("PitchBend_")
    )
    if not pitch_bend_ids:
        raise DatasetContractError(
            "GuitarREMI vocabulary must contain PitchBend tokens."
        )
    duration_token_ids = frozenset(
        token_id
        for token_id, token in vocabulary_by_id.items()
        if token.startswith("Duration_")
    )
    if not duration_token_ids:
        raise DatasetContractError(
            "GuitarREMI vocabulary must contain Duration tokens."
        )

    return {
        "path": tokenizer_path,
        "sha256": declared_sha256,
        "vocabulary_size": vocabulary_size,
        "vocabulary_ids": vocabulary_ids,
        "pad": pad,
        "bos": bos,
        "eos": eos,
        "technique_token_ids": technique_token_ids,
        "pitch_bend_ids": pitch_bend_ids,
        "duration_token_ids": duration_token_ids,
    }


def _normalize_technique_token_ids(value: object, name: str) -> dict[str, int]:
    mapping = _require_mapping(value, name)
    _require_exact_keys(mapping, set(TECHNIQUE_TYPES), name)
    normalized = {
        technique_type: _require_nonnegative_int(
            mapping[technique_type], f"{name}.{technique_type}"
        )
        for technique_type in TECHNIQUE_TYPES
    }
    if len(set(normalized.values())) != len(TECHNIQUE_TYPES):
        raise DatasetContractError(f"{name} IDs must be distinct.")
    return normalized


def _require_technique_coverage(value: object, name: str) -> str:
    coverage = _require_string(value, name)
    if coverage not in TECHNIQUE_COVERAGE:
        raise DatasetContractError(
            f"{name} must be one of {', '.join(TECHNIQUE_COVERAGE)}."
        )
    return coverage


def _normalize_techniques(
    value: object,
    name: str,
    *,
    num_notes: int,
    coverage: str,
) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, list):
        raise DatasetContractError(f"{name} must be a JSON array.")
    normalized: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for index, raw_annotation in enumerate(value):
        annotation_name = f"{name}[{index}]"
        annotation = _require_mapping(raw_annotation, annotation_name)
        _require_exact_keys(annotation, {"type", "note_index"}, annotation_name)
        technique_type = _require_string(
            annotation["type"], f"{annotation_name}.type"
        )
        if technique_type not in _TECHNIQUE_ORDER:
            raise DatasetContractError(
                f"{annotation_name}.type must be one of "
                f"{', '.join(TECHNIQUE_TYPES)}."
            )
        note_index = _require_nonnegative_int(
            annotation["note_index"], f"{annotation_name}.note_index"
        )
        if note_index >= num_notes:
            raise DatasetContractError(
                f"{annotation_name}.note_index {note_index} is outside the "
                f"{num_notes}-note sequence."
            )
        identity = (technique_type, note_index)
        if identity in seen:
            raise DatasetContractError(
                f"{name} contains duplicate technique {technique_type} "
                f"for note {note_index}."
            )
        seen.add(identity)
        normalized.append(identity)

    canonical = sorted(
        normalized,
        key=lambda item: (item[1], _TECHNIQUE_ORDER[item[0]]),
    )
    if normalized != canonical:
        raise DatasetContractError(
            f"{name} must be in canonical note/type order."
        )
    if coverage == "UNLABELED" and normalized:
        raise DatasetContractError(
            f"{name} must be empty when technique_coverage is UNLABELED."
        )

    by_note: dict[int, set[str]] = defaultdict(set)
    for technique_type, note_index in normalized:
        by_note[note_index].add(technique_type)
    for note_index, types in by_note.items():
        if {"SLIDE_UP", "SLIDE_DOWN"}.issubset(types):
            raise DatasetContractError(
                f"{name} note {note_index} cannot use SLIDE_UP and "
                "SLIDE_DOWN together."
            )
        if {"PALM_MUTE_ON", "PALM_MUTE_OFF"}.issubset(types):
            raise DatasetContractError(
                f"{name} note {note_index} cannot switch palm mute on and "
                "off together."
            )

    palm_muted = False
    for technique_type, note_index in normalized:
        if technique_type == "PALM_MUTE_ON":
            if palm_muted:
                raise DatasetContractError(
                    f"{name} has redundant PALM_MUTE_ON at note {note_index}."
                )
            palm_muted = True
        elif technique_type == "PALM_MUTE_OFF":
            if not palm_muted:
                raise DatasetContractError(
                    f"{name} has PALM_MUTE_OFF without an active mute at "
                    f"note {note_index}."
                )
            palm_muted = False
    return tuple(normalized)


def _validate_sequences(
    values: list[object],
    *,
    root: Path,
    run_dir: Path,
    preprocessing: object,
    tokenizer_info: Mapping[str, Any],
    verify_hashes: bool,
) -> tuple[tuple[_Sequence, ...], tuple[Mapping[str, Any], ...]]:
    preprocessing_map = _require_mapping(preprocessing, "preprocessing")
    preprocessing_run_id = _require_string(preprocessing_map["run_id"], "preprocessing.run_id")
    required = {
        "actual_note_duration_seconds",
        "instrument_index",
        "nominal_duration_seconds",
        "num_musical_tokens",
        "num_notes",
        "num_pitch_bend_tokens",
        "num_technique_tokens",
        "num_tokens",
        "phrase_index",
        "processed_midi",
        "processed_midi_sha256",
        "processed_midi_size_bytes",
        "programs",
        "round_trip_ok",
        "sequence_file",
        "sequence_id",
        "sequence_sha256",
        "sequence_size_bytes",
        "source_file",
        "source_sha256",
        "split",
        "token_error_ratio",
        "technique_coverage",
        "techniques",
        "track_number",
        "transpose_semitones",
    }
    sequences: list[_Sequence] = []
    records: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    seen_processed: set[str] = set()
    source_files: dict[str, tuple[str, str]] = {}
    source_content_splits: dict[str, str] = {}

    for index, raw_record in enumerate(values):
        name = f"sequences[{index}]"
        record = _require_mapping(raw_record, name)
        _require_exact_keys(record, required, name)
        sequence_id = _require_string(record["sequence_id"], f"{name}.sequence_id")
        if not _SEQUENCE_ID_PATTERN.fullmatch(sequence_id):
            raise DatasetContractError(f"{name}.sequence_id has an invalid format.")
        if sequence_id in seen_ids:
            raise DatasetContractError(f"Duplicate sequence_id: {sequence_id}")
        seen_ids.add(sequence_id)
        split = _require_split(record["split"], f"{name}.split")

        source_file = _declared_relative_path(record["source_file"], f"{name}.source_file")
        if source_file.suffix.lower() not in {".mid", ".midi"}:
            raise DatasetContractError(f"{name}.source_file must be a MIDI path.")
        source_sha256 = _require_sha256(record["source_sha256"], f"{name}.source_sha256")
        previous_source = source_files.setdefault(source_file.as_posix(), (source_sha256, split))
        if previous_source != (source_sha256, split):
            raise DatasetContractError(
                f"Source '{source_file}' has inconsistent content or split metadata."
            )
        previous_split = source_content_splits.setdefault(source_sha256, split)
        if previous_split != split:
            raise DatasetContractError("Identical source content appears in multiple splits.")

        processed = _declared_relative_path(record["processed_midi"], f"{name}.processed_midi")
        if processed.suffix.lower() not in {".mid", ".midi"} or processed.parent.name != split:
            raise DatasetContractError(
                f"{name}.processed_midi must be a MIDI in its declared split directory."
            )
        if processed.parent.parts[-3:] != ("runs", preprocessing_run_id, split):
            raise DatasetContractError(
                f"{name}.processed_midi does not reference preprocessing run {preprocessing_run_id}."
            )
        if processed.as_posix() in seen_processed:
            raise DatasetContractError(f"Duplicate processed_midi: {processed}")
        seen_processed.add(processed.as_posix())
        _require_sha256(record["processed_midi_sha256"], f"{name}.processed_midi_sha256")
        _require_positive_int(
            record["processed_midi_size_bytes"], f"{name}.processed_midi_size_bytes"
        )

        sequence_path = _resolve_declared_path(
            record["sequence_file"], root, f"{name}.sequence_file"
        )
        if sequence_path.parent != run_dir / split or sequence_path.suffix != ".json":
            raise DatasetContractError(
                f"{name}.sequence_file must be a JSON file directly inside its split directory."
            )
        if sequence_path.stem != sequence_id:
            raise DatasetContractError(
                f"{name}.sequence_file name must match sequence_id."
            )
        if sequence_path in seen_paths:
            raise DatasetContractError(f"Duplicate sequence_file: {sequence_path}")
        seen_paths.add(sequence_path)
        _reject_symlink_path(sequence_path, root, f"{name}.sequence_file")
        if not sequence_path.is_file():
            raise DatasetContractError(f"Declared sequence does not exist: {sequence_path}")
        declared_sequence_sha = _require_sha256(
            record["sequence_sha256"], f"{name}.sequence_sha256"
        )
        declared_sequence_size = _require_positive_int(
            record["sequence_size_bytes"], f"{name}.sequence_size_bytes"
        )
        sequence_file = _read_json_file(sequence_path, f"sequence '{sequence_id}'")
        if verify_hashes and (
            sequence_file.sha256 != declared_sequence_sha
            or sequence_file.size_bytes != declared_sequence_size
        ):
            raise DatasetContractError(
                f"Sequence '{sequence_id}' hash or size does not match the manifest."
            )
        sequence_payload = sequence_file.json
        _require_exact_keys(
            sequence_payload,
            {
                "schema_version",
                "sequence_id",
                "ids",
                "programs",
                "technique_coverage",
                "techniques",
            },
            f"sequence '{sequence_id}'",
        )
        if _require_int(
            sequence_payload["schema_version"],
            f"sequence '{sequence_id}'.schema_version",
        ) != TOKEN_SEQUENCE_SCHEMA_VERSION:
            raise DatasetContractError(
                f"Sequence '{sequence_id}' must use schema_version "
                f"{TOKEN_SEQUENCE_SCHEMA_VERSION}."
            )
        if sequence_payload["sequence_id"] != sequence_id:
            raise DatasetContractError(
                f"Sequence payload ID does not match manifest ID '{sequence_id}'."
            )
        programs = _normalize_programs(record["programs"], f"{name}.programs")
        payload_programs = _normalize_programs(
            sequence_payload["programs"], f"sequence '{sequence_id}'.programs"
        )
        if payload_programs != programs:
            raise DatasetContractError(
                f"Sequence '{sequence_id}' program metadata does not match its record."
            )
        num_notes = _require_positive_int(
            record["num_notes"], f"{name}.num_notes"
        )
        record_coverage = _require_technique_coverage(
            record["technique_coverage"], f"{name}.technique_coverage"
        )
        payload_coverage = _require_technique_coverage(
            sequence_payload["technique_coverage"],
            f"sequence '{sequence_id}'.technique_coverage",
        )
        if payload_coverage != record_coverage:
            raise DatasetContractError(
                f"Sequence '{sequence_id}' technique coverage does not match "
                "its record."
            )
        record_techniques = _normalize_techniques(
            record["techniques"],
            f"{name}.techniques",
            num_notes=num_notes,
            coverage=record_coverage,
        )
        payload_techniques = _normalize_techniques(
            sequence_payload["techniques"],
            f"sequence '{sequence_id}'.techniques",
            num_notes=num_notes,
            coverage=payload_coverage,
        )
        if payload_techniques != record_techniques:
            raise DatasetContractError(
                f"Sequence '{sequence_id}' techniques do not match its record."
            )
        ids = _normalize_ids(sequence_payload["ids"], f"sequence '{sequence_id}'.ids")
        if len(ids) < 3:
            raise DatasetContractError(
                f"Sequence '{sequence_id}' must contain BOS, musical content, and EOS."
            )
        pad = tokenizer_info["pad"]
        bos = tokenizer_info["bos"]
        eos = tokenizer_info["eos"]
        if ids[0] != bos or ids[-1] != eos:
            raise DatasetContractError(
                f"Sequence '{sequence_id}' must begin with BOS and end with EOS."
            )
        if pad in ids:
            raise DatasetContractError(
                f"Sequence '{sequence_id}' contains PAD before batching."
            )
        if bos in ids[1:] or eos in ids[:-1]:
            raise DatasetContractError(
                f"Sequence '{sequence_id}' contains an internal BOS or EOS token."
            )
        unknown = sorted(set(ids) - tokenizer_info["vocabulary_ids"])
        if unknown:
            raise DatasetContractError(
                f"Sequence '{sequence_id}' contains IDs outside the tokenizer vocabulary: "
                + ", ".join(str(token_id) for token_id in unknown)
                + "."
            )
        technique_token_ids = tokenizer_info["technique_token_ids"]
        technique_id_values = frozenset(technique_token_ids.values())
        actual_technique_ids = Counter(
            token_id
            for token_id in ids
            if token_id in technique_id_values
        )
        expected_technique_ids = Counter(
            technique_token_ids[technique_type]
            for technique_type, _ in record_techniques
        )
        if actual_technique_ids != expected_technique_ids:
            raise DatasetContractError(
                f"Sequence '{sequence_id}' technique token IDs do not match "
                "its canonical annotations."
            )
        num_technique_tokens = _require_nonnegative_int(
            record["num_technique_tokens"], f"{name}.num_technique_tokens"
        )
        if (
            num_technique_tokens != len(record_techniques)
            or num_technique_tokens != sum(actual_technique_ids.values())
        ):
            raise DatasetContractError(
                f"{name}.num_technique_tokens does not match sequence "
                f"'{sequence_id}'."
            )
        actual_pitch_bend_tokens = sum(
            token_id in tokenizer_info["pitch_bend_ids"] for token_id in ids
        )
        num_pitch_bend_tokens = _require_nonnegative_int(
            record["num_pitch_bend_tokens"], f"{name}.num_pitch_bend_tokens"
        )
        if num_pitch_bend_tokens != actual_pitch_bend_tokens:
            raise DatasetContractError(
                f"{name}.num_pitch_bend_tokens does not match PitchBend IDs "
                f"in sequence '{sequence_id}'."
            )

        num_tokens = _require_positive_int(record["num_tokens"], f"{name}.num_tokens")
        num_musical = _require_positive_int(
            record["num_musical_tokens"], f"{name}.num_musical_tokens"
        )
        if num_tokens != len(ids) or num_musical != len(ids) - 2:
            raise DatasetContractError(
                f"{name} token counts do not match sequence '{sequence_id}'."
            )
        _require_nonnegative_int(record["track_number"], f"{name}.track_number")
        _require_nonnegative_int(record["instrument_index"], f"{name}.instrument_index")
        if record["track_number"] != record["instrument_index"]:
            raise DatasetContractError(f"{name} track and instrument indices must match.")
        _require_nonnegative_int(record["phrase_index"], f"{name}.phrase_index")
        _require_int(record["transpose_semitones"], f"{name}.transpose_semitones")
        _require_positive_number(
            record["nominal_duration_seconds"], f"{name}.nominal_duration_seconds"
        )
        _require_positive_number(
            record["actual_note_duration_seconds"], f"{name}.actual_note_duration_seconds"
        )
        if record["round_trip_ok"] is not True:
            raise DatasetContractError(f"{name}.round_trip_ok must be true.")
        if _require_nonnegative_number(
            record["token_error_ratio"], f"{name}.token_error_ratio"
        ) != 0.0:
            raise DatasetContractError(f"{name}.token_error_ratio must be zero.")

        sequences.append(
            _Sequence(
                sequence_id=sequence_id,
                split=split,
                ids=ids,
                technique_coverage=record_coverage,
            )
        )
        records.append(record)

    order = [(SPLITS.index(sequence.split), str(record["sequence_file"])) for sequence, record in zip(sequences, records, strict=True)]
    if order != sorted(order):
        raise DatasetContractError(
            "Stage 2 sequences must be ordered by split and sequence_file."
        )
    return tuple(sequences), tuple(records)


def _validate_summary(value: object, records: Sequence[Mapping[str, Any]]) -> None:
    summary = _require_mapping(value, "summary")
    _require_exact_keys(
        summary,
        {
            "sequences",
            "total_tokens",
            "total_musical_tokens",
            "length",
            "by_split",
            "techniques",
            "pitch_bends",
        },
        "summary",
    )

    def expected_group(group: Sequence[Mapping[str, Any]]) -> dict[str, object]:
        lengths = [int(record["num_tokens"]) for record in group]
        musical = [int(record["num_musical_tokens"]) for record in group]
        return {
            "sequences": len(group),
            "total_tokens": sum(lengths),
            "total_musical_tokens": sum(musical),
            "min_tokens": min(lengths) if lengths else None,
            "max_tokens": max(lengths) if lengths else None,
            "mean_tokens": sum(lengths) / len(lengths) if lengths else None,
        }

    all_expected = expected_group(records)
    for field in ("sequences", "total_tokens", "total_musical_tokens"):
        if summary[field] != all_expected[field]:
            raise DatasetContractError(f"summary.{field} does not match sequences.")
    length = _require_mapping(summary["length"], "summary.length")
    _require_exact_keys(length, {"min_tokens", "max_tokens", "mean_tokens"}, "summary.length")
    for field in ("min_tokens", "max_tokens", "mean_tokens"):
        if not _numbers_equal(length[field], all_expected[field]):
            raise DatasetContractError(f"summary.length.{field} does not match sequences.")

    by_split = _require_mapping(summary["by_split"], "summary.by_split")
    _require_exact_keys(by_split, set(SPLITS), "summary.by_split")
    group_fields = {
        "sequences",
        "total_tokens",
        "total_musical_tokens",
        "min_tokens",
        "max_tokens",
        "mean_tokens",
    }
    for split in SPLITS:
        actual = _require_mapping(by_split[split], f"summary.by_split.{split}")
        _require_exact_keys(actual, group_fields, f"summary.by_split.{split}")
        expected = expected_group([record for record in records if record["split"] == split])
        for field in group_fields:
            if not _numbers_equal(actual[field], expected[field]):
                raise DatasetContractError(
                    f"summary.by_split.{split}.{field} does not match sequences."
                )

    techniques = _require_mapping(summary["techniques"], "summary.techniques")
    _require_exact_keys(
        techniques,
        {"total_tokens", "by_type", "coverage"},
        "summary.techniques",
    )
    expected_total_techniques = sum(
        int(record["num_technique_tokens"]) for record in records
    )
    actual_total_techniques = _require_nonnegative_int(
        techniques["total_tokens"], "summary.techniques.total_tokens"
    )
    if actual_total_techniques != expected_total_techniques:
        raise DatasetContractError(
            "summary.techniques.total_tokens does not match sequences."
        )
    by_type = _require_mapping(
        techniques["by_type"], "summary.techniques.by_type"
    )
    _require_exact_keys(
        by_type, set(TECHNIQUE_TYPES), "summary.techniques.by_type"
    )
    expected_by_type = Counter(
        annotation["type"]
        for record in records
        for annotation in record["techniques"]
    )
    for technique_type in TECHNIQUE_TYPES:
        actual_count = _require_nonnegative_int(
            by_type[technique_type],
            f"summary.techniques.by_type.{technique_type}",
        )
        if actual_count != expected_by_type[technique_type]:
            raise DatasetContractError(
                f"summary.techniques.by_type.{technique_type} does not "
                "match sequences."
            )
    coverage = _require_mapping(
        techniques["coverage"], "summary.techniques.coverage"
    )
    _require_exact_keys(
        coverage,
        {"complete_sequences", "unlabeled_sequences"},
        "summary.techniques.coverage",
    )
    expected_complete = sum(
        record["technique_coverage"] == "COMPLETE" for record in records
    )
    expected_unlabeled = len(records) - expected_complete
    actual_complete = _require_nonnegative_int(
        coverage["complete_sequences"],
        "summary.techniques.coverage.complete_sequences",
    )
    actual_unlabeled = _require_nonnegative_int(
        coverage["unlabeled_sequences"],
        "summary.techniques.coverage.unlabeled_sequences",
    )
    if (actual_complete, actual_unlabeled) != (
        expected_complete,
        expected_unlabeled,
    ):
        raise DatasetContractError(
            "summary.techniques.coverage does not match sequences."
        )

    pitch_bends = _require_mapping(summary["pitch_bends"], "summary.pitch_bends")
    _require_exact_keys(
        pitch_bends,
        {"total_tokens", "sequences_with_pitch_bends"},
        "summary.pitch_bends",
    )
    expected_pitch_bends = sum(
        int(record["num_pitch_bend_tokens"]) for record in records
    )
    expected_sequences_with_bends = sum(
        int(record["num_pitch_bend_tokens"]) > 0 for record in records
    )
    actual_pitch_bends = _require_nonnegative_int(
        pitch_bends["total_tokens"], "summary.pitch_bends.total_tokens"
    )
    actual_sequences_with_bends = _require_nonnegative_int(
        pitch_bends["sequences_with_pitch_bends"],
        "summary.pitch_bends.sequences_with_pitch_bends",
    )
    if (actual_pitch_bends, actual_sequences_with_bends) != (
        expected_pitch_bends,
        expected_sequences_with_bends,
    ):
        raise DatasetContractError(
            "summary.pitch_bends does not match sequences."
        )


def _resolve_project_root(manifest_path: Path, project_root: str | Path | None) -> Path:
    if project_root is not None:
        root = Path(project_root).expanduser().resolve()
        if not root.is_dir():
            raise DatasetContractError(f"project_root is not a directory: {root}")
        return root
    for candidate in manifest_path.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate.resolve()
    raise DatasetContractError(
        "Could not infer project_root from the manifest; pass project_root explicitly."
    )


def _read_json_file(path: Path, name: str) -> _FilePayload:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DatasetContractError(f"Could not read {name} '{path}': {exc}") from exc
    try:
        decoded = raw.decode("utf-8")
        parsed = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DatasetContractError(f"{name} is not valid strict UTF-8 JSON: {exc}") from exc
    mapping = _require_mapping(parsed, name)
    return _FilePayload(
        path=path,
        raw=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        json=mapping,
    )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _resolve_declared_path(value: object, root: Path, name: str) -> Path:
    relative = _declared_relative_path(value, name)
    candidate = root.joinpath(*relative.parts)
    _reject_symlink_path(candidate, root, name)
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise DatasetContractError(f"{name} escapes project_root.")
    return resolved


def _declared_relative_path(value: object, name: str) -> PurePosixPath:
    label = _require_string(value, name)
    if "\\" in label:
        raise DatasetContractError(f"{name} must use POSIX path separators.")
    path = PurePosixPath(label)
    if path.is_absolute():
        raise DatasetContractError(f"{name} must be relative to project_root.")
    if path.as_posix() != label or any(part in {"", ".", ".."} for part in path.parts):
        raise DatasetContractError(
            f"{name} must be a normalized relative path without traversal."
        )
    return path


def _reject_symlink_path(path: Path, root: Path, name: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:  # pragma: no cover - caller already resolves safely.
        raise DatasetContractError(f"{name} escapes project_root.") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise DatasetContractError(f"{name} cannot use symlinks.")


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetContractError(f"{name} must be a JSON object.")
    if not all(isinstance(key, str) for key in value):
        raise DatasetContractError(f"{name} keys must be strings.")
    return value


def _require_exact_keys(value: Mapping[str, Any], required: set[str], name: str) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing:
        raise DatasetContractError(f"{name} is missing field(s): {', '.join(missing)}.")
    if unknown:
        raise DatasetContractError(f"{name} has unknown field(s): {', '.join(unknown)}.")


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise DatasetContractError(f"{name} must be a non-empty string.")
    return value


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise DatasetContractError(f"{name} must be an integer.")
    return int(value)


def _require_nonnegative_int(value: object, name: str) -> int:
    converted = _require_int(value, name)
    if converted < 0:
        raise DatasetContractError(f"{name} must be non-negative.")
    return converted


def _require_positive_int(value: object, name: str) -> int:
    converted = _require_int(value, name)
    if converted <= 0:
        raise DatasetContractError(f"{name} must be positive.")
    return converted


def _require_sha256(value: object, name: str) -> str:
    converted = _require_string(value, name)
    if not _SHA256_PATTERN.fullmatch(converted):
        raise DatasetContractError(f"{name} must be 64 lowercase hexadecimal digits.")
    return converted


def _require_split(value: object, name: str) -> str:
    split = _require_string(value, name)
    if split not in SPLITS:
        raise DatasetContractError(f"{name} must be one of {', '.join(SPLITS)}.")
    return split


def _require_positive_number(value: object, name: str) -> float:
    converted = _require_nonnegative_number(value, name)
    if converted <= 0:
        raise DatasetContractError(f"{name} must be positive.")
    return converted


def _require_nonnegative_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetContractError(f"{name} must be a number.")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise DatasetContractError(f"{name} must be finite and non-negative.")
    return converted


def _normalize_ids(value: object, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DatasetContractError(f"{name} must be a sequence of integers.")
    return tuple(
        _require_nonnegative_int(token_id, f"{name}[{index}]")
        for index, token_id in enumerate(value)
    )


def _normalize_loss_mask(value: object, name: str) -> tuple[bool, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise DatasetContractError(f"{name} must be a sequence of booleans.")
    normalized: list[bool] = []
    for index, enabled in enumerate(value):
        if not isinstance(enabled, bool):
            raise DatasetContractError(f"{name}[{index}] must be boolean.")
        normalized.append(enabled)
    return tuple(normalized)


def _normalize_programs(value: object, name: str) -> tuple[tuple[int, bool], ...]:
    if not isinstance(value, list) or len(value) != 1:
        raise DatasetContractError(f"{name} must contain exactly one program descriptor.")
    descriptor = value[0]
    if not isinstance(descriptor, list) or len(descriptor) != 2:
        raise DatasetContractError(f"{name}[0] must be [program, is_drum].")
    program = _require_nonnegative_int(descriptor[0], f"{name}[0][0]")
    if program > 127:
        raise DatasetContractError(f"{name}[0][0] must be between 0 and 127.")
    if not isinstance(descriptor[1], bool) or descriptor[1]:
        raise DatasetContractError(f"{name}[0][1] must be false for guitar data.")
    return ((program, False),)


def _canonical_hash(payload: object) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _numbers_equal(actual: object, expected: object) -> bool:
    if expected is None:
        return actual is None
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return False
    return math.isfinite(float(actual)) and math.isclose(
        float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
    )
