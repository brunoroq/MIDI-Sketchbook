"""Synthetic contract tests for the Stage 3 next-token dataset."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pytest

from midi_idea_generator.dataset import (
    DatasetContractError,
    TokenizedSequenceDataset,
    collate_token_sequences,
    make_collate_fn,
)
from midi_idea_generator.tokenization_config import RemiTokenizerConfig
from midi_idea_generator.tokenizer import (
    build_tokenizer,
    get_special_token_ids,
    save_tokenizer,
)


_TOKENIZATION_RUN_ID = "0123456789abcdefabcd"
_PREPROCESSING_RUN_ID = "abcdef0123456789abcd"
_CONFIGURATION = {"tokenizer": {"fixture": "stage-three-schema-one"}}


@dataclass(slots=True)
class _SyntheticCorpus:
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    sequence_paths: dict[str, Path]
    special_ids: tuple[int, int, int]

    def write_manifest(self) -> None:
        _write_json(self.manifest_path, self.manifest)


def _canonical_hash(value: object) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fingerprint(path: Path) -> tuple[str, int]:
    raw = path.read_bytes()
    return sha256(raw).hexdigest(), len(raw)


def _group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [record["num_tokens"] for record in records]
    musical = [record["num_musical_tokens"] for record in records]
    return {
        "sequences": len(records),
        "total_tokens": sum(lengths),
        "total_musical_tokens": sum(musical),
        "min_tokens": min(lengths) if lengths else None,
        "max_tokens": max(lengths) if lengths else None,
        "mean_tokens": sum(lengths) / len(lengths) if lengths else None,
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    overall = _group_summary(records)
    return {
        "sequences": overall["sequences"],
        "total_tokens": overall["total_tokens"],
        "total_musical_tokens": overall["total_musical_tokens"],
        "length": {
            "min_tokens": overall["min_tokens"],
            "max_tokens": overall["max_tokens"],
            "mean_tokens": overall["mean_tokens"],
        },
        "by_split": {
            split: _group_summary(
                [record for record in records if record["split"] == split]
            )
            for split in ("train", "validation", "test")
        },
    }


def _make_synthetic_corpus(
    tmp_path: Path,
    *,
    split_lengths: dict[str, tuple[int, ...]] | None = None,
) -> _SyntheticCorpus:
    split_lengths = split_lengths or {
        "train": (5, 7),
        "validation": (6,),
        "test": (4,),
    }
    root = tmp_path / "synthetic-stage-three-project"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname='synthetic-stage-three-test'\n",
        encoding="utf-8",
    )
    run_dir = root / "data" / "tokenized" / "runs" / _TOKENIZATION_RUN_ID
    configuration_sha256 = _canonical_hash(_CONFIGURATION)
    tokenizer_path = run_dir / "tokenizer.json"
    tokenizer = build_tokenizer(RemiTokenizerConfig())
    save_tokenizer(
        tokenizer,
        tokenizer_path,
        additional_attributes={
            "stage": 2,
            "tokenization_schema_version": 1,
            "tokenization_run_id": _TOKENIZATION_RUN_ID,
            "configuration_sha256": configuration_sha256,
        },
    )
    tokenizer_sha256, tokenizer_size = _fingerprint(tokenizer_path)
    special = get_special_token_ids(tokenizer)
    musical_ids = [
        int(token_id)
        for token_id in tokenizer.vocab.values()
        if int(token_id) not in {special.pad, special.bos, special.eos}
    ]

    records: list[dict[str, Any]] = []
    sequence_paths: dict[str, Path] = {}
    serial = 0
    for split in ("train", "validation", "test"):
        for split_index, num_tokens in enumerate(split_lengths.get(split, ())):
            assert num_tokens >= 3
            sequence_id = f"{split}-riff-{split_index:02d}-{serial:012x}"
            musical_length = num_tokens - 2
            ids = [
                special.bos,
                *(
                    musical_ids[index % len(musical_ids)]
                    for index in range(musical_length)
                ),
                special.eos,
            ]
            sequence_path = run_dir / split / f"{sequence_id}.json"
            programs = [[30, False]]
            _write_json(
                sequence_path,
                {
                    "schema_version": 1,
                    "sequence_id": sequence_id,
                    "ids": ids,
                    "programs": programs,
                },
            )
            sequence_sha256, sequence_size = _fingerprint(sequence_path)
            source_file = f"data/raw/{split}-source-{split_index}.mid"
            processed_midi = (
                f"data/processed/runs/{_PREPROCESSING_RUN_ID}/{split}/"
                f"{split}-riff-{split_index}.mid"
            )
            record = {
                "actual_note_duration_seconds": 1.5,
                "instrument_index": 0,
                "nominal_duration_seconds": 2.0,
                "num_musical_tokens": musical_length,
                "num_notes": 3,
                "num_tokens": num_tokens,
                "phrase_index": split_index,
                "processed_midi": processed_midi,
                "processed_midi_sha256": sha256(
                    processed_midi.encode("utf-8")
                ).hexdigest(),
                "processed_midi_size_bytes": 128,
                "programs": programs,
                "round_trip_ok": True,
                "sequence_file": sequence_path.relative_to(root).as_posix(),
                "sequence_id": sequence_id,
                "sequence_sha256": sequence_sha256,
                "sequence_size_bytes": sequence_size,
                "source_file": source_file,
                "source_sha256": sha256(source_file.encode("utf-8")).hexdigest(),
                "split": split,
                "token_error_ratio": 0.0,
                "track_number": 0,
                "transpose_semitones": 0,
            }
            records.append(record)
            sequence_paths[sequence_id] = sequence_path
            serial += 1

    manifest = {
        "schema_version": 1,
        "tokenization_run_id": _TOKENIZATION_RUN_ID,
        "tokenized_run_dir": run_dir.relative_to(root).as_posix(),
        "preprocessing": {
            "manifest_path": "data/splits/manifest.json",
            "manifest_sha256": "1" * 64,
            "schema_version": 2,
            "run_id": _PREPROCESSING_RUN_ID,
            "configuration_sha256": "2" * 64,
        },
        "configuration_sha256": configuration_sha256,
        "configuration": deepcopy(_CONFIGURATION),
        "tool_versions": {
            "implementation_sha256": "3" * 64,
            "fixture": "1",
        },
        "tokenizer": {
            "type": "REMI",
            "path": tokenizer_path.relative_to(root).as_posix(),
            "sha256": tokenizer_sha256,
            "size_bytes": tokenizer_size,
            "vocabulary_sha256": _canonical_hash(tokenizer.vocab),
            "vocabulary_size": len(tokenizer.vocab),
            "special_token_ids": {
                "pad": special.pad,
                "bos": special.bos,
                "eos": special.eos,
            },
        },
        "summary": _summary(records),
        "sequences": records,
    }
    manifest_path = root / "data" / "tokenized" / "manifest.json"
    corpus = _SyntheticCorpus(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        sequence_paths=sequence_paths,
        special_ids=(special.pad, special.bos, special.eos),
    )
    corpus.write_manifest()
    return corpus


def test_dataset_uses_manifest_only_and_builds_next_token_pairs(tmp_path: Path) -> None:
    corpus = _make_synthetic_corpus(tmp_path)
    unlisted = (
        corpus.root
        / "data"
        / "tokenized"
        / "runs"
        / _TOKENIZATION_RUN_ID
        / "train"
        / "unlisted.json"
    )
    unlisted.write_text("not JSON and deliberately unlisted", encoding="utf-8")

    dataset = TokenizedSequenceDataset(corpus.manifest_path, "train")

    assert len(dataset) == 2
    assert unlisted.is_file()
    first = dataset[0]
    payload = json.loads(
        corpus.sequence_paths[first["sequence_id"]].read_text(encoding="utf-8")
    )
    assert first["input_ids"] == tuple(payload["ids"][:-1])
    assert first["target_ids"] == tuple(payload["ids"][1:])
    assert first["length"] == len(payload["ids"]) - 1
    assert first["split"] == "train"
    assert first["input_ids"][0] == dataset.bos_token_id
    assert first["target_ids"][-1] == dataset.eos_token_id
    assert dataset.pad_token_id == corpus.special_ids[0]
    assert dataset.tokenization_run_id == _TOKENIZATION_RUN_ID
    assert dataset.configuration_sha256 == corpus.manifest["configuration_sha256"]
    assert dataset.tokenizer_sha256 == corpus.manifest["tokenizer"]["sha256"]
    assert dataset.tokenization_manifest_sha256 == sha256(
        corpus.manifest_path.read_bytes()
    ).hexdigest()


def test_dataset_preserves_empty_splits(tmp_path: Path) -> None:
    corpus = _make_synthetic_corpus(tmp_path, split_lengths={"train": (5,)})

    validation = TokenizedSequenceDataset(
        corpus.manifest_path, "validation", project_root=corpus.root
    )

    assert len(validation) == 0
    assert validation.split == "validation"


def test_overlong_sequence_is_rejected_without_truncation(tmp_path: Path) -> None:
    corpus = _make_synthetic_corpus(tmp_path, split_lengths={"train": (9,)})

    with pytest.raises(DatasetContractError, match="has 9 tokens"):
        TokenizedSequenceDataset(
            corpus.manifest_path,
            "train",
            max_sequence_length=8,
        )

    dataset = TokenizedSequenceDataset(
        corpus.manifest_path,
        "train",
        max_sequence_length=9,
    )
    assert len(dataset[0]["input_ids"]) == 8


def test_dynamic_collate_pads_inputs_targets_and_mask(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    corpus = _make_synthetic_corpus(
        tmp_path,
        split_lengths={"train": (5, 8)},
    )
    dataset = TokenizedSequenceDataset(corpus.manifest_path, "train")

    batch = collate_token_sequences(
        [dataset[0], dataset[1]],
        dataset.pad_token_id,
    )

    assert batch["input_ids"].shape == (2, 7)
    assert batch["target_ids"].shape == (2, 7)
    assert batch["lengths"].tolist() == [4, 7]
    assert batch["attention_mask"].dtype == torch.bool
    assert batch["attention_mask"].tolist() == [
        [True, True, True, True, False, False, False],
        [True, True, True, True, True, True, True],
    ]
    assert batch["input_ids"][0, 4:].tolist() == [dataset.pad_token_id] * 3
    assert batch["target_ids"][0, 4:].tolist() == [dataset.pad_token_id] * 3
    assert batch["sequence_ids"] == [
        dataset[0]["sequence_id"],
        dataset[1]["sequence_id"],
    ]
    assert batch["splits"] == ["train", "train"]

    collator = make_collate_fn(dataset.pad_token_id)
    repeated = collator([dataset[0], dataset[1]])
    assert torch.equal(repeated["input_ids"], batch["input_ids"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("schema", "schema_version 1"),
        ("summary", "summary.sequences does not match"),
        ("sequence_traversal", "normalized relative path without traversal"),
        ("split_path", "declared split directory"),
        ("duplicate", "Duplicate sequence_id"),
        ("source_leakage", "Identical source content appears in multiple splits"),
    ],
)
def test_malformed_manifest_contract_is_rejected(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    corpus = _make_synthetic_corpus(tmp_path)
    if mutation == "schema":
        corpus.manifest["schema_version"] = 2
    elif mutation == "summary":
        corpus.manifest["summary"]["sequences"] += 1
    elif mutation == "sequence_traversal":
        corpus.manifest["sequences"][0]["sequence_file"] = "../escape.json"
    elif mutation == "split_path":
        corpus.manifest["sequences"][0]["split"] = "validation"
    elif mutation == "duplicate":
        duplicate = deepcopy(corpus.manifest["sequences"][0])
        corpus.manifest["sequences"].insert(1, duplicate)
        corpus.manifest["summary"] = _summary(corpus.manifest["sequences"])
    elif mutation == "source_leakage":
        train = corpus.manifest["sequences"][0]
        validation = next(
            record
            for record in corpus.manifest["sequences"]
            if record["split"] == "validation"
        )
        validation["source_sha256"] = train["source_sha256"]
    else:  # pragma: no cover - protects the parametrization itself.
        raise AssertionError(f"Unhandled mutation: {mutation}")
    corpus.write_manifest()

    with pytest.raises(DatasetContractError, match=message):
        TokenizedSequenceDataset(corpus.manifest_path, "train")


def test_tampered_tokenizer_is_rejected_by_fingerprint(tmp_path: Path) -> None:
    corpus = _make_synthetic_corpus(tmp_path)
    tokenizer_path = corpus.root / corpus.manifest["tokenizer"]["path"]
    tokenizer_path.write_bytes(tokenizer_path.read_bytes() + b"\n")

    with pytest.raises(DatasetContractError, match="Tokenizer hash or size"):
        TokenizedSequenceDataset(corpus.manifest_path, "train")


def test_tampered_sequence_is_rejected_by_fingerprint(tmp_path: Path) -> None:
    corpus = _make_synthetic_corpus(tmp_path)
    first = corpus.manifest["sequences"][0]
    sequence_path = corpus.root / first["sequence_file"]
    payload = json.loads(sequence_path.read_text(encoding="utf-8"))
    payload["ids"].insert(-1, payload["ids"][1])
    _write_json(sequence_path, payload)

    with pytest.raises(DatasetContractError, match="hash or size"):
        TokenizedSequenceDataset(corpus.manifest_path, "train")


def test_declared_sequence_cannot_be_a_symlink(tmp_path: Path) -> None:
    corpus = _make_synthetic_corpus(tmp_path)
    sequence_path = corpus.root / corpus.manifest["sequences"][0]["sequence_file"]
    real_path = sequence_path.with_suffix(".real.json")
    sequence_path.rename(real_path)
    sequence_path.symlink_to(real_path)

    with pytest.raises(DatasetContractError, match="cannot use symlinks"):
        TokenizedSequenceDataset(corpus.manifest_path, "train")


def test_stored_pad_or_invalid_token_id_is_rejected_after_valid_hash(
    tmp_path: Path,
) -> None:
    corpus = _make_synthetic_corpus(tmp_path)
    first = corpus.manifest["sequences"][0]
    sequence_path = corpus.root / first["sequence_file"]
    payload = json.loads(sequence_path.read_text(encoding="utf-8"))
    payload["ids"][1] = corpus.special_ids[0]
    _write_json(sequence_path, payload)
    sequence_sha256, sequence_size = _fingerprint(sequence_path)
    first["sequence_sha256"] = sequence_sha256
    first["sequence_size_bytes"] = sequence_size
    corpus.write_manifest()

    with pytest.raises(DatasetContractError, match="contains PAD before batching"):
        TokenizedSequenceDataset(corpus.manifest_path, "train")


def test_collate_rejects_pre_padded_or_inconsistent_samples() -> None:
    torch = pytest.importorskip("torch")
    del torch
    sample = {
        "input_ids": (1, 3),
        "target_ids": (3, 2),
        "length": 2,
        "sequence_id": "valid-sequence",
        "split": "train",
    }

    with pytest.raises(DatasetContractError, match="contains PAD"):
        collate_token_sequences(
            [{**sample, "input_ids": (1, 0)}],
            pad_token_id=0,
        )
    with pytest.raises(DatasetContractError, match="lengths must match"):
        collate_token_sequences(
            [{**sample, "length": 3}],
            pad_token_id=0,
        )
    with pytest.raises(DatasetContractError, match="empty batch"):
        collate_token_sequences([], pad_token_id=0)
