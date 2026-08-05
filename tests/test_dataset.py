"""Synthetic contract tests for the Stage 3 next-token dataset."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pytest

from midi_idea_generator.tonality import MODE_NAMES, TONALITY_METHODS, TONIC_NAMES
from midi_idea_generator.dataset import (
    DatasetContractError,
    TokenizedSequenceDataset,
    collate_token_sequences,
    make_collate_fn,
)
from midi_idea_generator.tokenization_config import RemiTokenizerConfig
from midi_idea_generator.tokenizer import (
    TECHNIQUE_TYPES,
    build_tokenizer,
    get_mode_token_ids,
    get_special_token_ids,
    get_technique_token_ids,
    get_tonic_token_ids,
    save_tokenizer,
)


_TOKENIZATION_RUN_ID = "0123456789abcdefabcd"
_PREPROCESSING_RUN_ID = "abcdef0123456789abcd"
_CONFIGURATION = {"tokenizer": {"fixture": "stage-three-schema-two"}}


@dataclass(slots=True)
class _SyntheticCorpus:
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    sequence_paths: dict[str, Path]
    special_ids: tuple[int, int, int]
    technique_ids: dict[str, int]
    tonic_ids: dict[str, int]
    mode_ids: dict[str, int]
    pitch_bend_ids: tuple[int, ...]
    duration_ids: tuple[int, ...]
    ordinary_ids: tuple[int, ...]
    bar_id: int

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
    technique_counts = {
        technique_type: sum(
            annotation["type"] == technique_type
            for record in records
            for annotation in record["techniques"]
        )
        for technique_type in TECHNIQUE_TYPES
    }
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
        "techniques": {
            "total_tokens": sum(
                record["num_technique_tokens"] for record in records
            ),
            "by_type": technique_counts,
            "coverage": {
                "complete_sequences": sum(
                    record["technique_coverage"] == "COMPLETE"
                    for record in records
                ),
                "unlabeled_sequences": sum(
                    record["technique_coverage"] == "UNLABELED"
                    for record in records
                ),
            },
        },
        "pitch_bends": {
            "total_tokens": sum(
                record["num_pitch_bend_tokens"] for record in records
            ),
            "sequences_with_pitch_bends": sum(
                record["num_pitch_bend_tokens"] > 0 for record in records
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


def _make_synthetic_corpus(
    tmp_path: Path,
    *,
    split_lengths: dict[str, tuple[int, ...]] | None = None,
) -> _SyntheticCorpus:
    split_lengths = split_lengths or {
        "train": (5, 7),
        "validation": (6,),
        "test": (5,),
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
            "tokenization_schema_version": 3,
            "tokenization_run_id": _TOKENIZATION_RUN_ID,
            "configuration_sha256": configuration_sha256,
        },
    )
    tokenizer_sha256, tokenizer_size = _fingerprint(tokenizer_path)
    special = get_special_token_ids(tokenizer)
    technique_ids = get_technique_token_ids(tokenizer)
    tonic_ids = get_tonic_token_ids(tokenizer)
    mode_ids = get_mode_token_ids(tokenizer)
    pitch_bend_ids = tuple(
        int(token_id)
        for token, token_id in tokenizer.vocab.items()
        if token.startswith("PitchBend_")
    )
    duration_ids = tuple(
        int(token_id)
        for token, token_id in tokenizer.vocab.items()
        if token.startswith("Duration_")
    )
    ordinary_ids = tuple(
        int(token_id)
        for token, token_id in tokenizer.vocab.items()
        if int(token_id) not in {special.pad, special.bos, special.eos}
        and int(token_id) not in set(technique_ids.values())
        and int(token_id) not in set(tonic_ids.values())
        and int(token_id) not in set(mode_ids.values())
        and not token.startswith("PitchBend_")
        and not token.startswith("Duration_")
    )
    base_pool = [
        int(token_id)
        for token, token_id in tokenizer.vocab.items()
        if int(token_id) not in {special.pad, special.bos, special.eos}
        and int(token_id) not in set(technique_ids.values())
        and int(token_id) not in set(tonic_ids.values())
        and int(token_id) not in set(mode_ids.values())
        and not token.startswith("PitchBend_")
        and not token.startswith("Bar_")
    ]
    bar_id = int(tokenizer.vocab["Bar_None"])
    tonality = {
        "tonic": "E",
        "mode": "PHRYGIAN",
        "method": "MANUAL",
        "tonic_confidence": None,
        "mode_confidence": None,
    }

    records: list[dict[str, Any]] = []
    sequence_paths: dict[str, Path] = {}
    serial = 0
    for split in ("train", "validation", "test"):
        for split_index, num_tokens in enumerate(split_lengths.get(split, ())):
            assert num_tokens >= 5
            sequence_id = f"{split}-riff-{split_index:02d}-{serial:012x}"
            musical_length = num_tokens - 2
            base_length = musical_length - 2
            ids = [
                special.bos,
                tonic_ids[tonality["tonic"]],
                mode_ids[tonality["mode"]],
                bar_id,
                *(base_pool[index % len(base_pool)] for index in range(base_length - 1)),
                special.eos,
            ]
            sequence_path = run_dir / split / f"{sequence_id}.json"
            programs = [[30, False]]
            _write_json(
                sequence_path,
                {
                    "schema_version": 3,
                    "sequence_id": sequence_id,
                    "ids": ids,
                    "programs": programs,
                    "technique_coverage": "UNLABELED",
                    "techniques": [],
                    "tonality": deepcopy(tonality),
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
                "num_pitch_bend_tokens": 0,
                "num_technique_tokens": 0,
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
                "technique_coverage": "UNLABELED",
                "techniques": [],
                "track_number": 0,
                "transpose_semitones": 0,
                "tonality": deepcopy(tonality),
            }
            records.append(record)
            sequence_paths[sequence_id] = sequence_path
            serial += 1

    manifest = {
        "schema_version": 3,
        "tokenization_run_id": _TOKENIZATION_RUN_ID,
        "tokenized_run_dir": run_dir.relative_to(root).as_posix(),
        "preprocessing": {
            "manifest_path": "data/splits/manifest.json",
            "manifest_sha256": "1" * 64,
            "schema_version": 4,
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
            "type": "ConditionedGuitarREMI",
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
            "technique_token_ids": technique_ids,
            "conditioning_schema_version": 1,
            "tonic_token_ids": tonic_ids,
            "mode_token_ids": mode_ids,
            "pitch_bend_sensitivity_semitones": 6,
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
        technique_ids=technique_ids,
        tonic_ids=tonic_ids,
        mode_ids=mode_ids,
        pitch_bend_ids=pitch_bend_ids,
        duration_ids=duration_ids,
        ordinary_ids=ordinary_ids,
        bar_id=bar_id,
    )
    corpus.write_manifest()
    return corpus


def _first_sequence(
    corpus: _SyntheticCorpus,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    record = corpus.manifest["sequences"][0]
    sequence_path = corpus.root / record["sequence_file"]
    payload = json.loads(sequence_path.read_text(encoding="utf-8"))
    return record, payload, sequence_path


def _publish_sequence_change(
    corpus: _SyntheticCorpus,
    record: dict[str, Any],
    payload: dict[str, Any],
    sequence_path: Path,
) -> None:
    _write_json(sequence_path, payload)
    sequence_sha256, sequence_size = _fingerprint(sequence_path)
    record["sequence_sha256"] = sequence_sha256
    record["sequence_size_bytes"] = sequence_size
    record["num_tokens"] = len(payload["ids"])
    record["num_musical_tokens"] = len(payload["ids"]) - 2
    corpus.manifest["summary"] = _summary(corpus.manifest["sequences"])
    corpus.write_manifest()


def _add_valid_guitar_metadata(corpus: _SyntheticCorpus) -> None:
    record, payload, sequence_path = _first_sequence(corpus)
    annotations = [
        {"type": "PALM_MUTE_ON", "note_index": 0},
        {"type": "DEAD_NOTE", "note_index": 1},
        {"type": "PALM_MUTE_OFF", "note_index": 2},
        {"type": "VIBRATO", "note_index": 2},
    ]
    record["technique_coverage"] = "COMPLETE"
    record["techniques"] = deepcopy(annotations)
    record["num_technique_tokens"] = len(annotations)
    record["num_pitch_bend_tokens"] = 2
    payload["technique_coverage"] = "COMPLETE"
    payload["techniques"] = deepcopy(annotations)
    payload["ids"][-1:-1] = [
        *(corpus.technique_ids[annotation["type"]] for annotation in annotations),
        corpus.pitch_bend_ids[0],
        corpus.pitch_bend_ids[-1],
    ]
    _publish_sequence_change(corpus, record, payload, sequence_path)


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
    assert corpus.special_ids[0] not in first["input_ids"]
    assert corpus.special_ids[0] not in first["target_ids"]
    assert len(first["unknown_technique_decision_mask"]) == first["length"]
    assert isinstance(dataset.technique_token_ids, frozenset)
    assert dataset.technique_token_ids == frozenset(corpus.technique_ids.values())
    assert dataset.tonic_token_ids == frozenset(corpus.tonic_ids.values())
    assert dataset.mode_token_ids == frozenset(corpus.mode_ids.values())
    assert len(dataset.token_type_by_id) == dataset.vocabulary_size
    assert dataset.token_type_by_id[dataset.pad_token_id] == "PAD"
    assert {
        dataset.token_type_by_id[token_id]
        for token_id in dataset.technique_token_ids
    } == {"Technique"}
    assert first["length"] == len(payload["ids"]) - 1
    assert first["split"] == "train"
    assert first["tonality"] == payload["tonality"]
    assert first["input_ids"][1] == corpus.tonic_ids["E"]
    assert first["input_ids"][2] == corpus.mode_ids["PHRYGIAN"]
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
    assert batch["unknown_technique_decision_mask"].shape == (2, 7)
    assert batch["lengths"].tolist() == [4, 7]
    assert batch["attention_mask"].dtype == torch.bool
    assert batch["unknown_technique_decision_mask"].dtype == torch.bool
    assert batch["attention_mask"].tolist() == [
        [True, True, True, True, False, False, False],
        [True, True, True, True, True, True, True],
    ]
    assert batch["input_ids"][0, 4:].tolist() == [dataset.pad_token_id] * 3
    assert batch["target_ids"][0, 4:].tolist() == [dataset.pad_token_id] * 3
    assert batch["unknown_technique_decision_mask"][0, 4:].tolist() == [
        False,
        False,
        False,
    ]
    assert batch["unknown_technique_decision_mask"][0, :4].tolist() == list(
        dataset[0]["unknown_technique_decision_mask"]
    )
    assert batch["unknown_technique_decision_mask"][1, :7].tolist() == list(
        dataset[1]["unknown_technique_decision_mask"]
    )
    assert int((batch["target_ids"] != dataset.pad_token_id).sum().item()) == sum(
        sample["length"] for sample in (dataset[0], dataset[1])
    )
    assert batch["sequence_ids"] == [
        dataset[0]["sequence_id"],
        dataset[1]["sequence_id"],
    ]
    assert batch["splits"] == ["train", "train"]

    collator = make_collate_fn(dataset.pad_token_id)
    repeated = collator([dataset[0], dataset[1]])
    assert torch.equal(repeated["input_ids"], batch["input_ids"])
    assert torch.equal(
        repeated["unknown_technique_decision_mask"],
        batch["unknown_technique_decision_mask"],
    )


def test_unlabeled_marks_duration_decisions_without_erasing_real_targets(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    corpus = _make_synthetic_corpus(
        tmp_path,
        split_lengths={"train": (7, 7)},
    )
    pad, bos, eos = corpus.special_ids
    ids = [
        bos,
        corpus.tonic_ids["E"],
        corpus.mode_ids["PHRYGIAN"],
        corpus.bar_id,
        corpus.duration_ids[0],
        corpus.ordinary_ids[1],
        eos,
    ]
    for record in corpus.manifest["sequences"]:
        sequence_path = corpus.root / record["sequence_file"]
        payload = json.loads(sequence_path.read_text(encoding="utf-8"))
        payload["ids"] = list(ids)
        if record["sequence_id"].startswith("train-riff-01"):
            record["technique_coverage"] = "COMPLETE"
            payload["technique_coverage"] = "COMPLETE"
        _publish_sequence_change(corpus, record, payload, sequence_path)

    dataset = TokenizedSequenceDataset(corpus.manifest_path, "train")
    unlabeled = dataset[0]
    complete = dataset[1]

    assert unlabeled["input_ids"] == tuple(ids[:-1])
    assert unlabeled["target_ids"] == tuple(ids[1:])
    assert pad not in unlabeled["target_ids"]
    assert unlabeled["unknown_technique_decision_mask"] == (
        False,
        False,
        False,
        False,
        True,
        False,
    )
    assert complete["unknown_technique_decision_mask"] == (False,) * 6

    batch = collate_token_sequences([unlabeled, complete], pad)

    assert batch["attention_mask"].tolist() == [[True] * 6, [True] * 6]
    assert batch["unknown_technique_decision_mask"].tolist() == [
        [False, False, False, False, True, False],
        [False, False, False, False, False, False],
    ]
    assert batch["target_ids"][0].tolist() == list(ids[1:])
    assert batch["target_ids"][1].tolist() == list(ids[1:])
    assert batch["target_ids"][0, 4].item() == ids[5]
    assert int((batch["target_ids"] != pad).sum().item()) == 12


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("schema", "schema_version 3"),
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
        "unknown_technique_decision_mask": (False, False),
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
    with pytest.raises(
        DatasetContractError,
        match="missing field.*unknown_technique_decision_mask",
    ):
        collate_token_sequences(
            [
                {
                    key: value
                    for key, value in sample.items()
                    if key != "unknown_technique_decision_mask"
                }
            ],
            pad_token_id=0,
        )
    with pytest.raises(DatasetContractError, match="must be boolean"):
        collate_token_sequences(
            [{**sample, "unknown_technique_decision_mask": (False, 1)}],
            pad_token_id=0,
        )
    with pytest.raises(DatasetContractError, match="lengths must match"):
        collate_token_sequences(
            [{**sample, "unknown_technique_decision_mask": (False,)}],
            pad_token_id=0,
        )
    with pytest.raises(DatasetContractError, match="empty batch"):
        collate_token_sequences([], pad_token_id=0)


def test_dataset_accepts_exact_guitar_technique_and_pitch_bend_contract(
    tmp_path: Path,
) -> None:
    corpus = _make_synthetic_corpus(tmp_path)
    _add_valid_guitar_metadata(corpus)

    dataset = TokenizedSequenceDataset(corpus.manifest_path, "train")

    assert len(dataset) == 2
    first = corpus.manifest["sequences"][0]
    assert first["num_technique_tokens"] == 4
    assert first["num_pitch_bend_tokens"] == 2
    assert corpus.manifest["summary"]["techniques"] == {
        "total_tokens": 4,
        "by_type": {
            "DEAD_NOTE": 1,
            "PALM_MUTE_ON": 1,
            "PALM_MUTE_OFF": 1,
            "SLIDE_UP": 0,
            "SLIDE_DOWN": 0,
            "VIBRATO": 1,
        },
        "coverage": {
            "complete_sequences": 1,
            "unlabeled_sequences": 3,
        },
    }
    assert corpus.manifest["summary"]["pitch_bends"] == {
        "total_tokens": 2,
        "sequences_with_pitch_bends": 1,
    }


def test_complete_coverage_can_confirm_an_empty_technique_list(
    tmp_path: Path,
) -> None:
    corpus = _make_synthetic_corpus(tmp_path)
    record, payload, sequence_path = _first_sequence(corpus)
    record["technique_coverage"] = "COMPLETE"
    payload["technique_coverage"] = "COMPLETE"
    _publish_sequence_change(corpus, record, payload, sequence_path)

    dataset = TokenizedSequenceDataset(corpus.manifest_path, "train")

    assert len(dataset) == 2
    assert corpus.manifest["summary"]["techniques"]["coverage"] == {
        "complete_sequences": 1,
        "unlabeled_sequences": 3,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("type", "tokenizer.type 'ConditionedGuitarREMI'"),
        ("missing_ids", "missing field"),
        ("unknown_id", "unknown field"),
        ("duplicate_ids", "IDs must be distinct"),
        ("wrong_id", "do not match the manifest"),
        ("sensitivity", "must be 6"),
        ("conditioning_schema", "conditioning_schema_version must be 1"),
        ("missing_tonic", "missing field"),
        ("wrong_mode", "do not match the manifest"),
        ("cross_duplicate", "mutually distinct"),
    ],
)
def test_guitar_tokenizer_manifest_contract_is_strict(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    corpus = _make_synthetic_corpus(tmp_path)
    tokenizer = corpus.manifest["tokenizer"]
    if mutation == "type":
        tokenizer["type"] = "REMI"
    elif mutation == "missing_ids":
        tokenizer["technique_token_ids"].pop("VIBRATO")
    elif mutation == "unknown_id":
        tokenizer["technique_token_ids"]["UNKNOWN"] = 999
    elif mutation == "duplicate_ids":
        tokenizer["technique_token_ids"]["VIBRATO"] = tokenizer[
            "technique_token_ids"
        ]["DEAD_NOTE"]
    elif mutation == "wrong_id":
        tokenizer["technique_token_ids"]["VIBRATO"] = max(
            tokenizer["technique_token_ids"].values()
        ) + 100
    elif mutation == "sensitivity":
        tokenizer["pitch_bend_sensitivity_semitones"] = 5
    elif mutation == "conditioning_schema":
        tokenizer["conditioning_schema_version"] = 2
    elif mutation == "missing_tonic":
        tokenizer["tonic_token_ids"].pop("E")
    elif mutation == "wrong_mode":
        tokenizer["mode_token_ids"]["BLUES"] += 1000
    elif mutation == "cross_duplicate":
        tokenizer["mode_token_ids"]["BLUES"] = tokenizer["tonic_token_ids"]["E"]
    else:  # pragma: no cover
        raise AssertionError(mutation)
    corpus.write_manifest()

    with pytest.raises(DatasetContractError, match=message):
        TokenizedSequenceDataset(corpus.manifest_path, "train")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_tonic", "fixed prefix positions"),
        ("duplicate_tonic", "fixed prefix positions"),
        ("swapped", "fixed prefix positions"),
        ("metadata", "fixed prefix positions"),
        ("nonbar", "immediately after Mode"),
    ],
)
def test_condition_prefix_exactly_matches_tonality_metadata(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    corpus = _make_synthetic_corpus(tmp_path)
    record, payload, sequence_path = _first_sequence(corpus)
    if mutation == "wrong_tonic":
        payload["ids"][1] = corpus.tonic_ids["G"]
    elif mutation == "duplicate_tonic":
        payload["ids"].insert(-1, corpus.tonic_ids["G"])
    elif mutation == "swapped":
        payload["ids"][1], payload["ids"][2] = (
            payload["ids"][2],
            payload["ids"][1],
        )
    elif mutation == "metadata":
        changed = {**record["tonality"], "tonic": "G"}
        record["tonality"] = deepcopy(changed)
        payload["tonality"] = deepcopy(changed)
    elif mutation == "nonbar":
        payload["ids"][3] = corpus.duration_ids[0]
    else:  # pragma: no cover
        raise AssertionError(mutation)
    _publish_sequence_change(corpus, record, payload, sequence_path)

    with pytest.raises(DatasetContractError, match=message):
        TokenizedSequenceDataset(corpus.manifest_path, "train")


def test_preprocessing_and_tokenizer_artifact_schema_versions_are_exact(
    tmp_path: Path,
) -> None:
    corpus = _make_synthetic_corpus(tmp_path)
    corpus.manifest["preprocessing"]["schema_version"] = 3
    corpus.write_manifest()
    with pytest.raises(DatasetContractError, match="must be 4"):
        TokenizedSequenceDataset(corpus.manifest_path, "train")

    embedded_root = tmp_path / "embedded"
    embedded_root.mkdir()
    corpus = _make_synthetic_corpus(embedded_root)
    tokenizer_path = corpus.root / corpus.manifest["tokenizer"]["path"]
    tokenizer_payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer_payload["tokenization_schema_version"] = 1
    _write_json(tokenizer_path, tokenizer_payload)
    tokenizer_sha256, tokenizer_size = _fingerprint(tokenizer_path)
    corpus.manifest["tokenizer"]["sha256"] = tokenizer_sha256
    corpus.manifest["tokenizer"]["size_bytes"] = tokenizer_size
    corpus.write_manifest()
    with pytest.raises(DatasetContractError, match="Stage 2 schema 3"):
        TokenizedSequenceDataset(corpus.manifest_path, "train")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("coverage_value", "must be one of UNLABELED, COMPLETE"),
        ("coverage_mismatch", "coverage does not match"),
        ("techniques_mismatch", "techniques do not match"),
        ("unlabeled_nonempty", "must be empty"),
        ("noncanonical", "canonical note/type order"),
        ("duplicate", "duplicate technique"),
        ("out_of_range", "outside the 3-note sequence"),
        ("boolean_index", "must be an integer"),
        ("unknown_key", "unknown field"),
        ("slide_conflict", "cannot use SLIDE_UP and SLIDE_DOWN"),
        ("palm_off_without_on", "without an active mute"),
    ],
)
def test_record_and_payload_technique_annotations_are_strict(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    corpus = _make_synthetic_corpus(tmp_path)
    _add_valid_guitar_metadata(corpus)
    record, payload, sequence_path = _first_sequence(corpus)
    if mutation == "coverage_value":
        record["technique_coverage"] = "PARTIAL"
    elif mutation == "coverage_mismatch":
        payload["technique_coverage"] = "UNLABELED"
    elif mutation == "techniques_mismatch":
        payload["techniques"] = payload["techniques"][:-1]
    elif mutation == "unlabeled_nonempty":
        record["technique_coverage"] = "UNLABELED"
        payload["technique_coverage"] = "UNLABELED"
    elif mutation == "noncanonical":
        record["techniques"] = list(reversed(record["techniques"]))
        payload["techniques"] = deepcopy(record["techniques"])
    elif mutation == "duplicate":
        record["techniques"].append(deepcopy(record["techniques"][0]))
        payload["techniques"] = deepcopy(record["techniques"])
    elif mutation == "out_of_range":
        record["techniques"][0]["note_index"] = 3
        payload["techniques"] = deepcopy(record["techniques"])
    elif mutation == "boolean_index":
        record["techniques"][0]["note_index"] = True
        payload["techniques"] = deepcopy(record["techniques"])
    elif mutation == "unknown_key":
        record["techniques"][0]["extra"] = 1
        payload["techniques"] = deepcopy(record["techniques"])
    elif mutation == "slide_conflict":
        annotations = [
            {"type": "SLIDE_UP", "note_index": 0},
            {"type": "SLIDE_DOWN", "note_index": 0},
        ]
        record["techniques"] = deepcopy(annotations)
        payload["techniques"] = deepcopy(annotations)
    elif mutation == "palm_off_without_on":
        annotations = [{"type": "PALM_MUTE_OFF", "note_index": 0}]
        record["techniques"] = deepcopy(annotations)
        payload["techniques"] = deepcopy(annotations)
    else:  # pragma: no cover
        raise AssertionError(mutation)
    record["num_technique_tokens"] = len(record["techniques"])
    _publish_sequence_change(corpus, record, payload, sequence_path)

    with pytest.raises(DatasetContractError, match=message):
        TokenizedSequenceDataset(corpus.manifest_path, "train")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_token", "token IDs do not match"),
        ("extra_token", "token IDs do not match"),
        ("wrong_type", "token IDs do not match"),
        ("wrong_count", "num_technique_tokens does not match"),
    ],
)
def test_sequence_ids_exactly_match_declared_techniques(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    corpus = _make_synthetic_corpus(tmp_path)
    _add_valid_guitar_metadata(corpus)
    record, payload, sequence_path = _first_sequence(corpus)
    if mutation == "missing_token":
        payload["ids"].remove(corpus.technique_ids["DEAD_NOTE"])
    elif mutation == "extra_token":
        payload["ids"].insert(-1, corpus.technique_ids["SLIDE_UP"])
    elif mutation == "wrong_type":
        index = payload["ids"].index(corpus.technique_ids["DEAD_NOTE"])
        payload["ids"][index] = corpus.technique_ids["SLIDE_UP"]
    elif mutation == "wrong_count":
        record["num_technique_tokens"] += 1
    else:  # pragma: no cover
        raise AssertionError(mutation)
    _publish_sequence_change(corpus, record, payload, sequence_path)

    with pytest.raises(DatasetContractError, match=message):
        TokenizedSequenceDataset(corpus.manifest_path, "train")


@pytest.mark.parametrize("mutation", ["extra_id", "wrong_count"])
def test_pitch_bend_count_is_derived_from_vocabulary_tokens(
    tmp_path: Path,
    mutation: str,
) -> None:
    corpus = _make_synthetic_corpus(tmp_path)
    _add_valid_guitar_metadata(corpus)
    record, payload, sequence_path = _first_sequence(corpus)
    if mutation == "extra_id":
        payload["ids"].insert(-1, corpus.pitch_bend_ids[1])
    else:
        record["num_pitch_bend_tokens"] += 1
    _publish_sequence_change(corpus, record, payload, sequence_path)

    with pytest.raises(DatasetContractError, match="num_pitch_bend_tokens"):
        TokenizedSequenceDataset(corpus.manifest_path, "train")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("technique_total", "techniques.total_tokens"),
        ("technique_type", "by_type.DEAD_NOTE"),
        ("coverage", "techniques.coverage"),
        ("pitch_total", "pitch_bends"),
        ("pitch_sequences", "pitch_bends"),
        ("tonic", "tonality.by_tonic.E"),
        ("method", "tonality.by_method.MANUAL"),
        ("unknown", "unknown field"),
    ],
)
def test_schema_three_summary_is_recomputed_strictly(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    corpus = _make_synthetic_corpus(tmp_path)
    _add_valid_guitar_metadata(corpus)
    summary = corpus.manifest["summary"]
    if mutation == "technique_total":
        summary["techniques"]["total_tokens"] += 1
    elif mutation == "technique_type":
        summary["techniques"]["by_type"]["DEAD_NOTE"] += 1
    elif mutation == "coverage":
        summary["techniques"]["coverage"]["complete_sequences"] += 1
    elif mutation == "pitch_total":
        summary["pitch_bends"]["total_tokens"] += 1
    elif mutation == "pitch_sequences":
        summary["pitch_bends"]["sequences_with_pitch_bends"] += 1
    elif mutation == "tonic":
        summary["tonality"]["by_tonic"]["E"] += 1
    elif mutation == "method":
        summary["tonality"]["by_method"]["MANUAL"] += 1
    elif mutation == "unknown":
        summary["techniques"]["unexpected"] = 0
    else:  # pragma: no cover
        raise AssertionError(mutation)
    corpus.write_manifest()

    with pytest.raises(DatasetContractError, match=message):
        TokenizedSequenceDataset(corpus.manifest_path, "train")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("payload_schema", "schema_version 3"),
        ("payload_missing", "missing field"),
        ("record_missing", "missing field"),
    ],
)
def test_sequence_schema_three_requires_all_new_fields(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    corpus = _make_synthetic_corpus(tmp_path)
    record, payload, sequence_path = _first_sequence(corpus)
    if mutation == "payload_schema":
        payload["schema_version"] = 2
    elif mutation == "payload_missing":
        payload.pop("technique_coverage")
    elif mutation == "record_missing":
        record.pop("num_pitch_bend_tokens")
    else:  # pragma: no cover
        raise AssertionError(mutation)
    if mutation == "record_missing":
        corpus.write_manifest()
    else:
        _publish_sequence_change(corpus, record, payload, sequence_path)

    with pytest.raises(DatasetContractError, match=message):
        TokenizedSequenceDataset(corpus.manifest_path, "train")
