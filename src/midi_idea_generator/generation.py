"""Grammar-constrained unconditional sampling for the Stage 3 GRU."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
from numbers import Integral
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import torch
from miditok import REMI
from torch import Tensor

from .generation_artifacts import GenerationArtifacts, write_generation_artifacts
from .generation_checkpoint import GenerationBundle, load_generation_bundle
from .generation_config import GenerationConfig, SamplingConfig
from .tokenizer import (
    DecodedMidi,
    TokenizationError,
    canonicalize_symbolic_token_ids,
    get_technique_token_ids,
)
from .utils import relative_label, write_json


MAX_SIMULTANEOUS_GUITAR_NOTES = 6
_SPECIAL_TYPES = {"PAD", "BOS", "EOS"}
_TECHNIQUE_INCOMPATIBLE_WITH_DEAD = {"SLIDE_UP", "SLIDE_DOWN", "VIBRATO"}
LOGGER = logging.getLogger(__name__)
GENERATION_MANIFEST_SCHEMA_VERSION = 1


class GenerationError(RuntimeError):
    """Raised when a requested unconditional sample cannot be produced safely."""


class _AttemptRejected(ValueError):
    """Internal marker for one invalid stochastic attempt."""


@dataclass(frozen=True, slots=True)
class SampledSequence:
    """One validated unconditional sample and its exact sampling provenance."""

    sample_index: int
    sample_seed: int
    attempts_used: int
    raw_token_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    tokens: tuple[str, ...]
    decoded: DecodedMidi

    @property
    def num_tokens(self) -> int:
        """Return the canonical serialized length, including BOS and EOS."""

        return len(self.token_ids)

    @property
    def num_notes(self) -> int:
        """Return the decoded note count."""

        return len(self.decoded.score.tracks[0].notes)

    @property
    def num_pitch_bends(self) -> int:
        """Return the decoded pitch-bend event count."""

        return len(self.decoded.score.tracks[0].pitch_bends)

    @property
    def num_techniques(self) -> int:
        """Return the decoded symbolic-technique annotation count."""

        return len(self.decoded.techniques)


@dataclass(frozen=True, slots=True)
class PublishedSample:
    """Final paths and statistics for one published generation."""

    sample_index: int
    sample_seed: int
    attempts_used: int
    midi_path: Path
    tokens_path: Path
    techniques_path: Path
    visualization_path: Path | None
    num_tokens: int
    num_notes: int
    num_pitch_bends: int
    num_techniques: int


@dataclass(frozen=True, slots=True)
class GenerationReport:
    """One atomically published unconditional-generation run."""

    generation_run_id: str
    device: str
    output_dir: Path
    manifest_path: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    training_run_id: str
    epoch: int
    samples: tuple[PublishedSample, ...]


@dataclass(frozen=True, slots=True)
class _VocabularyGrammar:
    tokenizer: REMI
    vocabulary_size: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    token_by_id: tuple[str, ...]
    type_by_id: tuple[str, ...]
    value_by_id: tuple[str, ...]
    ids_by_type: Mapping[str, tuple[int, ...]]
    successor_types: Mapping[str, frozenset[str]]
    technique_by_id: Mapping[int, str]


def sample_unconditional(
    bundle: GenerationBundle,
    sampling: SamplingConfig,
    *,
    program: int,
    sample_index: int,
    base_seed: int,
) -> SampledSequence:
    """Sample one sequence from BOS with bounded retries and strict decoding.

    There is no seed MIDI or musical prefix.  The fixed ``Bar / Position_0 /
    Tempo`` preamble is format scaffolding shared by every training sequence;
    all musical events are sampled autoregressively from the trained model.
    """

    if isinstance(program, bool) or not isinstance(program, int) or not 0 <= program <= 127:
        raise GenerationError("MIDI program must be an integer between 0 and 127.")
    if (
        isinstance(sample_index, bool)
        or not isinstance(sample_index, int)
        or sample_index < 1
    ):
        raise GenerationError("sample_index must be a positive integer.")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise GenerationError("Generation seed must be a non-negative integer.")

    grammar = _build_vocabulary_grammar(bundle)
    rejections: Counter[str] = Counter()
    for attempt in range(1, sampling.max_attempts_per_sample + 1):
        sample_seed = _attempt_seed(base_seed, sample_index, attempt)
        try:
            raw_ids = _sample_attempt(bundle, grammar, sampling, sample_seed)
            canonical_ids, decoded = canonicalize_symbolic_token_ids(
                bundle.tokenizer,
                raw_ids,
                ((program, False),),
            )
            _validate_decoded_sample(decoded)
        except (TokenizationError, _AttemptRejected) as exc:
            rejections[_concise_rejection(exc)] += 1
            continue
        tokens = tuple(grammar.token_by_id[token_id] for token_id in canonical_ids)
        return SampledSequence(
            sample_index=sample_index,
            sample_seed=sample_seed,
            attempts_used=attempt,
            raw_token_ids=raw_ids,
            token_ids=canonical_ids,
            tokens=tokens,
            decoded=decoded,
        )

    summary = "; ".join(
        f"{reason} ({count})" for reason, count in rejections.most_common(3)
    )
    detail = f" Rejections: {summary}." if summary else ""
    raise GenerationError(
        f"Could not produce valid sample {sample_index} after "
        f"{sampling.max_attempts_per_sample} attempts.{detail}"
    )


def run_generation(config: GenerationConfig) -> GenerationReport:
    """Load the exact trained bundle and atomically publish all requested samples."""

    bundle = load_generation_bundle(
        config.paths.checkpoint_path,
        config.paths.tokenization_manifest_path,
        config.project_root,
        config.device,
    )
    output_root = config.paths.output_dir
    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise GenerationError(
            f"Could not create generation output directory '{output_root}': {exc}"
        ) from exc
    if output_root.is_symlink() or not output_root.is_dir():
        raise GenerationError("Generation output root must be a regular directory.")

    created_at = datetime.now(UTC)
    run_id = _generation_run_id(config, bundle, created_at)
    final_dir = output_root / run_id
    if final_dir.exists() or final_dir.is_symlink():
        raise GenerationError(f"Generation run directory already exists: {final_dir}")
    try:
        staging_dir = Path(
            tempfile.mkdtemp(prefix=f".{run_id}-", dir=output_root)
        )
    except OSError as exc:
        raise GenerationError(
            f"Could not allocate generation staging directory: {exc}"
        ) from exc

    sample_records: list[dict[str, object]] = []
    staged_samples: list[tuple[SampledSequence, GenerationArtifacts]] = []
    try:
        for sample_index in range(1, config.generation.num_samples + 1):
            sampled = sample_unconditional(
                bundle,
                config.generation,
                program=config.midi.program,
                sample_index=sample_index,
                base_seed=config.seed,
            )
            LOGGER.info(
                "Generated sample %d/%d after %d attempt(s): %d tokens, %d notes",
                sample_index,
                config.generation.num_samples,
                sampled.attempts_used,
                sampled.num_tokens,
                sampled.num_notes,
            )
            stem = staging_dir / f"sample-{sample_index:03d}"
            provenance = _sample_provenance(config, bundle, sampled, run_id)
            artifacts = write_generation_artifacts(
                sampled.decoded,
                sampled.token_ids,
                sampled.tokens,
                provenance,
                stem,
                program=config.midi.program,
                visualization_enabled=config.visualization.enabled,
                dpi=config.visualization.dpi,
            )
            staged_samples.append((sampled, artifacts))
            sample_records.append(_sample_manifest_record(sampled, artifacts))

        manifest_payload = _generation_manifest(
            config,
            bundle,
            run_id=run_id,
            created_at=created_at,
            samples=sample_records,
        )
        write_json(staging_dir / "manifest.json", manifest_payload)
        os.rename(staging_dir, final_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    published = tuple(
        _published_sample(final_dir, sampled, artifacts)
        for sampled, artifacts in staged_samples
    )
    return GenerationReport(
        generation_run_id=run_id,
        device=str(bundle.device),
        output_dir=final_dir,
        manifest_path=final_dir / "manifest.json",
        checkpoint_path=bundle.checkpoint_path,
        checkpoint_sha256=bundle.checkpoint_sha256,
        training_run_id=bundle.training_run_id,
        epoch=bundle.epoch,
        samples=published,
    )


def _build_vocabulary_grammar(bundle: GenerationBundle) -> _VocabularyGrammar:
    vocabulary = bundle.tokenizer.vocab
    if not isinstance(vocabulary, Mapping):
        raise GenerationError("Generation requires one flat tokenizer vocabulary.")
    token_by_id: list[str | None] = [None] * bundle.vocabulary_size
    for token, token_id in vocabulary.items():
        if not isinstance(token, str):
            raise GenerationError("Tokenizer vocabulary contains a non-string token.")
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, Integral)
            or not 0 <= int(token_id) < bundle.vocabulary_size
            or token_by_id[int(token_id)] is not None
        ):
            raise GenerationError("Tokenizer vocabulary IDs are not contiguous and unique.")
        token_by_id[int(token_id)] = token
    if any(token is None for token in token_by_id):
        raise GenerationError("Tokenizer vocabulary IDs are not contiguous and unique.")
    resolved_tokens = tuple(str(token) for token in token_by_id)
    split = tuple(_split_token(token) for token in resolved_tokens)
    type_by_id = tuple(item[0] for item in split)
    value_by_id = tuple(item[1] for item in split)
    grouped: dict[str, list[int]] = defaultdict(list)
    for token_id, token_type in enumerate(type_by_id):
        grouped[token_type].append(token_id)

    graph = bundle.tokenizer.tokens_types_graph
    if not isinstance(graph, Mapping):
        raise GenerationError("Tokenizer does not expose a token-type graph.")
    successor_types: dict[str, frozenset[str]] = {}
    for token_type, successors in graph.items():
        if isinstance(token_type, str) and isinstance(successors, (set, list, tuple)):
            successor_types[token_type] = frozenset(
                successor
                for successor in successors
                if isinstance(successor, str) and successor in grouped
            )

    technique_by_id = {
        int(token_id): technique
        for technique, token_id in get_technique_token_ids(bundle.tokenizer).items()
    }
    required = {"Bar", "Position", "Tempo", "Pitch", "Duration"}
    missing = sorted(required - set(grouped))
    if missing:
        raise GenerationError(
            "Tokenizer is missing required token types: " + ", ".join(missing) + "."
        )
    if "Position_0" not in vocabulary:
        raise GenerationError("Tokenizer must define Position_0 for generation.")
    return _VocabularyGrammar(
        tokenizer=bundle.tokenizer,
        vocabulary_size=bundle.vocabulary_size,
        pad_token_id=bundle.pad_token_id,
        bos_token_id=bundle.bos_token_id,
        eos_token_id=bundle.eos_token_id,
        token_by_id=resolved_tokens,
        type_by_id=type_by_id,
        value_by_id=value_by_id,
        ids_by_type={key: tuple(value) for key, value in grouped.items()},
        successor_types=successor_types,
        technique_by_id=technique_by_id,
    )


def _sample_attempt(
    bundle: GenerationBundle,
    grammar: _VocabularyGrammar,
    sampling: SamplingConfig,
    sample_seed: int,
) -> tuple[int, ...]:
    if sampling.max_tokens < 7:
        raise _AttemptRejected("max_tokens is too small for a complete REMI phrase")

    ids = [grammar.bos_token_id]
    hidden: Tensor | None = None
    generator_device = bundle.device.type if bundle.device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(sample_seed)
    model = bundle.model
    model.eval()

    with torch.inference_mode():
        while len(ids) < sampling.max_tokens:
            input_ids = torch.tensor(
                [[ids[-1]]], dtype=torch.long, device=bundle.device
            )
            output = model(input_ids, hidden, return_hidden=True)
            if not isinstance(output, tuple) or len(output) != 2:
                raise GenerationError("GRU did not return logits and recurrent state.")
            logits, hidden = output
            if logits.shape != (1, 1, grammar.vocabulary_size):
                raise GenerationError("GRU returned logits with an invalid shape.")
            next_logits = logits[0, -1].float()
            if not torch.isfinite(next_logits).all():
                raise GenerationError("GRU returned non-finite generation logits.")

            forced_type = _closure_type(ids, grammar, sampling)
            if forced_type == "EOS":
                ids.append(grammar.eos_token_id)
                break
            allowed = _allowed_token_ids(ids, grammar, sampling)
            if forced_type is not None:
                allowed = tuple(
                    token_id
                    for token_id in allowed
                    if grammar.type_by_id[token_id] == forced_type
                )
            if not allowed:
                raise _AttemptRejected("no legal continuation is available")
            next_id = _draw_token(
                next_logits,
                allowed,
                ids,
                sampling,
                generator,
            )
            ids.append(next_id)
            if next_id == grammar.eos_token_id:
                break

    if ids[-1] != grammar.eos_token_id:
        raise _AttemptRejected("sequence reached max_tokens without a legal EOS")
    if len(ids) > sampling.max_tokens:
        raise _AttemptRejected("sequence exceeded max_tokens during closure")
    return tuple(ids)


def _allowed_token_ids(
    ids: Sequence[int],
    grammar: _VocabularyGrammar,
    sampling: SamplingConfig,
) -> tuple[int, ...]:
    musical_ids = ids[1:]
    if not musical_ids:
        return grammar.ids_by_type["Bar"]
    if len(musical_ids) == 1:
        position_zero = grammar.tokenizer.vocab["Position_0"]
        return (int(position_zero),)
    if len(musical_ids) == 2:
        return grammar.ids_by_type["Tempo"]

    previous_type = grammar.type_by_id[ids[-1]]
    types = set(grammar.successor_types.get(previous_type, ()))
    types.difference_update(_SPECIAL_TYPES | {"Tempo"})
    if previous_type == "Bar":
        types = {"Position"}

    bar_content = _tokens_since_bar(musical_ids, grammar)
    last_position = _last_position(bar_content, grammar)
    pitches = _pitches_at_current_position(bar_content, grammar)
    if len(pitches) >= sampling.max_simultaneous_notes:
        types.discard("Pitch")

    allowed = [
        token_id
        for token_type in types
        for token_id in grammar.ids_by_type.get(token_type, ())
    ]
    filtered: list[int] = []
    current_techniques = _current_note_techniques(musical_ids, grammar)
    palm_muted = _palm_mute_state(musical_ids, grammar)
    for token_id in allowed:
        token_type = grammar.type_by_id[token_id]
        if token_type == "Position":
            value = _integer_token_value(grammar.value_by_id[token_id], "Position")
            if value <= last_position:
                continue
        elif token_type == "Pitch":
            value = _integer_token_value(grammar.value_by_id[token_id], "Pitch")
            if value in pitches:
                continue
        elif token_type == "Technique" and not _technique_is_allowed(
            token_id,
            grammar,
            current_techniques=current_techniques,
            palm_muted=palm_muted,
        ):
            continue
        filtered.append(token_id)

    if (
        len(musical_ids) >= sampling.min_tokens
        and previous_type == "Bar"
        and len(ids) < sampling.max_tokens
    ):
        filtered.append(grammar.eos_token_id)
    return tuple(sorted(set(filtered)))


def _closure_type(
    ids: Sequence[int],
    grammar: _VocabularyGrammar,
    sampling: SamplingConfig,
) -> str | None:
    previous_type = grammar.type_by_id[ids[-1]]
    musical_count = len(ids) - 1
    if musical_count < sampling.min_tokens:
        return None
    steps = {
        "Bar": 1,
        "Duration": 2,
        "Technique": 2,
        "PitchBend": 2,
        "Pitch": 3,
        "Position": 4,
        "Tempo": 4,
    }.get(previous_type)
    if steps is None:
        raise _AttemptRejected(
            f"cannot close a sequence after token type {previous_type}"
        )
    if len(ids) + steps < sampling.max_tokens:
        return None
    return {
        "Bar": "EOS",
        "Duration": "Bar",
        "Technique": "Bar",
        "PitchBend": "Bar",
        "Pitch": "Duration",
        "Position": "Pitch",
        "Tempo": "Pitch",
    }[previous_type]


def _draw_token(
    logits: Tensor,
    allowed: Sequence[int],
    prior_ids: Sequence[int],
    sampling: SamplingConfig,
    generator: torch.Generator,
) -> int:
    adjusted = logits.clone()
    if sampling.repetition_penalty > 1.0:
        repeated = torch.tensor(
            sorted(set(prior_ids)), dtype=torch.long, device=adjusted.device
        )
        values = adjusted[repeated]
        adjusted[repeated] = torch.where(
            values < 0,
            values * sampling.repetition_penalty,
            values / sampling.repetition_penalty,
        )
    adjusted /= sampling.temperature

    mask = torch.ones_like(adjusted, dtype=torch.bool)
    allowed_tensor = torch.tensor(
        tuple(allowed), dtype=torch.long, device=adjusted.device
    )
    mask[allowed_tensor] = False
    adjusted[mask] = -torch.inf
    finite_count = int(torch.isfinite(adjusted).sum().item())
    if finite_count == 0:
        raise _AttemptRejected("all legal continuation logits were filtered")

    if sampling.top_k > 0 and finite_count > sampling.top_k:
        threshold = torch.topk(adjusted, sampling.top_k).values[-1]
        adjusted[adjusted < threshold] = -torch.inf
    probabilities = torch.softmax(adjusted, dim=0)
    if sampling.top_p < 1.0:
        sorted_probabilities, sorted_ids = torch.sort(
            probabilities, descending=True
        )
        cumulative_before = torch.cumsum(sorted_probabilities, dim=0) - (
            sorted_probabilities
        )
        remove = cumulative_before >= sampling.top_p
        adjusted[sorted_ids[remove]] = -torch.inf
        probabilities = torch.softmax(adjusted, dim=0)
    if not torch.isfinite(probabilities).all() or probabilities.sum().item() <= 0:
        raise _AttemptRejected("sampling probabilities are invalid")
    sampled = torch.multinomial(probabilities, 1, generator=generator)
    return int(sampled.item())


def _tokens_since_bar(
    musical_ids: Sequence[int], grammar: _VocabularyGrammar
) -> tuple[int, ...]:
    content: list[int] = []
    for token_id in reversed(musical_ids):
        if grammar.type_by_id[token_id] == "Bar":
            break
        content.append(token_id)
    content.reverse()
    return tuple(content)


def _last_position(
    bar_content: Sequence[int], grammar: _VocabularyGrammar
) -> int:
    for token_id in reversed(bar_content):
        if grammar.type_by_id[token_id] == "Position":
            return _integer_token_value(grammar.value_by_id[token_id], "Position")
    return -1


def _pitches_at_current_position(
    bar_content: Sequence[int], grammar: _VocabularyGrammar
) -> frozenset[int]:
    pitches: set[int] = set()
    for token_id in reversed(bar_content):
        token_type = grammar.type_by_id[token_id]
        if token_type == "Position":
            break
        if token_type == "Pitch":
            pitches.add(_integer_token_value(grammar.value_by_id[token_id], "Pitch"))
    return frozenset(pitches)


def _current_note_techniques(
    musical_ids: Sequence[int], grammar: _VocabularyGrammar
) -> frozenset[str]:
    techniques: set[str] = set()
    for token_id in reversed(musical_ids):
        if grammar.type_by_id[token_id] != "Technique":
            break
        technique = grammar.technique_by_id.get(token_id)
        if technique is None:
            raise GenerationError("Tokenizer contains an unknown Technique token.")
        techniques.add(technique)
    return frozenset(techniques)


def _palm_mute_state(
    musical_ids: Sequence[int], grammar: _VocabularyGrammar
) -> bool:
    palm_muted = False
    for token_id in musical_ids:
        technique = grammar.technique_by_id.get(token_id)
        if technique == "PALM_MUTE_ON":
            palm_muted = True
        elif technique == "PALM_MUTE_OFF":
            palm_muted = False
    return palm_muted


def _technique_is_allowed(
    token_id: int,
    grammar: _VocabularyGrammar,
    *,
    current_techniques: frozenset[str],
    palm_muted: bool,
) -> bool:
    technique = grammar.technique_by_id.get(token_id)
    if technique is None or technique in current_techniques:
        return False
    if technique == "SLIDE_UP" and "SLIDE_DOWN" in current_techniques:
        return False
    if technique == "SLIDE_DOWN" and "SLIDE_UP" in current_techniques:
        return False
    if technique == "PALM_MUTE_ON" and palm_muted:
        return False
    if technique == "PALM_MUTE_OFF" and not palm_muted:
        return False
    if technique == "DEAD_NOTE" and current_techniques.intersection(
        _TECHNIQUE_INCOMPATIBLE_WITH_DEAD
    ):
        return False
    if (
        technique in _TECHNIQUE_INCOMPATIBLE_WITH_DEAD
        and "DEAD_NOTE" in current_techniques
    ):
        return False
    return True


def _validate_decoded_sample(decoded: DecodedMidi) -> None:
    if len(decoded.score.tracks) != 1:
        raise _AttemptRejected("decoded sample does not contain exactly one track")
    track = decoded.score.tracks[0]
    if not track.notes:
        raise _AttemptRejected("decoded sample contains no notes")
    if bool(track.is_drum):
        raise _AttemptRejected("decoded sample unexpectedly uses a drum track")
    active_end_by_pitch: dict[int, int] = {}
    for note in sorted(
        track.notes,
        key=lambda item: (
            int(item.time),
            int(item.pitch),
            int(item.end),
            int(item.velocity),
        ),
    ):
        pitch = int(note.pitch)
        onset = int(note.time)
        if onset < active_end_by_pitch.get(pitch, onset):
            raise _AttemptRejected(
                "decoded sample contains overlapping notes of the same pitch"
            )
        active_end_by_pitch[pitch] = int(note.end)


def _split_token(token: str) -> tuple[str, str]:
    if "_" not in token:
        raise GenerationError(f"Tokenizer token has no type separator: {token!r}.")
    token_type, value = token.split("_", maxsplit=1)
    if not token_type or not value:
        raise GenerationError(f"Tokenizer token is malformed: {token!r}.")
    return token_type, value


def _integer_token_value(value: str, token_type: str) -> int:
    try:
        converted = int(value)
    except ValueError as exc:
        raise GenerationError(
            f"{token_type} token value is not an integer: {value!r}."
        ) from exc
    return converted


def _attempt_seed(base_seed: int, sample_index: int, attempt: int) -> int:
    # Large odd strides make each sample/attempt an independent reproducible
    # stream while keeping the value inside PyTorch's signed 64-bit seed range.
    return int(
        (base_seed + (sample_index - 1) * 1_000_003 + (attempt - 1) * 10_007)
        % (2**63 - 1)
    )


def _concise_rejection(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    return message[:180] if message else type(exc).__name__


def _generation_run_id(
    config: GenerationConfig,
    bundle: GenerationBundle,
    created_at: datetime,
) -> str:
    identity = {
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "tokenizer_sha256": bundle.tokenizer_sha256,
        "seed": config.seed,
        "sampling": asdict(config.generation),
        "program": config.midi.program,
    }
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    suffix = hashlib.sha256(encoded).hexdigest()[:10]
    timestamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{suffix}"


def _sample_provenance(
    config: GenerationConfig,
    bundle: GenerationBundle,
    sample: SampledSequence,
    run_id: str,
) -> dict[str, object]:
    return {
        "generation_mode": "unconditional",
        "generation_run_id": run_id,
        "sample_index": sample.sample_index,
        "base_seed": config.seed,
        "sample_seed": sample.sample_seed,
        "attempts_used": sample.attempts_used,
        "raw_num_tokens": len(sample.raw_token_ids),
        "canonicalized": sample.raw_token_ids != sample.token_ids,
        "checkpoint": {
            "file": relative_label(bundle.checkpoint_path, config.project_root),
            "sha256": bundle.checkpoint_sha256,
            "training_run_id": bundle.training_run_id,
            "epoch": bundle.epoch,
            "best_epoch": bundle.best_epoch,
            "best_validation_loss": bundle.best_validation_loss,
        },
        "tokenization": {
            "manifest": relative_label(bundle.manifest_path, config.project_root),
            "manifest_sha256": bundle.manifest_sha256,
            "tokenizer": relative_label(bundle.tokenizer_path, config.project_root),
            "tokenizer_sha256": bundle.tokenizer_sha256,
            "tokenization_run_id": bundle.tokenization_run_id,
            "vocabulary_size": bundle.vocabulary_size,
        },
        "sampling": asdict(config.generation),
    }


def _sample_manifest_record(
    sample: SampledSequence,
    artifacts: GenerationArtifacts,
) -> dict[str, object]:
    return {
        "sample_index": sample.sample_index,
        "sample_seed": sample.sample_seed,
        "attempts_used": sample.attempts_used,
        "raw_num_tokens": len(sample.raw_token_ids),
        "num_tokens": sample.num_tokens,
        "num_notes": sample.num_notes,
        "num_pitch_bends": sample.num_pitch_bends,
        "num_techniques": sample.num_techniques,
        "artifacts": {
            "midi": _artifact_manifest_reference(artifacts.midi),
            "tokens": _artifact_manifest_reference(artifacts.tokens),
            "techniques": _artifact_manifest_reference(artifacts.techniques),
            "visualization": (
                _artifact_manifest_reference(artifacts.visualization)
                if artifacts.visualization is not None
                else None
            ),
        },
    }


def _artifact_manifest_reference(artifact: object) -> dict[str, object]:
    path = getattr(artifact, "path", None)
    sha256 = getattr(artifact, "sha256", None)
    size_bytes = getattr(artifact, "size_bytes", None)
    if not isinstance(path, Path) or not isinstance(sha256, str) or not isinstance(
        size_bytes, int
    ):
        raise GenerationError("Artifact exporter returned invalid file metadata.")
    return {
        "file": path.name,
        "sha256": sha256,
        "size_bytes": size_bytes,
    }


def _generation_manifest(
    config: GenerationConfig,
    bundle: GenerationBundle,
    *,
    run_id: str,
    created_at: datetime,
    samples: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": GENERATION_MANIFEST_SCHEMA_VERSION,
        "generation_run_id": run_id,
        "generation_mode": "unconditional",
        "created_at_utc": created_at.isoformat(),
        "device": str(bundle.device),
        "checkpoint": {
            "path": relative_label(bundle.checkpoint_path, config.project_root),
            "sha256": bundle.checkpoint_sha256,
            "training_run_id": bundle.training_run_id,
            "epoch": bundle.epoch,
            "best_epoch": bundle.best_epoch,
            "best_validation_loss": bundle.best_validation_loss,
        },
        "tokenization": {
            "manifest_path": relative_label(
                bundle.manifest_path, config.project_root
            ),
            "manifest_sha256": bundle.manifest_sha256,
            "tokenizer_path": relative_label(
                bundle.tokenizer_path, config.project_root
            ),
            "tokenizer_sha256": bundle.tokenizer_sha256,
            "tokenization_run_id": bundle.tokenization_run_id,
            "vocabulary_size": bundle.vocabulary_size,
        },
        "configuration": {
            "seed": config.seed,
            "sampling": asdict(config.generation),
            "midi": asdict(config.midi),
            "visualization": asdict(config.visualization),
        },
        "summary": {
            "num_samples": len(samples),
            "num_notes": sum(int(sample["num_notes"]) for sample in samples),
            "num_pitch_bends": sum(
                int(sample["num_pitch_bends"]) for sample in samples
            ),
            "num_techniques": sum(
                int(sample["num_techniques"]) for sample in samples
            ),
        },
        "samples": [dict(sample) for sample in samples],
    }


def _published_sample(
    final_dir: Path,
    sample: SampledSequence,
    artifacts: GenerationArtifacts,
) -> PublishedSample:
    return PublishedSample(
        sample_index=sample.sample_index,
        sample_seed=sample.sample_seed,
        attempts_used=sample.attempts_used,
        midi_path=final_dir / artifacts.midi.path.name,
        tokens_path=final_dir / artifacts.tokens.path.name,
        techniques_path=final_dir / artifacts.techniques.path.name,
        visualization_path=(
            final_dir / artifacts.visualization.path.name
            if artifacts.visualization is not None
            else None
        ),
        num_tokens=sample.num_tokens,
        num_notes=sample.num_notes,
        num_pitch_bends=sample.num_pitch_bends,
        num_techniques=sample.num_techniques,
    )


__all__ = [
    "GENERATION_MANIFEST_SCHEMA_VERSION",
    "GenerationError",
    "GenerationReport",
    "MAX_SIMULTANEOUS_GUITAR_NOTES",
    "PublishedSample",
    "SampledSequence",
    "run_generation",
    "sample_unconditional",
]
