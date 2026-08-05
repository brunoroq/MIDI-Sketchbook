"""Tests for grammar-constrained unconditional sampling."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from midi_idea_generator.generation import (
    GenerationError,
    _build_vocabulary_grammar,
    _draw_token,
    sample_unconditional,
)
from midi_idea_generator.generation_checkpoint import GenerationBundle
from midi_idea_generator.generation_config import SamplingConfig
from midi_idea_generator.tokenization_config import RemiTokenizerConfig
from midi_idea_generator.tokenizer import build_tokenizer, get_special_token_ids


class _ScriptedModel:
    def __init__(self, vocabulary_size: int, targets: list[int]) -> None:
        self.vocabulary_size = vocabulary_size
        self.targets = targets
        self.calls = 0
        self.training = True

    def eval(self) -> "_ScriptedModel":
        self.training = False
        return self

    def __call__(
        self,
        input_ids: torch.Tensor,
        hidden: torch.Tensor | None,
        *,
        return_hidden: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert return_hidden is True
        assert input_ids.shape == (1, 1)
        logits = torch.full((1, 1, self.vocabulary_size), -1000.0)
        target = self.targets[min(self.calls, len(self.targets) - 1)]
        logits[0, 0, target] = 1000.0
        self.calls += 1
        return logits, torch.zeros((1, 1, 1))


def _token_with_prefix(vocabulary: dict[str, int], prefix: str) -> int:
    return next(token_id for token, token_id in vocabulary.items() if token.startswith(prefix))


def _bundle_and_targets() -> tuple[GenerationBundle, list[int]]:
    tokenizer = build_tokenizer(RemiTokenizerConfig())
    special = get_special_token_ids(tokenizer)
    vocabulary = tokenizer.vocab
    targets = [
        vocabulary["Bar_None"],
        vocabulary["Position_0"],
        _token_with_prefix(vocabulary, "Tempo_"),
        vocabulary["Pitch_60"],
        _token_with_prefix(vocabulary, "Duration_1.0."),
        vocabulary["Bar_None"],
        special.eos,
    ]
    model = _ScriptedModel(len(vocabulary), targets)
    bundle = GenerationBundle(
        model=model,  # type: ignore[arg-type]
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        checkpoint_path=Path("checkpoint.pt"),
        checkpoint_sha256="0" * 64,
        manifest_path=Path("manifest.json"),
        manifest_sha256="1" * 64,
        tokenizer_path=Path("tokenizer.json"),
        tokenizer_sha256="2" * 64,
        training_run_id="run",
        tokenization_run_id="a" * 20,
        epoch=1,
        best_epoch=1,
        best_validation_loss=1.0,
        vocabulary_size=len(vocabulary),
        pad_token_id=special.pad,
        bos_token_id=special.bos,
        eos_token_id=special.eos,
    )
    return bundle, targets


def _sampling(**overrides: object) -> SamplingConfig:
    values: dict[str, object] = {
        "min_tokens": 6,
        "max_tokens": 20,
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "max_simultaneous_notes": 3,
        "num_samples": 1,
        "max_attempts_per_sample": 2,
    }
    values.update(overrides)
    return SamplingConfig(**values)  # type: ignore[arg-type]


def test_unconditional_sampling_starts_at_bos_and_publishes_canonical_ids() -> None:
    bundle, _ = _bundle_and_targets()

    sample = sample_unconditional(
        bundle,
        _sampling(),
        program=29,
        sample_index=1,
        base_seed=42,
    )

    assert sample.raw_token_ids[0] == bundle.bos_token_id
    assert sample.raw_token_ids[-1] == bundle.eos_token_id
    assert sample.token_ids[0] == bundle.bos_token_id
    assert sample.token_ids[-1] == bundle.eos_token_id
    assert sample.tokens[:4] == (
        "BOS_None",
        "Bar_None",
        "Position_0",
        sample.tokens[3],
    )
    assert sample.tokens[3].startswith("Tempo_")
    assert sample.tokens[-2].startswith("Duration_")
    assert sample.raw_token_ids[-2] == bundle.tokenizer.vocab["Bar_None"]
    assert sample.token_ids[-2] != bundle.tokenizer.vocab["Bar_None"]


def test_unconditional_sampling_decodes_a_nonempty_guitar_track() -> None:
    bundle, _ = _bundle_and_targets()

    sample = sample_unconditional(
        bundle,
        _sampling(),
        program=29,
        sample_index=2,
        base_seed=7,
    )

    track = sample.decoded.score.tracks[0]
    assert track.program == 29
    assert [note.pitch for note in track.notes] == [60]
    assert sample.num_notes == 1
    assert sample.attempts_used == 1
    assert sample.num_pitch_bends == 0
    assert sample.num_techniques == 0


def test_draw_token_never_selects_a_disallowed_high_logit() -> None:
    logits = torch.tensor([100.0, 1.0, 0.0, -1.0])
    generator = torch.Generator(device="cpu").manual_seed(1)

    sampled = _draw_token(
        logits,
        allowed=(1, 2),
        prior_ids=(),
        sampling=_sampling(top_k=1),
        generator=generator,
    )

    assert sampled == 1


def test_generation_rejects_invalid_public_arguments() -> None:
    bundle, _ = _bundle_and_targets()

    with pytest.raises(GenerationError, match="program"):
        sample_unconditional(
            bundle,
            _sampling(),
            program=128,
            sample_index=1,
            base_seed=0,
        )
    with pytest.raises(GenerationError, match="sample_index"):
        sample_unconditional(
            bundle,
            _sampling(),
            program=29,
            sample_index=0,
            base_seed=0,
        )


def test_vocabulary_grammar_resolves_all_model_ids() -> None:
    bundle, _ = _bundle_and_targets()

    grammar = _build_vocabulary_grammar(bundle)

    assert len(grammar.token_by_id) == bundle.vocabulary_size
    assert grammar.token_by_id[grammar.bos_token_id] == "BOS_None"
    assert grammar.ids_by_type["Technique"]
    assert set(grammar.technique_by_id.values()) == {
        "DEAD_NOTE",
        "PALM_MUTE_ON",
        "PALM_MUTE_OFF",
        "SLIDE_UP",
        "SLIDE_DOWN",
        "VIBRATO",
    }
