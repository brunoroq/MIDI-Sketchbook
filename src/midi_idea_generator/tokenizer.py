"""Pure REMI tokenization boundary for Stage 2.

This module intentionally knows nothing about manifests, dataset splits, or
publishing runs.  It translates one validated, single-track MIDI into one
REMI token stream and enforces the invariants needed by the later dataset
pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral
from pathlib import Path
from typing import Any

from miditok import REMI, TokSequence, TokenizerConfig
from symusic import Score

from .tokenization_config import RemiTokenizerConfig


class TokenizationError(ValueError):
    """Raised when a MIDI cannot satisfy the Stage 2 token contract."""


@dataclass(frozen=True, slots=True)
class SpecialTokenIds:
    """Vocabulary-resolved identifiers for the three Stage 2 specials."""

    pad: int
    bos: int
    eos: int


@dataclass(frozen=True, slots=True)
class EncodedMidi:
    """One verified single-track MIDI represented as REMI identifiers."""

    ids: tuple[int, ...]
    musical_ids: tuple[int, ...]
    programs: tuple[tuple[int, bool], ...]
    num_notes: int
    token_error_ratio: float
    round_trip_ok: bool

    @property
    def num_tokens(self) -> int:
        """Return the serialized length, including BOS and EOS."""

        return len(self.ids)

    @property
    def num_musical_tokens(self) -> int:
        """Return the REMI content length, excluding BOS and EOS."""

        return len(self.musical_ids)


def build_tokenizer(config: RemiTokenizerConfig) -> REMI:
    """Build the deterministic, single-track REMI vocabulary for Stage 2.

    MidiTok 3.0.6.post1 builds pitch tokens for both configured endpoints, so
    the project's inclusive range can be passed through unchanged.  Features
    not represented by the Stage 1 guitar-riff contract are disabled
    explicitly here so that a library-default change cannot silently alter
    the vocabulary.
    """

    beat_res = {
        (entry.start_beat, entry.end_beat): entry.resolution
        for entry in config.beat_res
    }
    tokenizer_config = TokenizerConfig(
        pitch_range=(config.pitch_min, config.pitch_max),
        beat_res=beat_res,
        num_velocities=config.num_velocities,
        special_tokens=list(config.special_tokens),
        encode_ids_split=config.encode_ids_split,
        use_velocities=config.use_velocities,
        use_chords=False,
        use_rests=False,
        use_tempos=config.use_tempos,
        use_time_signatures=False,
        use_sustain_pedals=False,
        use_pitch_bends=False,
        use_programs=False,
        use_pitch_intervals=False,
        use_pitchdrum_tokens=False,
        num_tempos=config.num_tempos,
        tempo_range=(config.tempo_min, config.tempo_max),
        remove_duplicated_notes=True,
        delete_equal_successive_tempo_changes=True,
        one_token_stream_for_programs=False,
        program_changes=False,
        add_trailing_bars=config.add_trailing_bars,
        use_bar_end_tokens=False,
    )
    try:
        tokenizer = REMI(
            tokenizer_config=tokenizer_config,
            max_bar_embedding=config.max_bar_embedding,
        )
    except Exception as exc:  # MidiTok exposes several configuration errors.
        raise TokenizationError(f"Could not build the REMI tokenizer: {exc}") from exc

    if tokenizer.one_token_stream:
        raise TokenizationError(
            "Stage 2 requires independent instrument streams, but MidiTok "
            "configured a merged token stream."
        )
    get_special_token_ids(tokenizer)
    return tokenizer


def get_special_token_ids(tokenizer: REMI) -> SpecialTokenIds:
    """Resolve PAD/BOS/EOS from the vocabulary without assuming numeric IDs."""

    vocabulary = _vocabulary(tokenizer)

    def resolve(name: str) -> int:
        token = f"{name}_None"
        token_id = vocabulary.get(token)
        if isinstance(token_id, bool) or not isinstance(token_id, Integral):
            raise TokenizationError(
                f"The tokenizer vocabulary does not define required token '{token}'."
            )
        return int(token_id)

    special_ids = SpecialTokenIds(
        pad=resolve("PAD"), bos=resolve("BOS"), eos=resolve("EOS")
    )
    if len({special_ids.pad, special_ids.bos, special_ids.eos}) != 3:
        raise TokenizationError("PAD, BOS, and EOS must have distinct token IDs.")
    return special_ids


def encode_midi(tokenizer: REMI, path: str | Path) -> EncodedMidi:
    """Encode one single-track MIDI and verify a loss-aware round trip.

    Velocity and timing can be intentionally quantized by the tokenizer
    configuration.  The musical event identity checked here is therefore the
    note count and pitch multiset, followed by exact idempotence of the REMI
    content IDs after decoding and re-encoding.
    """

    midi_path = Path(path).expanduser().resolve()
    if not midi_path.is_file():
        raise TokenizationError(f"MIDI file does not exist: {midi_path}")

    try:
        score = Score(midi_path)
    except Exception as exc:
        raise TokenizationError(
            f"Could not parse MIDI '{midi_path}' ({type(exc).__name__}): {exc}"
        ) from exc

    track, note_pitches = _validated_single_track(score, source=str(midi_path))
    programs = ((int(track.program), bool(track.is_drum)),)

    sequence = _encode_single_sequence(tokenizer, score, source=str(midi_path))
    musical_ids = _validated_musical_ids(tokenizer, sequence.ids)
    error_ratio = _token_error_ratio(tokenizer, sequence)
    if error_ratio != 0.0:
        raise TokenizationError(
            f"MidiTok reported a token error ratio of {error_ratio:g} for "
            f"'{midi_path}'."
        )

    special_ids = get_special_token_ids(tokenizer)
    ids = (special_ids.bos, *musical_ids, special_ids.eos)
    if special_ids.pad in ids:
        raise TokenizationError("PAD must never be stored in an encoded MIDI.")

    decoded = decode_token_ids(tokenizer, ids, programs)
    _, decoded_pitches = _validated_single_track(
        decoded, source=f"decoded representation of '{midi_path}'"
    )
    if decoded_pitches != note_pitches:
        raise TokenizationError(
            f"REMI round trip changed the note count or pitches for '{midi_path}'."
        )

    round_trip_sequence = _encode_single_sequence(
        tokenizer, decoded, source=f"decoded representation of '{midi_path}'"
    )
    round_trip_ids = _validated_musical_ids(
        tokenizer, round_trip_sequence.ids
    )
    round_trip_error_ratio = _token_error_ratio(tokenizer, round_trip_sequence)
    if round_trip_error_ratio != 0.0:
        raise TokenizationError(
            "MidiTok reported token errors after decoding and re-encoding "
            f"'{midi_path}'."
        )
    if round_trip_ids != musical_ids:
        raise TokenizationError(
            f"REMI content IDs are not idempotent for '{midi_path}'."
        )

    return EncodedMidi(
        ids=ids,
        musical_ids=musical_ids,
        programs=programs,
        num_notes=len(note_pitches),
        token_error_ratio=error_ratio,
        round_trip_ok=True,
    )


def decode_token_ids(
    tokenizer: REMI,
    token_ids: Sequence[int],
    programs: Sequence[tuple[int, bool]],
) -> Score:
    """Decode one stored Stage 2 sequence after removing BOS and EOS.

    Stored sequences must contain exactly one BOS/EOS boundary pair and no PAD
    token.  ``programs`` remains out-of-band because program tokens are
    deliberately disabled for this single-instrument experiment.
    """

    ids = _normalize_ids(token_ids, name="token IDs")
    special_ids = get_special_token_ids(tokenizer)
    if len(ids) < 3:
        raise TokenizationError(
            "A stored token sequence must contain BOS, musical content, and EOS."
        )
    if ids[0] != special_ids.bos or ids[-1] != special_ids.eos:
        raise TokenizationError(
            "A stored token sequence must begin with BOS and end with EOS."
        )
    if special_ids.pad in ids:
        raise TokenizationError("PAD is only for batching and cannot be decoded.")
    if special_ids.bos in ids[1:] or special_ids.eos in ids[:-1]:
        raise TokenizationError("BOS and EOS may only occur at sequence boundaries.")

    musical_ids = _validated_musical_ids(tokenizer, ids[1:-1])
    normalized_programs = _normalize_programs(programs)
    try:
        score = tokenizer.decode(
            [list(musical_ids)], programs=list(normalized_programs)
        )
    except Exception as exc:
        raise TokenizationError(f"Could not decode REMI token IDs: {exc}") from exc
    _validated_single_track(score, source="decoded token sequence")
    return score


def save_tokenizer(
    tokenizer: REMI,
    path: str | Path,
    *,
    additional_attributes: Mapping[str, Any] | None = None,
) -> Path:
    """Persist a tokenizer as one reloadable MidiTok JSON file."""

    output_path = Path(path).expanduser().resolve()
    if output_path.suffix.lower() != ".json":
        raise TokenizationError("Tokenizer output path must use a .json extension.")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(
            output_path,
            additional_attributes=(
                dict(additional_attributes)
                if additional_attributes is not None
                else None
            ),
        )
    except Exception as exc:
        raise TokenizationError(
            f"Could not save tokenizer to '{output_path}': {exc}"
        ) from exc
    if not output_path.is_file():
        raise TokenizationError(
            f"MidiTok did not create tokenizer file '{output_path}'."
        )
    return output_path


def load_tokenizer(path: str | Path) -> REMI:
    """Reload a persisted Stage 2 REMI tokenizer and validate its boundaries."""

    tokenizer_path = Path(path).expanduser().resolve()
    if not tokenizer_path.is_file():
        raise TokenizationError(f"Tokenizer file does not exist: {tokenizer_path}")
    try:
        tokenizer = REMI(params=tokenizer_path)
    except Exception as exc:
        raise TokenizationError(
            f"Could not load tokenizer '{tokenizer_path}': {exc}"
        ) from exc
    if tokenizer.one_token_stream:
        raise TokenizationError(
            "Loaded tokenizer merges instruments into one stream, which Stage 2 "
            "does not support."
        )
    get_special_token_ids(tokenizer)
    return tokenizer


def _vocabulary(tokenizer: REMI) -> Mapping[str, int]:
    vocabulary = tokenizer.vocab
    if not isinstance(vocabulary, Mapping):
        raise TokenizationError("Stage 2 requires a single MidiTok vocabulary.")
    return vocabulary


def _validated_single_track(
    score: Score, *, source: str
) -> tuple[Any, tuple[int, ...]]:
    if len(score.tracks) != 1:
        raise TokenizationError(
            f"Stage 2 requires exactly one MIDI track; {source} has "
            f"{len(score.tracks)}."
        )
    track = score.tracks[0]
    if len(track.notes) == 0:
        raise TokenizationError(
            f"Stage 2 requires a non-empty track; {source} is empty."
        )
    if bool(track.is_drum):
        raise TokenizationError(
            f"Stage 2 only supports pitched instrumental tracks; {source} is drums."
        )
    program = int(track.program)
    if not 0 <= program <= 127:
        raise TokenizationError(
            f"Track program must be between 0 and 127; {source} uses {program}."
        )
    pitches = tuple(sorted(int(note.pitch) for note in track.notes))
    return track, pitches


def _encode_single_sequence(
    tokenizer: REMI, score: Score, *, source: str
) -> TokSequence:
    try:
        encoded = tokenizer.encode(score, encode_ids=False)
    except Exception as exc:
        raise TokenizationError(f"Could not encode {source}: {exc}") from exc
    if not isinstance(encoded, list) or len(encoded) != 1:
        stream_count = len(encoded) if isinstance(encoded, list) else "a merged stream"
        raise TokenizationError(
            f"Stage 2 expected exactly one TokSequence for {source}; got "
            f"{stream_count}."
        )
    sequence = encoded[0]
    if not isinstance(sequence, TokSequence):
        raise TokenizationError(f"MidiTok returned an invalid sequence for {source}.")
    if not sequence.ids:
        raise TokenizationError(f"MidiTok returned no musical tokens for {source}.")
    return sequence


def _normalize_ids(token_ids: Sequence[int], *, name: str) -> tuple[int, ...]:
    if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence):
        raise TokenizationError(f"{name.capitalize()} must be a sequence of integers.")
    normalized: list[int] = []
    for index, token_id in enumerate(token_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, Integral):
            raise TokenizationError(f"{name.capitalize()}[{index}] must be an integer.")
        normalized.append(int(token_id))
    return tuple(normalized)


def _validated_musical_ids(
    tokenizer: REMI, token_ids: Sequence[int]
) -> tuple[int, ...]:
    ids = _normalize_ids(token_ids, name="musical token IDs")
    if not ids:
        raise TokenizationError("A REMI sequence must contain musical tokens.")
    vocabulary = _vocabulary(tokenizer)
    vocabulary_ids = {int(token_id) for token_id in vocabulary.values()}
    unknown_ids = sorted(set(ids) - vocabulary_ids)
    if unknown_ids:
        raise TokenizationError(
            "Token sequence contains IDs outside the tokenizer vocabulary: "
            + ", ".join(str(token_id) for token_id in unknown_ids)
            + "."
        )
    special_ids = get_special_token_ids(tokenizer)
    forbidden = {special_ids.pad, special_ids.bos, special_ids.eos}.intersection(ids)
    if forbidden:
        raise TokenizationError(
            "Musical content cannot contain PAD, BOS, or EOS token IDs."
        )
    return ids


def _normalize_programs(
    programs: Sequence[tuple[int, bool]],
) -> tuple[tuple[int, bool], ...]:
    if isinstance(programs, (str, bytes)) or not isinstance(programs, Sequence):
        raise TokenizationError("Programs must contain one (program, is_drum) pair.")
    if len(programs) != 1:
        raise TokenizationError("Stage 2 requires exactly one program descriptor.")
    descriptor = programs[0]
    if not isinstance(descriptor, (tuple, list)) or len(descriptor) != 2:
        raise TokenizationError("Program descriptor must be (program, is_drum).")
    program, is_drum = descriptor
    if isinstance(program, bool) or not isinstance(program, Integral):
        raise TokenizationError("Program number must be an integer.")
    if not 0 <= int(program) <= 127:
        raise TokenizationError("Program number must be between 0 and 127.")
    if not isinstance(is_drum, bool):
        raise TokenizationError("Program is_drum flag must be true or false.")
    if is_drum:
        raise TokenizationError("Stage 2 does not support drum programs.")
    return ((int(program), is_drum),)


def _token_error_ratio(tokenizer: REMI, sequence: TokSequence) -> float:
    try:
        ratio = tokenizer.tokens_errors(sequence)
    except Exception as exc:
        raise TokenizationError(
            f"Could not validate REMI token transitions: {exc}"
        ) from exc
    if isinstance(ratio, list):
        if len(ratio) != 1:
            raise TokenizationError("MidiTok returned multiple token error ratios.")
        ratio = ratio[0]
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise TokenizationError("MidiTok returned an invalid token error ratio.")
    converted = float(ratio)
    if not math.isfinite(converted) or converted < 0:
        raise TokenizationError("MidiTok returned an invalid token error ratio.")
    return converted


__all__ = [
    "EncodedMidi",
    "SpecialTokenIds",
    "TokenizationError",
    "build_tokenizer",
    "decode_token_ids",
    "encode_midi",
    "get_special_token_ids",
    "load_tokenizer",
    "save_tokenizer",
]
