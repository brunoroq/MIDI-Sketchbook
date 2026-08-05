"""Pure REMI tokenization boundary for Stage 2.

This module intentionally knows nothing about manifests, dataset splits, or
publishing runs.  It translates one validated, single-track MIDI into one
REMI token stream and enforces the invariants needed by the later dataset
pipeline.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any

from miditok import Event, REMI, TokSequence, TokenizerConfig
from symusic import Score

from .tokenization_config import (
    GUITAR_TECHNIQUE_TOKENS,
    PITCH_BEND_RANGE,
    PITCH_BEND_SENSITIVITY_SEMITONES,
    RemiTokenizerConfig,
)
from .tonality import MODE_NAMES, TONIC_NAMES, normalize_mode, normalize_tonic


TECHNIQUE_TYPES: tuple[str, ...] = (
    "DEAD_NOTE",
    "PALM_MUTE_ON",
    "PALM_MUTE_OFF",
    "SLIDE_UP",
    "SLIDE_DOWN",
    "VIBRATO",
)
TECHNIQUE_TOKEN_BY_TYPE: dict[str, str] = dict(
    zip(TECHNIQUE_TYPES, GUITAR_TECHNIQUE_TOKENS, strict=True)
)
TECHNIQUE_TYPE_BY_TOKEN: dict[str, str] = {
    token: technique_type
    for technique_type, token in TECHNIQUE_TOKEN_BY_TYPE.items()
}
_TECHNIQUE_ORDER = {
    technique_type: index for index, technique_type in enumerate(TECHNIQUE_TYPES)
}
_TECHNIQUE_SCHEMA_VERSION = 1
CONDITIONING_SCHEMA_VERSION = 1
TONIC_TOKENS: tuple[str, ...] = tuple(f"Tonic_{name}" for name in TONIC_NAMES)
MODE_TOKENS: tuple[str, ...] = tuple(f"Mode_{name}" for name in MODE_NAMES)


class TokenizationError(ValueError):
    """Raised when a MIDI cannot satisfy the Stage 2 token contract."""


class GuitarREMI(REMI):
    """REMI extended with postfix tokens for note-level guitar techniques."""

    def _create_base_vocabulary(self) -> list[str]:
        vocabulary = super()._create_base_vocabulary()
        vocabulary.extend(GUITAR_TECHNIQUE_TOKENS)
        return vocabulary

    def _create_token_types_graph(self) -> dict[str, set[str]]:
        graph = super()._create_token_types_graph()
        if "Duration" not in graph:
            raise TokenizationError(
                "GuitarREMI requires Duration tokens for postfix techniques."
            )
        duration_successors = set(graph["Duration"])
        graph["Duration"].add("Technique")
        graph["Technique"] = duration_successors | {"Technique"}
        return graph


class ConditionedGuitarREMI(GuitarREMI):
    """Guitar REMI whose stored streams begin with tonic and mode tokens.

    The legacy :class:`GuitarREMI` class intentionally remains unchanged so
    tokenizer JSON files belonging to older checkpoints can still recreate
    their original vocabulary.  The two conditioning events are a wrapper
    protocol: they are stripped before MidiTok validates or decodes REMI.
    """

    def _create_base_vocabulary(self) -> list[str]:
        vocabulary = super()._create_base_vocabulary()
        vocabulary.extend(TONIC_TOKENS)
        vocabulary.extend(MODE_TOKENS)
        return vocabulary

    def _create_token_types_graph(self) -> dict[str, set[str]]:
        graph = super()._create_token_types_graph()
        graph["Tonic"] = {"Mode"}
        graph["Mode"] = {"Bar"}
        return graph


@dataclass(frozen=True, slots=True)
class SpecialTokenIds:
    """Vocabulary-resolved identifiers for the three Stage 2 specials."""

    pad: int
    bos: int
    eos: int


@dataclass(frozen=True, slots=True)
class TechniqueAnnotation:
    """One canonical technique attached to a processed note by stable index."""

    type: str
    note_index: int


@dataclass(frozen=True, slots=True)
class DecodedMidi:
    """Decoded score together with technique information MIDI cannot express."""

    score: Score
    techniques: tuple[TechniqueAnnotation, ...]
    tonic: str
    mode: str


@dataclass(frozen=True, slots=True)
class EncodedMidi:
    """One verified single-track MIDI represented as REMI identifiers."""

    ids: tuple[int, ...]
    musical_ids: tuple[int, ...]
    programs: tuple[tuple[int, bool], ...]
    num_notes: int
    num_pitch_bends: int
    techniques: tuple[TechniqueAnnotation, ...]
    tonic: str
    mode: str
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


def build_tokenizer(config: RemiTokenizerConfig) -> ConditionedGuitarREMI:
    """Build the deterministic, single-track REMI vocabulary for Stage 2.

    MidiTok 3.0.6.post1 builds pitch tokens for both configured endpoints, so
    the project's inclusive range can be passed through unchanged.  Features
    not represented by the Stage 1 guitar-riff contract are disabled
    explicitly here so that a library-default change cannot silently alter
    the vocabulary.
    """

    if not config.use_pitch_bends:
        raise TokenizationError("GuitarREMI requires native PitchBend tokens.")
    if config.pitch_bend_range != PITCH_BEND_RANGE:
        raise TokenizationError(
            f"GuitarREMI requires pitch_bend_range={PITCH_BEND_RANGE}."
        )
    if (
        config.pitch_bend_sensitivity_semitones
        != PITCH_BEND_SENSITIVITY_SEMITONES
    ):
        raise TokenizationError(
            "GuitarREMI requires a six-semitone pitch-bend sensitivity."
        )
    if config.technique_tokens != GUITAR_TECHNIQUE_TOKENS:
        raise TokenizationError(
            "GuitarREMI requires the canonical guitar-technique vocabulary."
        )

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
        use_pitch_bends=config.use_pitch_bends,
        use_programs=False,
        use_pitch_intervals=False,
        use_pitchdrum_tokens=False,
        num_tempos=config.num_tempos,
        tempo_range=(config.tempo_min, config.tempo_max),
        pitch_bend_range=config.pitch_bend_range,
        remove_duplicated_notes=True,
        delete_equal_successive_tempo_changes=True,
        one_token_stream_for_programs=False,
        program_changes=False,
        add_trailing_bars=config.add_trailing_bars,
        use_bar_end_tokens=False,
    )
    try:
        tokenizer = ConditionedGuitarREMI(
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
    tokenizer.guitar_technique_schema_version = _TECHNIQUE_SCHEMA_VERSION
    tokenizer.guitar_technique_tokens = list(GUITAR_TECHNIQUE_TOKENS)
    tokenizer.pitch_bend_sensitivity_semitones = (
        config.pitch_bend_sensitivity_semitones
    )
    tokenizer.conditioning_schema_version = CONDITIONING_SCHEMA_VERSION
    tokenizer.tonic_names = list(TONIC_NAMES)
    tokenizer.mode_names = list(MODE_NAMES)
    _validate_guitar_tokenizer(tokenizer)
    _validate_conditioned_tokenizer(tokenizer)
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


def get_technique_token_ids(tokenizer: REMI) -> dict[str, int]:
    """Resolve canonical technique aliases to ordinary vocabulary identifiers."""

    vocabulary = _vocabulary(tokenizer)
    resolved: dict[str, int] = {}
    for technique_type, token in TECHNIQUE_TOKEN_BY_TYPE.items():
        token_id = vocabulary.get(token)
        if isinstance(token_id, bool) or not isinstance(token_id, Integral):
            raise TokenizationError(
                f"The tokenizer vocabulary does not define technique token '{token}'."
            )
        resolved[technique_type] = int(token_id)
    if len(set(resolved.values())) != len(resolved):
        raise TokenizationError("Guitar technique tokens must have distinct IDs.")
    return resolved


def get_tonic_token_ids(tokenizer: REMI) -> dict[str, int]:
    """Resolve canonical tonic names to conditioned-vocabulary IDs."""

    return _get_condition_token_ids(tokenizer, "Tonic", TONIC_NAMES)


def get_mode_token_ids(tokenizer: REMI) -> dict[str, int]:
    """Resolve canonical mode names to conditioned-vocabulary IDs."""

    return _get_condition_token_ids(tokenizer, "Mode", MODE_NAMES)


def _get_condition_token_ids(
    tokenizer: REMI, token_type: str, names: Sequence[str]
) -> dict[str, int]:
    vocabulary = _vocabulary(tokenizer)
    resolved: dict[str, int] = {}
    for name in names:
        token = f"{token_type}_{name}"
        token_id = vocabulary.get(token)
        if isinstance(token_id, bool) or not isinstance(token_id, Integral):
            raise TokenizationError(
                f"The tokenizer vocabulary does not define {token_type.lower()} "
                f"token '{token}'."
            )
        resolved[name] = int(token_id)
    if len(set(resolved.values())) != len(resolved):
        raise TokenizationError(f"{token_type} tokens must have distinct IDs.")
    return resolved


def encode_midi(
    tokenizer: REMI,
    path: str | Path,
    *,
    techniques: Sequence[TechniqueAnnotation | Mapping[str, object]] = (),
    tonic: str = "UNKNOWN",
    mode: str = "UNKNOWN",
) -> EncodedMidi:
    """Encode one single-track MIDI plus lossless symbolic guitar techniques.

    Bends remain native MidiTok ``PitchBend`` events. Techniques not expressible
    in MIDI are supplied by the authoritative Stage 1 manifest and serialized
    immediately after the ``Duration`` belonging to their note.
    """

    normalized_tonic, normalized_mode = _normalize_condition(tonic, mode)
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
    normalized_techniques = _normalize_techniques(
        techniques, num_notes=len(track.notes), source=str(midi_path)
    )
    programs = ((int(track.program), bool(track.is_drum)),)

    preprocessed = _preprocess_score(tokenizer, score, source=str(midi_path))
    preprocessed_track, _ = _validated_single_track(
        preprocessed, source=f"preprocessed representation of '{midi_path}'"
    )
    note_mapping = _map_note_indices_after_preprocessing(
        track,
        preprocessed_track,
        normalized_techniques,
        source=str(midi_path),
    )
    sequence = _encode_single_sequence(
        tokenizer,
        preprocessed,
        source=str(midi_path),
        no_preprocess_score=True,
    )
    sequence = _insert_technique_events(
        tokenizer,
        sequence,
        normalized_techniques,
        note_mapping,
        source=str(midi_path),
    )
    base_musical_ids = _validated_musical_ids(tokenizer, sequence.ids)
    error_ratio = _token_error_ratio(tokenizer, sequence)
    if error_ratio != 0.0:
        raise TokenizationError(
            f"MidiTok reported a token error ratio of {error_ratio:g} for "
            f"'{midi_path}'."
        )

    special_ids = get_special_token_ids(tokenizer)
    if isinstance(tokenizer, ConditionedGuitarREMI):
        condition_ids = _condition_ids(
            tokenizer, normalized_tonic, normalized_mode
        )
    else:
        condition_ids = ()
    musical_ids = (*condition_ids, *base_musical_ids)
    ids = (special_ids.bos, *musical_ids, special_ids.eos)
    if special_ids.pad in ids:
        raise TokenizationError("PAD must never be stored in an encoded MIDI.")

    decoded = decode_symbolic_token_ids(tokenizer, ids, programs)
    decoded_track, decoded_pitches = _validated_single_track(
        decoded.score, source=f"decoded representation of '{midi_path}'"
    )
    if decoded_pitches != note_pitches:
        raise TokenizationError(
            f"REMI round trip changed the note count or pitches for '{midi_path}'."
        )
    if decoded.techniques != normalized_techniques:
        raise TokenizationError(
            f"REMI round trip changed guitar techniques for '{midi_path}'."
        )
    if (decoded.tonic, decoded.mode) != (normalized_tonic, normalized_mode):
        raise TokenizationError(
            f"REMI round trip changed tonal conditioning for '{midi_path}'."
        )

    round_trip_sequence = _encode_score_with_techniques(
        tokenizer,
        decoded.score,
        decoded.techniques,
        source=f"decoded representation of '{midi_path}'",
    )
    round_trip_ids = _validated_musical_ids(tokenizer, round_trip_sequence.ids)
    round_trip_error_ratio = _token_error_ratio(tokenizer, round_trip_sequence)
    if round_trip_error_ratio != 0.0:
        raise TokenizationError(
            "MidiTok reported token errors after decoding and re-encoding "
            f"'{midi_path}'."
        )
    if round_trip_ids != base_musical_ids:
        raise TokenizationError(
            f"REMI content IDs are not idempotent for '{midi_path}'."
        )

    return EncodedMidi(
        ids=ids,
        musical_ids=musical_ids,
        programs=programs,
        num_notes=len(note_pitches),
        num_pitch_bends=len(decoded_track.pitch_bends),
        techniques=normalized_techniques,
        tonic=normalized_tonic,
        mode=normalized_mode,
        token_error_ratio=error_ratio,
        round_trip_ok=True,
    )


def decode_symbolic_token_ids(
    tokenizer: REMI,
    token_ids: Sequence[int],
    programs: Sequence[tuple[int, bool]],
) -> DecodedMidi:
    """Decode stored IDs while retaining postfix guitar-technique annotations."""

    stored_musical_ids = _validated_stored_ids(tokenizer, token_ids)
    tonic, mode, musical_ids = _split_condition_prefix(
        tokenizer, stored_musical_ids
    )
    normalized_programs = _normalize_programs(programs)
    sequence = TokSequence(ids=list(musical_ids))
    try:
        tokenizer.complete_sequence(sequence)
    except Exception as exc:
        raise TokenizationError(f"Could not resolve REMI token IDs: {exc}") from exc
    error_ratio = _token_error_ratio(tokenizer, sequence)
    if error_ratio != 0.0:
        raise TokenizationError(
            f"Stored REMI token sequence has an error ratio of {error_ratio:g}."
        )

    try:
        score = tokenizer.decode(
            [list(musical_ids)], programs=list(normalized_programs)
        )
    except Exception as exc:
        raise TokenizationError(f"Could not decode REMI token IDs: {exc}") from exc
    track, _ = _validated_single_track(score, source="decoded token sequence")

    tokens = tuple(str(token) for token in sequence.tokens)
    base_tokens = tuple(
        token for token in tokens if token not in TECHNIQUE_TYPE_BY_TOKEN
    )
    base_sequence = _encode_single_sequence(
        tokenizer,
        score,
        source="decoded token sequence",
        no_preprocess_score=True,
    )
    if tuple(base_sequence.tokens) != base_tokens:
        raise TokenizationError(
            "REMI content without technique tokens is not idempotent after decoding."
        )
    techniques = _extract_postfix_techniques(
        tokens, base_sequence, track, source="decoded token sequence"
    )
    return DecodedMidi(
        score=score,
        techniques=techniques,
        tonic=tonic,
        mode=mode,
    )


def canonicalize_symbolic_token_ids(
    tokenizer: REMI,
    token_ids: Sequence[int],
    programs: Sequence[tuple[int, bool]],
) -> tuple[tuple[int, ...], DecodedMidi]:
    """Canonicalize a valid generated stream and decode it.

    Autoregressive generation can end on a syntactically legal trailing
    ``Bar`` token which carries no event in MIDI.  MidiTok consequently drops
    that token when it re-encodes the decoded score.  Stored corpus sequences
    are already canonical and :func:`decode_symbolic_token_ids` deliberately
    rejects such a mismatch; generation instead needs one explicit
    normalization boundary before artifacts are published.

    The input must still have valid REMI transitions and valid postfix guitar
    techniques.  Only the score's canonical re-encoding is returned, so this
    function cannot be used to conceal malformed model output.
    """

    stored_musical_ids = _validated_stored_ids(tokenizer, token_ids)
    tonic, mode, musical_ids = _split_condition_prefix(
        tokenizer, stored_musical_ids
    )
    normalized_programs = _normalize_programs(programs)
    sequence = TokSequence(ids=list(musical_ids))
    try:
        tokenizer.complete_sequence(sequence)
    except Exception as exc:
        raise TokenizationError(
            f"Could not resolve generated REMI token IDs: {exc}"
        ) from exc
    error_ratio = _token_error_ratio(tokenizer, sequence)
    if error_ratio != 0.0:
        raise TokenizationError(
            "Generated REMI token sequence has an error ratio of "
            f"{error_ratio:g}."
        )

    tokens = tuple(str(token) for token in sequence.tokens)
    vocabulary = _vocabulary(tokenizer)
    base_tokens = tuple(
        token for token in tokens if token not in TECHNIQUE_TYPE_BY_TOKEN
    )
    try:
        base_ids = tuple(int(vocabulary[token]) for token in base_tokens)
    except (KeyError, TypeError, ValueError) as exc:
        raise TokenizationError(
            "Generated REMI stream contains an unknown base token."
        ) from exc
    base_sequence = TokSequence(ids=list(base_ids))
    try:
        tokenizer.complete_sequence(base_sequence)
        score = tokenizer.decode(
            [list(base_ids)], programs=list(normalized_programs)
        )
    except Exception as exc:
        raise TokenizationError(
            f"Could not decode generated REMI token IDs: {exc}"
        ) from exc
    track, _ = _validated_single_track(score, source="generated token sequence")
    canonical_base = _encode_single_sequence(
        tokenizer,
        score,
        source="generated token sequence",
        no_preprocess_score=True,
    )
    normalized_tokens = list(tokens)
    normalized_base_tokens = list(base_tokens)
    canonical_base_tokens = tuple(str(token) for token in canonical_base.tokens)
    while (
        tuple(normalized_base_tokens) != canonical_base_tokens
        and normalized_base_tokens
        and normalized_base_tokens[-1] == "Bar_None"
        and normalized_tokens
        and normalized_tokens[-1] == "Bar_None"
    ):
        normalized_base_tokens.pop()
        normalized_tokens.pop()
    if tuple(normalized_base_tokens) != canonical_base_tokens:
        raise TokenizationError(
            "Canonicalization changed generated musical events, not only a "
            "redundant trailing Bar."
        )
    techniques = _extract_postfix_techniques(
        normalized_tokens,
        canonical_base,
        track,
        source="generated token sequence",
    )
    canonical = _encode_score_with_techniques(
        tokenizer,
        score,
        techniques,
        source="generated token sequence",
    )
    canonical_musical_ids = _validated_musical_ids(tokenizer, canonical.ids)
    if _token_error_ratio(tokenizer, canonical) != 0.0:
        raise TokenizationError(
            "Canonical generated REMI token sequence contains invalid transitions."
        )
    special_ids = get_special_token_ids(tokenizer)
    condition_ids = (
        _condition_ids(tokenizer, tonic, mode)
        if isinstance(tokenizer, ConditionedGuitarREMI)
        else ()
    )
    canonical_ids = (
        special_ids.bos,
        *condition_ids,
        *canonical_musical_ids,
        special_ids.eos,
    )
    decoded = decode_symbolic_token_ids(
        tokenizer,
        canonical_ids,
        normalized_programs,
    )
    return canonical_ids, decoded


def decode_token_ids(
    tokenizer: REMI,
    token_ids: Sequence[int],
    programs: Sequence[tuple[int, bool]],
) -> Score:
    """Compatibility wrapper returning only the MIDI-representable score."""

    return decode_symbolic_token_ids(tokenizer, token_ids, programs).score


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
    attributes = (
        dict(additional_attributes) if additional_attributes is not None else {}
    )
    protected_serialization_fields = {
        "_model",
        "_vocab_base",
        "_vocab_base_byte_to_token",
        "config",
        "hf_tokenizers_version",
        "miditok_version",
        "symusic_version",
        "tokenization",
    }
    protected_overrides = sorted(
        protected_serialization_fields.intersection(attributes)
    )
    if protected_overrides:
        raise TokenizationError(
            "Additional tokenizer attributes cannot override serialization field(s): "
            + ", ".join(protected_overrides)
            + "."
        )
    if isinstance(tokenizer, GuitarREMI):
        reserved = {
            "guitar_technique_schema_version": _TECHNIQUE_SCHEMA_VERSION,
            "guitar_technique_tokens": list(GUITAR_TECHNIQUE_TOKENS),
            "pitch_bend_sensitivity_semitones": (
                PITCH_BEND_SENSITIVITY_SEMITONES
            ),
        }
        if isinstance(tokenizer, ConditionedGuitarREMI):
            reserved.update(
                {
                    "conditioning_schema_version": CONDITIONING_SCHEMA_VERSION,
                    "tonic_names": list(TONIC_NAMES),
                    "mode_names": list(MODE_NAMES),
                }
            )
        conflicts = sorted(
            key
            for key, value in reserved.items()
            if key in attributes and attributes[key] != value
        )
        if conflicts:
            raise TokenizationError(
                "Additional tokenizer attributes cannot override reserved field(s): "
                + ", ".join(conflicts)
                + "."
            )
        attributes = {**reserved, **attributes}
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(
            output_path,
            additional_attributes=attributes or None,
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
        payload = json.loads(tokenizer_path.read_bytes())
        if not isinstance(payload, Mapping):
            raise TypeError("tokenizer JSON root is not an object")
        tokenization = payload.get("tokenization")
        if tokenization == "ConditionedGuitarREMI":
            tokenizer = ConditionedGuitarREMI(params=tokenizer_path)
        elif tokenization == "GuitarREMI":
            tokenizer = GuitarREMI(params=tokenizer_path)
        elif tokenization == "REMI":
            tokenizer = REMI(params=tokenizer_path)
        else:
            raise ValueError(
                "tokenization must be 'ConditionedGuitarREMI', "
                "'GuitarREMI', or legacy 'REMI'"
            )
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
    if isinstance(tokenizer, GuitarREMI):
        _validate_guitar_tokenizer(tokenizer)
    if isinstance(tokenizer, ConditionedGuitarREMI):
        _validate_conditioned_tokenizer(tokenizer)
    return tokenizer


def _vocabulary(tokenizer: REMI) -> Mapping[str, int]:
    vocabulary = tokenizer.vocab
    if not isinstance(vocabulary, Mapping):
        raise TokenizationError("Stage 2 requires a single MidiTok vocabulary.")
    return vocabulary


def _validate_guitar_tokenizer(tokenizer: GuitarREMI) -> None:
    special_tokens = tuple(tokenizer.config.special_tokens)
    if special_tokens != ("PAD_None", "BOS_None", "EOS_None"):
        raise TokenizationError(
            "GuitarREMI requires PAD, BOS, and EOS as its only special tokens."
        )
    if not tokenizer.config.use_pitch_bends:
        raise TokenizationError("GuitarREMI must enable native PitchBend tokens.")
    if tuple(tokenizer.config.pitch_bend_range) != PITCH_BEND_RANGE:
        raise TokenizationError(
            f"GuitarREMI pitch_bend_range must be {PITCH_BEND_RANGE}."
        )
    if 0 not in {int(value) for value in tokenizer.pitch_bends}:
        raise TokenizationError("GuitarREMI pitch-bend vocabulary must contain zero.")
    if (
        getattr(tokenizer, "guitar_technique_schema_version", None)
        != _TECHNIQUE_SCHEMA_VERSION
    ):
        raise TokenizationError("GuitarREMI technique schema metadata is missing.")
    if tuple(getattr(tokenizer, "guitar_technique_tokens", ())) != (
        GUITAR_TECHNIQUE_TOKENS
    ):
        raise TokenizationError("GuitarREMI technique vocabulary metadata is invalid.")
    if (
        getattr(tokenizer, "pitch_bend_sensitivity_semitones", None)
        != PITCH_BEND_SENSITIVITY_SEMITONES
    ):
        raise TokenizationError(
            "GuitarREMI pitch-bend sensitivity metadata must be six semitones."
        )
    get_special_token_ids(tokenizer)
    get_technique_token_ids(tokenizer)


def _validate_conditioned_tokenizer(tokenizer: ConditionedGuitarREMI) -> None:
    if (
        getattr(tokenizer, "conditioning_schema_version", None)
        != CONDITIONING_SCHEMA_VERSION
    ):
        raise TokenizationError(
            "ConditionedGuitarREMI conditioning schema metadata is missing."
        )
    if tuple(getattr(tokenizer, "tonic_names", ())) != TONIC_NAMES:
        raise TokenizationError(
            "ConditionedGuitarREMI tonic vocabulary metadata is invalid."
        )
    if tuple(getattr(tokenizer, "mode_names", ())) != MODE_NAMES:
        raise TokenizationError(
            "ConditionedGuitarREMI mode vocabulary metadata is invalid."
        )
    get_tonic_token_ids(tokenizer)
    get_mode_token_ids(tokenizer)


def _normalize_condition(tonic: object, mode: object) -> tuple[str, str]:
    try:
        normalized_tonic = normalize_tonic(tonic)
        normalized_mode = normalize_mode(mode)
    except (TypeError, ValueError) as exc:
        raise TokenizationError(f"Invalid tonal conditioning: {exc}") from exc
    if normalized_tonic == "UNKNOWN" and normalized_mode != "UNKNOWN":
        raise TokenizationError(
            "Mode must be UNKNOWN when tonic is UNKNOWN."
        )
    return normalized_tonic, normalized_mode


def _condition_ids(
    tokenizer: ConditionedGuitarREMI, tonic: str, mode: str
) -> tuple[int, int]:
    normalized_tonic, normalized_mode = _normalize_condition(tonic, mode)
    return (
        get_tonic_token_ids(tokenizer)[normalized_tonic],
        get_mode_token_ids(tokenizer)[normalized_mode],
    )


def _split_condition_prefix(
    tokenizer: REMI, musical_ids: Sequence[int]
) -> tuple[str, str, tuple[int, ...]]:
    ids = tuple(musical_ids)
    if not isinstance(tokenizer, ConditionedGuitarREMI):
        return "UNKNOWN", "UNKNOWN", ids

    tonic_ids = get_tonic_token_ids(tokenizer)
    mode_ids = get_mode_token_ids(tokenizer)
    tonic_name_by_id = {token_id: name for name, token_id in tonic_ids.items()}
    mode_name_by_id = {token_id: name for name, token_id in mode_ids.items()}
    tonic_positions = [
        index for index, token_id in enumerate(ids) if token_id in tonic_name_by_id
    ]
    mode_positions = [
        index for index, token_id in enumerate(ids) if token_id in mode_name_by_id
    ]
    if tonic_positions != [0] or mode_positions != [1]:
        raise TokenizationError(
            "A conditioned sequence must contain exactly one Tonic token at "
            "the first musical position and one Mode token immediately after it."
        )
    if len(ids) < 3:
        raise TokenizationError(
            "A conditioned sequence must contain Tonic, Mode, and REMI content."
        )
    vocabulary_by_id = {int(value): key for key, value in _vocabulary(tokenizer).items()}
    first_base = vocabulary_by_id.get(ids[2], "")
    if not first_base.startswith("Bar_"):
        raise TokenizationError(
            "A conditioned sequence's first REMI token after Mode must be Bar."
        )
    tonic = tonic_name_by_id[ids[0]]
    mode = mode_name_by_id[ids[1]]
    return (*_normalize_condition(tonic, mode), ids[2:])


def _normalize_techniques(
    techniques: Sequence[TechniqueAnnotation | Mapping[str, object]],
    *,
    num_notes: int,
    source: str,
) -> tuple[TechniqueAnnotation, ...]:
    if isinstance(techniques, (str, bytes)) or not isinstance(techniques, Sequence):
        raise TokenizationError(f"Techniques for {source} must be a sequence.")

    normalized: list[TechniqueAnnotation] = []
    seen: set[tuple[int, str]] = set()
    for index, value in enumerate(techniques):
        name = f"techniques[{index}]"
        if isinstance(value, TechniqueAnnotation):
            technique_type = value.type
            note_index = value.note_index
        elif isinstance(value, Mapping):
            if set(value) != {"type", "note_index"}:
                raise TokenizationError(
                    f"{name} must contain exactly 'type' and 'note_index'."
                )
            technique_type = value["type"]
            note_index = value["note_index"]
        else:
            raise TokenizationError(
                f"{name} must be a TechniqueAnnotation or mapping."
            )
        if not isinstance(technique_type, str) or technique_type not in (
            TECHNIQUE_TOKEN_BY_TYPE
        ):
            raise TokenizationError(
                f"{name}.type must be one of {', '.join(TECHNIQUE_TYPES)}."
            )
        if isinstance(note_index, bool) or not isinstance(note_index, Integral):
            raise TokenizationError(f"{name}.note_index must be an integer.")
        converted_index = int(note_index)
        if not 0 <= converted_index < num_notes:
            raise TokenizationError(
                f"{name}.note_index {converted_index} is outside the "
                f"{num_notes}-note track."
            )
        identity = (converted_index, technique_type)
        if identity in seen:
            raise TokenizationError(
                f"Duplicate technique {technique_type} for note {converted_index}."
            )
        seen.add(identity)
        normalized.append(TechniqueAnnotation(technique_type, converted_index))

    normalized.sort(key=lambda item: (item.note_index, _TECHNIQUE_ORDER[item.type]))
    grouped: dict[int, set[str]] = defaultdict(set)
    for technique in normalized:
        grouped[technique.note_index].add(technique.type)
    for note_index, types in grouped.items():
        if {"SLIDE_UP", "SLIDE_DOWN"}.issubset(types):
            raise TokenizationError(
                f"Note {note_index} cannot use SLIDE_UP and SLIDE_DOWN together."
            )
        if {"PALM_MUTE_ON", "PALM_MUTE_OFF"}.issubset(types):
            raise TokenizationError(
                f"Note {note_index} cannot switch palm mute on and off together."
            )

    palm_muted = False
    for technique in normalized:
        if technique.type == "PALM_MUTE_ON":
            if palm_muted:
                raise TokenizationError(
                    f"PALM_MUTE_ON at note {technique.note_index} is redundant."
                )
            palm_muted = True
        elif technique.type == "PALM_MUTE_OFF":
            if not palm_muted:
                raise TokenizationError(
                    f"PALM_MUTE_OFF at note {technique.note_index} has no active mute."
                )
            palm_muted = False
    return tuple(normalized)


def _canonical_notes(track: Any) -> tuple[Any, ...]:
    return tuple(
        sorted(
            track.notes,
            key=lambda note: (
                int(note.time),
                int(note.pitch),
                int(note.end),
                int(note.velocity),
            ),
        )
    )


def _preprocess_score(tokenizer: REMI, score: Score, *, source: str) -> Score:
    try:
        return tokenizer.preprocess_score(score)
    except Exception as exc:
        raise TokenizationError(f"Could not preprocess {source}: {exc}") from exc


def _map_note_indices_after_preprocessing(
    original_track: Any,
    preprocessed_track: Any,
    techniques: Sequence[TechniqueAnnotation],
    *,
    source: str,
) -> dict[int, Any]:
    if not techniques:
        return {}
    original_notes = _canonical_notes(original_track)
    by_pitch_original: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    for note_index, note in enumerate(original_notes):
        by_pitch_original[int(note.pitch)].append((note_index, note))
    by_pitch_preprocessed: dict[int, list[Any]] = defaultdict(list)
    for note in preprocessed_track.notes:
        by_pitch_preprocessed[int(note.pitch)].append(note)
    for notes in by_pitch_preprocessed.values():
        notes.sort(key=lambda note: (int(note.time), int(note.end), int(note.velocity)))

    requested = {technique.note_index for technique in techniques}
    mapping: dict[int, Any] = {}
    for pitch, original_group in by_pitch_original.items():
        original_group.sort(
            key=lambda item: (
                int(item[1].time),
                int(item[1].end),
                int(item[1].velocity),
            )
        )
        preprocessed_group = by_pitch_preprocessed.get(pitch, [])
        if len(original_group) != len(preprocessed_group):
            if any(note_index in requested for note_index, _ in original_group):
                raise TokenizationError(
                    f"Tokenizer preprocessing removed or merged a technique-bearing "
                    f"pitch {pitch} note in {source}."
                )
            continue
        for (note_index, _), preprocessed_note in zip(
            original_group, preprocessed_group, strict=True
        ):
            if note_index in requested:
                mapping[note_index] = preprocessed_note
    missing = sorted(requested - set(mapping))
    if missing:
        raise TokenizationError(
            "Could not map technique-bearing note index(es) after preprocessing: "
            + ", ".join(str(index) for index in missing)
            + "."
        )
    return mapping


def _insert_technique_events(
    tokenizer: REMI,
    sequence: TokSequence,
    techniques: Sequence[TechniqueAnnotation],
    note_mapping: Mapping[int, Any],
    *,
    source: str,
) -> TokSequence:
    if not techniques:
        return sequence
    if not isinstance(tokenizer, GuitarREMI):
        raise TokenizationError("Guitar techniques require a GuitarREMI tokenizer.")

    pitch_event_indexes: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, event in enumerate(sequence.events):
        if event.type_ == "Pitch":
            pitch_event_indexes[
                (int(event.time), int(event.value), int(event.desc))
            ].append(index)

    insertions: dict[int, list[Event]] = defaultdict(list)
    for technique in techniques:
        note = note_mapping[technique.note_index]
        identity = (int(note.time), int(note.pitch), int(note.end))
        matches = pitch_event_indexes.get(identity, [])
        if len(matches) != 1:
            raise TokenizationError(
                f"Technique-bearing note {technique.note_index} in {source} matched "
                f"{len(matches)} REMI Pitch events instead of one."
            )
        pitch_index = matches[0]
        duration_index = pitch_index + (2 if tokenizer.config.use_velocities else 1)
        if (
            duration_index >= len(sequence.events)
            or sequence.events[duration_index].type_ != "Duration"
        ):
            raise TokenizationError(
                f"Technique-bearing note {technique.note_index} in {source} has no "
                "postfix Duration event."
            )
        token = TECHNIQUE_TOKEN_BY_TYPE[technique.type]
        token_value = token.split("_", maxsplit=1)[1]
        pitch_event = sequence.events[pitch_index]
        insertions[duration_index].append(
            Event(
                "Technique",
                token_value,
                int(pitch_event.time),
                int(pitch_event.program),
                int(pitch_event.desc),
            )
        )

    events: list[Event] = []
    for index, event in enumerate(sequence.events):
        events.append(event)
        events.extend(insertions.get(index, ()))
    enriched = TokSequence(events=events)
    try:
        tokenizer.complete_sequence(enriched)
    except Exception as exc:
        raise TokenizationError(
            f"Could not add guitar-technique tokens for {source}: {exc}"
        ) from exc
    return enriched


def _encode_score_with_techniques(
    tokenizer: REMI,
    score: Score,
    techniques: Sequence[TechniqueAnnotation],
    *,
    source: str,
) -> TokSequence:
    track, _ = _validated_single_track(score, source=source)
    normalized = _normalize_techniques(
        techniques, num_notes=len(track.notes), source=source
    )
    preprocessed = _preprocess_score(tokenizer, score, source=source)
    preprocessed_track, _ = _validated_single_track(
        preprocessed, source=f"preprocessed {source}"
    )
    mapping = _map_note_indices_after_preprocessing(
        track, preprocessed_track, normalized, source=source
    )
    sequence = _encode_single_sequence(
        tokenizer,
        preprocessed,
        source=source,
        no_preprocess_score=True,
    )
    return _insert_technique_events(
        tokenizer, sequence, normalized, mapping, source=source
    )


def _extract_postfix_techniques(
    tokens: Sequence[str],
    base_sequence: TokSequence,
    track: Any,
    *,
    source: str,
) -> tuple[TechniqueAnnotation, ...]:
    canonical_notes = _canonical_notes(track)
    note_index_by_identity: dict[tuple[int, int, int], int] = {}
    for note_index, note in enumerate(canonical_notes):
        identity = (int(note.time), int(note.pitch), int(note.end))
        if identity in note_index_by_identity:
            raise TokenizationError(
                f"{source} contains ambiguous decoded note identities."
            )
        note_index_by_identity[identity] = note_index

    annotations: list[TechniqueAnnotation] = []
    base_index = -1
    current_note_identity: tuple[int, int, int] | None = None
    for token in tokens:
        if token.startswith("Technique_"):
            technique_type = TECHNIQUE_TYPE_BY_TOKEN.get(token)
            if technique_type is None:
                raise TokenizationError(f"Unknown guitar-technique token '{token}'.")
            if current_note_identity is None:
                raise TokenizationError(
                    f"Technique token '{token}' is not postfix to a Duration."
                )
            note_index = note_index_by_identity.get(current_note_identity)
            if note_index is None:
                raise TokenizationError(
                    f"Technique token '{token}' cannot be matched to a decoded note."
                )
            annotations.append(TechniqueAnnotation(technique_type, note_index))
            continue

        current_note_identity = None
        base_index += 1
        if base_index >= len(base_sequence.events):
            raise TokenizationError("Technique filtering desynchronized REMI events.")
        event = base_sequence.events[base_index]
        if event.type_ != "Duration":
            continue
        pitch_index = base_index - 1
        if (
            pitch_index >= 0
            and base_sequence.events[pitch_index].type_ == "Velocity"
        ):
            pitch_index -= 1
        if pitch_index < 0 or base_sequence.events[pitch_index].type_ != "Pitch":
            continue
        pitch_event = base_sequence.events[pitch_index]
        current_note_identity = (
            int(pitch_event.time),
            int(pitch_event.value),
            int(pitch_event.desc),
        )

    if base_index + 1 != len(base_sequence.events):
        raise TokenizationError("Technique filtering desynchronized REMI tokens.")
    return _normalize_techniques(
        annotations, num_notes=len(canonical_notes), source=source
    )


def _validated_stored_ids(
    tokenizer: REMI, token_ids: Sequence[int]
) -> tuple[int, ...]:
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
    return _validated_musical_ids(tokenizer, ids[1:-1])


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
    tokenizer: REMI,
    score: Score,
    *,
    source: str,
    no_preprocess_score: bool = False,
) -> TokSequence:
    try:
        encoded = tokenizer.encode(
            score,
            encode_ids=False,
            no_preprocess_score=no_preprocess_score,
        )
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
    "CONDITIONING_SCHEMA_VERSION",
    "ConditionedGuitarREMI",
    "DecodedMidi",
    "EncodedMidi",
    "GuitarREMI",
    "MODE_TOKENS",
    "SpecialTokenIds",
    "TECHNIQUE_TOKEN_BY_TYPE",
    "TECHNIQUE_TYPES",
    "TechniqueAnnotation",
    "TONIC_TOKENS",
    "TokenizationError",
    "build_tokenizer",
    "canonicalize_symbolic_token_ids",
    "decode_symbolic_token_ids",
    "decode_token_ids",
    "encode_midi",
    "get_special_token_ids",
    "get_mode_token_ids",
    "get_technique_token_ids",
    "get_tonic_token_ids",
    "load_tokenizer",
    "save_tokenizer",
]
