"""Unit tests for the pure Stage 2 REMI adapter."""

from __future__ import annotations

import json
from pathlib import Path

from miditok import REMI, TokenizerConfig
import pretty_midi
import pytest

from midi_idea_generator.tonality import MODE_NAMES, TONIC_NAMES
from midi_idea_generator.tokenization_config import (
    GUITAR_TECHNIQUE_TOKENS,
    PITCH_BEND_RANGE,
    PITCH_BEND_SENSITIVITY_SEMITONES,
    RemiTokenizerConfig,
)
from midi_idea_generator.tokenizer import (
    CONDITIONING_SCHEMA_VERSION,
    ConditionedGuitarREMI,
    GuitarREMI,
    MODE_TOKENS,
    TECHNIQUE_TOKEN_BY_TYPE,
    TECHNIQUE_TYPES,
    TONIC_TOKENS,
    TechniqueAnnotation,
    TokenizationError,
    build_tokenizer,
    decode_symbolic_token_ids,
    decode_token_ids,
    encode_midi,
    get_mode_token_ids,
    get_special_token_ids,
    get_technique_token_ids,
    get_tonic_token_ids,
    load_tokenizer,
    save_tokenizer,
)


def _write_guitar_midi(
    path: Path,
    make_instrument,
    write_midi_file,
    *,
    velocities: tuple[int, int] = (36, 112),
) -> Path:
    instrument = make_instrument(
        [
            (21, 0.0, 0.25, velocities[0]),
            (64, 0.5, 1.0, velocities[1]),
            (108, 1.5, 2.0, velocities[0]),
        ],
        program=30,
        name="Distortion Guitar",
    )
    return write_midi_file(path, [instrument], tempo_bpm=137.0)


def test_build_tokenizer_translates_config_and_disables_unused_features() -> None:
    tokenizer = build_tokenizer(RemiTokenizerConfig())

    assert type(tokenizer) is ConditionedGuitarREMI
    assert tokenizer.one_token_stream is False
    assert tokenizer.config.pitch_range == (21, 108)
    assert tokenizer.config.beat_res == {(0, 4): 24, (4, 16): 4}
    assert tokenizer.config.use_velocities is False
    assert tokenizer.config.use_tempos is True
    assert tokenizer.config.tempo_range == (40.0, 250.0)
    assert tokenizer.config.num_tempos == 32
    assert tokenizer.config.num_velocities == 16
    assert tokenizer.config.remove_duplicated_notes is True
    assert tokenizer.config.delete_equal_successive_tempo_changes is True
    assert tokenizer.config.additional_params["add_trailing_bars"] is True
    assert tokenizer.config.use_pitch_bends is True
    assert tuple(tokenizer.config.pitch_bend_range) == PITCH_BEND_RANGE
    assert len(tokenizer.pitch_bends) == PITCH_BEND_RANGE[2]
    assert 0 in tokenizer.pitch_bends
    assert (
        tokenizer.pitch_bend_sensitivity_semitones
        == PITCH_BEND_SENSITIVITY_SEMITONES
    )
    assert all(
        getattr(tokenizer.config, feature) is False
        for feature in (
            "use_chords",
            "use_rests",
            "use_time_signatures",
            "use_sustain_pedals",
            "use_programs",
            "use_pitch_intervals",
            "use_pitchdrum_tokens",
        )
    )


def test_techniques_are_ordinary_postfix_tokens_with_a_strict_graph() -> None:
    tokenizer = build_tokenizer(RemiTokenizerConfig())
    technique_ids = get_technique_token_ids(tokenizer)

    assert tuple(technique_ids) == TECHNIQUE_TYPES
    vocabulary = tuple(tokenizer.vocab)
    assert vocabulary[-len(MODE_TOKENS) :] == MODE_TOKENS
    tonic_start = -(len(MODE_TOKENS) + len(TONIC_TOKENS))
    assert vocabulary[tonic_start : -len(MODE_TOKENS)] == TONIC_TOKENS
    technique_start = tonic_start - len(GUITAR_TECHNIQUE_TOKENS)
    assert vocabulary[technique_start:tonic_start] == GUITAR_TECHNIQUE_TOKENS
    assert tokenizer.config.special_tokens == ["PAD_None", "BOS_None", "EOS_None"]
    assert "Technique" in tokenizer.tokens_types_graph["Duration"]
    assert tokenizer.tokens_types_graph["Technique"] == (
        tokenizer.tokens_types_graph["Duration"] - {"Technique"}
    ) | {"Technique"}
    musical_predecessors = {
        token_type
        for token_type, successors in tokenizer.tokens_types_graph.items()
        if "Technique" in successors and token_type not in {"PAD", "BOS", "EOS"}
    }
    assert musical_predecessors == {"Duration", "Technique"}
    assert tokenizer.tokens_types_graph["Tonic"] - {"PAD", "EOS"} == {"Mode"}
    assert tokenizer.tokens_types_graph["Mode"] - {"PAD", "EOS"} == {"Bar"}
    assert tuple(get_tonic_token_ids(tokenizer)) == TONIC_NAMES
    assert tuple(get_mode_token_ids(tokenizer)) == MODE_NAMES


@pytest.mark.parametrize(
    "config",
    [
        RemiTokenizerConfig(use_pitch_bends=False),
        RemiTokenizerConfig(pitch_bend_range=(-8192, 8191, 32)),
        RemiTokenizerConfig(pitch_bend_sensitivity_semitones=2),
        RemiTokenizerConfig(technique_tokens=("Technique_VIBRATO",)),
    ],
)
def test_build_rejects_noncanonical_guitar_language(
    config: RemiTokenizerConfig,
) -> None:
    with pytest.raises(TokenizationError, match="requires"):
        build_tokenizer(config)


def test_encode_adds_resolved_boundaries_without_pad_and_round_trips(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    midi_path = _write_guitar_midi(
        tmp_path / "riff.mid", make_instrument, write_midi_file
    )
    tokenizer = build_tokenizer(RemiTokenizerConfig())

    encoded = encode_midi(tokenizer, midi_path, tonic="E", mode="PHRYGIAN")
    special_ids = get_special_token_ids(tokenizer)

    assert encoded.ids[0] == special_ids.bos
    assert encoded.ids[-1] == special_ids.eos
    assert special_ids.pad not in encoded.ids
    assert encoded.ids[1:-1] == encoded.musical_ids
    assert tokenizer[encoded.ids[1]] == "Tonic_E"
    assert tokenizer[encoded.ids[2]] == "Mode_PHRYGIAN"
    assert tokenizer[encoded.ids[3]] == "Bar_None"
    assert encoded.num_tokens == len(encoded.ids)
    assert encoded.num_musical_tokens == len(encoded.musical_ids)
    assert encoded.num_tokens == encoded.num_musical_tokens + 2
    assert encoded.programs == ((30, False),)
    assert encoded.num_notes == 3
    assert encoded.num_pitch_bends == 0
    assert encoded.techniques == ()
    assert (encoded.tonic, encoded.mode) == ("E", "PHRYGIAN")
    assert encoded.token_error_ratio == 0.0
    assert encoded.round_trip_ok is True

    musical_tokens = [tokenizer[token_id] for token_id in encoded.musical_ids]
    assert any(token.startswith("Tempo_") for token in musical_tokens)
    assert not any(token.startswith("Velocity_") for token in musical_tokens)

    decoded = decode_token_ids(tokenizer, encoded.ids, encoded.programs)
    assert len(decoded.tracks) == 1
    assert decoded.tracks[0].program == 30
    assert [note.pitch for note in decoded.tracks[0].notes] == [21, 64, 108]


def test_native_pitch_bends_include_exact_zero_and_round_trip(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    instrument = make_instrument([(60, 0.0, 1.0)], program=30)
    instrument.pitch_bends.extend(
        [
            pretty_midi.PitchBend(pitch=0, time=0.0),
            pretty_midi.PitchBend(pitch=4096, time=0.25),
            pretty_midi.PitchBend(pitch=8191, time=0.5),
            pretty_midi.PitchBend(pitch=0, time=0.75),
        ]
    )
    path = write_midi_file(tmp_path / "bends.mid", [instrument])
    tokenizer = build_tokenizer(RemiTokenizerConfig())

    encoded = encode_midi(tokenizer, path)
    tokens = [tokenizer[token_id] for token_id in encoded.musical_ids]
    decoded = decode_symbolic_token_ids(tokenizer, encoded.ids, encoded.programs)

    bend_tokens = [token for token in tokens if token.startswith("PitchBend_")]
    assert bend_tokens == [
        "PitchBend_0",
        "PitchBend_4095",
        "PitchBend_8191",
        "PitchBend_0",
    ]
    assert encoded.num_pitch_bends == 4
    assert [bend.value for bend in decoded.score.tracks[0].pitch_bends] == [
        0,
        4095,
        8191,
        0,
    ]
    assert decoded.techniques == ()


def test_condition_prefix_is_exact_and_survives_symbolic_round_trip(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    path = _write_guitar_midi(
        tmp_path / "conditioned.mid", make_instrument, write_midi_file
    )
    tokenizer = build_tokenizer(RemiTokenizerConfig())
    encoded = encode_midi(tokenizer, path, tonic="f#", mode="natural minor")
    decoded = decode_symbolic_token_ids(tokenizer, encoded.ids, encoded.programs)

    assert (encoded.tonic, encoded.mode) == ("F_SHARP", "MINOR")
    assert (decoded.tonic, decoded.mode) == ("F_SHARP", "MINOR")
    assert [tokenizer[token_id] for token_id in encoded.ids[:4]] == [
        "BOS_None",
        "Tonic_F_SHARP",
        "Mode_MINOR",
        "Bar_None",
    ]

    missing_tonic = (encoded.ids[0], *encoded.ids[2:])
    with pytest.raises(TokenizationError, match="exactly one Tonic"):
        decode_symbolic_token_ids(tokenizer, missing_tonic, encoded.programs)

    reversed_prefix = (
        encoded.ids[0],
        encoded.ids[2],
        encoded.ids[1],
        *encoded.ids[3:],
    )
    with pytest.raises(TokenizationError, match="exactly one Tonic"):
        decode_symbolic_token_ids(tokenizer, reversed_prefix, encoded.programs)

    duplicated_tonic = (*encoded.ids[:-1], encoded.ids[1], encoded.ids[-1])
    with pytest.raises(TokenizationError, match="exactly one Tonic"):
        decode_symbolic_token_ids(tokenizer, duplicated_tonic, encoded.programs)

    with pytest.raises(TokenizationError, match="Invalid tonal conditioning"):
        encode_midi(tokenizer, path, tonic="H", mode="MINOR")
    with pytest.raises(TokenizationError, match="Mode must be UNKNOWN"):
        encode_midi(tokenizer, path, tonic="UNKNOWN", mode="MINOR")


def test_all_techniques_are_postfix_and_survive_symbolic_round_trip(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    # Simultaneous notes have deliberately different durations. This guards
    # against associating note_index with Symusic's internal event order.
    instrument = make_instrument(
        [
            (60, 0.0, 1.0),
            (64, 0.0, 0.5),
            (67, 1.0, 1.5),
            (69, 2.0, 2.5),
        ],
        program=30,
    )
    path = write_midi_file(tmp_path / "techniques.mid", [instrument])
    tokenizer = build_tokenizer(RemiTokenizerConfig())
    techniques = [
        {"type": "VIBRATO", "note_index": 3},
        {"type": "SLIDE_DOWN", "note_index": 3},
        {"type": "PALM_MUTE_OFF", "note_index": 3},
        {"type": "SLIDE_UP", "note_index": 2},
        {"type": "DEAD_NOTE", "note_index": 1},
        {"type": "PALM_MUTE_ON", "note_index": 0},
    ]

    encoded = encode_midi(tokenizer, path, techniques=techniques)
    tokens = [tokenizer[token_id] for token_id in encoded.musical_ids]
    decoded = decode_symbolic_token_ids(tokenizer, encoded.ids, encoded.programs)

    expected = (
        TechniqueAnnotation("PALM_MUTE_ON", 0),
        TechniqueAnnotation("DEAD_NOTE", 1),
        TechniqueAnnotation("SLIDE_UP", 2),
        TechniqueAnnotation("PALM_MUTE_OFF", 3),
        TechniqueAnnotation("SLIDE_DOWN", 3),
        TechniqueAnnotation("VIBRATO", 3),
    )
    assert encoded.techniques == expected
    assert decoded.techniques == expected
    assert decode_token_ids(tokenizer, encoded.ids, encoded.programs).tracks[0].notes

    for token in GUITAR_TECHNIQUE_TOKENS:
        index = tokens.index(token)
        assert tokens[index - 1].startswith("Duration_") or tokens[
            index - 1
        ].startswith("Technique_")

    dead_index = tokens.index(TECHNIQUE_TOKEN_BY_TYPE["DEAD_NOTE"])
    assert tokens[dead_index - 1].startswith("Duration_")
    assert tokens[dead_index - 2] == "Pitch_64"
    palm_on_index = tokens.index(TECHNIQUE_TOKEN_BY_TYPE["PALM_MUTE_ON"])
    assert tokens[palm_on_index - 2] == "Pitch_60"


def test_velocity_tokens_are_optional_and_still_idempotent(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    midi_path = _write_guitar_midi(
        tmp_path / "dynamic.mid", make_instrument, write_midi_file
    )
    tokenizer = build_tokenizer(RemiTokenizerConfig(use_velocities=True))

    encoded = encode_midi(
        tokenizer,
        midi_path,
        techniques=({"type": "VIBRATO", "note_index": 1},),
    )
    musical_tokens = [tokenizer[token_id] for token_id in encoded.musical_ids]

    assert any(token.startswith("Velocity_") for token in musical_tokens)
    assert encoded.techniques == (TechniqueAnnotation("VIBRATO", 1),)
    assert encoded.round_trip_ok is True
    assert encoded.token_error_ratio == 0.0


def test_saved_tokenizer_reloads_with_identical_encoding(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    midi_path = _write_guitar_midi(
        tmp_path / "riff.mid", make_instrument, write_midi_file
    )
    tokenizer = build_tokenizer(RemiTokenizerConfig())
    before = encode_midi(tokenizer, midi_path)

    tokenizer_path = save_tokenizer(
        tokenizer,
        tmp_path / "artifact" / "tokenizer.json",
        additional_attributes={"stage": 2},
    )
    restored = load_tokenizer(tokenizer_path)
    after = encode_midi(restored, midi_path)

    assert type(restored) is ConditionedGuitarREMI
    assert restored.vocab == tokenizer.vocab
    assert get_special_token_ids(restored) == get_special_token_ids(tokenizer)
    assert get_technique_token_ids(restored) == get_technique_token_ids(tokenizer)
    payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    assert payload["tokenization"] == "ConditionedGuitarREMI"
    assert payload["conditioning_schema_version"] == CONDITIONING_SCHEMA_VERSION
    assert payload["tonic_names"] == list(TONIC_NAMES)
    assert payload["mode_names"] == list(MODE_NAMES)
    assert payload["guitar_technique_tokens"] == list(GUITAR_TECHNIQUE_TOKENS)
    assert payload["pitch_bend_sensitivity_semitones"] == 6
    assert after.ids == before.ids
    assert after.programs == before.programs


def test_save_rejects_reserved_metadata_override_and_loads_legacy_remi(
    tmp_path: Path,
) -> None:
    tokenizer = build_tokenizer(RemiTokenizerConfig())
    with pytest.raises(TokenizationError, match="reserved field"):
        save_tokenizer(
            tokenizer,
            tmp_path / "conflict.json",
            additional_attributes={"pitch_bend_sensitivity_semitones": 2},
        )
    with pytest.raises(TokenizationError, match="serialization field"):
        save_tokenizer(
            tokenizer,
            tmp_path / "wrong-class.json",
            additional_attributes={"tokenization": "REMI"},
        )

    legacy = REMI(
        TokenizerConfig(
            special_tokens=["PAD", "BOS", "EOS"],
            use_velocities=False,
        )
    )
    legacy_path = tmp_path / "legacy.json"
    legacy.save(legacy_path)

    restored = load_tokenizer(legacy_path)

    assert type(restored) is REMI
    with pytest.raises(TokenizationError, match="does not define technique token"):
        get_technique_token_ids(restored)

    legacy_guitar = GuitarREMI(tokenizer_config=tokenizer.config)
    legacy_guitar_path = save_tokenizer(
        legacy_guitar, tmp_path / "legacy-guitar.json"
    )
    restored_guitar = load_tokenizer(legacy_guitar_path)

    assert type(restored_guitar) is GuitarREMI
    with pytest.raises(TokenizationError, match="does not define tonic token"):
        get_tonic_token_ids(restored_guitar)


def test_load_rejects_corrupt_guitar_metadata(tmp_path: Path) -> None:
    path = save_tokenizer(
        build_tokenizer(RemiTokenizerConfig()), tmp_path / "tokenizer.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pitch_bend_sensitivity_semitones"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TokenizationError, match="sensitivity metadata"):
        load_tokenizer(path)


def test_decode_rejects_missing_boundaries_and_pad(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    midi_path = _write_guitar_midi(
        tmp_path / "riff.mid", make_instrument, write_midi_file
    )
    tokenizer = build_tokenizer(RemiTokenizerConfig())
    encoded = encode_midi(tokenizer, midi_path)
    special_ids = get_special_token_ids(tokenizer)

    with pytest.raises(TokenizationError, match="begin with BOS"):
        decode_token_ids(tokenizer, encoded.musical_ids, encoded.programs)
    with pytest.raises(TokenizationError, match="PAD is only for batching"):
        decode_token_ids(
            tokenizer,
            (encoded.ids[0], special_ids.pad, *encoded.ids[1:]),
            encoded.programs,
        )


def test_decode_rejects_invalid_program_descriptor(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    midi_path = _write_guitar_midi(
        tmp_path / "riff.mid", make_instrument, write_midi_file
    )
    tokenizer = build_tokenizer(RemiTokenizerConfig())
    encoded = encode_midi(tokenizer, midi_path)

    with pytest.raises(TokenizationError, match="exactly one program"):
        decode_token_ids(tokenizer, encoded.ids, ())
    with pytest.raises(TokenizationError, match="does not support drum"):
        decode_token_ids(tokenizer, encoded.ids, ((30, True),))


@pytest.mark.parametrize(
    ("techniques", "message"),
    [
        ([{"type": "TAPPING", "note_index": 0}], "must be one of"),
        ([{"type": "VIBRATO", "note_index": 9}], "outside the 3-note track"),
        ([{"type": "VIBRATO", "note_index": True}], "must be an integer"),
        ([{"type": "VIBRATO", "note_index": 0, "extra": 1}], "exactly"),
        (
            [
                {"type": "VIBRATO", "note_index": 0},
                {"type": "VIBRATO", "note_index": 0},
            ],
            "Duplicate technique",
        ),
        (
            [
                {"type": "SLIDE_UP", "note_index": 0},
                {"type": "SLIDE_DOWN", "note_index": 0},
            ],
            "cannot use SLIDE_UP and SLIDE_DOWN",
        ),
        ([{"type": "PALM_MUTE_OFF", "note_index": 0}], "has no active mute"),
        (
            [
                {"type": "PALM_MUTE_ON", "note_index": 0},
                {"type": "PALM_MUTE_ON", "note_index": 1},
            ],
            "is redundant",
        ),
        (
            [
                {"type": "PALM_MUTE_ON", "note_index": 0},
                {"type": "PALM_MUTE_OFF", "note_index": 0},
            ],
            "cannot switch palm mute",
        ),
    ],
)
def test_encode_rejects_invalid_technique_annotations(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
    techniques,
    message: str,
) -> None:
    path = _write_guitar_midi(
        tmp_path / "invalid-techniques.mid", make_instrument, write_midi_file
    )

    with pytest.raises(TokenizationError, match=message):
        encode_midi(
            build_tokenizer(RemiTokenizerConfig()),
            path,
            techniques=techniques,
        )


def test_decode_rejects_nonpostfix_and_semantically_invalid_techniques(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    path = _write_guitar_midi(
        tmp_path / "malformed-techniques.mid", make_instrument, write_midi_file
    )
    tokenizer = build_tokenizer(RemiTokenizerConfig())
    encoded = encode_midi(tokenizer, path)
    special = get_special_token_ids(tokenizer)
    tokens = [tokenizer[token_id] for token_id in encoded.musical_ids]

    pitch_index = next(
        index for index, token in enumerate(tokens) if token.startswith("Pitch_")
    )
    orphan_tokens = list(tokens)
    orphan_tokens.insert(pitch_index, TECHNIQUE_TOKEN_BY_TYPE["VIBRATO"])
    orphan_ids = tuple(tokenizer.vocab[token] for token in orphan_tokens)
    with pytest.raises(TokenizationError, match="error ratio"):
        decode_symbolic_token_ids(
            tokenizer,
            (special.bos, *orphan_ids, special.eos),
            encoded.programs,
        )

    valid = encode_midi(
        tokenizer,
        path,
        techniques=({"type": "VIBRATO", "note_index": 0},),
    )
    palm_off_id = get_technique_token_ids(tokenizer)["PALM_MUTE_OFF"]
    invalid_semantics = tuple(
        palm_off_id
        if token_id == get_technique_token_ids(tokenizer)["VIBRATO"]
        else token_id
        for token_id in valid.ids
    )
    with pytest.raises(TokenizationError, match="has no active mute"):
        decode_symbolic_token_ids(tokenizer, invalid_semantics, valid.programs)


def test_encode_rejects_multitrack_midi(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    first = make_instrument([(60, 0.0, 0.5)], program=25, name="Guitar 1")
    second = make_instrument([(67, 0.5, 1.0)], program=26, name="Guitar 2")
    midi_path = write_midi_file(tmp_path / "two.mid", [first, second])

    with pytest.raises(TokenizationError, match="exactly one MIDI track"):
        encode_midi(build_tokenizer(RemiTokenizerConfig()), midi_path)


def test_encode_rejects_empty_and_drum_tracks(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    empty_path = write_midi_file(
        tmp_path / "empty.mid", [make_instrument([], program=30)]
    )
    drum_path = write_midi_file(
        tmp_path / "drums.mid",
        [make_instrument([(36, 0.0, 0.5)], is_drum=True)],
    )
    tokenizer = build_tokenizer(RemiTokenizerConfig())

    with pytest.raises(TokenizationError, match="exactly one MIDI track|non-empty"):
        encode_midi(tokenizer, empty_path)
    with pytest.raises(TokenizationError, match="pitched instrumental tracks"):
        encode_midi(tokenizer, drum_path)


def test_encode_rejects_notes_that_tokenizer_would_drop(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    instrument = make_instrument(
        [(20, 0.0, 0.5), (60, 0.5, 1.0)], program=30
    )
    midi_path = write_midi_file(tmp_path / "outside-range.mid", [instrument])

    with pytest.raises(TokenizationError, match="changed the note count or pitches"):
        encode_midi(build_tokenizer(RemiTokenizerConfig()), midi_path)


def test_invalid_files_and_tokenizer_paths_raise_adapter_errors(tmp_path: Path) -> None:
    tokenizer = build_tokenizer(RemiTokenizerConfig())
    corrupt_midi = tmp_path / "corrupt.mid"
    corrupt_midi.write_bytes(b"not a MIDI")

    with pytest.raises(TokenizationError, match="Could not parse MIDI"):
        encode_midi(tokenizer, corrupt_midi)
    with pytest.raises(TokenizationError, match="must use a .json extension"):
        save_tokenizer(tokenizer, tmp_path / "tokenizer.txt")
    with pytest.raises(TokenizationError, match="does not exist"):
        load_tokenizer(tmp_path / "missing.json")
