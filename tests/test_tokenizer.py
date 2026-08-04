"""Unit tests for the pure Stage 2 REMI adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from midi_idea_generator.tokenization_config import RemiTokenizerConfig
from midi_idea_generator.tokenizer import (
    TokenizationError,
    build_tokenizer,
    decode_token_ids,
    encode_midi,
    get_special_token_ids,
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
    assert all(
        getattr(tokenizer.config, feature) is False
        for feature in (
            "use_chords",
            "use_rests",
            "use_time_signatures",
            "use_sustain_pedals",
            "use_pitch_bends",
            "use_programs",
            "use_pitch_intervals",
            "use_pitchdrum_tokens",
        )
    )


def test_encode_adds_resolved_boundaries_without_pad_and_round_trips(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    midi_path = _write_guitar_midi(
        tmp_path / "riff.mid", make_instrument, write_midi_file
    )
    tokenizer = build_tokenizer(RemiTokenizerConfig())

    encoded = encode_midi(tokenizer, midi_path)
    special_ids = get_special_token_ids(tokenizer)

    assert encoded.ids[0] == special_ids.bos
    assert encoded.ids[-1] == special_ids.eos
    assert special_ids.pad not in encoded.ids
    assert encoded.ids[1:-1] == encoded.musical_ids
    assert encoded.num_tokens == len(encoded.ids)
    assert encoded.num_musical_tokens == len(encoded.musical_ids)
    assert encoded.num_tokens == encoded.num_musical_tokens + 2
    assert encoded.programs == ((30, False),)
    assert encoded.num_notes == 3
    assert encoded.token_error_ratio == 0.0
    assert encoded.round_trip_ok is True

    musical_tokens = [tokenizer[token_id] for token_id in encoded.musical_ids]
    assert any(token.startswith("Tempo_") for token in musical_tokens)
    assert not any(token.startswith("Velocity_") for token in musical_tokens)

    decoded = decode_token_ids(tokenizer, encoded.ids, encoded.programs)
    assert len(decoded.tracks) == 1
    assert decoded.tracks[0].program == 30
    assert [note.pitch for note in decoded.tracks[0].notes] == [21, 64, 108]


def test_velocity_tokens_are_optional_and_still_idempotent(
    tmp_path: Path, make_instrument, write_midi_file
) -> None:
    midi_path = _write_guitar_midi(
        tmp_path / "dynamic.mid", make_instrument, write_midi_file
    )
    tokenizer = build_tokenizer(RemiTokenizerConfig(use_velocities=True))

    encoded = encode_midi(tokenizer, midi_path)
    musical_tokens = [tokenizer[token_id] for token_id in encoded.musical_ids]

    assert any(token.startswith("Velocity_") for token in musical_tokens)
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

    assert restored.vocab == tokenizer.vocab
    assert get_special_token_ids(restored) == get_special_token_ids(tokenizer)
    assert after.ids == before.ids
    assert after.programs == before.programs


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
