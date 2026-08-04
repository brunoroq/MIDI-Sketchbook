"""Tests for synthetic MIDI discovery, reading, and inspection."""

from __future__ import annotations

from pathlib import Path

import mido
import pretty_midi
import pytest

from midi_idea_generator.config import TrackSelectionConfig, ValidationConfig
from midi_idea_generator.midi_io import (
    MidiReadError,
    MidiWriteError,
    discover_midi_files,
    inspect_midi,
    read_midi,
    write_midi,
)


def test_discover_and_read_programmatically_generated_midis(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    input_dir = tmp_path / "raw"
    first = write_midi_file(
        input_dir / "A.mid",
        [make_instrument([(60, 0.0, 0.5), (64, 0.5, 1.0)])],
    )
    second = write_midi_file(
        input_dir / "nested" / "b.MIDI",
        [make_instrument([(67, 0.0, 0.25)])],
    )
    (input_dir / "ignore.txt").write_text("not MIDI", encoding="utf-8")
    (input_dir / "also-ignore.mid.bak").write_bytes(b"not MIDI either")

    discovered = discover_midi_files(input_dir)
    loaded = read_midi(first)

    assert discovered == [first.resolve(), second.resolve()]
    assert len(loaded.instruments) == 1
    assert [note.pitch for note in loaded.instruments[0].notes] == [60, 64]
    assert loaded.get_end_time() == pytest.approx(1.0)


def test_inspect_midi_selects_most_notes_from_valid_non_drum_tracks(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    lead = make_instrument(
        [(60, 0.0, 0.5), (62, 0.5, 1.0), (64, 1.0, 1.5)],
        program=29,
        name="lead",
    )
    drums = make_instrument(
        [(36, 0.0, 0.25), (38, 0.25, 0.5), (42, 0.5, 0.75), (36, 0.75, 1.0)],
        is_drum=True,
        name="drums",
    )
    path = write_midi_file(tmp_path / "selection.mid", [lead, drums])

    inspection = inspect_midi(path, ValidationConfig())

    assert inspection.readable is True
    assert inspection.compatible is True
    assert inspection.discard_reason is None
    assert inspection.selected_track == 0
    assert inspection.tempo_bpm == pytest.approx(120.0)
    assert inspection.tempo_change_count == 1
    assert inspection.time_signatures == ("4/4",)
    assert inspection.duration_seconds == pytest.approx(1.5)
    assert len(inspection.tracks) == 2
    assert inspection.tracks[0].valid is True
    assert inspection.tracks[0].num_notes == 3
    assert inspection.tracks[0].min_pitch == 60
    assert inspection.tracks[0].max_pitch == 64
    assert inspection.tracks[1].valid is False
    assert "Drum tracks are excluded" in inspection.tracks[1].issues
    serialized = inspection.to_dict(source_label="fixture/selection.mid")
    assert serialized["source_file"] == "fixture/selection.mid"
    assert serialized["tracks"][1]["issues"] == ["Drum tracks are excluded"]


def test_inspect_midi_reports_meter_pitch_bend_and_range_incompatibilities(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    instrument = make_instrument([(20, 0.0, 0.5), (60, 0.5, 1.0)])
    instrument.pitch_bends.append(pretty_midi.PitchBend(pitch=256, time=0.25))
    path = write_midi_file(
        tmp_path / "incompatible.mid",
        [instrument],
        time_signature=(3, 4),
    )

    inspection = inspect_midi(path, ValidationConfig())

    assert inspection.readable is True
    assert inspection.compatible is False
    assert inspection.selected_track is None
    assert any("Unsupported time signature" in issue for issue in inspection.issues)
    assert any("No valid instrumental track" in issue for issue in inspection.issues)
    assert any("outside 21-108" in issue for issue in inspection.tracks[0].issues)
    assert "Track contains pitch bends" in inspection.tracks[0].issues
    assert inspection.discard_reason is not None


def test_inspection_rejects_control_changes_that_would_be_lost(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    instrument = make_instrument([(60, 0.0, 1.0)])
    instrument.control_changes.append(
        pretty_midi.ControlChange(number=64, value=127, time=0.25)
    )
    path = write_midi_file(tmp_path / "sustain.mid", [instrument])

    inspection = inspect_midi(path, ValidationConfig())

    assert inspection.compatible is False
    assert any(
        "unsupported MIDI control changes" in issue
        for issue in inspection.tracks[0].issues
    )


def test_inspection_accepts_guitar_pro_pitch_bend_range_setup(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    instrument = make_instrument([(60, 0.0, 1.0)])
    instrument.control_changes.extend(
        [
            pretty_midi.ControlChange(number=101, value=0, time=0.0),
            pretty_midi.ControlChange(number=100, value=0, time=0.0),
            pretty_midi.ControlChange(number=6, value=6, time=0.0),
        ]
    )
    path = write_midi_file(tmp_path / "guitar-pro-rpn.mid", [instrument])

    inspection = inspect_midi(path, ValidationConfig())

    assert inspection.compatible is True
    assert inspection.tracks[0].issues == ()


def test_inspection_accepts_neutral_pitchwheel_resets(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    instrument = make_instrument([(60, 0.0, 1.0)])
    instrument.pitch_bends.extend(
        [
            pretty_midi.PitchBend(pitch=0, time=0.0),
            pretty_midi.PitchBend(pitch=0, time=0.5),
        ]
    )
    path = write_midi_file(tmp_path / "neutral-pitchwheel.mid", [instrument])

    inspection = inspect_midi(path, ValidationConfig())

    assert inspection.compatible is True
    assert inspection.tracks[0].has_pitch_bends is False


def test_inspection_rejects_lyrics(
    tmp_path: Path,
    make_instrument,
) -> None:
    path = tmp_path / "lyrics.mid"
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    midi.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    midi.lyrics.append(pretty_midi.Lyric(text="synthetic words", time=0.0))
    midi.instruments.append(make_instrument([(60, 0.0, 0.5)]))
    midi.write(str(path))

    inspection = inspect_midi(path, ValidationConfig())

    assert inspection.readable is True
    assert inspection.compatible is False
    assert "Lyrics are unsupported" in inspection.discard_reason


def test_inspect_midi_honors_missing_signature_and_explicit_track_selection(
    tmp_path: Path,
    make_instrument,
    write_midi_file,
) -> None:
    path = write_midi_file(
        tmp_path / "implicit-meter.mid",
        [
            make_instrument([(60, 0.0, 0.5), (62, 0.5, 1.0)]),
            make_instrument([(72, 0.0, 0.5)]),
        ],
        time_signature=None,
    )

    allowed = inspect_midi(
        path,
        ValidationConfig(allow_missing_time_signature=True),
        TrackSelectionConfig(mode="index", track_index=1),
    )
    rejected = inspect_midi(
        path,
        ValidationConfig(allow_missing_time_signature=False),
    )

    assert allowed.compatible is True
    assert allowed.selected_track == 1
    assert allowed.time_signatures == ("4/4 (implicit)",)
    assert rejected.compatible is False
    assert "Time signature is missing" in rejected.issues


def test_corrupt_midi_is_wrapped_and_inspection_does_not_raise(
    tmp_path: Path,
) -> None:
    corrupt = tmp_path / "corrupt.mid"
    corrupt.write_bytes(b"this is deliberately not a MIDI stream")

    with pytest.raises(MidiReadError, match="Could not parse.*corrupt.mid"):
        read_midi(corrupt)

    inspection = inspect_midi(corrupt, ValidationConfig())

    assert inspection.source_file == corrupt.resolve()
    assert inspection.readable is False
    assert inspection.compatible is False
    assert inspection.selected_track is None
    assert inspection.tracks == ()
    assert inspection.tempo_bpm is None
    assert inspection.discard_reason is not None
    assert "Could not parse" in inspection.discard_reason


def test_inspection_rejects_asynchronous_type_two_midi(tmp_path: Path) -> None:
    path = tmp_path / "asynchronous.mid"
    raw = mido.MidiFile(type=2, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.extend(
        [
            mido.Message("note_on", note=60, velocity=90, time=0),
            mido.Message("note_off", note=60, velocity=0, time=480),
        ]
    )
    raw.tracks.append(track)
    raw.save(path)

    inspection = inspect_midi(path, ValidationConfig())

    assert inspection.compatible is False
    assert inspection.readable is True
    assert "Unsupported asynchronous" in inspection.discard_reason


def test_inspection_rejects_global_metadata_outside_track_zero(
    tmp_path: Path,
) -> None:
    path = tmp_path / "misplaced-tempo.mid"
    raw = mido.MidiFile(type=1, ticks_per_beat=480)
    raw.tracks.append(mido.MidiTrack([mido.MetaMessage("end_of_track", time=0)]))
    musical_track = mido.MidiTrack(
        [
            mido.MetaMessage("set_tempo", tempo=500_000, time=0),
            mido.Message("note_on", note=60, velocity=90, time=0),
            mido.Message("note_off", note=60, velocity=0, time=480),
        ]
    )
    raw.tracks.append(musical_track)
    raw.save(path)

    inspection = inspect_midi(path, ValidationConfig())

    assert inspection.compatible is False
    assert inspection.readable is True
    assert "Global metadata appears outside track 0" in inspection.discard_reason


def test_atomic_writer_rejects_ambiguous_same_pitch_overlap(
    tmp_path: Path,
    make_instrument,
) -> None:
    output = tmp_path / "ambiguous.mid"
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0, resolution=480)
    midi.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    midi.instruments.append(
        make_instrument([(60, 0.0, 2.0), (60, 1.0, 3.0)])
    )

    with pytest.raises(MidiWriteError, match="Could not validate written MIDI"):
        write_midi(midi, output)

    assert not output.exists()


def test_inspection_accepts_exact_duplicate_unison_note_events(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate-unison.mid"
    raw = mido.MidiFile(type=0, ticks_per_beat=480)
    raw.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("set_tempo", tempo=500_000, time=0),
                mido.MetaMessage(
                    "time_signature", numerator=4, denominator=4, time=0
                ),
                mido.Message("note_on", note=59, velocity=76, time=0),
                mido.Message("note_on", note=59, velocity=76, time=0),
                mido.Message("note_off", note=59, velocity=64, time=240),
                mido.Message("note_off", note=59, velocity=64, time=0),
            ]
        )
    )
    raw.save(path)

    inspection = inspect_midi(path, ValidationConfig())

    assert inspection.compatible is True
    assert inspection.tracks[0].num_notes == 1
    assert inspection.tracks[0].raw_note_events == 2
    assert inspection.tracks[0].duplicate_notes_collapsed == 1


def test_inspection_rejects_simultaneous_duplicates_with_different_ends(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ambiguous-unison.mid"
    raw = mido.MidiFile(type=0, ticks_per_beat=480)
    raw.tracks.append(
        mido.MidiTrack(
            [
                mido.Message("note_on", note=59, velocity=76, time=0),
                mido.Message("note_on", note=59, velocity=76, time=0),
                mido.Message("note_off", note=59, velocity=64, time=120),
                mido.Message("note_off", note=59, velocity=64, time=120),
            ]
        )
    )
    raw.save(path)

    inspection = inspect_midi(path, ValidationConfig())

    assert inspection.compatible is False
    assert "Overlapping note-on" in inspection.discard_reason


def test_inspection_rejects_dangling_note_on(tmp_path: Path) -> None:
    path = tmp_path / "dangling.mid"
    raw = mido.MidiFile(type=0, ticks_per_beat=480)
    raw.tracks.append(
        mido.MidiTrack(
            [
                mido.Message("note_on", note=60, velocity=90, time=0),
                mido.Message("note_off", note=60, velocity=0, time=480),
                mido.Message("note_on", note=64, velocity=90, time=0),
            ]
        )
    )
    raw.save(path)

    inspection = inspect_midi(path, ValidationConfig())

    assert inspection.readable is True
    assert inspection.compatible is False
    assert "Dangling note-on" in inspection.discard_reason


def test_writer_tolerates_one_microsecond_tempo_rounding(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "awkward-tempo-source.mid"
    output_path = tmp_path / "awkward-tempo-output.mid"
    raw = mido.MidiFile(type=0, ticks_per_beat=480)
    raw.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("set_tempo", tempo=999_999, time=0),
                mido.MetaMessage(
                    "time_signature", numerator=4, denominator=4, time=0
                ),
                mido.Message("note_on", note=60, velocity=90, time=0),
                mido.Message("note_off", note=60, velocity=0, time=480),
            ]
        )
    )
    raw.save(source_path)

    write_midi(read_midi(source_path), output_path)

    assert output_path.is_file()
    assert len(read_midi(output_path).instruments[0].notes) == 1


def test_inspection_rejects_zero_ppq_without_aborting(tmp_path: Path) -> None:
    path = tmp_path / "zero-ppq.mid"
    raw = mido.MidiFile(type=0, ticks_per_beat=480)
    raw.tracks.append(
        mido.MidiTrack(
            [
                mido.Message("note_on", note=60, velocity=90, time=0),
                mido.Message("note_off", note=60, velocity=0, time=480),
            ]
        )
    )
    raw.save(path)
    malformed = bytearray(path.read_bytes())
    malformed[12:14] = b"\x00\x00"
    path.write_bytes(malformed)

    inspection = inspect_midi(path, ValidationConfig())

    assert inspection.readable is True
    assert inspection.compatible is False
    assert "positive pulses-per-quarter" in inspection.discard_reason
