"""Projection tests for guitar techniques across Stage 1 transforms."""

from __future__ import annotations

from pathlib import Path

import pretty_midi

from midi_idea_generator.config import ProcessingConfig
from midi_idea_generator.preprocessing import (
    normalize_instrument,
    split_instrument_into_phrases,
)
from midi_idea_generator.technique_processing import project_phrase_techniques
from midi_idea_generator.techniques import (
    NoteRef,
    NoteTechniques,
    PalmMuteRange,
    Technique,
    TechniqueSidecar,
    TechniqueType,
)


def _source_fixture() -> tuple[
    pretty_midi.PrettyMIDI,
    pretty_midi.Instrument,
    TechniqueSidecar,
]:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0, resolution=480)
    midi.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    instrument = pretty_midi.Instrument(program=29, name="guitar")
    for pitch, start_tick, end_tick in (
        (60, 960, 1200),
        (62, 1200, 1440),
        (64, 1440, 1680),
    ):
        instrument.notes.append(
            pretty_midi.Note(
                velocity=90,
                pitch=pitch,
                start=midi.tick_to_time(start_tick),
                end=midi.tick_to_time(end_tick),
            )
        )
    midi.instruments.append(instrument)
    refs = tuple(
        NoteRef(
            onset_tick=int(midi.time_to_tick(note.start)),
            end_tick=int(midi.time_to_tick(note.end)),
            pitch=note.pitch,
            velocity=note.velocity,
        )
        for note in instrument.notes
    )
    sidecar = TechniqueSidecar(
        path=Path("riff.mid.techniques.json"),
        fingerprint="0" * 64,
        size_bytes=1,
        source_midi="riff.mid",
        source_sha256="1" * 64,
        ticks_per_quarter=480,
        instrument_index=0,
        coverage="COMPLETE",
        note_techniques=(
            NoteTechniques(
                refs[0],
                (
                    Technique(TechniqueType.DEAD_NOTE),
                    Technique(TechniqueType.PALM_MUTE),
                ),
            ),
            NoteTechniques(
                refs[1],
                (
                    Technique(TechniqueType.PALM_MUTE),
                    Technique(TechniqueType.SLIDE_UP, target_pitch=65),
                    Technique(TechniqueType.VIBRATO),
                ),
            ),
        ),
        palm_mute_ranges=(PalmMuteRange(960, 1440),),
    )
    return midi, instrument, sidecar


def test_projects_quantized_notes_and_fragment_local_palm_mute_state() -> None:
    midi, instrument, sidecar = _source_fixture()
    processing = ProcessingConfig(
        phrase_bars=2,
        remove_initial_silence=True,
        quantize=True,
        subdivisions_per_beat=4,
        include_partial_final_phrase=True,
        min_notes_per_phrase=1,
    )
    normalized = normalize_instrument(
        instrument,
        120.0,
        processing,
        resolution=480,
    )
    phrase = split_instrument_into_phrases(
        normalized,
        120.0,
        processing,
        resolution=480,
    )[0]

    annotations = project_phrase_techniques(
        source_midi=midi,
        instrument_index=0,
        sidecar=sidecar,
        normalized_instrument=normalized,
        phrase=phrase,
        tempo_bpm=120.0,
        processing=processing,
        semitones=1,
        pitch_min=21,
        pitch_max=108,
    )

    assert annotations is not None
    assert [annotation.as_dict() for annotation in annotations] == [
        {"type": "DEAD_NOTE", "note_index": 0},
        {"type": "PALM_MUTE_ON", "note_index": 0},
        {"type": "SLIDE_UP", "note_index": 1},
        {"type": "VIBRATO", "note_index": 1},
        {"type": "PALM_MUTE_OFF", "note_index": 2},
    ]


def test_skips_transposition_when_slide_target_leaves_pitch_range() -> None:
    midi, instrument, sidecar = _source_fixture()
    processing = ProcessingConfig(
        phrase_bars=2,
        remove_initial_silence=False,
        quantize=False,
        subdivisions_per_beat=4,
        include_partial_final_phrase=True,
        min_notes_per_phrase=1,
    )
    normalized = normalize_instrument(
        instrument,
        120.0,
        processing,
        resolution=480,
    )
    phrase = split_instrument_into_phrases(
        normalized,
        120.0,
        processing,
        resolution=480,
    )[0]

    assert project_phrase_techniques(
        source_midi=midi,
        instrument_index=0,
        sidecar=sidecar,
        normalized_instrument=normalized,
        phrase=phrase,
        tempo_bpm=120.0,
        processing=processing,
        semitones=1,
        pitch_min=21,
        pitch_max=65,
    ) is None


def test_absent_sidecar_stays_unannotated() -> None:
    midi, instrument, _ = _source_fixture()
    processing = ProcessingConfig(
        phrase_bars=2,
        remove_initial_silence=False,
        quantize=False,
        subdivisions_per_beat=4,
        include_partial_final_phrase=True,
        min_notes_per_phrase=1,
    )
    normalized = normalize_instrument(instrument, 120.0, processing, resolution=480)
    phrase = split_instrument_into_phrases(
        normalized, 120.0, processing, resolution=480
    )[0]

    assert project_phrase_techniques(
        source_midi=midi,
        instrument_index=0,
        sidecar=None,
        normalized_instrument=normalized,
        phrase=phrase,
        tempo_bpm=120.0,
        processing=processing,
        semitones=0,
        pitch_min=21,
        pitch_max=108,
    ) == ()
