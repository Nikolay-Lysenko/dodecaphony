"""
Test `dodecaphony.rendering` module.

Author: Nikolay Lysenko
"""


from typing import Any

import pretty_midi
import pytest
import yaml

from dodecaphony.fragment import Fragment, ToneRowInstance, override_calculated_attributes
from dodecaphony.rendering import (
    create_lilypond_file_from_fragment,
    create_midi_from_fragment,
    create_sinethesizer_instruments,
    create_tsv_events_from_fragment,
    create_wav_from_tsv_events,
    create_yaml_from_fragment,
)
from .conftest import MEASURE_DURATIONS_BY_N_EVENTS


@pytest.mark.parametrize(
    "fragment, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                    [[2.0, 4.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'])],
                    [ToneRowInstance(['F', 'G#', 'E', 'F#', 'A', 'D', 'C', 'D#', 'C#', 'G', 'A#', 'B'])],
                ],
                grouped_mutable_pauses_indices=[[11], []],
                grouped_immutable_pauses_indices=[[], []],
                n_beats=16,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0], 1: [1]},
                mutable_temporal_content_indices=[0, 1],
                mutable_independent_tone_row_instances_indices=[(0, 0), (1, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `expected`
            (
                "\\version \"2.22.1\"\n"
                "\\layout {\n"
                "    indent = #0\n"
                "}\n"
                "\\new StaffGroup <<\n"
                "    \\new Staff <<\n"
                "        \\clef treble\n"
                "        \\time 4/4\n"
                "        \\key c \\major\n"
                "        {b'4 ais'4 g'4 cis''4 dis''2 c''2 d''4 a'4 fis'4 e'4 gis'2 r4 f'4}\n"
                "    >>\n"
                "    \\new Staff <<\n"
                "        \\clef bass\n"
                "        \\time 4/4\n"
                "        \\key c \\major\n"
                "        {f'2 gis2~ gis2 e'4 fis'4 a'4 d'4 c'4 dis'4 cis'4 g'4 ais4 b4}\n"
                "    >>\n"
                ">>"
            )
        ),
    ]
)
def test_create_lilypond_file_from_fragment(
        path_to_tmp_file: str, fragment: Fragment, expected: str
) -> None:
    """Test `create_lilypond_file_from_fragment` function."""
    override_calculated_attributes(fragment)
    create_lilypond_file_from_fragment(fragment, path_to_tmp_file)
    with open(path_to_tmp_file) as in_file:
        result = in_file.read()
        assert result == expected


@pytest.mark.parametrize(
    "fragment, note_number, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0]],
                    [[2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'])],
                    [ToneRowInstance(['F', 'G#', 'E', 'F#', 'A', 'D', 'C', 'D#', 'C#', 'G', 'A#', 'B'])],
                ],
                grouped_mutable_pauses_indices=[[], []],
                grouped_immutable_pauses_indices=[[], []],
                n_beats=16,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0], 1: [1]},
                mutable_temporal_content_indices=[0, 1],
                mutable_independent_tone_row_instances_indices=[(0, 0), (1, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `note_number`
            5,
            # `expected`
            {
                'pitch': 72,
                'start': 4.0,
                'end': 5.0
            }
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0]],
                    [[2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'D', 'A', 'F#', 'E', 'G#', 'F'])],
                    [ToneRowInstance(['F', 'G#', 'E', 'F#', 'A', 'D', 'C', 'D#', 'C#', 'G', 'A#', 'B'])],
                ],
                grouped_mutable_pauses_indices=[[5], []],
                grouped_immutable_pauses_indices=[[], []],
                n_beats=16,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0], 1: [1]},
                mutable_temporal_content_indices=[0, 1],
                mutable_independent_tone_row_instances_indices=[(0, 0), (1, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `note_number`
            5,
            # `expected`
            {
                'pitch': 74,
                'start': 5.0,
                'end': 5.5
            }
        ),
    ]
)
def test_create_midi_from_fragment(
        path_to_tmp_file: str, fragment: Fragment, note_number: int, expected: dict[str, float]
) -> None:
    """Test `create_midi_from_fragment` function."""
    override_calculated_attributes(fragment)
    create_midi_from_fragment(
        fragment,
        path_to_tmp_file,
        beat_in_seconds=0.5,
        instruments={k: 0 for k in fragment.line_ids},
        velocity=100,
        opening_silence_in_seconds=1,
        trailing_silence_in_seconds=1
    )
    midi_data = pretty_midi.PrettyMIDI(path_to_tmp_file)
    instrument = midi_data.instruments[0]
    midi_note = instrument.notes[note_number]
    result = {
        'pitch': midi_note.pitch,
        'start': midi_note.start,
        'end': midi_note.end
    }
    assert result == expected


@pytest.mark.parametrize(
    "fragment, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0]],
                    [[2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'])],
                    [ToneRowInstance(['F', 'G#', 'E', 'F#', 'A', 'D', 'C', 'D#', 'C#', 'G', 'A#', 'B'])],
                ],
                grouped_mutable_pauses_indices=[[], []],
                grouped_immutable_pauses_indices=[[], []],
                n_beats=16,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0], 1: [1]},
                mutable_temporal_content_indices=[0, 1],
                mutable_independent_tone_row_instances_indices=[(0, 0), (1, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `expected`
            (
                "instrument\tstart_time\tduration\tfrequency\tvelocity\teffects\tline_id\n"
                "breathy_open_diapason\t1.0\t1.0\tF4\t1.0\t\t2\n"
                "breathy_open_diapason\t1.0\t0.5\tB4\t1.0\t\t1\n"
                "breathy_open_diapason\t1.5\t0.5\tA#4\t1.0\t\t1\n"
                "breathy_open_diapason\t2.0\t1.0\tG#3\t1.0\t\t2\n"
                "breathy_open_diapason\t2.0\t0.5\tG4\t1.0\t\t1\n"
                "breathy_open_diapason\t2.5\t0.5\tC#5\t1.0\t\t1\n"
                "breathy_open_diapason\t3.0\t0.5\tE4\t1.0\t\t2\n"
                "breathy_open_diapason\t3.0\t1.0\tD#5\t1.0\t\t1\n"
                "breathy_open_diapason\t3.5\t0.5\tF#4\t1.0\t\t2\n"
                "breathy_open_diapason\t4.0\t0.5\tA4\t1.0\t\t2\n"
                "breathy_open_diapason\t4.0\t1.0\tC5\t1.0\t\t1\n"
                "breathy_open_diapason\t4.5\t0.5\tD4\t1.0\t\t2\n"
                "breathy_open_diapason\t5.0\t1.0\tC4\t1.0\t\t2\n"
                "breathy_open_diapason\t5.0\t0.5\tD5\t1.0\t\t1\n"
                "breathy_open_diapason\t5.5\t0.5\tA4\t1.0\t\t1\n"
                "breathy_open_diapason\t6.0\t1.0\tD#4\t1.0\t\t2\n"
                "breathy_open_diapason\t6.0\t0.5\tF#4\t1.0\t\t1\n"
                "breathy_open_diapason\t6.5\t0.5\tE4\t1.0\t\t1\n"
                "breathy_open_diapason\t7.0\t0.5\tC#4\t1.0\t\t2\n"
                "breathy_open_diapason\t7.0\t1.0\tG#4\t1.0\t\t1\n"
                "breathy_open_diapason\t7.5\t0.5\tG4\t1.0\t\t2\n"
                "breathy_open_diapason\t8.0\t0.5\tA#3\t1.0\t\t2\n"
                "breathy_open_diapason\t8.0\t1.0\tF4\t1.0\t\t1\n"
                "breathy_open_diapason\t8.5\t0.5\tB3\t1.0\t\t2\n"
            )
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0]],
                    [[2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'])],
                    [ToneRowInstance(['G#', 'E', 'F#', 'A', 'D', 'C', 'D#', 'C#', 'G', 'A#', 'B'])],
                ],
                grouped_mutable_pauses_indices=[[], [0]],
                grouped_immutable_pauses_indices=[[], []],
                n_beats=16,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0], 1: [1]},
                mutable_temporal_content_indices=[0, 1],
                mutable_independent_tone_row_instances_indices=[(0, 0), (1, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `expected`
            (
                "instrument\tstart_time\tduration\tfrequency\tvelocity\teffects\tline_id\n"
                "breathy_open_diapason\t1.0\t0.5\tB4\t1.0\t\t1\n"
                "breathy_open_diapason\t1.5\t0.5\tA#4\t1.0\t\t1\n"
                "breathy_open_diapason\t2.0\t1.0\tG#3\t1.0\t\t2\n"
                "breathy_open_diapason\t2.0\t0.5\tG4\t1.0\t\t1\n"
                "breathy_open_diapason\t2.5\t0.5\tC#5\t1.0\t\t1\n"
                "breathy_open_diapason\t3.0\t0.5\tE4\t1.0\t\t2\n"
                "breathy_open_diapason\t3.0\t1.0\tD#5\t1.0\t\t1\n"
                "breathy_open_diapason\t3.5\t0.5\tF#4\t1.0\t\t2\n"
                "breathy_open_diapason\t4.0\t0.5\tA4\t1.0\t\t2\n"
                "breathy_open_diapason\t4.0\t1.0\tC5\t1.0\t\t1\n"
                "breathy_open_diapason\t4.5\t0.5\tD4\t1.0\t\t2\n"
                "breathy_open_diapason\t5.0\t1.0\tC4\t1.0\t\t2\n"
                "breathy_open_diapason\t5.0\t0.5\tD5\t1.0\t\t1\n"
                "breathy_open_diapason\t5.5\t0.5\tA4\t1.0\t\t1\n"
                "breathy_open_diapason\t6.0\t1.0\tD#4\t1.0\t\t2\n"
                "breathy_open_diapason\t6.0\t0.5\tF#4\t1.0\t\t1\n"
                "breathy_open_diapason\t6.5\t0.5\tE4\t1.0\t\t1\n"
                "breathy_open_diapason\t7.0\t0.5\tC#4\t1.0\t\t2\n"
                "breathy_open_diapason\t7.0\t1.0\tG#4\t1.0\t\t1\n"
                "breathy_open_diapason\t7.5\t0.5\tG4\t1.0\t\t2\n"
                "breathy_open_diapason\t8.0\t0.5\tA#3\t1.0\t\t2\n"
                "breathy_open_diapason\t8.0\t1.0\tF4\t1.0\t\t1\n"
                "breathy_open_diapason\t8.5\t0.5\tB3\t1.0\t\t2\n"
            )
        ),
    ]
)
def test_create_tsv_events_from_fragment(
        path_to_tmp_file: str, fragment: Fragment, expected: str
) -> None:
    """Test `create_tsv_events_from_fragment` function."""
    override_calculated_attributes(fragment)
    create_tsv_events_from_fragment(
        fragment,
        path_to_tmp_file,
        beat_in_seconds=0.5,
        instruments={k: 'breathy_open_diapason' for k in fragment.line_ids},
        effects={k: '' for k in fragment.line_ids},
        velocity=1.0,
        opening_silence_in_seconds=1.0
    )
    with open(path_to_tmp_file) as in_file:
        result = in_file.read()
        assert result == expected


@pytest.mark.parametrize(
    "tsv_content, trailing_silence_in_seconds",
    [
        (
            [
                'instrument\tstart_time\tduration\tfrequency\tvelocity\teffects\tline_id',
                'breathy_open_diapason\t1\t1\tA0\t1\t\t1',
                'breathy_open_diapason\t2\t1\t27.5\t1\t[{"name": "tremolo", "frequency": 1}]\t1'
            ],
            1.0
        ),
    ]
)
def test_create_wav_from_tsv_events(
        path_to_tmp_file: str, path_to_another_tmp_file: str,
        tsv_content: list[str], trailing_silence_in_seconds: float
) -> None:
    """Test `create_wav_from_tsv_events` function."""
    with open(path_to_tmp_file, 'w') as tmp_tsv_file:
        for line in tsv_content:
            tmp_tsv_file.write(line + '\n')
    instruments_registry = create_sinethesizer_instruments()
    create_wav_from_tsv_events(
        path_to_tmp_file,
        path_to_another_tmp_file,
        instruments_registry,
        trailing_silence_in_seconds
    )


@pytest.mark.parametrize(
    "fragment, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0]],
                    [[2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'])],
                    [ToneRowInstance(['G#', 'E', 'F#', 'A', 'D', 'C', 'D#', 'C#', 'G', 'A#', 'B'])],
                ],
                grouped_mutable_pauses_indices=[[], [0]],
                grouped_immutable_pauses_indices=[[], []],
                n_beats=16,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0], 1: [1]},
                mutable_temporal_content_indices=[0, 1],
                mutable_independent_tone_row_instances_indices=[(0, 0), (1, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `expected`
            {
                'groups': [
                    {
                        'melodic_line_indices': [0],
                        'tone_row_instances': [{'pitch_classes': ['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']}],
                        'n_pauses': 0,
                        'immutable_pauses_indices': [],
                    },
                    {
                        'melodic_line_indices': [1],
                        'tone_row_instances': [{'pitch_classes': ['G#', 'E', 'F#', 'A', 'D', 'C', 'D#', 'C#', 'G', 'A#', 'B']}],
                        'n_pauses': 1,
                        'immutable_pauses_indices': [0],
                    },
                ],
                'temporal_content': {
                    0: {'durations': [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0]]},
                    1: {'durations': [[2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0]]},
                },
            }
        ),
    ]
)
def test_create_yaml_from_fragment(
        path_to_tmp_file: str, fragment: Fragment, expected: dict[str, Any]
) -> None:
    """Test `create_yaml_from_fragment` function."""
    create_yaml_from_fragment(fragment, path_to_tmp_file)
    with open(path_to_tmp_file) as in_file:
        result = yaml.load(in_file, Loader=yaml.FullLoader)
    assert result == expected
