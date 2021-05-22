"""
Test `dodecaphony.rendering` module.

Author: Nikolay Lysenko
"""


import pretty_midi
import pytest

from dodecaphony.fragment import Event, Fragment, override_calculated_attributes
from dodecaphony.rendering import (
    create_lilypond_file_from_fragment,
    create_midi_from_fragment,
    create_sinethesizer_instruments,
    create_tsv_events_from_fragment,
    create_wav_from_tsv_events,
)


@pytest.mark.parametrize(
    "fragment, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
                    [2.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                sonic_content=[
                    ['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'pause', 'F'],
                    ['F', 'G#', 'E', 'F#', 'A', 'D', 'C', 'D#', 'C#', 'G', 'A#', 'B'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=16,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[1, 1],
                n_tone_row_instances_by_group=[1, 1]
            ),
            # `expected`
            (
                "\\version \"2.18.2\"\n"
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
    fragment = override_calculated_attributes(fragment)
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
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
                    [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
                ],
                sonic_content=[
                    ['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                    ['F', 'G#', 'E', 'F#', 'A', 'D', 'C', 'D#', 'C#', 'G', 'A#', 'B'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=16,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[1, 1],
                n_tone_row_instances_by_group=[1, 1]
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
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
                    [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
                ],
                sonic_content=[
                    ['B', 'A#', 'G', 'C#', 'D#', 'pause', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                    ['F', 'G#', 'E', 'F#', 'A', 'D', 'C', 'D#', 'C#', 'G', 'A#', 'B'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=16,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[1, 1],
                n_tone_row_instances_by_group=[1, 1]
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
    fragment = override_calculated_attributes(fragment)
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
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
                    [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
                ],
                sonic_content=[
                    ['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                    ['F', 'G#', 'E', 'F#', 'A', 'D', 'C', 'D#', 'C#', 'G', 'A#', 'B'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=16,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[1, 1],
                n_tone_row_instances_by_group=[1, 1]
            ),
            # `expected`
            (
                "instrument\tstart_time\tduration\tfrequency\tvelocity\teffects\tline_id\n"
                "additive_mellow_pipe\t1.0\t1.0\tF4\t1.0\t\t2\n"
                "additive_mellow_pipe\t1.0\t0.5\tB4\t1.0\t\t1\n"
                "additive_mellow_pipe\t1.5\t0.5\tA#4\t1.0\t\t1\n"
                "additive_mellow_pipe\t2.0\t1.0\tG#3\t1.0\t\t2\n"
                "additive_mellow_pipe\t2.0\t0.5\tG4\t1.0\t\t1\n"
                "additive_mellow_pipe\t2.5\t0.5\tC#5\t1.0\t\t1\n"
                "additive_mellow_pipe\t3.0\t0.5\tE4\t1.0\t\t2\n"
                "additive_mellow_pipe\t3.0\t1.0\tD#5\t1.0\t\t1\n"
                "additive_mellow_pipe\t3.5\t0.5\tF#4\t1.0\t\t2\n"
                "additive_mellow_pipe\t4.0\t0.5\tA4\t1.0\t\t2\n"
                "additive_mellow_pipe\t4.0\t1.0\tC5\t1.0\t\t1\n"
                "additive_mellow_pipe\t4.5\t0.5\tD4\t1.0\t\t2\n"
                "additive_mellow_pipe\t5.0\t1.0\tC4\t1.0\t\t2\n"
                "additive_mellow_pipe\t5.0\t0.5\tD5\t1.0\t\t1\n"
                "additive_mellow_pipe\t5.5\t0.5\tA4\t1.0\t\t1\n"
                "additive_mellow_pipe\t6.0\t1.0\tD#4\t1.0\t\t2\n"
                "additive_mellow_pipe\t6.0\t0.5\tF#4\t1.0\t\t1\n"
                "additive_mellow_pipe\t6.5\t0.5\tE4\t1.0\t\t1\n"
                "additive_mellow_pipe\t7.0\t0.5\tC#4\t1.0\t\t2\n"
                "additive_mellow_pipe\t7.0\t1.0\tG#4\t1.0\t\t1\n"
                "additive_mellow_pipe\t7.5\t0.5\tG4\t1.0\t\t2\n"
                "additive_mellow_pipe\t8.0\t0.5\tA#3\t1.0\t\t2\n"
                "additive_mellow_pipe\t8.0\t1.0\tF4\t1.0\t\t1\n"
                "additive_mellow_pipe\t8.5\t0.5\tB3\t1.0\t\t2\n"
            )
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
                    [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
                ],
                sonic_content=[
                    ['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                    ['pause', 'G#', 'E', 'F#', 'A', 'D', 'C', 'D#', 'C#', 'G', 'A#', 'B'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=16,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[1, 1],
                n_tone_row_instances_by_group=[1, 1]
            ),
            # `expected`
            (
                "instrument\tstart_time\tduration\tfrequency\tvelocity\teffects\tline_id\n"
                "additive_mellow_pipe\t1.0\t0.5\tB4\t1.0\t\t1\n"
                "additive_mellow_pipe\t1.5\t0.5\tA#4\t1.0\t\t1\n"
                "additive_mellow_pipe\t2.0\t1.0\tG#3\t1.0\t\t2\n"
                "additive_mellow_pipe\t2.0\t0.5\tG4\t1.0\t\t1\n"
                "additive_mellow_pipe\t2.5\t0.5\tC#5\t1.0\t\t1\n"
                "additive_mellow_pipe\t3.0\t0.5\tE4\t1.0\t\t2\n"
                "additive_mellow_pipe\t3.0\t1.0\tD#5\t1.0\t\t1\n"
                "additive_mellow_pipe\t3.5\t0.5\tF#4\t1.0\t\t2\n"
                "additive_mellow_pipe\t4.0\t0.5\tA4\t1.0\t\t2\n"
                "additive_mellow_pipe\t4.0\t1.0\tC5\t1.0\t\t1\n"
                "additive_mellow_pipe\t4.5\t0.5\tD4\t1.0\t\t2\n"
                "additive_mellow_pipe\t5.0\t1.0\tC4\t1.0\t\t2\n"
                "additive_mellow_pipe\t5.0\t0.5\tD5\t1.0\t\t1\n"
                "additive_mellow_pipe\t5.5\t0.5\tA4\t1.0\t\t1\n"
                "additive_mellow_pipe\t6.0\t1.0\tD#4\t1.0\t\t2\n"
                "additive_mellow_pipe\t6.0\t0.5\tF#4\t1.0\t\t1\n"
                "additive_mellow_pipe\t6.5\t0.5\tE4\t1.0\t\t1\n"
                "additive_mellow_pipe\t7.0\t0.5\tC#4\t1.0\t\t2\n"
                "additive_mellow_pipe\t7.0\t1.0\tG#4\t1.0\t\t1\n"
                "additive_mellow_pipe\t7.5\t0.5\tG4\t1.0\t\t2\n"
                "additive_mellow_pipe\t8.0\t0.5\tA#3\t1.0\t\t2\n"
                "additive_mellow_pipe\t8.0\t1.0\tF4\t1.0\t\t1\n"
                "additive_mellow_pipe\t8.5\t0.5\tB3\t1.0\t\t2\n"
            )
        ),
    ]
)
def test_create_tsv_events_from_fragment(
        path_to_tmp_file: str, fragment: Fragment, expected: str
) -> None:
    """Test `create_tsv_events_from_fragment` function."""
    fragment = override_calculated_attributes(fragment)
    create_tsv_events_from_fragment(
        fragment,
        path_to_tmp_file,
        beat_in_seconds=0.5,
        instruments={k: 'additive_mellow_pipe' for k in fragment.line_ids},
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
                'additive_mellow_pipe\t1\t1\tA0\t1\t\t1',
                'additive_mellow_pipe\t2\t1\t1\t1\t[{"name": "tremolo", "frequency": 1}]\t1'
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
    n_melodic_lines = 1
    instruments_registry = create_sinethesizer_instruments(n_melodic_lines)
    create_wav_from_tsv_events(
        path_to_tmp_file,
        path_to_another_tmp_file,
        instruments_registry,
        trailing_silence_in_seconds
    )
