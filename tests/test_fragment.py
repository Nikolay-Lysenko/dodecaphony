"""
Test `dodecaphony.fragment` module.

Author: Nikolay Lysenko
"""


from collections import Counter

import pytest

from dodecaphony.fragment import (
    Event,
    Fragment,
    FragmentParams,
    SUPPORTED_DURATIONS,
    distribute_pitch_classes,
    find_initial_sonic_content,
    find_initial_temporal_content,
    find_sonorities,
    initialize_fragment,
    set_pitches_of_lower_lines,
    set_pitches_of_upper_line,
    split_time_span,
    validate,
)


@pytest.mark.parametrize(
    "fragment, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [
                        [
                            Event(line_index=0, start_time=0.0, duration=4.0),
                        ],
                    ],
                    [
                        [
                            Event(line_index=1, start_time=0.0, duration=3.0),
                            Event(line_index=1, start_time=3.0, duration=1.0),
                        ],
                        [
                            Event(line_index=2, start_time=0.0, duration=2.0),
                            Event(line_index=2, start_time=2.0, duration=2.0),
                        ],
                    ]
                ],
                sonic_content=[
                    ['C'],
                    ['D', 'E', 'F', 'G'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=4,
                line_ids=[1, 2, 3],
                upper_line_highest_position=88,
                upper_line_lowest_position=1,
                n_tone_row_instances_by_group=[0, 0]
            ),
            # `expected`
            [
                [
                    Event(line_index=0, start_time=0.0, duration=4.0, pitch_class='C'),
                ],
                [
                    Event(line_index=1, start_time=0.0, duration=3.0, pitch_class='D'),
                    Event(line_index=1, start_time=3.0, duration=1.0, pitch_class='G'),
                ],
                [
                    Event(line_index=2, start_time=0.0, duration=2.0, pitch_class='E'),
                    Event(line_index=2, start_time=2.0, duration=2.0, pitch_class='F'),
                ]
            ]
        ),
    ]
)
def test_distribute_pitch_classes(fragment: Fragment, expected: list[list[Event]]) -> None:
    """Test `distribute_pitch_classes` function."""
    result = distribute_pitch_classes(fragment)
    assert result == expected


@pytest.mark.parametrize(
    "params, expected_n_pauses_by_group",
    [
        (
            FragmentParams(
                tone_row=['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                groups=[
                    {'n_melodic_lines': 1, 'n_tone_row_instances': 100},
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_measures=100,
                line_ids=[1],
                upper_line_highest_note='E6',
                upper_line_lowest_note='E4',
                pauses_fraction=0.1
            ),
            [133]
        ),
    ]
)
def test_find_initial_sonic_content(
        params: FragmentParams, expected_n_pauses_by_group: list[int]
) -> None:
    """Test `find_initial_sonic_content` function."""
    sonic_content = find_initial_sonic_content(params)
    zipped = zip(sonic_content, expected_n_pauses_by_group)
    for i, (line_content, expected_n_pauses) in enumerate(zipped):
        counter = Counter(line_content)
        for pitch_class in params.tone_row:
            assert counter[pitch_class] == params.groups[i]['n_tone_row_instances']
        assert counter['pause'] == expected_n_pauses


@pytest.mark.parametrize(
    "params, expected_n_events_by_line",
    [
        (
            FragmentParams(
                tone_row=['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                groups=[
                    {'n_melodic_lines': 1, 'n_tone_row_instances': 1},
                    {'n_melodic_lines': 3, 'n_tone_row_instances': 6},
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_measures=8,
                line_ids=[1, 2, 3, 4],
                upper_line_highest_note='E6',
                upper_line_lowest_note='E4',
                pauses_fraction=0.1
            ),
            [13, 27, 27, 26]
        ),
    ]
)
def test_find_initial_temporal_content(
        params: FragmentParams, expected_n_events_by_line: list[int]
) -> None:
    """Test `find_initial_temporal_content` function."""
    temporal_content = find_initial_temporal_content(params)
    assert len(temporal_content) == len(params.groups)
    for group_content, group_params in zip(temporal_content, params.groups):
        assert len(group_content) == group_params['n_melodic_lines']
    n_events_by_line = [len(x) for group_content in temporal_content for x in group_content]
    assert n_events_by_line == expected_n_events_by_line


@pytest.mark.parametrize(
    "melodic_lines, expected",
    [
        (
            # `melodic_lines`
            [
                [
                    Event(line_index=0, start_time=0.0, duration=3.0),
                    Event(line_index=0, start_time=3.0, duration=1.0),
                ],
                [
                    Event(line_index=1, start_time=0.0, duration=2.0),
                    Event(line_index=1, start_time=2.0, duration=2.0),
                ],
                [
                    Event(line_index=2, start_time=0.0, duration=2.0),
                    Event(line_index=2, start_time=2.0, duration=2.0),
                ],
            ],
            # `expected`
            [
                [
                    Event(line_index=0, start_time=0.0, duration=3.0),
                    Event(line_index=1, start_time=0.0, duration=2.0),
                    Event(line_index=2, start_time=0.0, duration=2.0),
                ],
                [
                    Event(line_index=0, start_time=0.0, duration=3.0),
                    Event(line_index=1, start_time=2.0, duration=2.0),
                    Event(line_index=2, start_time=2.0, duration=2.0),
                ],
                [
                    Event(line_index=0, start_time=3.0, duration=1.0),
                    Event(line_index=1, start_time=2.0, duration=2.0),
                    Event(line_index=2, start_time=2.0, duration=2.0),
                ],
            ]
        ),
    ]
)
def test_find_sonorities(melodic_lines: list[list[Event]], expected: list[list[Event]]) -> None:
    """Test `find_sonorities` function."""
    result = find_sonorities(melodic_lines)
    assert result == expected


@pytest.mark.parametrize(
    "params",
    [
        (
            FragmentParams(
                tone_row=['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                groups=[
                    {'n_melodic_lines': 1, 'n_tone_row_instances': 1},
                    {'n_melodic_lines': 3, 'n_tone_row_instances': 6},
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_measures=8,
                line_ids=[1, 2, 3, 4],
                upper_line_highest_note='E6',
                upper_line_lowest_note='E4',
                pauses_fraction=0.1
            )
        ),
    ]
)
def test_initialize_fragment(params: FragmentParams) -> None:
    """Test `initialize_fragment` function."""
    fragment = initialize_fragment(params)
    for melodic_line in fragment.melodic_lines:
        for event in melodic_line:
            assert event.position_in_semitones is not None or event.pitch_class == 'pause'


@pytest.mark.parametrize(
    "fragment, max_interval, default_shift, expected_melodic_lines, expected_sonorities",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [
                        [
                            Event(line_index=0, start_time=0.0, duration=1.0),
                            Event(line_index=0, start_time=1.0, duration=1.0),
                            Event(line_index=0, start_time=2.0, duration=1.0),
                            Event(line_index=0, start_time=3.0, duration=1.0),
                        ],
                    ],
                    [
                        [
                            Event(line_index=1, start_time=0.0, duration=1.0),
                            Event(line_index=1, start_time=1.0, duration=1.0),
                            Event(line_index=1, start_time=2.0, duration=1.0),
                            Event(line_index=1, start_time=3.0, duration=1.0),
                        ],
                    ]
                ],
                sonic_content=[
                    ['C', 'A', 'D', 'F'],
                    ['D', 'B', 'G', 'A'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=4,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_tone_row_instances_by_group=[0, 0]
            ),
            # `max_interval`
            16,
            # `default_shift`
            7,
            # `expected_melodic_lines`
            [
                [
                    Event(line_index=0, start_time=0.0, duration=1.0, pitch_class='C', position_in_semitones=51),
                    Event(line_index=0, start_time=1.0, duration=1.0, pitch_class='A', position_in_semitones=48),
                    Event(line_index=0, start_time=2.0, duration=1.0, pitch_class='D', position_in_semitones=53),
                    Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='F', position_in_semitones=44),
                ],
                [
                    Event(line_index=1, start_time=0.0, duration=1.0, pitch_class='D', position_in_semitones=41),
                    Event(line_index=1, start_time=1.0, duration=1.0, pitch_class='B', position_in_semitones=38),
                    Event(line_index=1, start_time=2.0, duration=1.0, pitch_class='G', position_in_semitones=46),
                    Event(line_index=1, start_time=3.0, duration=1.0, pitch_class='A', position_in_semitones=36),
                ],
            ],
            # `expected_sonorities`
            [
                [
                    Event(line_index=0, start_time=0.0, duration=1.0, pitch_class='C', position_in_semitones=51),
                    Event(line_index=1, start_time=0.0, duration=1.0, pitch_class='D', position_in_semitones=41),
                ],
                [
                    Event(line_index=0, start_time=1.0, duration=1.0, pitch_class='A', position_in_semitones=48),
                    Event(line_index=1, start_time=1.0, duration=1.0, pitch_class='B', position_in_semitones=38),
                ],
                [
                    Event(line_index=0, start_time=2.0, duration=1.0, pitch_class='D', position_in_semitones=53),
                    Event(line_index=1, start_time=2.0, duration=1.0, pitch_class='G', position_in_semitones=46),
                ],
                [
                    Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='F', position_in_semitones=44),
                    Event(line_index=1, start_time=3.0, duration=1.0, pitch_class='A', position_in_semitones=36),
                ],
            ]
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [
                        [
                            Event(line_index=0, start_time=0.0, duration=2.0),
                            Event(line_index=0, start_time=2.0, duration=1.0),
                            Event(line_index=0, start_time=3.0, duration=1.0),
                        ],
                    ],
                    [
                        [
                            Event(line_index=1, start_time=0.0, duration=2.0),
                            Event(line_index=1, start_time=2.0, duration=1.0),
                            Event(line_index=1, start_time=3.0, duration=1.0),
                        ],
                    ],
                    [
                        [
                            Event(line_index=2, start_time=0.0, duration=1.0),
                            Event(line_index=2, start_time=1.0, duration=1.0),
                            Event(line_index=2, start_time=2.0, duration=1.0),
                            Event(line_index=2, start_time=3.0, duration=1.0),
                        ],
                    ]
                ],
                sonic_content=[
                    ['C', 'D', 'F'],
                    ['C', 'D', 'F'],
                    ['G', 'B', 'G', 'A'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=4,
                line_ids=[1, 2, 3],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_tone_row_instances_by_group=[0, 0]
            ),
            # `max_interval`
            16,
            # `default_shift`
            7,
            # `expected_melodic_lines`
            [
                [
                    Event(line_index=0, start_time=0.0, duration=2.0, pitch_class='C', position_in_semitones=51),
                    Event(line_index=0, start_time=2.0, duration=1.0, pitch_class='D', position_in_semitones=53),
                    Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='F', position_in_semitones=44),
                ],
                [
                    Event(line_index=1, start_time=0.0, duration=2.0, pitch_class='C', position_in_semitones=39),
                    Event(line_index=1, start_time=2.0, duration=1.0, pitch_class='D', position_in_semitones=41),
                    Event(line_index=1, start_time=3.0, duration=1.0, pitch_class='F', position_in_semitones=32),
                ],
                [
                    Event(line_index=2, start_time=0.0, duration=1.0, pitch_class='G', position_in_semitones=34),
                    Event(line_index=2, start_time=1.0, duration=1.0, pitch_class='B', position_in_semitones=38),
                    Event(line_index=2, start_time=2.0, duration=1.0, pitch_class='G', position_in_semitones=34),
                    Event(line_index=2, start_time=3.0, duration=1.0, pitch_class='A', position_in_semitones=24),
                ],
            ],
            # `expected_sonorities`
            [
                [
                    Event(line_index=0, start_time=0.0, duration=2.0, pitch_class='C', position_in_semitones=51),
                    Event(line_index=1, start_time=0.0, duration=2.0, pitch_class='C', position_in_semitones=39),
                    Event(line_index=2, start_time=0.0, duration=1.0, pitch_class='G', position_in_semitones=34),
                ],
                [
                    Event(line_index=0, start_time=0.0, duration=2.0, pitch_class='C', position_in_semitones=51),
                    Event(line_index=1, start_time=0.0, duration=2.0, pitch_class='C', position_in_semitones=39),
                    Event(line_index=2, start_time=1.0, duration=1.0, pitch_class='B', position_in_semitones=38),
                ],
                [
                    Event(line_index=0, start_time=2.0, duration=1.0, pitch_class='D', position_in_semitones=53),
                    Event(line_index=1, start_time=2.0, duration=1.0, pitch_class='D', position_in_semitones=41),
                    Event(line_index=2, start_time=2.0, duration=1.0, pitch_class='G', position_in_semitones=34),
                ],
                [
                    Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='F', position_in_semitones=44),
                    Event(line_index=1, start_time=3.0, duration=1.0, pitch_class='F', position_in_semitones=32),
                    Event(line_index=2, start_time=3.0, duration=1.0, pitch_class='A', position_in_semitones=24),
                ],
            ]
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [
                        [
                            Event(line_index=0, start_time=0.0, duration=1.0),
                            Event(line_index=0, start_time=1.0, duration=1.0),
                            Event(line_index=0, start_time=2.0, duration=1.0),
                            Event(line_index=0, start_time=3.0, duration=1.0),
                        ],
                    ],
                    [
                        [
                            Event(line_index=1, start_time=0.0, duration=1.0),
                            Event(line_index=1, start_time=1.0, duration=1.0),
                            Event(line_index=1, start_time=2.0, duration=1.0),
                            Event(line_index=1, start_time=3.0, duration=1.0),
                        ],
                    ],
                    [
                        [
                            Event(line_index=2, start_time=0.0, duration=1.0),
                            Event(line_index=2, start_time=1.0, duration=1.0),
                            Event(line_index=2, start_time=2.0, duration=1.0),
                            Event(line_index=2, start_time=3.0, duration=1.0),
                        ],
                    ]
                ],
                sonic_content=[
                    ['C', 'A', 'D', 'F'],
                    ['D', 'pause', 'G', 'A'],
                    ['D', 'B', 'G', 'A'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=4,
                line_ids=[1, 2, 3],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_tone_row_instances_by_group=[0, 0, 0]
            ),
            # `max_interval`
            16,
            # `default_shift`
            24,
            # `expected_melodic_lines`
            [
                [
                    Event(line_index=0, start_time=0.0, duration=1.0, pitch_class='C', position_in_semitones=51),
                    Event(line_index=0, start_time=1.0, duration=1.0, pitch_class='A', position_in_semitones=48),
                    Event(line_index=0, start_time=2.0, duration=1.0, pitch_class='D', position_in_semitones=53),
                    Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='F', position_in_semitones=44),
                ],
                [
                    Event(line_index=1, start_time=0.0, duration=1.0, pitch_class='D', position_in_semitones=41),
                    Event(line_index=1, start_time=1.0, duration=1.0, pitch_class='pause', position_in_semitones=None),
                    Event(line_index=1, start_time=2.0, duration=1.0, pitch_class='G', position_in_semitones=46),
                    Event(line_index=1, start_time=3.0, duration=1.0, pitch_class='A', position_in_semitones=36),
                ],
                [
                    Event(line_index=2, start_time=0.0, duration=1.0, pitch_class='D', position_in_semitones=29),
                    Event(line_index=2, start_time=1.0, duration=1.0, pitch_class='B', position_in_semitones=14),
                    Event(line_index=2, start_time=2.0, duration=1.0, pitch_class='G', position_in_semitones=22),
                    Event(line_index=2, start_time=3.0, duration=1.0, pitch_class='A', position_in_semitones=24),
                ],
            ],
            # `expected_sonorities`
            [
                [
                    Event(line_index=0, start_time=0.0, duration=1.0, pitch_class='C', position_in_semitones=51),
                    Event(line_index=1, start_time=0.0, duration=1.0, pitch_class='D', position_in_semitones=41),
                    Event(line_index=2, start_time=0.0, duration=1.0, pitch_class='D', position_in_semitones=29),
                ],
                [
                    Event(line_index=0, start_time=1.0, duration=1.0, pitch_class='A', position_in_semitones=48),
                    Event(line_index=1, start_time=1.0, duration=1.0, pitch_class='pause', position_in_semitones=None),
                    Event(line_index=2, start_time=1.0, duration=1.0, pitch_class='B', position_in_semitones=14),
                ],
                [
                    Event(line_index=0, start_time=2.0, duration=1.0, pitch_class='D', position_in_semitones=53),
                    Event(line_index=1, start_time=2.0, duration=1.0, pitch_class='G', position_in_semitones=46),
                    Event(line_index=2, start_time=2.0, duration=1.0, pitch_class='G', position_in_semitones=22),
                ],
                [
                    Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='F', position_in_semitones=44),
                    Event(line_index=1, start_time=3.0, duration=1.0, pitch_class='A', position_in_semitones=36),
                    Event(line_index=2, start_time=3.0, duration=1.0, pitch_class='A', position_in_semitones=24),
                ],
            ]
        ),
    ]
)
def test_set_pitches_of_lower_lines(
        fragment: Fragment,
        max_interval: int,
        default_shift: int,
        expected_melodic_lines: list[list[Event]],
        expected_sonorities: list[list[Event]]
) -> None:
    """Test `set_pitches_of_lower_lines` function."""
    # Below three lines are added instead of setting all arguments initially,
    # because `sonorities` and `melodic_lines` must reference to the same events.
    fragment.melodic_lines = distribute_pitch_classes(fragment)
    fragment.sonorities = find_sonorities(fragment.melodic_lines)
    fragment = set_pitches_of_upper_line(fragment)

    fragment = set_pitches_of_lower_lines(fragment, max_interval, default_shift)
    assert fragment.melodic_lines == expected_melodic_lines
    assert fragment.sonorities == expected_sonorities


@pytest.mark.parametrize(
    "fragment, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [
                        [
                            Event(line_index=0, start_time=0.0, duration=1.0),
                            Event(line_index=0, start_time=1.0, duration=1.0),
                            Event(line_index=0, start_time=2.0, duration=1.0),
                            Event(line_index=0, start_time=3.0, duration=1.0),
                        ],
                    ],
                    [
                        [
                            Event(line_index=1, start_time=0.0, duration=1.0),
                            Event(line_index=1, start_time=1.0, duration=1.0),
                            Event(line_index=1, start_time=2.0, duration=1.0),
                            Event(line_index=1, start_time=3.0, duration=1.0),
                        ],
                    ]
                ],
                sonic_content=[
                    ['C', 'A', 'D', 'F'],
                    ['D', 'B', 'G', 'A'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=4,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_tone_row_instances_by_group=[0, 0],
                melodic_lines=[
                    [
                        Event(line_index=0, start_time=0.0, duration=1.0, pitch_class='C'),
                        Event(line_index=0, start_time=1.0, duration=1.0, pitch_class='A'),
                        Event(line_index=0, start_time=2.0, duration=1.0, pitch_class='D'),
                        Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='F'),
                    ],
                    [
                        Event(line_index=1, start_time=0.0, duration=1.0, pitch_class='D'),
                        Event(line_index=1, start_time=1.0, duration=1.0, pitch_class='B'),
                        Event(line_index=1, start_time=2.0, duration=1.0, pitch_class='G'),
                        Event(line_index=1, start_time=3.0, duration=1.0, pitch_class='A'),
                    ],
                ],
            ),
            # `expected`
            [
                [
                    Event(line_index=0, start_time=0.0, duration=1.0, pitch_class='C', position_in_semitones=51),
                    Event(line_index=0, start_time=1.0, duration=1.0, pitch_class='A', position_in_semitones=48),
                    Event(line_index=0, start_time=2.0, duration=1.0, pitch_class='D', position_in_semitones=53),
                    Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='F', position_in_semitones=44),
                ],
                [
                    Event(line_index=1, start_time=0.0, duration=1.0, pitch_class='D'),
                    Event(line_index=1, start_time=1.0, duration=1.0, pitch_class='B'),
                    Event(line_index=1, start_time=2.0, duration=1.0, pitch_class='G'),
                    Event(line_index=1, start_time=3.0, duration=1.0, pitch_class='A'),
                ],
            ],
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [
                        [
                            Event(line_index=0, start_time=0.0, duration=1.0),
                            Event(line_index=0, start_time=1.0, duration=1.0),
                            Event(line_index=0, start_time=2.0, duration=1.0),
                            Event(line_index=0, start_time=3.0, duration=1.0),
                        ],
                    ],
                    [
                        [
                            Event(line_index=1, start_time=0.0, duration=1.0),
                            Event(line_index=1, start_time=1.0, duration=1.0),
                            Event(line_index=1, start_time=2.0, duration=1.0),
                            Event(line_index=1, start_time=3.0, duration=1.0),
                        ],
                    ]
                ],
                sonic_content=[
                    ['pause', 'A', 'D', 'F'],
                    ['D', 'B', 'G', 'A'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=4,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_tone_row_instances_by_group=[0, 0],
                melodic_lines=[
                    [
                        Event(line_index=0, start_time=0.0, duration=1.0, pitch_class='pause'),
                        Event(line_index=0, start_time=1.0, duration=1.0, pitch_class='A'),
                        Event(line_index=0, start_time=2.0, duration=1.0, pitch_class='D'),
                        Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='F'),
                    ],
                    [
                        Event(line_index=1, start_time=0.0, duration=1.0, pitch_class='D'),
                        Event(line_index=1, start_time=1.0, duration=1.0, pitch_class='B'),
                        Event(line_index=1, start_time=2.0, duration=1.0, pitch_class='G'),
                        Event(line_index=1, start_time=3.0, duration=1.0, pitch_class='A'),
                    ],
                ],
            ),
            # `expected`
            [
                [
                    Event(line_index=0, start_time=0.0, duration=1.0, pitch_class='pause', position_in_semitones=None),
                    Event(line_index=0, start_time=1.0, duration=1.0, pitch_class='A', position_in_semitones=48),
                    Event(line_index=0, start_time=2.0, duration=1.0, pitch_class='D', position_in_semitones=53),
                    Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='F', position_in_semitones=44),
                ],
                [
                    Event(line_index=1, start_time=0.0, duration=1.0, pitch_class='D'),
                    Event(line_index=1, start_time=1.0, duration=1.0, pitch_class='B'),
                    Event(line_index=1, start_time=2.0, duration=1.0, pitch_class='G'),
                    Event(line_index=1, start_time=3.0, duration=1.0, pitch_class='A'),
                ],
            ],
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [
                        [
                            Event(line_index=0, start_time=0.0, duration=1.0),
                            Event(line_index=0, start_time=1.0, duration=1.0),
                            Event(line_index=0, start_time=2.0, duration=1.0),
                            Event(line_index=0, start_time=3.0, duration=1.0),
                        ],
                    ],
                    [
                        [
                            Event(line_index=1, start_time=0.0, duration=1.0),
                            Event(line_index=1, start_time=1.0, duration=1.0),
                            Event(line_index=1, start_time=2.0, duration=1.0),
                            Event(line_index=1, start_time=3.0, duration=1.0),
                        ],
                    ]
                ],
                sonic_content=[
                    ['C', 'pause', 'D', 'F'],
                    ['D', 'B', 'G', 'A'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=4,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_tone_row_instances_by_group=[0, 0],
                melodic_lines=[
                    [
                        Event(line_index=0, start_time=0.0, duration=1.0, pitch_class='C'),
                        Event(line_index=0, start_time=1.0, duration=1.0, pitch_class='pause'),
                        Event(line_index=0, start_time=2.0, duration=1.0, pitch_class='D'),
                        Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='F'),
                    ],
                    [
                        Event(line_index=1, start_time=0.0, duration=1.0, pitch_class='D'),
                        Event(line_index=1, start_time=1.0, duration=1.0, pitch_class='B'),
                        Event(line_index=1, start_time=2.0, duration=1.0, pitch_class='G'),
                        Event(line_index=1, start_time=3.0, duration=1.0, pitch_class='A'),
                    ],
                ],
            ),
            # `expected`
            [
                [
                    Event(line_index=0, start_time=0.0, duration=1.0, pitch_class='C', position_in_semitones=51),
                    Event(line_index=0, start_time=1.0, duration=1.0, pitch_class='pause', position_in_semitones=None),
                    Event(line_index=0, start_time=2.0, duration=1.0, pitch_class='D', position_in_semitones=53),
                    Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='F', position_in_semitones=44),
                ],
                [
                    Event(line_index=1, start_time=0.0, duration=1.0, pitch_class='D'),
                    Event(line_index=1, start_time=1.0, duration=1.0, pitch_class='B'),
                    Event(line_index=1, start_time=2.0, duration=1.0, pitch_class='G'),
                    Event(line_index=1, start_time=3.0, duration=1.0, pitch_class='A'),
                ],
            ],
        ),
    ]
)
def test_set_pitches_of_upper_line(fragment: Fragment, expected: list[list[Event]]) -> None:
    """Test `set_pitches_of_upper_line` function."""
    fragment = set_pitches_of_upper_line(fragment)
    assert fragment.melodic_lines == expected


@pytest.mark.parametrize(
    "n_measures, n_events, meter_numerator",
    [
        (2, 9, 4),
        (8, 51, 3),
    ]
)
def test_split_time_span(n_measures: int, n_events: int, meter_numerator: float) -> None:
    """Test `split_time_span` function."""
    durations = split_time_span(n_measures, n_events, meter_numerator)
    assert len(durations) == n_events
    assert sum(durations) == n_measures * meter_numerator
    for duration in durations:
        assert duration in SUPPORTED_DURATIONS


@pytest.mark.parametrize(
    "n_measures, n_events, meter_numerator, match",
    [
        (4, 3, 4, "Average duration of an event is longer than semibreve."),
        (1, 20, 4, "The number of events is so high that some of them are too short.")
    ]
)
def test_split_time_span_with_invalid_arguments(
    n_measures: int, n_events: int, meter_numerator: float, match: str
) -> None:
    """Test `split_time_span` function with invalid arguments."""
    with pytest.raises(ValueError, match=match):
        split_time_span(n_measures, n_events, meter_numerator)


@pytest.mark.parametrize(
    "params, match",
    [
        (
            FragmentParams(
                tone_row=['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                groups=[{'n_melodic_lines': 1, 'n_tone_row_instances': 2}],
                meter_numerator=4,
                meter_denominator=4,
                n_measures=8,
                line_ids=[1, 2],
                upper_line_highest_note='E6',
                upper_line_lowest_note='E4',
                pauses_fraction=0.0
            ),
            "Number of lines in `groups` is not equal to that in `line_ids`."
        ),
        (
            FragmentParams(
                tone_row=['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                groups=[{'n_melodic_lines': 2, 'n_tone_row_instances': 2}],
                meter_numerator=4,
                meter_denominator=4,
                n_measures=8,
                line_ids=[1, 1],
                upper_line_highest_note='E6',
                upper_line_lowest_note='E4',
                pauses_fraction=0.0
            ),
            "IDs of melodic lines must be unique."
        ),
        (
            FragmentParams(
                tone_row=['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                groups=[{'n_melodic_lines': 2, 'n_tone_row_instances': 2}],
                meter_numerator=5,
                meter_denominator=4,
                n_measures=8,
                line_ids=[1, 2],
                upper_line_highest_note='E6',
                upper_line_lowest_note='E4',
                pauses_fraction=0.0
            ),
            "Meter numerator = 5 is not supported."
        ),
    ]
)
def test_validate(params: FragmentParams, match: str) -> None:
    """Test `validate` function."""
    with pytest.raises(ValueError, match=match):
        validate(params)
