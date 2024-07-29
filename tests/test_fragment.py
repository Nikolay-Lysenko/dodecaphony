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
    Sonority,
    ToneRowInstance,
    create_initial_grouped_tone_row_instances,
    create_initial_temporal_content,
    find_initial_pauses_indices,
    find_mutable_temporal_content_indices,
    group_durations_by_measures,
    initialize_fragment,
    override_calculated_attributes,
    set_melodic_lines_and_their_pitch_classes,
    set_pitches_of_lower_lines,
    set_pitches_of_upper_line,
    set_sonic_content,
    set_sonorities,
    split_time_span,
    update_dependent_tone_row_instance,
    update_dependent_tone_row_instances,
    validate,
)
from .conftest import MEASURE_DURATIONS, MEASURE_DURATIONS_BY_N_EVENTS


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            # `params`
            FragmentParams(
                tone_row=['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                groups=[
                    {
                        'melodic_line_indices': [0],
                        'tone_row_instances': [
                            {
                                'pitch_classes': ['A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E'],
                                'immutable': True,
                            },
                            {
                                'pitch_classes': None,
                                'dependence': {
                                    'group_index': 0,
                                    'tone_row_instance_index': 0,
                                    'transformation': 'reversion',
                                }
                            }
                        ],
                        'n_pauses': 1
                    },
                ],
                n_measures=8,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations=MEASURE_DURATIONS,
                line_ids=[1],
                upper_line_highest_note='E6',
                upper_line_lowest_note='E4',
                temporal_content={}
            ),
            # `expected`
            (
                [
                    [
                        ToneRowInstance(
                            ['A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E']
                        ),
                        ToneRowInstance(
                            ['E', 'G', 'D#', 'F', 'G#', 'C#', 'B', 'D', 'C', 'F#', 'A', 'A#'],
                            independent_instance_indices=(0, 0),
                            dependence_name='reversion',
                            dependence_params={}
                        )
                    ]
                ],
                [],
                []
            )
        ),
        (
            # `params`
            FragmentParams(
                tone_row=['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                groups=[
                    {
                        'melodic_line_indices': [0],
                        'tone_row_instances': [
                            {
                                'pitch_classes': ['A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E'],
                            },
                            {
                                'pitch_classes': None,
                                'dependence': {
                                    'group_index': 0,
                                    'tone_row_instance_index': 0,
                                    'transformation': 'reversion',
                                }
                            }
                        ],
                        'n_pauses': 1
                    },
                ],
                n_measures=8,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations=MEASURE_DURATIONS,
                line_ids=[1],
                upper_line_highest_note='E6',
                upper_line_lowest_note='E4',
                temporal_content={}
            ),
            # `expected`
            (
                [
                    [
                        ToneRowInstance(
                            ['A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E']
                        ),
                        ToneRowInstance(
                            ['E', 'G', 'D#', 'F', 'G#', 'C#', 'B', 'D', 'C', 'F#', 'A', 'A#'],
                            independent_instance_indices=(0, 0),
                            dependence_name='reversion',
                            dependence_params={}
                        )
                    ]
                ],
                [(0, 0)],
                [(0, 1)]
            )
        ),
    ]
)
def test_create_initial_grouped_tone_row_instances(
        params: FragmentParams,
        expected: tuple[list[list[ToneRowInstance]], list[tuple[int, int]], list[tuple[int, int]]]
) -> None:
    """Test `create_initial_grouped_tone_row_instances` function."""
    result = create_initial_grouped_tone_row_instances(params)
    assert result == expected


@pytest.mark.parametrize(
    "params, expected_n_events_by_line",
    [
        (
            # `params`
            FragmentParams(
                tone_row=['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                groups=[
                    {'melodic_line_indices': [0], 'tone_row_instances': [{}], 'n_pauses': 1},
                    {'melodic_line_indices': [1, 2, 3], 'tone_row_instances': [{}, {}, {}, {}, {}, {}], 'n_pauses': 8},
                ],
                n_measures=8,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations=MEASURE_DURATIONS,
                line_ids=[1, 2, 3, 4],
                upper_line_highest_note='E6',
                upper_line_lowest_note='E4',
                temporal_content={}
            ),
            # `expected_n_events_by_line`
            [13, 27, 27, 26]
        ),
        (
            # `params`
            FragmentParams(
                tone_row=['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                groups=[
                    {'melodic_line_indices': [0], 'tone_row_instances': [{}], 'n_pauses': 1},
                    {'melodic_line_indices': [1, 2, 3], 'tone_row_instances': [{}, {}, {}, {}, {}, {}], 'n_pauses': 8},
                ],
                n_measures=8,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations=MEASURE_DURATIONS,
                line_ids=[1, 2, 3, 4],
                upper_line_highest_note='E6',
                upper_line_lowest_note='E4',
                temporal_content={1: {'durations': [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]}}
            ),
            # `expected_n_events_by_line`
            [13, 8, 36, 36]
        ),
    ]
)
def test_create_initial_temporal_content(
        params: FragmentParams, expected_n_events_by_line: list[int]
) -> None:
    """Test `create_initial_temporal_content` function."""
    temporal_content = create_initial_temporal_content(params, MEASURE_DURATIONS_BY_N_EVENTS)
    assert len(temporal_content) == len(params.line_ids)
    n_events_by_line = [sum(len(y) for y in x) for x in temporal_content]
    assert n_events_by_line == expected_n_events_by_line


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            # `params`
            FragmentParams(
                tone_row=['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                groups=[
                    {
                        'melodic_line_indices': [0],
                        'tone_row_instances': [{}],
                        'n_pauses': 1,
                        'immutable_pauses_indices': [5]
                    },
                ],
                n_measures=8,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations=MEASURE_DURATIONS,
                line_ids=[1],
                upper_line_highest_note='E6',
                upper_line_lowest_note='E4',
                temporal_content={}
            ),
            # `expected`
            ([[]], [[5]])
        ),
    ]
)
def test_find_initial_pauses_indices(
        params: FragmentParams, expected: tuple[list[list[int]], list[list[int]]]
) -> None:
    """Test `find_initial_pauses_indices` function."""
    result = find_initial_pauses_indices(params)
    assert result == expected


@pytest.mark.parametrize(
    "params, expected",
    [
        (
            # `params`
            FragmentParams(
                tone_row=['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                groups=[
                    {'melodic_line_indices': [0], 'tone_row_instances': [{}], 'n_pauses': 1},
                    {'melodic_line_indices': [1, 2, 3], 'tone_row_instances': [{}, {}, {}, {}, {}, {}], 'n_pauses': 8},
                ],
                n_measures=8,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations=MEASURE_DURATIONS,
                line_ids=[1, 2, 3, 4],
                upper_line_highest_note='E6',
                upper_line_lowest_note='E4',
                temporal_content={
                    1: {
                        'durations': [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                        'immutable': True
                    }
                },
            ),
            # `expected`
            [0, 2, 3]
        ),
    ]
)
def test_find_mutable_temporal_content_indices(
        params: FragmentParams, expected: list[int]
) -> None:
    """Test `find_mutable_temporal_content_indices` function."""
    result = find_mutable_temporal_content_indices(params)
    assert result == expected


@pytest.mark.parametrize(
    "durations, meter_numerator, expected",
    [
        ([2.0, 4.0, 1.0, 1.0], 4, [[2.0, 4.0], [2.0, 1.0, 1.0]]),
    ]
)
def test_group_durations_by_measures(
        durations: list[float], meter_numerator: int, expected: list[list[float]]
) -> None:
    """Test `group_durations_by_measures` function."""
    result = group_durations_by_measures(durations, meter_numerator)
    assert result == expected


@pytest.mark.parametrize(
    "params",
    [
        (
            FragmentParams(
                tone_row=['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                groups=[
                    {'melodic_line_indices': [0], 'tone_row_instances': [{}], 'n_pauses': 1},
                    {'melodic_line_indices': [1, 2, 3], 'tone_row_instances': [{}, {}, {}, {}, {}, {}],
                     'n_pauses': 8},
                ],
                n_measures=8,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations=MEASURE_DURATIONS,
                line_ids=[1, 2, 3, 4],
                upper_line_highest_note='E6',
                upper_line_lowest_note='E4',
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
    "fragment, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[4.0]],
                    [[3.0, 1.0]],
                    [[2.0, 2.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['C'])],
                    [ToneRowInstance(['D']), ToneRowInstance(['E']), ToneRowInstance(['F']), ToneRowInstance(['G'])],
                ],
                grouped_mutable_pauses_indices=[[], []],
                grouped_immutable_pauses_indices=[[], []],
                n_beats=4,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2, 3],
                upper_line_highest_position=88,
                upper_line_lowest_position=1,
                tone_row_len=12,
                group_index_to_line_indices={0: [0], 1: [1, 2]},
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_independent_tone_row_instances_indices=[(0, 0), (1, 0), (1, 1), (1, 2), (1, 3)],
                mutable_dependent_tone_row_instances_indices=[],
                sonic_content=[
                    ['C'],
                    ['D', 'E', 'F', 'G']
                ]
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
def test_set_melodic_lines_and_their_pitch_classes(fragment: Fragment, expected: list[list[Event]]) -> None:
    """Test `set_melodic_lines_and_their_pitch_classes` function."""
    set_melodic_lines_and_their_pitch_classes(fragment)
    assert fragment.melodic_lines == expected


@pytest.mark.parametrize(
    "fragment, max_interval, default_shift, expected_melodic_lines, expected_sonorities",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['C', 'A', 'D', 'F'])],
                    [ToneRowInstance(['D', 'B', 'G', 'A'])],
                ],
                grouped_mutable_pauses_indices=[[], []],
                grouped_immutable_pauses_indices=[[], []],
                n_beats=4,
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
                mutable_dependent_tone_row_instances_indices=[],
                sonic_content=[
                    ['C', 'A', 'D', 'F'],
                    ['D', 'B', 'G', 'A'],
                ]
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
                Sonority(
                    [
                        Event(line_index=0, start_time=0.0, duration=1.0, pitch_class='C', position_in_semitones=51),
                        Event(line_index=1, start_time=0.0, duration=1.0, pitch_class='D', position_in_semitones=41),
                    ],
                    [
                        Event(line_index=0, start_time=0.0, duration=1.0, pitch_class='C', position_in_semitones=51),
                        Event(line_index=1, start_time=0.0, duration=1.0, pitch_class='D', position_in_semitones=41),
                    ],
                    0.0,
                    1.0
                ),
                Sonority(
                    [
                        Event(line_index=0, start_time=1.0, duration=1.0, pitch_class='A', position_in_semitones=48),
                        Event(line_index=1, start_time=1.0, duration=1.0, pitch_class='B', position_in_semitones=38),
                    ],
                    [
                        Event(line_index=0, start_time=1.0, duration=1.0, pitch_class='A', position_in_semitones=48),
                        Event(line_index=1, start_time=1.0, duration=1.0, pitch_class='B', position_in_semitones=38),
                    ],
                    1.0,
                    2.0
                ),
                Sonority(
                    [
                        Event(line_index=0, start_time=2.0, duration=1.0, pitch_class='D', position_in_semitones=53),
                        Event(line_index=1, start_time=2.0, duration=1.0, pitch_class='G', position_in_semitones=46),
                    ],
                    [
                        Event(line_index=0, start_time=2.0, duration=1.0, pitch_class='D', position_in_semitones=53),
                        Event(line_index=1, start_time=2.0, duration=1.0, pitch_class='G', position_in_semitones=46),
                    ],
                    2.0,
                    3.0
                ),
                Sonority(
                    [
                        Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='F', position_in_semitones=44),
                        Event(line_index=1, start_time=3.0, duration=1.0, pitch_class='A', position_in_semitones=36),
                    ],
                    [
                        Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='F', position_in_semitones=44),
                        Event(line_index=1, start_time=3.0, duration=1.0, pitch_class='A', position_in_semitones=36),
                    ],
                    3.0,
                    4.0
                ),
            ]
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[2.0, 1.0, 1.0]],
                    [[2.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['C', 'D', 'F'])],
                    [ToneRowInstance(['C', 'D', 'F'])],
                    [ToneRowInstance(['G', 'B', 'G', 'A'])],
                ],
                grouped_mutable_pauses_indices=[[], [], []],
                grouped_immutable_pauses_indices=[[], [], []],
                n_beats=4,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2, 3],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0], 1: [1], 2: [2]},
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_independent_tone_row_instances_indices=[(0, 0), (1, 0), (2, 0)],
                mutable_dependent_tone_row_instances_indices=[],
                sonic_content=[
                    ['C', 'D', 'F'],
                    ['C', 'D', 'F'],
                    ['G', 'B', 'G', 'A'],
                ]
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
                Sonority(
                    [
                        Event(line_index=0, start_time=0.0, duration=2.0, pitch_class='C', position_in_semitones=51),
                        Event(line_index=1, start_time=0.0, duration=2.0, pitch_class='C', position_in_semitones=39),
                        Event(line_index=2, start_time=0.0, duration=1.0, pitch_class='G', position_in_semitones=34),
                    ],
                    [
                        Event(line_index=0, start_time=0.0, duration=2.0, pitch_class='C', position_in_semitones=51),
                        Event(line_index=1, start_time=0.0, duration=2.0, pitch_class='C', position_in_semitones=39),
                        Event(line_index=2, start_time=0.0, duration=1.0, pitch_class='G', position_in_semitones=34),
                    ],
                    0.0,
                    1.0
                ),
                Sonority(
                    [
                        Event(line_index=0, start_time=0.0, duration=2.0, pitch_class='C', position_in_semitones=51),
                        Event(line_index=1, start_time=0.0, duration=2.0, pitch_class='C', position_in_semitones=39),
                        Event(line_index=2, start_time=1.0, duration=1.0, pitch_class='B', position_in_semitones=38),
                    ],
                    [
                        Event(line_index=0, start_time=0.0, duration=2.0, pitch_class='C', position_in_semitones=51),
                        Event(line_index=1, start_time=0.0, duration=2.0, pitch_class='C', position_in_semitones=39),
                        Event(line_index=2, start_time=1.0, duration=1.0, pitch_class='B', position_in_semitones=38),
                    ],
                    1.0,
                    2.0
                ),
                Sonority(
                    [
                        Event(line_index=0, start_time=2.0, duration=1.0, pitch_class='D', position_in_semitones=53),
                        Event(line_index=1, start_time=2.0, duration=1.0, pitch_class='D', position_in_semitones=41),
                        Event(line_index=2, start_time=2.0, duration=1.0, pitch_class='G', position_in_semitones=34),
                    ],
                    [
                        Event(line_index=0, start_time=2.0, duration=1.0, pitch_class='D', position_in_semitones=53),
                        Event(line_index=1, start_time=2.0, duration=1.0, pitch_class='D', position_in_semitones=41),
                        Event(line_index=2, start_time=2.0, duration=1.0, pitch_class='G', position_in_semitones=34),
                    ],
                    2.0,
                    3.0
                ),
                Sonority(
                    [
                        Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='F', position_in_semitones=44),
                        Event(line_index=1, start_time=3.0, duration=1.0, pitch_class='F', position_in_semitones=32),
                        Event(line_index=2, start_time=3.0, duration=1.0, pitch_class='A', position_in_semitones=24),
                    ],
                    [
                        Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='F', position_in_semitones=44),
                        Event(line_index=1, start_time=3.0, duration=1.0, pitch_class='F', position_in_semitones=32),
                        Event(line_index=2, start_time=3.0, duration=1.0, pitch_class='A', position_in_semitones=24),
                    ],
                    3.0,
                    4.0
                ),
            ]
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['C', 'A', 'D', 'F'])],
                    [ToneRowInstance(['D', 'G', 'A'])],
                    [ToneRowInstance(['D', 'B', 'G', 'A'])],
                ],
                grouped_mutable_pauses_indices=[[], [1], []],
                grouped_immutable_pauses_indices=[[], [], []],
                n_beats=4,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2, 3],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0], 1: [1], 2: [2]},
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_independent_tone_row_instances_indices=[(0, 0), (1, 0), (2, 0)],
                mutable_dependent_tone_row_instances_indices=[],
                sonic_content=[
                    ['C', 'A', 'D', 'F'],
                    ['D', 'pause', 'G', 'A'],
                    ['D', 'B', 'G', 'A'],
                ]
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
                Sonority(
                    [
                        Event(line_index=0, start_time=0.0, duration=1.0, pitch_class='C', position_in_semitones=51),
                        Event(line_index=1, start_time=0.0, duration=1.0, pitch_class='D', position_in_semitones=41),
                        Event(line_index=2, start_time=0.0, duration=1.0, pitch_class='D', position_in_semitones=29),
                    ],
                    [
                        Event(line_index=0, start_time=0.0, duration=1.0, pitch_class='C', position_in_semitones=51),
                        Event(line_index=1, start_time=0.0, duration=1.0, pitch_class='D', position_in_semitones=41),
                        Event(line_index=2, start_time=0.0, duration=1.0, pitch_class='D', position_in_semitones=29),
                    ],
                    0.0,
                    1.0
                ),
                Sonority(
                    [
                        Event(line_index=0, start_time=1.0, duration=1.0, pitch_class='A', position_in_semitones=48),
                        Event(line_index=1, start_time=1.0, duration=1.0, pitch_class='pause', position_in_semitones=None),
                        Event(line_index=2, start_time=1.0, duration=1.0, pitch_class='B', position_in_semitones=14),
                    ],
                    [
                        Event(line_index=0, start_time=1.0, duration=1.0, pitch_class='A', position_in_semitones=48),
                        Event(line_index=2, start_time=1.0, duration=1.0, pitch_class='B', position_in_semitones=14),
                    ],
                    1.0,
                    2.0
                ),
                Sonority(
                    [
                        Event(line_index=0, start_time=2.0, duration=1.0, pitch_class='D', position_in_semitones=53),
                        Event(line_index=1, start_time=2.0, duration=1.0, pitch_class='G', position_in_semitones=46),
                        Event(line_index=2, start_time=2.0, duration=1.0, pitch_class='G', position_in_semitones=22),
                    ],
                    [
                        Event(line_index=0, start_time=2.0, duration=1.0, pitch_class='D', position_in_semitones=53),
                        Event(line_index=1, start_time=2.0, duration=1.0, pitch_class='G', position_in_semitones=46),
                        Event(line_index=2, start_time=2.0, duration=1.0, pitch_class='G', position_in_semitones=22),
                    ],
                    2.0,
                    3.0
                ),
                Sonority(
                    [
                        Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='F', position_in_semitones=44),
                        Event(line_index=1, start_time=3.0, duration=1.0, pitch_class='A', position_in_semitones=36),
                        Event(line_index=2, start_time=3.0, duration=1.0, pitch_class='A', position_in_semitones=24),
                    ],
                    [
                        Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='F', position_in_semitones=44),
                        Event(line_index=1, start_time=3.0, duration=1.0, pitch_class='A', position_in_semitones=36),
                        Event(line_index=2, start_time=3.0, duration=1.0, pitch_class='A', position_in_semitones=24),
                    ],
                    3.0,
                    4.0
                ),
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
    # Below three lines of code are added instead of setting all arguments initially,
    # because `sonorities` and `melodic_lines` must reference to the same events.
    set_melodic_lines_and_their_pitch_classes(fragment)
    set_sonorities(fragment)
    set_pitches_of_upper_line(fragment)

    set_pitches_of_lower_lines(fragment, max_interval, default_shift)
    assert fragment.melodic_lines == expected_melodic_lines
    assert fragment.sonorities == expected_sonorities


@pytest.mark.parametrize(
    "fragment, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['C', 'A', 'D', 'F'])],
                    [ToneRowInstance(['D', 'B', 'G', 'A'])],
                ],
                grouped_mutable_pauses_indices=[[], []],
                grouped_immutable_pauses_indices=[[], []],
                n_beats=4,
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
                mutable_dependent_tone_row_instances_indices=[],
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
                    [[1.0, 1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['A', 'D', 'F'])],
                    [ToneRowInstance(['D', 'B', 'G', 'A'])],
                ],
                grouped_mutable_pauses_indices=[[0], []],
                grouped_immutable_pauses_indices=[[], []],
                n_beats=4,
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
                mutable_dependent_tone_row_instances_indices=[],
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
                    [[1.0, 1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['C', 'D', 'F'])],
                    [ToneRowInstance(['D', 'B', 'G', 'A'])],
                ],
                grouped_mutable_pauses_indices=[[1], []],
                grouped_immutable_pauses_indices=[[], []],
                n_beats=4,
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
                mutable_dependent_tone_row_instances_indices=[],
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
    set_pitches_of_upper_line(fragment)
    assert fragment.melodic_lines == expected


@pytest.mark.parametrize(
    "fragment, indices_to_pitch_class",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                    [[2.0, 2.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                    [ToneRowInstance(['A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E'])],
                ],
                grouped_mutable_pauses_indices=[[12], []],
                grouped_immutable_pauses_indices=[[], [1]],
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
                mutable_dependent_tone_row_instances_indices=[],
            ),
            # `indices_to_pitch_class`
            {(0, 12): 'pause', (1, 1): 'pause'}
        ),
    ]
)
def test_set_sonic_content(
        fragment: Fragment, indices_to_pitch_class: dict[tuple[int, int], str]
) -> None:
    """Test `set_sonic_content` function."""
    override_calculated_attributes(fragment)
    set_sonic_content(fragment)
    assert len(fragment.sonic_content) == len(fragment.grouped_tone_row_instances)
    zipped = zip(fragment.sonic_content, fragment.grouped_tone_row_instances)
    for line_content, tone_row_instances in zipped:
        counter = Counter(line_content)
        for pitch_class in tone_row_instances[0].pitch_classes:
            assert counter[pitch_class] == len(tone_row_instances)
    for indices, pitch_class in indices_to_pitch_class.items():
        assert fragment.sonic_content[indices[0]][indices[1]] == pitch_class


@pytest.mark.parametrize(
    "melodic_lines, expected",
    [
        (
            # `melodic_lines`
            [
                [
                    Event(line_index=0, start_time=0.0, duration=3.0, pitch_class='C'),
                    Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='D'),
                ],
                [
                    Event(line_index=1, start_time=0.0, duration=2.0, pitch_class='pause'),
                    Event(line_index=1, start_time=2.0, duration=2.0, pitch_class='pause'),
                ],
                [
                    Event(line_index=2, start_time=0.0, duration=2.0, pitch_class='E'),
                    Event(line_index=2, start_time=2.0, duration=2.0, pitch_class='F'),
                ],
            ],
            # `expected`
            [
                Sonority(
                    [
                        Event(line_index=0, start_time=0.0, duration=3.0, pitch_class='C'),
                        Event(line_index=1, start_time=0.0, duration=2.0, pitch_class='pause'),
                        Event(line_index=2, start_time=0.0, duration=2.0, pitch_class='E'),
                    ],
                    [
                        Event(line_index=0, start_time=0.0, duration=3.0, pitch_class='C'),
                        Event(line_index=2, start_time=0.0, duration=2.0, pitch_class='E'),
                    ],
                    0.0,
                    2.0
                ),
                Sonority(
                    [
                        Event(line_index=0, start_time=0.0, duration=3.0, pitch_class='C'),
                        Event(line_index=1, start_time=2.0, duration=2.0, pitch_class='pause'),
                        Event(line_index=2, start_time=2.0, duration=2.0, pitch_class='F'),
                    ],
                    [
                        Event(line_index=0, start_time=0.0, duration=3.0, pitch_class='C'),
                        Event(line_index=2, start_time=2.0, duration=2.0, pitch_class='F'),
                    ],
                    2.0,
                    3.0
                ),
                Sonority(
                    [
                        Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='D'),
                        Event(line_index=1, start_time=2.0, duration=2.0, pitch_class='pause'),
                        Event(line_index=2, start_time=2.0, duration=2.0, pitch_class='F'),
                    ],
                    [
                        Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='D'),
                        Event(line_index=2, start_time=2.0, duration=2.0, pitch_class='F'),
                    ],
                    3.0,
                    4.0
                ),
            ]
        ),
    ]
)
def test_set_sonorities(melodic_lines: list[list[Event]], expected: list[list[Event]]) -> None:
    """Test `set_sonorities` function."""
    # It is an arbitrary stub of fragment with number of lines equal to that of `melodic_lines`.
    fragment = Fragment(
        grouped_tone_row_instances=[[]],
        grouped_mutable_pauses_indices=[[]],
        grouped_immutable_pauses_indices=[[]],
        n_beats=4,
        meter_numerator=4,
        meter_denominator=4,
        measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
        line_ids=[x + 1 for x in range(len(melodic_lines))],
        upper_line_highest_position=88,
        upper_line_lowest_position=1,
        tone_row_len=12,
        group_index_to_line_indices={},
        mutable_temporal_content_indices=[],
        mutable_independent_tone_row_instances_indices=[],
        mutable_dependent_tone_row_instances_indices=[],
        temporal_content=[[] for _ in melodic_lines],
    )
    fragment.melodic_lines = melodic_lines
    set_sonorities(fragment)
    assert fragment.sonorities == expected


@pytest.mark.parametrize(
    "n_measures, n_events, measure_durations_by_n_events",
    [
        (2, 9, MEASURE_DURATIONS_BY_N_EVENTS),
        (8, 51, MEASURE_DURATIONS_BY_N_EVENTS),
    ]
)
def test_split_time_span(
        n_measures: int, n_events: int, measure_durations_by_n_events: dict[int, list[list[float]]]
) -> None:
    """Test `split_time_span` function."""
    durations = split_time_span(n_measures, n_events, measure_durations_by_n_events)
    actual_n_events = 0
    for current_measure_durations in durations:
        assert current_measure_durations in MEASURE_DURATIONS
        actual_n_events += len(current_measure_durations)
    assert actual_n_events == n_events


@pytest.mark.parametrize(
    "n_measures, n_events, measure_durations_by_n_events, match",
    [
        (4, 3, MEASURE_DURATIONS_BY_N_EVENTS, "Average duration of an event is longer than semibreve."),
        (1, 20, MEASURE_DURATIONS_BY_N_EVENTS, "The number of events is too high.")
    ]
)
def test_split_time_span_with_invalid_arguments(
        n_measures: int, n_events: int,
        measure_durations_by_n_events: dict[int, list[list[float]]], match: str
) -> None:
    """Test `split_time_span` function with invalid arguments."""
    with pytest.raises(ValueError, match=match):
        split_time_span(n_measures, n_events, measure_durations_by_n_events)


@pytest.mark.parametrize(
    "tone_row_instance, pitch_classes, expected",
    [
        (
            # `tone_row_instance`
            ToneRowInstance(
                pitch_classes=['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                independent_instance_indices=(0, 0),
                dependence_name='rotation',
                dependence_params={'shift': -1}
            ),
            # `pitch_classes`
            ['A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E'],
            # `expected`
            ['A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E', 'A#'],
        ),
    ]
)
def test_update_dependent_tone_row_instance(
        tone_row_instance: ToneRowInstance, pitch_classes: list[str], expected: list[str]
) -> None:
    """Test `update_dependent_tone_row_instance` function."""
    update_dependent_tone_row_instance(tone_row_instance, pitch_classes)
    assert tone_row_instance.pitch_classes == expected


@pytest.mark.parametrize(
    "fragment, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                    [[2.0, 2.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [
                        ToneRowInstance(
                            ['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'],
                            independent_instance_indices=(1, 0),
                            dependence_name='rotation',
                            dependence_params={'shift': 3},
                        )
                    ],
                    [
                        ToneRowInstance(
                            ['A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E']
                        )
                    ],
                ],
                grouped_mutable_pauses_indices=[[12], []],
                grouped_immutable_pauses_indices=[[], [1]],
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
                mutable_independent_tone_row_instances_indices=[(1, 0)],
                mutable_dependent_tone_row_instances_indices=[(0, 0)],
            ),
            # `expected`
            [
                [
                    ToneRowInstance(
                        ['D#', 'G', 'E', 'A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F'],
                        independent_instance_indices=(1, 0),
                        dependence_name='rotation',
                        dependence_params={'shift': 3},
                    )
                ],
                [
                    ToneRowInstance(
                        ['A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E']
                    )
                ],
            ],
        ),
    ]
)
def test_update_dependent_tone_row_instances(
        fragment: Fragment, expected: list[list[ToneRowInstance]]
) -> None:
    """Test `update_dependent_tone_row_instances` function."""
    update_dependent_tone_row_instances(fragment)
    assert fragment.grouped_tone_row_instances == expected


@pytest.mark.parametrize(
    "params, match",
    [
        (
            FragmentParams(
                tone_row=['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                groups=[{'melodic_line_indices': [0], 'tone_row_instances': [{}, {}], 'n_pauses': 0}],
                n_measures=8,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations=MEASURE_DURATIONS,
                line_ids=[1, 2],
                upper_line_highest_note='E6',
                upper_line_lowest_note='E4'
            ),
            "Number of lines in `groups` is not equal to that in `line_ids`."
        ),
        (
            FragmentParams(
                tone_row=['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                groups=[{'melodic_line_indices': [0], 'tone_row_instances': [{}, {}], 'n_pauses': 0}],
                n_measures=8,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations=MEASURE_DURATIONS,
                line_ids=[1, 1],
                upper_line_highest_note='E6',
                upper_line_lowest_note='E4'
            ),
            "IDs of melodic lines must be unique."
        ),
        (
            FragmentParams(
                tone_row=['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                groups=[{'melodic_line_indices': [0, 1], 'tone_row_instances': [{}, {}], 'n_pauses': 0}],
                n_measures=8,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations=MEASURE_DURATIONS,
                line_ids=[1, 2],
                upper_line_highest_note='E6',
                upper_line_lowest_note='E4',
                temporal_content={
                    0: {'durations': [1.0 for _ in range(40)]},
                    1: {'durations': [1.0]},
                }
            ),
            "A line with index 0 has duration 40.0 beats, whereas duration of the fragment is set to 32 beats."
        ),
        (
            FragmentParams(
                tone_row=['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                groups=[{'melodic_line_indices': [0, 1], 'tone_row_instances': [{}], 'n_pauses': 0}],
                n_measures=2,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations=MEASURE_DURATIONS,
                line_ids=[1, 2],
                upper_line_highest_note='E6',
                upper_line_lowest_note='E4',
                temporal_content={
                    0: {'durations': [2.0, 4.0, 0.5, 0.5, 0.5, 0.5], 'immutable': True},
                    1: {'durations': [2.0, 4.0, 0.5, 0.5, 0.5, 0.5]},
                }
            ),
            "Violations: line_index=1, crossed_bars=\{4\}."
        ),
    ]
)
def test_validate(params: FragmentParams, match: str) -> None:
    """Test `validate` function."""
    with pytest.raises(ValueError, match=match):
        validate(params)


@pytest.mark.parametrize(
    "first_temporal_content, second_temporal_content, "
    "first_grouped_tone_row_instances, second_grouped_tone_row_instances, "
    "expected",
    [
        (
            [[[1.0 for _ in range(4)] for __ in range(3)]],
            [[[1.0 for _ in range(4)] for __ in range(3)]],
            [[ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'])]],
            [[ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'])]],
            True
        ),
    ]
)
def test_equality_of_fragments(
        first_temporal_content: list[list[list[float]]],
        second_temporal_content: list[list[list[float]]],
        first_grouped_tone_row_instances: list[list[ToneRowInstance]],
        second_grouped_tone_row_instances: list[list[ToneRowInstance]],
        expected: bool
) -> None:
    """Test `__eq__` method of `Fragment` class."""
    first_fragment = Fragment(
        first_temporal_content,
        first_grouped_tone_row_instances,
        grouped_mutable_pauses_indices=[[]],
        grouped_immutable_pauses_indices=[[]],
        n_beats=12,
        meter_numerator=4,
        meter_denominator=4,
        measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
        line_ids=[1],
        upper_line_highest_position=88,
        upper_line_lowest_position=0,
        tone_row_len=12,
        group_index_to_line_indices={0: [0]},
        mutable_temporal_content_indices=[],
        mutable_independent_tone_row_instances_indices=[],
        mutable_dependent_tone_row_instances_indices=[]
    )
    override_calculated_attributes(first_fragment)
    second_fragment = Fragment(
        second_temporal_content,
        second_grouped_tone_row_instances,
        grouped_mutable_pauses_indices=[[]],
        grouped_immutable_pauses_indices=[[]],
        n_beats=12,
        meter_numerator=4,
        meter_denominator=4,
        measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
        line_ids=[1],
        upper_line_highest_position=88,
        upper_line_lowest_position=0,
        tone_row_len=12,
        group_index_to_line_indices={0: [0]},
        mutable_temporal_content_indices=[],
        mutable_independent_tone_row_instances_indices=[],
        mutable_dependent_tone_row_instances_indices=[]
    )
    override_calculated_attributes(second_fragment)
    result = first_fragment == second_fragment
    assert result == expected
