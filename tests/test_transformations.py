"""
Test `dodecaphony.transformations` module.

Author: Nikolay Lysenko
"""


import pytest
from copy import deepcopy

from dodecaphony.fragment import Event, Fragment, ToneRowInstance, override_calculated_attributes
from dodecaphony.transformations import (
    apply_crossmeasure_event_transfer,
    apply_inversion,
    apply_line_durations_change,
    apply_measure_durations_change,
    apply_pause_shift,
    apply_reversion,
    apply_rotation,
    apply_transposition,
    create_transformations_registry,
    transform,
)
from .conftest import MEASURE_DURATIONS_BY_N_EVENTS


@pytest.mark.parametrize(
    "fragment, expected_n_changes",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                    ],
                ],
                grouped_mutable_pauses_indices=[[]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=12,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0]},
                mutable_temporal_content_indices=[0],
                mutable_independent_tone_row_instances_indices=[(0, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `expected_n_changes`
            2
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[4.0], [4.0], [4.0]],
                ],
                grouped_tone_row_instances=[
                    [
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                    ],
                ],
                grouped_mutable_pauses_indices=[[]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=12,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0]},
                mutable_temporal_content_indices=[0],
                mutable_independent_tone_row_instances_indices=[(0, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `expected_n_changes`
            0
        ),
    ]
)
def test_apply_crossmeasure_event_transfer(fragment: Fragment, expected_n_changes: int) -> None:
    """Test `apply_crossmeasure_event_transfer` function."""
    initial_temporal_content = deepcopy(fragment.temporal_content)
    fragment = apply_crossmeasure_event_transfer(fragment)
    override_calculated_attributes(fragment)
    n_changes = 0
    zipped = zip(initial_temporal_content, fragment.temporal_content)
    for initial_line_temporal_content, line_temporal_content in zipped:
        nested_zipped = zip(initial_line_temporal_content, line_temporal_content)
        for initial_measure_durations, measure_durations in nested_zipped:
            if initial_measure_durations != measure_durations:
                n_changes += 1
                valid_values = (
                    MEASURE_DURATIONS_BY_N_EVENTS.get(len(initial_measure_durations) - 1, [])
                    + MEASURE_DURATIONS_BY_N_EVENTS.get(len(initial_measure_durations) + 1, [])
                )
                assert measure_durations in valid_values
    assert n_changes == expected_n_changes


@pytest.mark.parametrize(
    "fragment, expected_options",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                    [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
                ],
                grouped_tone_row_instances=[
                    [
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                    ],
                    [
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                    ],
                ],
                grouped_mutable_pauses_indices=[[12], []],
                grouped_immutable_pauses_indices=[[], []],
                n_beats=24,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0], 1: [1]},
                mutable_temporal_content_indices=[0, 1],
                mutable_independent_tone_row_instances_indices=[(0, 0), (0, 1), (1, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `expected_options`
            [
                [
                    [
                        'B', 'C', 'D#', 'A', 'G', 'A#', 'G#', 'C#', 'E', 'F#', 'D', 'F', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'B', 'C', 'D#', 'A', 'G', 'A#', 'G#', 'C#', 'E', 'F#', 'D', 'F'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'B', 'C', 'D#', 'A', 'G', 'A#', 'G#', 'C#', 'E', 'F#', 'D', 'F'
                    ],
                ],
            ]
        ),
    ]
)
def test_apply_inversion(fragment: Fragment, expected_options: list[list[list[str]]]) -> None:
    """Test `apply_inversion` function."""
    fragment = apply_inversion(fragment)
    override_calculated_attributes(fragment)
    assert fragment.sonic_content in expected_options


@pytest.mark.parametrize(
    "fragment, expected_n_changes",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                    [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
                ],
                grouped_tone_row_instances=[
                    [
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                    ],
                    [
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                    ],
                ],
                grouped_mutable_pauses_indices=[[12], []],
                grouped_immutable_pauses_indices=[[], []],
                n_beats=24,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0], 1: [1]},
                mutable_temporal_content_indices=[0, 1],
                mutable_independent_tone_row_instances_indices=[(0, 0), (0, 1), (1, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `expected_n_changes`
            1
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[4.0], [4.0], [4.0]],
                ],
                grouped_tone_row_instances=[
                    [
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                    ],
                ],
                grouped_mutable_pauses_indices=[[]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=12,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0]},
                mutable_temporal_content_indices=[0],
                mutable_independent_tone_row_instances_indices=[(0, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `expected_n_changes`
            0
        ),
    ]
)
def test_apply_line_durations_change(fragment: Fragment, expected_n_changes: int) -> None:
    """Test `apply_line_durations_change` function."""
    fragment = apply_line_durations_change(fragment)
    override_calculated_attributes(fragment)
    for line_temporal_content in fragment.temporal_content:
        for measure_durations in line_temporal_content:
            valid_values = MEASURE_DURATIONS_BY_N_EVENTS[len(measure_durations)]
            assert measure_durations in valid_values


@pytest.mark.parametrize(
    "fragment, expected_n_changes",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                    [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
                ],
                grouped_tone_row_instances=[
                    [
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                    ],
                    [
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                    ],
                ],
                grouped_mutable_pauses_indices=[[12], []],
                grouped_immutable_pauses_indices=[[], []],
                n_beats=24,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0], 1: [1]},
                mutable_temporal_content_indices=[0, 1],
                mutable_independent_tone_row_instances_indices=[(0, 0), (0, 1), (1, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `expected_n_changes`
            1
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[4.0], [4.0], [4.0]],
                ],
                grouped_tone_row_instances=[
                    [
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                    ],
                ],
                grouped_mutable_pauses_indices=[[]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=12,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0]},
                mutable_temporal_content_indices=[0],
                mutable_independent_tone_row_instances_indices=[(0, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `expected_n_changes`
            0
        ),
    ]
)
def test_apply_measure_durations_change(fragment: Fragment, expected_n_changes: int) -> None:
    """Test `apply_measure_durations_change` function."""
    initial_temporal_content = deepcopy(fragment.temporal_content)
    fragment = apply_measure_durations_change(fragment)
    override_calculated_attributes(fragment)
    n_changes = 0
    zipped = zip(initial_temporal_content, fragment.temporal_content)
    for initial_line_temporal_content, line_temporal_content in zipped:
        nested_zipped = zip(initial_line_temporal_content, line_temporal_content)
        for initial_measure_durations, measure_durations in nested_zipped:
            if initial_measure_durations != measure_durations:
                n_changes += 1
                valid_values = MEASURE_DURATIONS_BY_N_EVENTS[len(initial_measure_durations)]
                assert measure_durations in valid_values
    assert n_changes == expected_n_changes


@pytest.mark.parametrize(
    "fragment, expected_options",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                    ],
                ],
                grouped_mutable_pauses_indices=[[12]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=24,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0]},
                mutable_temporal_content_indices=[0],
                mutable_independent_tone_row_instances_indices=[(0, 0), (0, 1)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `expected_options`
            [
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'pause', 'F',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F',
                        'B', 'pause', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
            ]
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                    ],
                ],
                grouped_mutable_pauses_indices=[[24]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=24,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0]},
                mutable_temporal_content_indices=[0],
                mutable_independent_tone_row_instances_indices=[(0, 0), (0, 1)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `expected_options`
            [
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'pause', 'F'
                    ],
                ],
            ]
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                    ],
                ],
                grouped_mutable_pauses_indices=[[0]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=24,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0]},
                mutable_temporal_content_indices=[0],
                mutable_independent_tone_row_instances_indices=[(0, 0), (0, 1)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `expected_options`
            [
                [
                    [
                        'B', 'pause', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
            ]
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                    ],
                ],
                grouped_mutable_pauses_indices=[[12, 13]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=24,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0]},
                mutable_temporal_content_indices=[0],
                mutable_independent_tone_row_instances_indices=[(0, 0), (0, 1)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `expected_options`
            [
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'pause', 'F',
                        'pause', 'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'B', 'pause', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
            ]
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'])],
                ],
                grouped_mutable_pauses_indices=[[]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=24,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1],
                upper_line_highest_position=88,
                upper_line_lowest_position=1,
                tone_row_len=12,
                group_index_to_line_indices={0: [0]},
                mutable_temporal_content_indices=[0],
                mutable_independent_tone_row_instances_indices=[(0, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `expected_options`
            [
                [
                    ['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                ],
            ]
        ),
    ]
)
def test_apply_pause_shift(fragment: Fragment, expected_options: list[list[list[str]]]) -> None:
    """Test `apply_pause_shift` function."""
    override_calculated_attributes(fragment)
    fragment = apply_pause_shift(fragment)
    override_calculated_attributes(fragment)
    assert fragment.sonic_content in expected_options


@pytest.mark.parametrize(
    "fragment, expected_options",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                    [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
                ],
                grouped_tone_row_instances=[
                    [
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                    ],
                    [
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                    ],
                ],
                grouped_mutable_pauses_indices=[[12], []],
                grouped_immutable_pauses_indices=[[], []],
                n_beats=24,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0], 1: [1]},
                mutable_temporal_content_indices=[0, 1],
                mutable_independent_tone_row_instances_indices=[(0, 0), (0, 1), (1, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `expected_options`
            [
                [
                    [
                        'F', 'G#', 'E', 'F#', 'A', 'D', 'C', 'D#', 'C#', 'G', 'A#', 'B', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'F', 'G#', 'E', 'F#', 'A', 'D', 'C', 'D#', 'C#', 'G', 'A#', 'B'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'F', 'G#', 'E', 'F#', 'A', 'D', 'C', 'D#', 'C#', 'G', 'A#', 'B'
                    ],
                ],
            ]
        ),
    ]
)
def test_apply_reversion(fragment: Fragment, expected_options: list[list[list[str]]]) -> None:
    """Test `apply_reversion` function."""
    fragment = apply_reversion(fragment)
    override_calculated_attributes(fragment)
    assert fragment.sonic_content in expected_options


@pytest.mark.parametrize(
    "fragment, max_rotation, expected_options",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                    [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
                ],
                grouped_tone_row_instances=[
                    [
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                    ],
                    [
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                    ],
                ],
                grouped_mutable_pauses_indices=[[12], []],
                grouped_immutable_pauses_indices=[[], []],
                n_beats=24,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0], 1: [1]},
                mutable_temporal_content_indices=[0, 1],
                mutable_independent_tone_row_instances_indices=[(0, 0), (0, 1), (1, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `max_rotation`
            1,
            # `expected_options`
            [
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'B', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'F', 'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'B'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'F', 'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'B'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'F', 'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#'
                    ],
                ],
            ]
        ),
    ]
)
def test_apply_rotation(
        fragment: Fragment, max_rotation: int, expected_options: list[list[list[str]]]
) -> None:
    """Test `apply_rotation` function."""
    fragment = apply_rotation(fragment, max_rotation)
    override_calculated_attributes(fragment)
    assert fragment.sonic_content in expected_options


@pytest.mark.parametrize(
    "fragment, max_transposition, expected_options",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                    [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
                ],
                grouped_tone_row_instances=[
                    [
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                    ],
                    [
                        ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F']),
                    ],
                ],
                grouped_mutable_pauses_indices=[[12], []],
                grouped_immutable_pauses_indices=[[], []],
                n_beats=24,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0], 1: [1]},
                mutable_temporal_content_indices=[0, 1],
                mutable_independent_tone_row_instances_indices=[(0, 0), (0, 1), (1, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `max_transposition`
            1,
            # `expected_options`
            [
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E'
                    ],
                ],
                [
                    [
                        'C', 'B', 'G#', 'D', 'E', 'C#', 'D#', 'A#', 'G', 'F', 'A', 'F#', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'C', 'B', 'G#', 'D', 'E', 'C#', 'D#', 'A#', 'G', 'F', 'A', 'F#'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'C', 'B', 'G#', 'D', 'E', 'C#', 'D#', 'A#', 'G', 'F', 'A', 'F#'
                    ],
                ],
            ]
        ),
    ]
)
def test_apply_transposition(
        fragment: Fragment, max_transposition: int, expected_options: list[list[list[str]]]
) -> None:
    """Test `apply_transposition` function."""
    fragment = apply_transposition(fragment, max_transposition)
    override_calculated_attributes(fragment)
    assert fragment.sonic_content in expected_options


@pytest.mark.parametrize(
    "fragment, n_transformations, transformation_names, max_rotation, max_transposition, "
    "expected_options",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'])],
                ],
                grouped_mutable_pauses_indices=[[]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=12,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0]},
                mutable_temporal_content_indices=[0],
                mutable_independent_tone_row_instances_indices=[(0, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `n_transformations`
            1,
            # `transformation_names`
            ['inversion', 'reversion'],
            # `max_rotation`
            1,
            # `max_transposition`
            1,
            # `expected_options`
            [
                [
                    [
                        Event(line_index=0, start_time=0.0, duration=1.0, pitch_class='B', position_in_semitones=50),
                        Event(line_index=0, start_time=1.0, duration=1.0, pitch_class='C', position_in_semitones=51),
                        Event(line_index=0, start_time=2.0, duration=1.0, pitch_class='D#', position_in_semitones=54),
                        Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='A', position_in_semitones=48),
                        Event(line_index=0, start_time=4.0, duration=1.0, pitch_class='G', position_in_semitones=46),
                        Event(line_index=0, start_time=5.0, duration=1.0, pitch_class='A#', position_in_semitones=49),
                        Event(line_index=0, start_time=6.0, duration=1.0, pitch_class='G#', position_in_semitones=47),
                        Event(line_index=0, start_time=7.0, duration=1.0, pitch_class='C#', position_in_semitones=52),
                        Event(line_index=0, start_time=8.0, duration=1.0, pitch_class='E', position_in_semitones=55),
                        Event(line_index=0, start_time=9.0, duration=1.0, pitch_class='F#', position_in_semitones=45),
                        Event(line_index=0, start_time=10.0, duration=1.0, pitch_class='D', position_in_semitones=41),
                        Event(line_index=0, start_time=11.0, duration=1.0, pitch_class='F', position_in_semitones=44),
                    ]
                ],
                [
                    [
                        Event(line_index=0, start_time=0.0, duration=1.0, pitch_class='F', position_in_semitones=44),
                        Event(line_index=0, start_time=1.0, duration=1.0, pitch_class='G#', position_in_semitones=47),
                        Event(line_index=0, start_time=2.0, duration=1.0, pitch_class='E', position_in_semitones=43),
                        Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='F#', position_in_semitones=45),
                        Event(line_index=0, start_time=4.0, duration=1.0, pitch_class='A', position_in_semitones=48),
                        Event(line_index=0, start_time=5.0, duration=1.0, pitch_class='D', position_in_semitones=53),
                        Event(line_index=0, start_time=6.0, duration=1.0, pitch_class='C', position_in_semitones=51),
                        Event(line_index=0, start_time=7.0, duration=1.0, pitch_class='D#', position_in_semitones=54),
                        Event(line_index=0, start_time=8.0, duration=1.0, pitch_class='C#', position_in_semitones=52),
                        Event(line_index=0, start_time=9.0, duration=1.0, pitch_class='G', position_in_semitones=46),
                        Event(line_index=0, start_time=10.0, duration=1.0, pitch_class='A#', position_in_semitones=49),
                        Event(line_index=0, start_time=11.0, duration=1.0, pitch_class='B', position_in_semitones=50),
                    ]
                ],
            ]
        ),
    ]
)
def test_transform(
        fragment: Fragment, n_transformations: int, transformation_names: list[str],
        max_rotation: int, max_transposition: int, expected_options: list[list[list[Event]]]
) -> None:
    """Test `transform` function."""
    registry = create_transformations_registry(max_rotation, max_transposition)
    transformation_probabilities = [1 / len(transformation_names) for _ in transformation_names]
    fragment = transform(
        fragment, n_transformations, registry, transformation_names, transformation_probabilities
    )
    assert fragment.melodic_lines in expected_options
