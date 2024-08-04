"""
Test `dodecaphony.scoring_functions.rhythm` module.

Author: Nikolay Lysenko
"""


import pytest

from dodecaphony.scoring_functions.rhythm import (
    evaluate_cadence_duration,
    evaluate_presence_of_required_pauses,
    evaluate_rhythmic_homogeneity,
    evaluate_rhythmic_intensity_by_positions,
)
from dodecaphony.fragment import Fragment, ToneRowInstance, override_calculated_attributes
from tests.conftest import MEASURE_DURATIONS_BY_N_EVENTS


@pytest.mark.parametrize(
    "fragment, min_desired_duration, last_sonority_weight, last_notes_weight, expected",
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
                    [ToneRowInstance(['A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E'])],
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
            # `min_desired_duration`
            4,
            # `last_sonority_weight`
            0.9,
            # `last_notes_weight`
            0.1,
            # `expected`
            -0.7375
        ),
    ]
)
def test_evaluate_cadence_duration(
        fragment: Fragment, min_desired_duration: float,
        last_sonority_weight: float, last_notes_weight: float, expected: float
) -> None:
    """Test `evaluate_cadence_duration` function."""
    override_calculated_attributes(fragment)
    result = evaluate_cadence_duration(
        fragment, min_desired_duration, last_sonority_weight, last_notes_weight
    )
    assert result == expected


@pytest.mark.parametrize(
    "fragment, pauses, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0]],
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                ],
                grouped_mutable_pauses_indices=[[6], [12]],
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
            # `pauses`
            [(7.0, 8.0), (10.0, 11.0)],
            # `expected`
            -0.75
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0]],
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                ],
                grouped_mutable_pauses_indices=[[6], [12]],
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
            # `pauses`
            [(7.0, 8.0), (10.0, 11.0), (13.0, 16.0)],
            # `expected`
            -0.8
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[4.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 0.5, 0.5, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                ],
                grouped_mutable_pauses_indices=[[6], [12]],
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
            # `pauses`
            [(0, 1), (2, 3), (10, 11)],
            # `expected`
            -5 / 6
        ),
    ]
)
def test_evaluate_presence_of_required_pauses(
        fragment: Fragment, pauses: list[tuple[float, float]], expected: float
) -> None:
    """Test `evaluate_presence_of_required_pauses` function."""
    override_calculated_attributes(fragment)
    result = evaluate_presence_of_required_pauses(fragment, pauses)
    assert result == expected


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
                    [ToneRowInstance(['A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E'])],
                ],
                grouped_mutable_pauses_indices=[[12], []],
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
            -(1 / 9 + (0.6 + 2 / 3 + 1 / 7) / 6)
        ),
    ]
)
def test_evaluate_rhythmic_homogeneity(fragment: Fragment, expected: float) -> None:
    """Test `evaluate_rhythmic_homogeneity` function."""
    override_calculated_attributes(fragment)
    result = evaluate_rhythmic_homogeneity(fragment)
    assert round(result, 10) == round(expected, 10)


@pytest.mark.parametrize(
    "fragment, positions, ranges, half_life, max_intensity_factor, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[4.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 0.5, 0.5, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                ],
                grouped_mutable_pauses_indices=[[6], [3]],
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
            # `positions`
            [3, 4],
            # `ranges`
            [
                [(0.5, 1.0), (0.0, 0.0)],
                [(1.0, 1.0), (0.0, 0.25)],
            ],
            # `half_life`
            1.0,
            # `max_intensity_factor`
            1.0 / 3.142558801568057,
            # `expected`
            -0.5
        ),
    ]
)
def test_evaluate_rhythmic_intensity_by_positions(
        fragment: Fragment, positions: list[float], ranges: list[list[tuple[float, float]]],
        half_life: float, max_intensity_factor: float, expected: float
) -> None:
    """Test `evaluate_rhythmic_intensity_by_positions` function."""
    override_calculated_attributes(fragment)
    result = evaluate_rhythmic_intensity_by_positions(
        fragment, positions, ranges, half_life, max_intensity_factor
    )
    assert round(result, 10) == round(expected, 10)
