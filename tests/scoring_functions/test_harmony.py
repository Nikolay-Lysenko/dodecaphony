"""
Test `dodecaphony.scoring_functions.harmony` module.

Author: Nikolay Lysenko
"""


from typing import Any, Optional

import pytest

from dodecaphony.scoring_functions.harmony import (
    evaluate_absence_of_doubled_pitch_classes,
    evaluate_absence_of_false_octaves,
    evaluate_absence_of_simultaneous_skips,
    evaluate_absence_of_voice_crossing,
    evaluate_dissonances_preparation_and_resolution,
    evaluate_harmony_dynamic_by_positions,
    evaluate_harmony_dynamic_by_time_intervals,
    evaluate_local_diatonicity_at_all_lines_level,
    evaluate_motion_to_perfect_consonances,
    evaluate_movement_to_final_sonority,
    evaluate_pitch_class_distribution_among_lines,
    evaluate_presence_of_vertical_intervals,
    evaluate_sonic_intensity_by_positions,
    find_indices_of_dissonating_events,
    find_sonority_type
)
from dodecaphony.fragment import Event, Fragment, ToneRowInstance, override_calculated_attributes
from tests.conftest import MEASURE_DURATIONS_BY_N_EVENTS


@pytest.mark.parametrize(
    "fragment, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                    [[2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'])],
                    [ToneRowInstance(['B', 'C', 'D#', 'A', 'G', 'A#', 'G#', 'C#', 'E', 'F#', 'D', 'F'])],
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
            -0.125
        ),
    ]
)
def test_evaluate_absence_of_doubled_pitch_classes(fragment: Fragment, expected: float) -> None:
    """Test `evaluate_absence_of_doubled_pitch_classes` function."""
    override_calculated_attributes(fragment)
    result = evaluate_absence_of_doubled_pitch_classes(fragment)
    assert result == expected


@pytest.mark.parametrize(
    "fragment, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 2.0, 0.5, 0.5]],
                    [[1.0, 1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['C', 'C#', 'D', 'C#', 'D#', 'C', 'C#', 'E', 'F', 'F', 'F#', 'G'])],
                ],
                grouped_mutable_pauses_indices=[[]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=4,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2, 3],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0, 1, 2]},
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_independent_tone_row_instances_indices=[(0, 0), (0, 1), (0, 2)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `expected`
            -0.5
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 2.0, 0.5, 0.5]],
                    [[1.0, 1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['C', 'C#', 'D', 'C#', 'D#', 'A', 'C#', 'E', 'F', 'F', 'F#', 'G'])],
                ],
                grouped_mutable_pauses_indices=[[]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=4,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2, 3],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0, 1, 2]},
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_independent_tone_row_instances_indices=[(0, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `expected`
            -0.25
        ),
    ]
)
def test_evaluate_absence_of_false_octaves(fragment: Fragment, expected: float) -> None:
    """Test `evaluate_absence_of_false_octaves` function."""
    override_calculated_attributes(fragment)
    result = evaluate_absence_of_false_octaves(fragment)
    assert result == expected


@pytest.mark.parametrize(
    "fragment, min_skip_in_semitones, max_skips_share, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                    [[2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'])],
                    [ToneRowInstance(['B', 'C', 'D#', 'A', 'G', 'A#', 'G#', 'C#', 'E', 'F#', 'D', 'F'])],
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
            # `min_skip_in_semitones`
            7,
            # `max_skips_share`
            0.1,
            # `expected`
            -2 / 15
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 0.5, 0.5]],
                    [[1.0, 1.0, 1.0, 0.5, 0.5]],
                    [[1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['C', 'C#', 'D', 'B', 'D#', 'F', 'A', 'F#', 'E', 'G#', 'A#', 'G'])],
                ],
                grouped_mutable_pauses_indices=[[4, 6]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=4,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2, 3],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0, 1, 2]},
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_independent_tone_row_instances_indices=[(0, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `min_skip_in_semitones`
            7,
            # `max_skips_share`
            0.1,
            # `expected`
            0.0
        ),
    ]
)
def test_evaluate_absence_of_simultaneous_skips(
        fragment: Fragment, min_skip_in_semitones: int, max_skips_share: float, expected: float
) -> None:
    """Test `evaluate_absence_of_simultaneous_skips` function."""
    override_calculated_attributes(fragment)
    result = evaluate_absence_of_simultaneous_skips(
        fragment, min_skip_in_semitones, max_skips_share
    )
    assert result == expected


@pytest.mark.parametrize(
    "fragment, n_semitones_to_penalty, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                    [[2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
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
            # `n_semitones_to_penalty`
            {
                0: 0.5,
                -1: 0.55,
                -2: 0.6,
                -3: 0.65,
                -4: 0.7,
                -5: 0.75,
                -6: 0.8,
                -7: 0.85,
                -8: 0.9,
                -9: 0.95,
                -10: 1
            },
            # `expected`
            -0.55 / 15
        ),
    ]
)
def test_evaluate_absence_of_voice_crossing(
        fragment: Fragment, n_semitones_to_penalty: dict[int, float], expected: float) -> None:
    """Test `evaluate_absence_of_voice_crossing` function."""
    override_calculated_attributes(fragment)
    result = evaluate_absence_of_voice_crossing(fragment, n_semitones_to_penalty)
    assert round(result, 10) == round(expected, 10)


@pytest.mark.parametrize(
    "fragment, n_semitones_to_pt_and_ngh_preparation_penalty, "
    "n_semitones_to_pt_and_ngh_resolution_penalty, n_semitones_to_suspension_resolution_penalty, "
    "expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[4.0], [4.0], [4.0], [4.0]],
                    [[4.0], [4.0], [4.0], [4.0]],
                    [[4.0], [4.0], [4.0], [4.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['G', 'E', 'C', 'G#', 'D#', 'C#', 'A', 'F', 'D', 'A#', 'F#', 'B'])],
                ],
                grouped_mutable_pauses_indices=[[]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=16,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[0, 1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0, 1, 2]},
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_independent_tone_row_instances_indices=[(0, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `n_semitones_to_pt_and_ngh_preparation_penalty`
            {
                -4: 0.2,
                -3: 0.1,
                -2: 0.0,
                -1: 0.0,
                0: 0.0,
                1: 0.0,
                2: 0.0,
                3: 0.1,
                4: 0.2,
            },
            # `n_semitones_to_pt_and_ngh_resolution_penalty`
            {
                -4: 0.2,
                -3: 0.1,
                -2: 0.0,
                -1: 0.0,
                0: 0.0,
                1: 0.0,
                2: 0.0,
                3: 0.1,
                4: 0.2,
            },
            # `n_semitones_to_suspension_resolution_penalty`
            {
                -5: 0.3,
                -4: 0.2,
                -3: 0.1,
                -2: 0.0,
                -1: 0.0,
                0: 0.0,
                1: 0.1,
                2: 0.2,
                3: 0.3,
            },
            # `expected`
            -0.1 / 9
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[4.0], [4.0], [4.0], [4.0]],
                    [[4.0], [4.0], [4.0], [4.0]],
                    [[4.0], [2.0, 2.0], [4.0], [4.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['G', 'E', 'C', 'G#', 'D#', 'C#', 'A', 'F', 'D', 'A#', 'F#', 'B'])],
                ],
                grouped_mutable_pauses_indices=[[5]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=16,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[0, 1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0, 1, 2]},
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_independent_tone_row_instances_indices=[(0, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `n_semitones_to_pt_and_ngh_preparation_penalty`
            {
                -4: 0.2,
                -3: 0.1,
                -2: 0.0,
                -1: 0.0,
                0: 0.0,
                1: 0.0,
                2: 0.0,
                3: 0.1,
                4: 0.2,
            },
            # `n_semitones_to_pt_and_ngh_resolution_penalty`
            {
                -4: 0.2,
                -3: 0.1,
                -2: 0.0,
                -1: 0.0,
                0: 0.0,
                1: 0.0,
                2: 0.0,
                3: 0.1,
                4: 0.2,
            },
            # `n_semitones_to_suspension_resolution_penalty`
            {
                -5: 0.3,
                -4: 0.2,
                -3: 0.1,
                -2: 0.0,
                -1: 0.0,
                0: 0.0,
                1: 0.1,
                2: 0.2,
                3: 0.3,
            },
            # `expected`
            -0.01
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[4.0], [4.0], [4.0], [4.0]],
                    [[4.0], [4.0], [4.0], [4.0]],
                    [[4.0], [2.0, 2.0], [4.0, 4.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['G', 'E', 'C', 'G#', 'D#', 'C#', 'A', 'F', 'D', 'A#', 'F#', 'B'])],
                ],
                grouped_mutable_pauses_indices=[[6]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=16,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[0, 1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0, 1, 2]},
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_independent_tone_row_instances_indices=[(0, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `n_semitones_to_pt_and_ngh_preparation_penalty`
            {
                -4: 0.2,
                -3: 0.1,
                -2: 0.0,
                -1: 0.0,
                0: 0.0,
                1: 0.0,
                2: 0.0,
                3: 0.1,
                4: 0.2,
            },
            # `n_semitones_to_pt_and_ngh_resolution_penalty`
            {
                -4: 0.2,
                -3: 0.1,
                -2: 0.0,
                -1: 0.0,
                0: 0.0,
                1: 0.0,
                2: 0.0,
                3: 0.1,
                4: 0.2,
            },
            # `n_semitones_to_suspension_resolution_penalty`
            {
                -5: 0.3,
                -4: 0.2,
                -3: 0.1,
                -2: 0.0,
                -1: 0.0,
                0: 0.0,
                1: 0.1,
                2: 0.2,
                3: 0.3,
            },
            # `expected`
            -0.01
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[4.0], [4.0], [4.0], [4.0]],
                    [[4.0], [4.0], [4.0], [4.0]],
                    [[4.0], [4.0], [4.0], [4.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['G#', 'D#', 'C#', 'A#', 'F#', 'B', 'G', 'E', 'C', 'A', 'F', 'D'])],
                ],
                grouped_mutable_pauses_indices=[[]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=16,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[0, 1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0, 1, 2]},
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_independent_tone_row_instances_indices=[(0, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `n_semitones_to_pt_and_ngh_preparation_penalty`
            {
                -4: 0.2,
                -3: 0.1,
                -2: 0.0,
                -1: 0.0,
                0: 0.0,
                1: 0.0,
                2: 0.0,
                3: 0.1,
                4: 0.2,
            },
            # `n_semitones_to_pt_and_ngh_resolution_penalty`
            {
                -4: 0.2,
                -3: 0.1,
                -2: 0.0,
                -1: 0.0,
                0: 0.0,
                1: 0.0,
                2: 0.0,
                3: 0.1,
                4: 0.2,
            },
            # `n_semitones_to_suspension_resolution_penalty`
            {
                -5: 0.3,
                -4: 0.2,
                -3: 0.1,
                -2: 0.0,
                -1: 0.0,
                0: 0.0,
                1: 0.1,
                2: 0.2,
                3: 0.3,
            },
            # `expected`
            -0.2 / 9
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[2.0, 4.0], [2.0, 2.0], [4.0]],
                    [[2.0, 2.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['E', 'C', 'F', 'D', 'D#', 'B', 'G#', 'C#', 'F#', 'G', 'A', 'A#'])],
                ],
                grouped_mutable_pauses_indices=[[]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=12,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[0, 1],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0, 1]},
                mutable_temporal_content_indices=[0, 1],
                mutable_independent_tone_row_instances_indices=[(0, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `n_semitones_to_pt_and_ngh_preparation_penalty`
            {
                -4: 0.2,
                -3: 0.1,
                -2: 0.0,
                -1: 0.0,
                0: 0.0,
                1: 0.0,
                2: 0.0,
                3: 0.1,
                4: 0.2,
            },
            # `n_semitones_to_pt_and_ngh_resolution_penalty`
            {
                -4: 0.2,
                -3: 0.1,
                -2: 0.0,
                -1: 0.0,
                0: 0.0,
                1: 0.0,
                2: 0.0,
                3: 0.1,
                4: 0.2,
            },
            # `n_semitones_to_suspension_resolution_penalty`
            {
                -5: 0.3,
                -4: 0.2,
                -3: 0.1,
                -2: 0.0,
                -1: 0.0,
                0: 0.0,
                1: 0.1,
                2: 0.2,
                3: 0.3,
            },
            # `expected`
            -0.1
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[2.0, 4.0], [2.0, 1.0, 1.0], [4.0]],
                    [[2.0, 2.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['E', 'C', 'F', 'D', 'D#', 'B', 'G#', 'C#', 'F#', 'G', 'A', 'A#'])],
                ],
                grouped_mutable_pauses_indices=[[5]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=12,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[0, 1],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0, 1]},
                mutable_temporal_content_indices=[0, 1],
                mutable_independent_tone_row_instances_indices=[(0, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `n_semitones_to_pt_and_ngh_preparation_penalty`
            {
                -4: 0.2,
                -3: 0.1,
                -2: 0.0,
                -1: 0.0,
                0: 0.0,
                1: 0.0,
                2: 0.0,
                3: 0.1,
                4: 0.2,
            },
            # `n_semitones_to_pt_and_ngh_resolution_penalty`
            {
                -4: 0.2,
                -3: 0.1,
                -2: 0.0,
                -1: 0.0,
                0: 0.0,
                1: 0.0,
                2: 0.0,
                3: 0.1,
                4: 0.2,
            },
            # `n_semitones_to_suspension_resolution_penalty`
            {
                -5: 0.3,
                -4: 0.2,
                -3: 0.1,
                -2: 0.0,
                -1: 0.0,
                0: 0.0,
                1: 0.1,
                2: 0.2,
                3: 0.3,
            },
            # `expected`
            0
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[2.0, 2.0], [2.0, 6.0], [4.0]],
                    [[2.0, 2.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['E', 'C', 'F', 'D', 'D#', 'B', 'G#', 'C#', 'F#', 'G', 'A', 'A#'])],
                ],
                grouped_mutable_pauses_indices=[[]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=12,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[0, 1],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0, 1]},
                mutable_temporal_content_indices=[0, 1],
                mutable_independent_tone_row_instances_indices=[(0, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `n_semitones_to_pt_and_ngh_preparation_penalty`
            {
                -4: 0.2,
                -3: 0.1,
                -2: 0.0,
                -1: 0.0,
                0: 0.0,
                1: 0.0,
                2: 0.0,
                3: 0.1,
                4: 0.2,
            },
            # `n_semitones_to_pt_and_ngh_resolution_penalty`
            {
                -4: 0.2,
                -3: 0.1,
                -2: 0.0,
                -1: 0.0,
                0: 0.0,
                1: 0.0,
                2: 0.0,
                3: 0.1,
                4: 0.2,
            },
            # `n_semitones_to_suspension_resolution_penalty`
            {
                -5: 0.3,
                -4: 0.2,
                -3: 0.1,
                -2: 0.0,
                -1: 0.0,
                0: 0.0,
                1: 0.1,
                2: 0.2,
                3: 0.3,
            },
            # `expected`
            -0.2
        ),
    ]
)
def test_evaluate_dissonances_preparation_and_resolution(
        fragment: Fragment,
        n_semitones_to_pt_and_ngh_preparation_penalty: dict[int, float],
        n_semitones_to_pt_and_ngh_resolution_penalty: dict[int, float],
        n_semitones_to_suspension_resolution_penalty: dict[int, float],
        expected: float
) -> None:
    """Test `evaluate_dissonances_preparation_and_resolution` function."""
    override_calculated_attributes(fragment)
    result = evaluate_dissonances_preparation_and_resolution(
        fragment,
        n_semitones_to_pt_and_ngh_preparation_penalty,
        n_semitones_to_pt_and_ngh_resolution_penalty,
        n_semitones_to_suspension_resolution_penalty
    )
    assert result == expected


@pytest.mark.parametrize(
    "fragment, regular_positions, ad_hoc_positions, ranges, n_semitones_to_stability, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0, 0.5, 0.5]],
                    [[1.0, 1.0, 1.0, 0.5, 0.5]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                ],
                grouped_mutable_pauses_indices=[[12, 13]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=4,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2, 3],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0, 1, 2]},
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_independent_tone_row_instances_indices=[(0, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `regular_positions`
            [
                {'name': 'downbeat', 'denominator': 4, 'remainder': 0},
                {'name': 'middle', 'denominator': 4, 'remainder': 2},
            ],
            # `ad_hoc_positions`
            [
                {'name': '2nd_beat', 'time': 1.0},
            ],
            # `ranges`
            {
                'downbeat': [0.75, 1.0],
                'middle': [0.5, 0.9],
                '2nd_beat': [0.8, 1.0],
                'default': [0.1, 0.9],
            },
            # `n_semitones_to_stability`
            {
                0: 1.0,
                1: 0.2,
                2: 0.2,
                3: 0.7,
                4: 0.8,
                5: 0.5,
                6: 0.0,
                7: 0.9,
                8: 0.6,
                9: 0.6,
                10: 0.2,
                11: 0.2,
            },
            # `expected`
            -0.1766666667
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0, 0.5, 0.5]],
                    [[1.0, 1.0, 1.0, 0.5, 0.5]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                ],
                grouped_mutable_pauses_indices=[[12, 13]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=4,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2, 3],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0, 1, 2]},
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_independent_tone_row_instances_indices=[(0, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `regular_positions`
            [
                {'name': 'downbeat', 'denominator': 4, 'remainder': 0},
                {'name': 'middle', 'denominator': 4, 'remainder': 2},
            ],
            # `ad_hoc_positions`
            [
                {'name': '2nd_beat', 'time': 1.0},
                {'name': 'ending', 'time': -0.01},
            ],
            # `ranges`
            {
                'downbeat': [0.75, 1.0],
                'middle': [0.5, 0.9],
                '2nd_beat': [0.8, 1.0],
                'ending': [0.0, 0.1],
                'default': [0.1, 0.9],
            },
            # `n_semitones_to_stability`
            {
                0: 1.0,
                1: 0.2,
                2: 0.2,
                3: 0.7,
                4: 0.8,
                5: 0.5,
                6: 0.0,
                7: 0.9,
                8: 0.6,
                9: 0.6,
                10: 0.2,
                11: 0.2,
            },
            # `expected`
            -0.3366666667
        ),
    ]
)
def test_evaluate_harmony_dynamic_by_positions(
        fragment: Fragment, regular_positions: list[dict[str, Any]],
        ad_hoc_positions: list[dict[str, Any]], ranges: dict[str, tuple[float, float]],
        n_semitones_to_stability: dict[int, float], expected: float
) -> None:
    """Test `evaluate_harmony_dynamic_by_positions` function."""
    override_calculated_attributes(fragment)
    result = evaluate_harmony_dynamic_by_positions(
        fragment, regular_positions, ad_hoc_positions, ranges, n_semitones_to_stability
    )
    assert round(result, 10) == expected


@pytest.mark.parametrize(
    "fragment, intervals, ranges, n_semitones_to_stability, expected",
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
            # `intervals`
            [(0, 0.75), (3, 4)],
            # `ranges`
            [(0.0, 0.5), (0.5, 1.0)],
            # `n_semitones_to_stability`
            {
                0: 1.0,
                1: 0.2,
                2: 0.2,
                3: 0.7,
                4: 0.8,
                5: 0.5,
                6: 0.0,
                7: 0.9,
                8: 0.6,
                9: 0.6,
                10: 0.2,
                11: 0.2,
            },
            # `expected`
            -(0.75 * 0.5 + 0.3) / 1.75
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
            # `intervals`
            [(0, 1.25), (7, 8)],
            # `ranges`
            [(0, 0.5), (0.5, 1.0)],
            # `n_semitones_to_stability`
            {
                0: 1.0,
                1: 0.2,
                2: 0.2,
                3: 0.7,
                4: 0.8,
                5: 0.5,
                6: 0.0,
                7: 0.9,
                8: 0.6,
                9: 0.6,
                10: 0.2,
                11: 0.2,
            },
            # `expected`
            -0.8 / 2.25
        ),
    ]
)
def test_evaluate_harmony_dynamic_by_time_intervals(
        fragment: Fragment, intervals: list[tuple[float, float]],
        ranges: list[tuple[float, float]], n_semitones_to_stability: dict[int, float],
        expected: float
) -> None:
    """Test `evaluate_harmony_dynamic_by_time_intervals` function."""
    override_calculated_attributes(fragment)
    result = evaluate_harmony_dynamic_by_time_intervals(
        fragment, intervals, ranges, n_semitones_to_stability
    )
    assert round(result, 10) == round(expected, 10)


@pytest.mark.parametrize(
    "fragment, depth, scale_types, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                    [[2.0, 4.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
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
            # `depth`
            2,
            # `scale_types`
            None,
            # `expected`
            -1 / 56
        ),
    ]
)
def test_evaluate_local_diatonicity_at_all_lines_level(
        fragment: Fragment, depth: int, scale_types: Optional[list[str]], expected: float
) -> None:
    """Test `evaluate_local_diatonicity_at_all_lines_level` function."""
    override_calculated_attributes(fragment)
    result = evaluate_local_diatonicity_at_all_lines_level(fragment, depth, scale_types)
    assert round(result, 10) == round(expected, 10)


@pytest.mark.parametrize(
    "fragment, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                    [[2.0, 4.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                    [[2.0, 4.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                    [ToneRowInstance(['A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E'])],
                    [ToneRowInstance(['A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E', 'A#'])],
                ],
                grouped_mutable_pauses_indices=[[12], [], []],
                grouped_immutable_pauses_indices=[[], [], []],
                n_beats=16,
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
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `expected`
            -1 / 7
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 0.5, 0.5]],
                    [[1.0, 2.0, 0.5, 0.5]],
                    [[1.0, 1.0, 1.0, 0.5, 0.5]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['A#', 'B', 'C#', 'D', 'G', 'C', 'D#', 'E', 'F', 'F#', 'G#', 'A'])],
                ],
                grouped_mutable_pauses_indices=[[5, 6]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=4,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2, 3],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0, 1, 2]},
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_independent_tone_row_instances_indices=[(0, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `expected`
            -0.25
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[2.0, 1.0, 0.5, 0.5]],
                    [[2.0, 1.0, 0.5, 0.5]],
                    [[1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['G', 'C', 'C#', 'D', 'D#', 'B', 'F', 'F#', 'E', 'A', 'A#', 'A'])],
                ],
                grouped_mutable_pauses_indices=[[]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=4,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2, 3],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0, 1, 2]},
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_independent_tone_row_instances_indices=[(0, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `expected`
            -0.5
        ),
    ]
)
def test_evaluate_motion_to_perfect_consonances(fragment: Fragment, expected: float) -> None:
    """Test `evaluate_motion_to_perfect_consonances` function."""
    override_calculated_attributes(fragment)
    result = evaluate_motion_to_perfect_consonances(fragment)
    assert round(result, 10) == round(expected, 10)


@pytest.mark.parametrize(
    "fragment, contrary_motion_term, conjunct_motion_term, bass_downward_skip_term, expected",
    [
        (
            # `Fragment`
            Fragment(
                temporal_content=[
                    [[2.0, 4.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                    [[2.0, 4.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                    [ToneRowInstance(['A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E'])],
                    [ToneRowInstance(['B', 'C#', 'G#', 'F', 'D#', 'G', 'E', 'A#', 'A', 'F#', 'C', 'D'])],
                ],
                grouped_mutable_pauses_indices=[[], [], [0]],
                grouped_immutable_pauses_indices=[[], [], []],
                n_beats=16,
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
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `contrary_motion_term`
            0.6,
            # `conjunct_motion_term`
            0.3,
            # `bass_downward_skip_term`
            0.1,
            # `expected`
            -0.4
        ),
        (
            Fragment(
                temporal_content=[
                    [[2.0, 4.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                    [[2.0, 4.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                    [ToneRowInstance(['A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E'])],
                    [ToneRowInstance(['B', 'C#', 'G#', 'F', 'D#', 'G', 'E', 'A#', 'A', 'F#', 'C', 'D'])],
                ],
                grouped_mutable_pauses_indices=[[], [], [11]],
                grouped_immutable_pauses_indices=[[], [], []],
                n_beats=16,
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
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `contrary_motion_term`
            0.5,
            # `conjunct_motion_term`
            0.3,
            # `bass_downward_skip_term`
            0.2,
            # `expected`
            -1.0
        ),
        (
            Fragment(
                temporal_content=[
                    [[2.0, 4.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                    [[2.0, 4.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                    [ToneRowInstance(['A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E'])],
                    [ToneRowInstance(['D#', 'G', 'E', 'A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F'])],
                ],
                grouped_mutable_pauses_indices=[[], [], [0]],
                grouped_immutable_pauses_indices=[[], [], []],
                n_beats=16,
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
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `contrary_motion_term`
            0.5,
            # `conjunct_motion_term`
            0.3,
            # `bass_downward_skip_term`
            0.2,
            # `expected`
            -0.8
        ),
        (
            Fragment(
                temporal_content=[
                    [[2.0, 4.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                ],
                grouped_mutable_pauses_indices=[[]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=16,
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
            # `contrary_motion_term`
            0.6,
            # `conjunct_motion_term`
            0.3,
            # `bass_downward_skip_term`
            0.1,
            # `expected`
            -0.6
        ),
        (
            Fragment(
                temporal_content=[
                    [[2.0, 4.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                ],
                grouped_mutable_pauses_indices=[[]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=16,
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
            # `contrary_motion_term`
            0.7,
            # `conjunct_motion_term`
            0.3,
            # `bass_downward_skip_term`
            0.0,
            # `expected`
            -1.0
        ),
    ]
)
def test_evaluate_movement_to_final_sonority(
        fragment: Fragment, contrary_motion_term: float, conjunct_motion_term: float,
        bass_downward_skip_term: float, expected: float
) -> None:
    """Test `evaluate_movement_to_final_sonority` function."""
    override_calculated_attributes(fragment)
    result = evaluate_movement_to_final_sonority(
        fragment, contrary_motion_term, conjunct_motion_term, bass_downward_skip_term
    )
    assert result == expected


@pytest.mark.parametrize(
    "fragment, line_id_to_banned_pitch_classes, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0]],
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                ],
                grouped_mutable_pauses_indices=[[4]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=8,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0, 1]},
                mutable_temporal_content_indices=[0, 1],
                mutable_independent_tone_row_instances_indices=[(0, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `line_id_to_banned_pitch_classes`
            {1: ['G', 'D#', 'A#', 'E']},
            # `expected`
            -0.25
        ),
    ]
)
def test_evaluate_pitch_class_distribution_among_lines(
        fragment: Fragment, line_id_to_banned_pitch_classes: dict[int, list[str]], expected: float
) -> None:
    """Test `evaluate_pitch_class_distribution_among_lines` function."""
    override_calculated_attributes(fragment)
    result = evaluate_pitch_class_distribution_among_lines(
        fragment, line_id_to_banned_pitch_classes
    )
    assert round(result, 10) == round(expected, 10)


@pytest.mark.parametrize(
    "fragment, intervals, min_n_weighted_occurrences, regular_positions, ad_hoc_positions, "
    "position_weights, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                    [[2.0, 4.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                    [[2.0, 4.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                    [ToneRowInstance(['A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E'])],
                    [ToneRowInstance(['A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E', 'A#'])],
                ],
                grouped_mutable_pauses_indices=[[12], [], []],
                grouped_immutable_pauses_indices=[[], [], []],
                n_beats=16,
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
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `intervals`
            [12, 10],
            # `min_n_weighted_occurrences`
            5.0,
            # `regular_positions`
            [
                {'name': 'downbeat', 'denominator': 4, 'remainder': 0},
                {'name': 'middle', 'denominator': 4, 'remainder': 2},
            ],
            # `ad_hoc_positions`
            [
                {'name': 'beginning', 'time': 0.0},
                {'name': 'ending', 'time': -0.01},
            ],
            # `position_weights`
            {
                "beginning": 2.0,
                "ending": 2.0,
                "downbeat": 1.0,
                "middle": 1.0,
                "default": 0.5,
            },
            # `expected`
            -0.9
        ),
    ]
)
def test_evaluate_presence_of_vertical_intervals(
        fragment: Fragment, intervals: list[int], min_n_weighted_occurrences: float,
        regular_positions: list[dict[str, Any]], ad_hoc_positions: list[dict[str, Any]],
        position_weights: dict[str, float], expected: float
) -> None:
    """Test `evaluate_presence_of_vertical_intervals` function."""
    override_calculated_attributes(fragment)
    result = evaluate_presence_of_vertical_intervals(
        fragment, intervals, min_n_weighted_occurrences, regular_positions, ad_hoc_positions,
        position_weights
    )
    assert result == expected


@pytest.mark.parametrize(
    "fragment, positions, ranges, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0, 0.5, 0.5]],
                    [[1.0, 1.0, 1.0, 0.5, 0.5]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                ],
                grouped_mutable_pauses_indices=[[1, 13]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=4,
                meter_numerator=4,
                meter_denominator=4,
                measure_durations_by_n_events=MEASURE_DURATIONS_BY_N_EVENTS,
                line_ids=[1, 2, 3],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                tone_row_len=12,
                group_index_to_line_indices={0: [0, 1, 2]},
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_independent_tone_row_instances_indices=[(0, 0)],
                mutable_dependent_tone_row_instances_indices=[]
            ),
            # `positions`
            [0.0, 1.0, 2.0, 3.0, 3.5],
            # `ranges`
            [(1, 1), (0, 3), (2, 3), (0, 2), (3, 3)],
            # `expected`
            -0.6
        )
    ]
)
def test_evaluate_sonic_intensity_by_positions(
        fragment: Fragment, positions: list[float], ranges: list[tuple[float, float]],
        expected: float
) -> None:
    """Test `evaluate_sonic_intensity_by_positions` function."""
    override_calculated_attributes(fragment)
    result = evaluate_sonic_intensity_by_positions(fragment, positions, ranges)
    assert round(result, 10) == round(expected, 10)


@pytest.mark.parametrize(
    "sonority_events, sonority_start_time, n_melodic_lines, meter_numerator, expected",
    [
        (
            # `sonority_events`
            [
                Event(line_index=0, start_time=2.0, duration=4.0, pitch_class='E', position_in_semitones=67),
                Event(line_index=1, start_time=4.0, duration=2.0, pitch_class='D', position_in_semitones=65),
                Event(line_index=2, start_time=4.0, duration=1.0, pitch_class='C', position_in_semitones=63),
            ],
            # `sonority_start_time`
            4.0,
            # `n_melodic_lines`
            3,
            # `meter_numerator`
            4,
            # `expected`
            ({1, 2}, {0})
        ),
        (
            # `sonority_events`
            [
                Event(line_index=0, start_time=2.0, duration=4.0, pitch_class='G', position_in_semitones=70),
                Event(line_index=1, start_time=4.0, duration=2.0, pitch_class='E', position_in_semitones=67),
                Event(line_index=2, start_time=4.0, duration=1.0, pitch_class='C', position_in_semitones=63),
            ],
            # `sonority_start_time`
            4.0,
            # `n_melodic_lines`
            3,
            # `meter_numerator`
            4,
            # `expected`
            (set(), set())
        ),
        (
            # `sonority_events`
            [
                Event(line_index=0, start_time=2.0, duration=2.0, pitch_class='E', position_in_semitones=67),
                Event(line_index=1, start_time=2.0, duration=2.0, pitch_class='D', position_in_semitones=65),
                Event(line_index=2, start_time=3.0, duration=1.0, pitch_class='C', position_in_semitones=63),
            ],
            # `sonority_start_time`
            3.0,
            # `n_melodic_lines`
            3,
            # `meter_numerator`
            4,
            # `expected`
            ({2}, set())
        ),
        (
            # `sonority_events`
            [
                Event(line_index=0, start_time=4.0, duration=2.0, pitch_class='E', position_in_semitones=67),
                Event(line_index=1, start_time=2.0, duration=4.0, pitch_class='D', position_in_semitones=65),
                Event(line_index=2, start_time=4.0, duration=1.0, pitch_class='C', position_in_semitones=63),
            ],
            # `sonority_start_time`
            4.0,
            # `n_melodic_lines`
            3,
            # `meter_numerator`
            4,
            # `expected`
            (set(), {1})
        ),
        (
            # `sonority_events`
            [
                Event(line_index=1, start_time=3.0, duration=1.0, pitch_class='D', position_in_semitones=65),
                Event(line_index=2, start_time=2.0, duration=2.0, pitch_class='C', position_in_semitones=63),
            ],
            # `sonority_start_time`
            3.0,
            # `n_melodic_lines`
            3,
            # `meter_numerator`
            4,
            # `expected`
            ({1}, set())
        ),
        (
            # `sonority_events`
            [
                Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='E#', position_in_semitones=68),
                Event(line_index=1, start_time=2.0, duration=2.0, pitch_class='D', position_in_semitones=65),
                Event(line_index=2, start_time=2.0, duration=2.0, pitch_class='C', position_in_semitones=63),
            ],
            # `sonority_start_time`
            3.0,
            # `n_melodic_lines`
            3,
            # `meter_numerator`
            4,
            # `expected`
            ({0}, set())
        ),
        (
            # `sonority_events`
            [
                Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='E#', position_in_semitones=68),
                Event(line_index=1, start_time=2.0, duration=2.0, pitch_class='D', position_in_semitones=65),
                Event(line_index=2, start_time=2.0, duration=2.0, pitch_class='C', position_in_semitones=63),
            ],
            # `sonority_start_time`
            3.0,
            # `n_melodic_lines`
            4,
            # `meter_numerator`
            4,
            # `expected`
            (set(), set())
        ),
    ]
)
def test_find_indices_of_dissonating_events(
        sonority_events: list[Event], sonority_start_time: float,
        n_melodic_lines: int, meter_numerator: int, expected: tuple[set[int], set[int]]
) -> None:
    """Test `find_indices_of_dissonating_events` function."""
    result = find_indices_of_dissonating_events(
        sonority_events, sonority_start_time, n_melodic_lines, meter_numerator
    )
    assert result == expected


@pytest.mark.parametrize(
    "sonority_start, sonority_end, regular_positions, ad_hoc_positions, n_beats, expected",
    [
        (
            # `sonority_start`
            3.5,
            # `sonority_end`
            3.75,
            # `regular_positions`
            [{'name': 't3.5', 'remainder': 3.5, 'denominator': 4}],
            # `ad_hoc_positions`
            [],
            # `n_beats`
            4,
            # `expected`
            't3.5'
        ),
        (
            # `sonority_start`
            3,
            # `sonority_end`
            5,
            # `regular_positions`
            [{'name': 'downbeat', 'remainder': 0, 'denominator': 4}],
            # `ad_hoc_positions`
            [{'name': 'ending', 'time': -0.01}],
            # `n_beats`
            8,
            # `expected`
            'downbeat'
        ),
        (
            # `sonority_start`
            7,
            # `sonority_end`
            8,
            # `regular_positions`
            [{'name': 't3', 'remainder': 3, 'denominator': 4}],
            # `ad_hoc_positions`
            [{'name': 'ending', 'time': -0.01}],
            # `n_beats`
            8,
            # `expected`
            'ending'
        ),
        (
            # `sonority_start`
            3,
            # `sonority_end`
            9,
            # `regular_positions`
            [
                {'name': 'downbeat', 'remainder': 0, 'denominator': 4},
                {'name': 'middle', 'remainder': 2, 'denominator': 4},
            ],
            # `ad_hoc_positions`
            [{'name': 'ending', 'time': -0.01}],
            # `n_beats`
            16,
            # `expected`
            'downbeat'
        ),
        (
            # `sonority_start`
            3,
            # `sonority_end`
            9,
            # `regular_positions`
            [
                {'name': 'middle', 'remainder': 2, 'denominator': 4},
                {'name': 'downbeat', 'remainder': 0, 'denominator': 4},
            ],
            # `ad_hoc_positions`
            [{'name': 'ending', 'time': -0.01}],
            # `n_beats`
            16,
            # `expected`
            'middle'
        ),
    ]
)
def test_find_sonority_type(
        sonority_start: float, sonority_end: float, regular_positions: list[dict[str, Any]],
        ad_hoc_positions: list[dict[str, Any]], n_beats: int, expected: str
) -> None:
    """Test `find_sonority_type` function."""
    result = find_sonority_type(
        sonority_start, sonority_end, regular_positions, ad_hoc_positions, n_beats
    )
    assert result == expected
