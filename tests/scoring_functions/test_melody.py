"""
Test `dodecaphony.scoring_functions.melody` module.

Author: Nikolay Lysenko
"""


from typing import Any, Optional

import pytest

from dodecaphony.scoring_functions.melody import (
    evaluate_absence_of_aimless_fluctuations,
    evaluate_climax_explicity,
    evaluate_direction_change_after_large_skip,
    evaluate_local_diatonicity_at_line_level,
    evaluate_pitch_class_prominence,
    evaluate_presence_of_intervallic_motif,
    evaluate_smoothness_of_voice_leading,
    evaluate_stackability,
    evaluate_transitions,
)
from dodecaphony.fragment import Fragment, ToneRowInstance, override_calculated_attributes
from tests.conftest import MEASURE_DURATIONS_BY_N_EVENTS


@pytest.mark.parametrize(
    "fragment, penalties, window_size, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
                    [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['C#', 'D#', 'C', 'E', 'B', 'D', 'F#', 'G#', 'F', 'A', 'G', 'A#'])],
                ],
                grouped_mutable_pauses_indices=[[]],
                grouped_immutable_pauses_indices=[[]],
                n_beats=12,
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
            # `penalties`
            {2: 1, 6: 0.5},
            # `window_size`
            3,
            # `expected`
            -0.6875
        ),
    ]
)
def test_evaluate_absence_of_aimless_fluctuations(
        fragment: Fragment, penalties: dict[int, float], window_size: int, expected: float
) -> None:
    """Test `evaluate_absence_of_aimless_fluctuations` function."""
    override_calculated_attributes(fragment)
    result = evaluate_absence_of_aimless_fluctuations(fragment, penalties, window_size)
    assert result == expected


@pytest.mark.parametrize(
    "fragment, height_penalties, duplication_penalty, expected",
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
            # `height_penalties`
            {
                2: 1.0,
                3: 0.8,
                4: 0.6,
                5: 0.5,
                6: 0.4,
                7: 0.3,
                8: 0.2,
                9: 0.1,
                10: 0.0,
            },
            # `duplication_penalty`
            0.5,
            # `expected`
            -0.2
        )
    ]
)
def test_evaluate_climax_explicity(
        fragment: Fragment, height_penalties: dict[int, float], duplication_penalty: float,
        expected: float
) -> None:
    """Test `evaluate_climax_explicity` function."""
    override_calculated_attributes(fragment)
    result = evaluate_climax_explicity(fragment, height_penalties, duplication_penalty)
    assert result == expected


@pytest.mark.parametrize(
    "fragment, min_skip_in_semitones, max_opposite_move_in_semitones, "
    "large_opposite_move_relative_penalty, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                    [[2.0, 2.0], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'])],
                    [ToneRowInstance(['C', 'F', 'D', 'F#', 'G', 'B', 'C#', 'G#', 'A', 'D#', 'A#', 'E'])],
                ],
                grouped_mutable_pauses_indices=[[2], [2]],
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
            5,
            # `max_opposite_move_in_semitones`
            2,
            # `large_opposite_move_relative_penalty`
            0.8,
            # `expected`
            -14 / 45
        )
    ]
)
def test_evaluate_direction_change_after_large_skip(
        fragment: Fragment, min_skip_in_semitones: int, max_opposite_move_in_semitones: int,
        large_opposite_move_relative_penalty: float, expected: float,
) -> None:
    """Test `evaluate_direction_change_after_large_skip` function."""
    override_calculated_attributes(fragment)
    result = evaluate_direction_change_after_large_skip(fragment)
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
            7,
            # `scale_types`
            None,
            # `expected`
            -97 / 546
        ),
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
                grouped_mutable_pauses_indices=[[2], []],
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
            7,
            # `scale_types`
            None,
            # `expected`
            -2 / 13
        ),
    ]
)
def test_evaluate_local_diatonicity_at_line_level(
        fragment: Fragment, depth: int, scale_types: Optional[list[str]], expected: float
) -> None:
    """Test `evaluate_local_diatonicity_at_line_level` function."""
    override_calculated_attributes(fragment)
    result = evaluate_local_diatonicity_at_line_level(fragment, depth, scale_types)
    assert round(result, 10) == round(expected, 10)


@pytest.mark.parametrize(
    "fragment, pitch_class_to_prominence_range, regular_positions, ad_hoc_positions, "
    "event_type_to_weight, default_weight, expected",
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
            # `pitch_class_to_prominence_range`
            {'B': (0.15, 1.0)},
            # `regular_positions`
            [
                {'name': 'downbeat', 'denominator': 4, 'remainder': 0},
                {'name': 'middle', 'denominator': 4, 'remainder': 2},
            ],
            # `ad_hoc_positions`
            [
                {'name': 'ending', 'time': -0.01},
            ],
            # `event_type_to_weight`
            {'downbeat': 1.0},
            # `default_weight`
            0,
            # `expected`
            -0.05
        ),
    ]
)
def test_evaluate_pitch_class_prominence(
        fragment: Fragment, pitch_class_to_prominence_range: dict[str, tuple[float, float]],
        regular_positions: list[dict[str, Any]], ad_hoc_positions: list[dict[str, Any]],
        event_type_to_weight: dict[str, float], default_weight: float, expected: float
) -> None:
    """Test `evaluate_pitch_class_prominence` function."""
    override_calculated_attributes(fragment)
    result = evaluate_pitch_class_prominence(
        fragment, pitch_class_to_prominence_range, regular_positions, ad_hoc_positions,
        event_type_to_weight, default_weight
    )
    assert round(result, 10) == round(expected, 10)


@pytest.mark.parametrize(
    "fragment, motif, min_n_occurrences, inversion, reversion, elision, inverted_elision, "
    "reverted_elision, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                ],
                grouped_mutable_pauses_indices=[[1], [2]],
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
            # `motif`
            [-2, 6, 2, -3],
            # `min_n_occurrences`
            [2, 2],
            # `inversion`
            False,
            # `reversion`
            False,
            # `elision`
            False,
            # `inverted_elision`
            False,
            # `reverted_elision`
            False,
            # `expected`
            -0.75
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0], [1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0]],
                ],
                grouped_tone_row_instances=[
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                    [ToneRowInstance(['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'])],
                ],
                grouped_mutable_pauses_indices=[[1], [2]],
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
            # `motif`
            [-2, 6, 2, -3],
            # `min_n_occurrences`
            [2, 2],
            # `inversion`
            True,
            # `reversion`
            True,
            # `elision`
            True,
            # `inverted_elision`
            True,
            # `reverted_elision`
            False,
            # `expected`
            -0.5
        ),
    ]
)
def test_evaluate_presence_of_intervallic_motif(
        fragment: Fragment, motif: list[int], min_n_occurrences: list[int],
        inversion: bool, reversion: bool, elision: bool, inverted_elision: bool,
        reverted_elision: bool, expected: float
) -> None:
    """Test `evaluate_presence_of_intervallic_motif` function."""
    override_calculated_attributes(fragment)
    result = evaluate_presence_of_intervallic_motif(
        fragment, motif, min_n_occurrences, inversion, reversion,
        elision, inverted_elision, reverted_elision
    )
    assert result == expected


@pytest.mark.parametrize(
    "fragment, penalty_deduction_per_line, n_semitones_to_penalty, expected",
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
            # `penalty_deduction_per_line`
            0.2,
            # `n_semitones_to_penalty`
            {
                0: 0.2,
                1: 0.0,
                2: 0.0,
                3: 0.1,
                4: 0.2,
                5: 0.3,
                6: 0.4,
                7: 0.5,
                8: 0.6,
                9: 0.7,
                10: 0.8,
                11: 0.9,
                12: 1.0,
            },
            # `expected`
            -1 / 6
        ),
    ]
)
def test_evaluate_smoothness_of_voice_leading(
        fragment: Fragment, penalty_deduction_per_line: float,
        n_semitones_to_penalty: dict[int, float], expected: float
) -> None:
    """Test `evaluate_smoothness_of_voice_leading` function."""
    override_calculated_attributes(fragment)
    result = evaluate_smoothness_of_voice_leading(
        fragment, penalty_deduction_per_line, n_semitones_to_penalty
    )
    assert round(result, 10) == round(expected, 10)


@pytest.mark.parametrize(
    "fragment, n_semitones_to_penalty, expected",
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
                grouped_mutable_pauses_indices=[[0, 13]],
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
            # `n_semitones_to_penalty`
            {
                0: 0.2,
                1: 0.0,
                2: 0.0,
                3: 0.1,
                4: 0.2,
                5: 0.3,
                6: 0.4,
                7: 0.5,
                8: 0.6,
                9: 0.7,
                10: 0.8,
                11: 0.9,
                12: 1.0,
            },
            # `expected`
            -2 / 15
        ),
    ]
)
def test_evaluate_stackability(
        fragment: Fragment, n_semitones_to_penalty: dict[int, float], expected: float
) -> None:
    """Test `evaluate_stackability` function."""
    override_calculated_attributes(fragment)
    result = evaluate_stackability(fragment, n_semitones_to_penalty)
    assert result == expected


@pytest.mark.parametrize(
    "fragment, n_semitones_to_penalty, left_end_notes, right_end_notes, expected",
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
                grouped_mutable_pauses_indices=[[0, 13]],
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
            # `n_semitones_to_penalty`
            {
                0: 0.2,
                1: 0.0,
                2: 0.0,
                3: 0.1,
                4: 0.2,
                5: 0.3,
                6: 0.4,
                7: 0.5,
                8: 0.6,
                9: 0.7,
                10: 0.8,
                11: 0.9,
                12: 1.0,
            },
            # `left_end_notes`
            ['G5', 'G4', 'G3'],
            # `right_end_notes`
            ['G5', 'G4', 'G3'],
            # `expected`
            -4/ 15
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
                grouped_mutable_pauses_indices=[[0, 13]],
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
            # `n_semitones_to_penalty`
            {
                0: 0.2,
                1: 0.0,
                2: 0.0,
                3: 0.1,
                4: 0.2,
                5: 0.3,
                6: 0.4,
                7: 0.5,
                8: 0.6,
                9: 0.7,
                10: 0.8,
                11: 0.9,
                12: 1.0,
            },
            # `left_end_notes`
            None,
            # `right_end_notes`
            ['G5', 'G4', 'G3'],
            # `expected`
            -1 / 3
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
                grouped_mutable_pauses_indices=[[0, 13]],
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
            # `n_semitones_to_penalty`
            {
                0: 0.2,
                1: 0.0,
                2: 0.0,
                3: 0.1,
                4: 0.2,
                5: 0.3,
                6: 0.4,
                7: 0.5,
                8: 0.6,
                9: 0.7,
                10: 0.8,
                11: 0.9,
                12: 1.0,
            },
            # `left_end_notes`
            ['G5', 'G4', 'G3'],
            # `right_end_notes`
            None,
            # `expected`
            -0.2
        ),
    ]
)
def test_evaluate_transitions(
        fragment: Fragment, n_semitones_to_penalty: dict[int, float],
        left_end_notes: Optional[list[str]], right_end_notes: Optional[list[str]], expected: float
) -> None:
    """Test `evaluate_transitions` function."""
    override_calculated_attributes(fragment)
    result = evaluate_transitions(
        fragment, n_semitones_to_penalty, left_end_notes, right_end_notes
    )
    assert round(result, 8) == round(expected, 8)
