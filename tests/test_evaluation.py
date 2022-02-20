"""
Test `dodecaphony.evaluation` module.

Author: Nikolay Lysenko
"""


from typing import Any, Optional

import pytest

from dodecaphony.evaluation import (
    evaluate,
    evaluate_absence_of_aimless_fluctuations,
    evaluate_absence_of_doubled_pitch_classes,
    evaluate_absence_of_simultaneous_skips,
    evaluate_absence_of_voice_crossing,
    evaluate_cadence_duration,
    evaluate_climax_explicity,
    evaluate_consistency_of_rhythm_with_meter,
    evaluate_dissonances_preparation_and_resolution,
    evaluate_harmony_dynamic,
    evaluate_local_diatonicity,
    evaluate_presence_of_intervallic_motif,
    evaluate_rhythmic_homogeneity,
    evaluate_smoothness_of_voice_leading,
    evaluate_stackability,
    find_indices_of_dissonating_events,
    find_sonority_type,
    parse_scoring_sets_registry,
    weight_score,
)
from dodecaphony.fragment import Event, Fragment, override_calculated_attributes


@pytest.mark.parametrize(
    "fragment, penalties, window_size, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                ],
                sonic_content=[
                    ['C#', 'D#', 'C', 'E', 'B', 'D', 'F#', 'G#', 'F', 'A', 'G', 'A#'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=12,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[2],
                n_tone_row_instances_by_group=[1],
                mutable_temporal_content_indices=[0, 1],
                mutable_sonic_content_indices=[0],
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
    fragment = override_calculated_attributes(fragment)
    result = evaluate_absence_of_aimless_fluctuations(fragment, penalties, window_size)
    assert result == expected


@pytest.mark.parametrize(
    "fragment, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
                    [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
                ],
                sonic_content=[
                    ['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause'],
                    ['B', 'C', 'D#', 'A', 'G', 'A#', 'G#', 'C#', 'E', 'F#', 'D', 'F'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=16,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[1, 1],
                n_tone_row_instances_by_group=[1, 1],
                mutable_temporal_content_indices=[0, 1],
                mutable_sonic_content_indices=[0, 1],
            ),
            # `expected`
            -0.125
        ),
    ]
)
def test_evaluate_absence_of_doubled_pitch_classes(fragment: Fragment, expected: float) -> None:
    """Test `evaluate_absence_of_doubled_pitch_classes` function."""
    fragment = override_calculated_attributes(fragment)
    result = evaluate_absence_of_doubled_pitch_classes(fragment)
    assert result == expected


@pytest.mark.parametrize(
    "fragment, min_skip_in_semitones, max_skips_share, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
                    [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
                ],
                sonic_content=[
                    ['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause'],
                    ['B', 'C', 'D#', 'A', 'G', 'A#', 'G#', 'C#', 'E', 'F#', 'D', 'F'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=16,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[1, 1],
                n_tone_row_instances_by_group=[1, 1],
                mutable_temporal_content_indices=[0, 1],
                mutable_sonic_content_indices=[0, 1],
            ),
            # `min_skip_in_semitones`
            7,
            # `max_skips_share`
            0.1,
            # `expected`
            -2 / 15
        ),
    ]
)
def test_evaluate_absence_of_simultaneous_skips(
        fragment: Fragment, min_skip_in_semitones: int, max_skips_share: float, expected: float
) -> None:
    """Test `evaluate_absence_of_simultaneous_skips` function."""
    fragment = override_calculated_attributes(fragment)
    result = evaluate_absence_of_simultaneous_skips(
        fragment, min_skip_in_semitones, max_skips_share
    )
    assert result == expected


@pytest.mark.parametrize(
    "fragment, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
                    [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
                ],
                sonic_content=[
                    ['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F', 'pause'],
                    ['A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=16,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[1, 1],
                n_tone_row_instances_by_group=[1, 1],
                mutable_temporal_content_indices=[0, 1],
                mutable_sonic_content_indices=[0, 1],
            ),
            # `expected`
            -0.0625
        ),
    ]
)
def test_evaluate_absence_of_voice_crossing(fragment: Fragment, expected: float) -> None:
    """Test `evaluate_absence_of_voice_crossing` function."""
    fragment = override_calculated_attributes(fragment)
    result = evaluate_absence_of_voice_crossing(fragment)
    assert result == expected


@pytest.mark.parametrize(
    "fragment, max_duration, last_sonority_weight, last_notes_weight, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
                    [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
                ],
                sonic_content=[
                    ['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'],
                    ['A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=16,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[1, 1],
                n_tone_row_instances_by_group=[1, 1],
                mutable_temporal_content_indices=[0, 1],
                mutable_sonic_content_indices=[0, 1],
            ),
            # `max_duration`
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
        fragment: Fragment, max_duration: float,
        last_sonority_weight: float, last_notes_weight: float, expected: float
) -> None:
    """Test `evaluate_cadence_duration` function."""
    fragment = override_calculated_attributes(fragment)
    result = evaluate_cadence_duration(
        fragment, max_duration, last_sonority_weight, last_notes_weight
    )
    assert result == expected


@pytest.mark.parametrize(
    "fragment, height_penalties, duplication_penalty, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
                    [2.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                sonic_content=[
                    ['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F', 'pause'],
                    ['A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=16,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[1, 1],
                n_tone_row_instances_by_group=[1, 1],
                mutable_temporal_content_indices=[0, 1],
                mutable_sonic_content_indices=[0, 1],
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
    fragment = override_calculated_attributes(fragment)
    result = evaluate_climax_explicity(fragment, height_penalties, duplication_penalty)
    assert result == expected


@pytest.mark.parametrize(
    "fragment, consistent_patterns, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
                    [2.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                sonic_content=[
                    ['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F', 'pause'],
                    ['A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=16,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[1, 1],
                n_tone_row_instances_by_group=[1, 1],
                mutable_temporal_content_indices=[0, 1],
                mutable_sonic_content_indices=[0, 1],
            ),
            # `consistent_patterns`
            [
                [2.0, 2.0],
                [2.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            # `expected`
            -0.125
        ),
    ]
)
def test_evaluate_consistency_of_rhythm_with_meter(
        fragment: Fragment, consistent_patterns: list[list[float]], expected: float
) -> None:
    """Test `evaluate_consistency_of_rhythm_with_meter` function."""
    fragment = override_calculated_attributes(fragment)
    result = evaluate_consistency_of_rhythm_with_meter(fragment, consistent_patterns)
    assert result == expected


@pytest.mark.parametrize(
    "fragment, n_semitones_to_pt_and_ngh_preparation_penalty, "
    "n_semitones_to_pt_and_ngh_resolution_penalty, n_semitones_to_suspension_resolution_penalty, "
    "expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [4.0, 4.0, 4.0, 4.0],
                    [4.0, 4.0, 4.0, 4.0],
                    [4.0, 4.0, 4.0, 4.0],
                ],
                sonic_content=[
                    ['G', 'E', 'C', 'G#', 'D#', 'C#', 'A', 'F', 'D', 'A#', 'F#', 'B']
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=16,
                line_ids=[0, 1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[3],
                n_tone_row_instances_by_group=[1],
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_sonic_content_indices=[0],
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
                    [4.0, 4.0, 4.0, 4.0],
                    [4.0, 4.0, 4.0, 4.0],
                    [4.0, 2.0, 2.0, 4.0, 4.0],
                ],
                sonic_content=[
                    ['G', 'E', 'C', 'G#', 'D#', 'pause', 'C#', 'A', 'F', 'D', 'A#', 'F#', 'B']
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=16,
                line_ids=[0, 1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[3],
                n_tone_row_instances_by_group=[1],
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_sonic_content_indices=[0],
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
                    [4.0, 4.0, 4.0, 4.0],
                    [4.0, 4.0, 4.0, 4.0],
                    [4.0, 2.0, 2.0, 4.0, 4.0],
                ],
                sonic_content=[
                    ['G', 'E', 'C', 'G#', 'D#', 'C#', 'pause', 'A', 'F', 'D', 'A#', 'F#', 'B']
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=16,
                line_ids=[0, 1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[3],
                n_tone_row_instances_by_group=[1],
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_sonic_content_indices=[0],
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
                    [4.0, 4.0, 4.0, 4.0],
                    [4.0, 4.0, 4.0, 4.0],
                    [4.0, 4.0, 4.0, 4.0],
                ],
                sonic_content=[
                    ['G#', 'D#', 'C#', 'A#', 'F#', 'B', 'G', 'E', 'C', 'A', 'F', 'D']
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=16,
                line_ids=[0, 1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[3],
                n_tone_row_instances_by_group=[1],
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_sonic_content_indices=[0],
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
                    [2.0, 4.0, 2.0, 4.0],
                    [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
                ],
                sonic_content=[
                    ['E', 'C', 'F', 'D', 'D#', 'B', 'G#', 'C#', 'F#', 'G', 'A', 'A#']
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=16,
                line_ids=[0, 1],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[2],
                n_tone_row_instances_by_group=[1],
                mutable_temporal_content_indices=[0, 1],
                mutable_sonic_content_indices=[0],
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
                    [2.0, 4.0, 1.0, 1.0, 4.0],
                    [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
                ],
                sonic_content=[
                    ['E', 'C', 'F', 'D', 'D#', 'pause', 'B', 'G#', 'C#', 'F#', 'G', 'A', 'A#']
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=16,
                line_ids=[0, 1],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[2],
                n_tone_row_instances_by_group=[1],
                mutable_temporal_content_indices=[0, 1],
                mutable_sonic_content_indices=[0],
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
                    [2.0, 2.0, 2.0, 6.0],
                    [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
                ],
                sonic_content=[
                    ['E', 'C', 'F', 'D', 'D#', 'B', 'G#', 'C#', 'F#', 'G', 'A', 'A#']
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=16,
                line_ids=[0, 1],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[2],
                n_tone_row_instances_by_group=[1],
                mutable_temporal_content_indices=[0, 1],
                mutable_sonic_content_indices=[0],
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
    fragment = override_calculated_attributes(fragment)
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
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.5, 0.5],
                    [1.0, 1.0, 1.0, 0.5, 0.5],
                ],
                sonic_content=[
                    [
                        'B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F',
                        'pause', 'pause'
                    ]
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=4,
                line_ids=[1, 2, 3],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[3],
                n_tone_row_instances_by_group=[1],
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_sonic_content_indices=[0],
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
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.5, 0.5],
                    [1.0, 1.0, 1.0, 0.5, 0.5],
                ],
                sonic_content=[
                    [
                        'B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F',
                        'pause', 'pause'
                    ]
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=4,
                line_ids=[1, 2, 3],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[3],
                n_tone_row_instances_by_group=[1],
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_sonic_content_indices=[0],
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
def test_evaluate_harmony_dynamic(
        fragment: Fragment, regular_positions: list[dict[str, Any]],
        ad_hoc_positions: list[dict[str, Any]], ranges: dict[str, tuple[float, float]],
        n_semitones_to_stability: dict[int, float], expected: float
) -> None:
    """Test `evaluate_harmony_dynamic` function."""
    fragment = override_calculated_attributes(fragment)
    result = evaluate_harmony_dynamic(
        fragment, regular_positions, ad_hoc_positions, ranges, n_semitones_to_stability
    )
    assert round(result, 10) == expected


@pytest.mark.parametrize(
    "fragment, depth, scale_types, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
                    [2.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                sonic_content=[
                    ['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F', 'pause'],
                    ['A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=16,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[1, 1],
                n_tone_row_instances_by_group=[1, 1],
                mutable_temporal_content_indices=[0, 1],
                mutable_sonic_content_indices=[0, 1],
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
def test_evaluate_local_diatonicity(
        fragment: Fragment, depth: int, scale_types: Optional[list[str]], expected: float
) -> None:
    """Test `evaluate_local_diatonicity` function."""
    fragment = override_calculated_attributes(fragment)
    result = evaluate_local_diatonicity(fragment, depth, scale_types)
    assert round(result, 10) == round(expected, 10)


@pytest.mark.parametrize(
    "fragment, motif, min_n_occurrences, inversion, reversion, elision, inverted_elision, "
    "reverted_elision, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
                ],
                sonic_content=[
                    ['B', 'pause', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'],
                    ['B', 'A', 'pause', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=16,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[1, 1],
                n_tone_row_instances_by_group=[1, 1],
                mutable_temporal_content_indices=[0, 1],
                mutable_sonic_content_indices=[0, 1],
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
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
                ],
                sonic_content=[
                    ['B', 'pause', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'],
                    ['B', 'A', 'pause', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=16,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[1, 1],
                n_tone_row_instances_by_group=[1, 1],
                mutable_temporal_content_indices=[0, 1],
                mutable_sonic_content_indices=[0, 1],
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
    fragment = override_calculated_attributes(fragment)
    result = evaluate_presence_of_intervallic_motif(
        fragment, motif, min_n_occurrences, inversion, reversion,
        elision, inverted_elision, reverted_elision
    )
    assert result == expected


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
                    ['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F', 'pause'],
                    ['A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=16,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[1, 1],
                n_tone_row_instances_by_group=[1, 1],
                mutable_temporal_content_indices=[0, 1],
                mutable_sonic_content_indices=[0, 1],
            ),
            # `expected`
            -(1 / 9 + (0.6 + 2 / 3 + 1 / 7) / 6)
        ),
    ]
)
def test_evaluate_rhythmic_homogeneity(fragment: Fragment, expected: float) -> None:
    """Test `evaluate_rhythmic_homogeneity` function."""
    fragment = override_calculated_attributes(fragment)
    result = evaluate_rhythmic_homogeneity(fragment)
    assert round(result, 10) == round(expected, 10)


@pytest.mark.parametrize(
    "fragment, penalty_deduction_per_line, n_semitones_to_penalty, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.5, 0.5],
                    [1.0, 1.0, 1.0, 0.5, 0.5],
                ],
                sonic_content=[
                    [
                        'B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F',
                        'pause', 'pause'
                    ]
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=4,
                line_ids=[1, 2, 3],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[3],
                n_tone_row_instances_by_group=[1],
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_sonic_content_indices=[0],
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
    fragment = override_calculated_attributes(fragment)
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
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.5, 0.5],
                    [1.0, 1.0, 1.0, 0.5, 0.5],
                ],
                sonic_content=[
                    [
                        'pause', 'B', 'A', 'G', 'C#', 'D#', 'C',
                        'D', 'A#', 'F#', 'E', 'G#', 'F', 'pause'
                    ]
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=4,
                line_ids=[1, 2, 3],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[3],
                n_tone_row_instances_by_group=[1],
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_sonic_content_indices=[0],
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
    fragment = override_calculated_attributes(fragment)
    result = evaluate_stackability(fragment, n_semitones_to_penalty)
    assert result == expected


@pytest.mark.parametrize(
    "sonority, meter_numerator, expected",
    [
        (
            # `sonority`
            [
                Event(line_index=0, start_time=2.0, duration=4.0, pitch_class='E', position_in_semitones=67),
                Event(line_index=1, start_time=4.0, duration=2.0, pitch_class='D', position_in_semitones=65),
                Event(line_index=2, start_time=4.0, duration=1.0, pitch_class='C', position_in_semitones=63),
            ],
            # `meter_numerator`
            4,
            # `expected`
            ({1, 2}, {0})
        ),
        (
            # `sonority`
            [
                Event(line_index=0, start_time=2.0, duration=4.0, pitch_class='G', position_in_semitones=70),
                Event(line_index=1, start_time=4.0, duration=2.0, pitch_class='E', position_in_semitones=67),
                Event(line_index=2, start_time=4.0, duration=1.0, pitch_class='C', position_in_semitones=63),
            ],
            # `meter_numerator`
            4,
            # `expected`
            (set(), set())
        ),
        (
            # `sonority`
            [
                Event(line_index=0, start_time=2.0, duration=2.0, pitch_class='E', position_in_semitones=67),
                Event(line_index=1, start_time=2.0, duration=2.0, pitch_class='D', position_in_semitones=65),
                Event(line_index=2, start_time=3.0, duration=1.0, pitch_class='C', position_in_semitones=63),
            ],
            # `meter_numerator`
            4,
            # `expected`
            ({2}, set())
        ),
        (
            # `sonority`
            [
                Event(line_index=0, start_time=4.0, duration=2.0, pitch_class='E', position_in_semitones=67),
                Event(line_index=1, start_time=2.0, duration=4.0, pitch_class='D', position_in_semitones=65),
                Event(line_index=2, start_time=4.0, duration=1.0, pitch_class='C', position_in_semitones=63),
            ],
            # `meter_numerator`
            4,
            # `expected`
            (set(), {1})
        ),
        (
            # `sonority`
            [
                Event(line_index=0, start_time=2.0, duration=2.0, pitch_class='pause', position_in_semitones=None),
                Event(line_index=1, start_time=3.0, duration=1.0, pitch_class='D', position_in_semitones=65),
                Event(line_index=2, start_time=2.0, duration=2.0, pitch_class='C', position_in_semitones=63),
            ],
            # `meter_numerator`
            4,
            # `expected`
            ({1}, set())
        ),
        (
            # `sonority`
            [
                Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='E#', position_in_semitones=68),
                Event(line_index=1, start_time=2.0, duration=2.0, pitch_class='D', position_in_semitones=65),
                Event(line_index=2, start_time=2.0, duration=2.0, pitch_class='C', position_in_semitones=63),
            ],
            # `meter_numerator`
            4,
            # `expected`
            ({0}, set())
        ),
    ]
)
def test_find_indices_of_dissonating_events(
        sonority: list[Event], meter_numerator: int, expected: tuple[set[int], set[int]]
) -> None:
    """Test `find_indices_of_dissonating_events` function."""
    result = find_indices_of_dissonating_events(sonority, meter_numerator)
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


@pytest.mark.parametrize(
    "fragment, params, scoring_sets, expected",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 0.5, 0.5],
                    [1.0, 1.0, 1.0, 0.5, 0.5],
                ],
                sonic_content=[
                    [
                        'B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A#', 'F#', 'E', 'G#', 'F',
                        'pause', 'pause'
                    ]
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=4,
                line_ids=[1, 2, 3],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[3],
                n_tone_row_instances_by_group=[1],
                mutable_temporal_content_indices=[0, 1, 2],
                mutable_sonic_content_indices=[0],
            ),
            # `params`
            [
                {
                    'name': 'default',
                    'scoring_functions': [
                        {
                            'name': 'smoothness_of_voice_leading',
                            'weights': {0.0: 0.5},
                            'penalty_deduction_per_line': 0.2,
                            'n_semitones_to_penalty': {
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
                        },
                    ],
                }
            ],
            # `scoring_sets`
            ['default'],
            # `expected`
            -1 / 12
        ),
    ]
)
def test_parse_scoring_sets_registry(
        fragment: Fragment, params: list[dict[str, Any]], scoring_sets: list[str], expected: float
) -> None:
    """Test `parse_scoring_sets_registry` function."""
    fragment = override_calculated_attributes(fragment)
    registry = parse_scoring_sets_registry(params)
    result = evaluate(fragment, scoring_sets, registry)
    assert round(result, 10) == round(expected, 10)


@pytest.mark.parametrize(
    "unweighted_score, weights, expected",
    [
        (-0.3, {0.0: 1.0, -0.2: 2.0, -0.5: 3.0}, -0.4),
        (-0.7, {0.0: 1.0, -0.2: 2.0, -0.5: 3.0}, -1.4),
    ]
)
def test_weight_score(
        unweighted_score: float, weights: dict[float, float], expected: float
) -> None:
    """Test `weight_score` function."""
    result = weight_score(unweighted_score, weights)
    assert round(result, 8) == expected
