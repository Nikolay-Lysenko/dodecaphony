"""
Test `dodecaphony.evaluation` module.

Author: Nikolay Lysenko
"""


from typing import Any

import pytest

from dodecaphony.evaluation import (
    evaluate,
    evaluate_absence_of_doubled_pitch_classes,
    evaluate_absence_of_voice_crossing,
    evaluate_cadence_duration,
    evaluate_climax_explicity,
    evaluate_consistency_of_rhythm_with_meter,
    evaluate_harmony_dynamic,
    evaluate_smoothness_of_voice_leading,
    parse_scoring_sets_registry,
)
from dodecaphony.fragment import Fragment, override_calculated_attributes


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
                n_tone_row_instances_by_group=[1, 1]
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
                n_tone_row_instances_by_group=[1, 1]
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
                n_tone_row_instances_by_group=[1, 1]
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
                n_tone_row_instances_by_group=[1, 1]
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
                n_tone_row_instances_by_group=[1, 1]
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
                n_tone_row_instances_by_group=[1]
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
                n_tone_row_instances_by_group=[1]
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
                n_tone_row_instances_by_group=[1]
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
                n_tone_row_instances_by_group=[1]
            ),
            # `params`
            [
                {
                    'name': 'default',
                    'scoring_functions': [
                        {
                            'name': 'smoothness_of_voice_leading',
                            'weight': 0.5,
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
