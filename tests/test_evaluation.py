"""
Test `dodecaphony.evaluation` module.

Author: Nikolay Lysenko
"""


from typing import Any, Optional

import pytest

from dodecaphony.evaluation import (
    evaluate,
    parse_scoring_sets_registry,
    weight_score,
)
from dodecaphony.fragment import Event, Fragment, ToneRowInstance, override_calculated_attributes
from .conftest import MEASURE_DURATIONS_BY_N_EVENTS


@pytest.mark.parametrize(
    "fragment, params, scoring_sets, expected",
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
    override_calculated_attributes(fragment)
    registry = parse_scoring_sets_registry(params)
    result, _ = evaluate(fragment, scoring_sets, registry)
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
