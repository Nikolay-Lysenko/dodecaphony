"""
Test `dodecaphony.optimization` module.

Author: Nikolay Lysenko
"""


from typing import Any

import pytest

from dodecaphony.evaluation import parse_scoring_sets_registry
from dodecaphony.fragment import Event, Fragment
from dodecaphony.optimization import optimize_with_local_search
from dodecaphony.transformations import create_transformations_registry


@pytest.mark.parametrize(
    "fragment, n_iterations, n_trials_per_iteration, default_n_transformations_per_trial, "
    "n_transformations_increment, max_n_transformations_per_trial, beam_width, max_transposition, "
    "transformation_probabilities, scoring_sets, scoring_sets_params",
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
                            Event(line_index=0, start_time=4.0, duration=1.0),
                            Event(line_index=0, start_time=5.0, duration=1.0),
                            Event(line_index=0, start_time=6.0, duration=1.0),
                            Event(line_index=0, start_time=7.0, duration=1.0),
                            Event(line_index=0, start_time=8.0, duration=1.0),
                            Event(line_index=0, start_time=9.0, duration=1.0),
                            Event(line_index=0, start_time=10.0, duration=1.0),
                            Event(line_index=0, start_time=11.0, duration=0.5),
                            Event(line_index=0, start_time=11.5, duration=0.5),
                            Event(line_index=0, start_time=12.0, duration=1.0),
                            Event(line_index=0, start_time=13.0, duration=1.0),
                            Event(line_index=0, start_time=14.0, duration=1.0),
                            Event(line_index=0, start_time=15.0, duration=1.0),
                            Event(line_index=0, start_time=16.0, duration=1.0),
                            Event(line_index=0, start_time=17.0, duration=1.0),
                            Event(line_index=0, start_time=18.0, duration=1.0),
                            Event(line_index=0, start_time=19.0, duration=1.0),
                            Event(line_index=0, start_time=20.0, duration=1.0),
                            Event(line_index=0, start_time=21.0, duration=1.0),
                            Event(line_index=0, start_time=22.0, duration=1.0),
                            Event(line_index=0, start_time=23.0, duration=1.0),
                        ]
                    ],
                    [
                        [
                            Event(line_index=1, start_time=0.0, duration=2.0),
                            Event(line_index=1, start_time=2.0, duration=2.0),
                            Event(line_index=1, start_time=4.0, duration=2.0),
                            Event(line_index=1, start_time=6.0, duration=2.0),
                            Event(line_index=1, start_time=8.0, duration=2.0),
                            Event(line_index=1, start_time=10.0, duration=2.0),
                            Event(line_index=1, start_time=12.0, duration=2.0),
                            Event(line_index=1, start_time=14.0, duration=2.0),
                            Event(line_index=1, start_time=16.0, duration=2.0),
                            Event(line_index=1, start_time=18.0, duration=2.0),
                            Event(line_index=1, start_time=20.0, duration=2.0),
                            Event(line_index=1, start_time=22.0, duration=2.0),
                        ]
                    ]
                ],
                sonic_content=[
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=24,
                line_ids=[1, 2],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_tone_row_instances_by_group=[2, 1]
            ),
            # `n_iterations`
            10,
            # `n_trials_per_iteration`
            10,
            # `default_n_transformations_per_trial`
            1,
            # `n_transformations_increment`
            1,
            # `max_n_transformations_per_trial`
            2,
            # `beam_width`
            1,
            # `max_transposition`
            1,
            # `transformation_probabilities`
            {
                'duration_change': 0.25,
                'inversion': 0.25,
                'reversion': 0.25,
                'transposition': 0.25,
            },
            # `scoring_sets`
            ['default'],
            # `scoring_sets_params`
            [
                {
                    'name': 'default',
                    'scoring_functions': [
                        {
                            'name': 'smoothness_of_voice_leading',
                            'weight': 1.0,
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
        ),
    ]
)
def test_optimize_with_local_search(
        fragment: Fragment, n_iterations: int, n_trials_per_iteration: int,
        default_n_transformations_per_trial: int, n_transformations_increment: int,
        max_n_transformations_per_trial: int, beam_width: int, max_transposition: int,
        transformation_probabilities: dict[str, float], scoring_sets: list[str],
        scoring_sets_params: list[dict[str, Any]]
) -> None:
    """Test `optimize_with_local_search` function."""
    transformations_registry = create_transformations_registry(max_transposition)
    scoring_sets_registry = parse_scoring_sets_registry(scoring_sets_params)
    optimize_with_local_search(
        fragment, n_iterations, n_trials_per_iteration, default_n_transformations_per_trial,
        n_transformations_increment, max_n_transformations_per_trial, beam_width,
        transformations_registry, transformation_probabilities, scoring_sets, scoring_sets_registry
    )
