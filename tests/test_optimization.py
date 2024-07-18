"""
Test `dodecaphony.optimization` module.

Author: Nikolay Lysenko
"""


from typing import Any

import pytest

from dodecaphony.evaluation import parse_scoring_sets_registry
from dodecaphony.fragment import Fragment, ToneRowInstance, override_calculated_attributes
from dodecaphony.optimization import (
    Record,
    Task,
    create_tasks,
    optimize_with_variable_neighborhood_search,
    select_distinct_best_records
)
from dodecaphony.transformations import create_transformations_registry
from .conftest import MEASURE_DURATIONS_BY_N_EVENTS


@pytest.mark.parametrize(
    "incumbent_solutions, n_trials_per_iteration, paralleling_params, expected",
    [
        (
            # `incumbent_solutions`
            [1, 2, 3, 4, 5],
            # `n_trials_per_iteration`
            1000,
            # `paralleling_params`
            {'n_processes': 8},
            # `expected`
            [
                [Task(1, 625)],
                [Task(1, 375), Task(2, 250)],
                [Task(2, 625)],
                [Task(2, 125), Task(3, 500)],
                [Task(3, 500), Task(4, 125)],
                [Task(4, 625)],
                [Task(4, 250), Task(5, 375)],
                [Task(5, 625)],
            ]
        ),
        (
            # `incumbent_solutions`
            [1, 2, 3, 4, 5],
            # `n_trials_per_iteration`
            1000,
            # `paralleling_params`
            {'n_processes': 6},
            # `expected`
            [
                [Task(1, 834)],
                [Task(1, 166), Task(2, 668)],
                [Task(2, 332), Task(3, 502)],
                [Task(3, 498), Task(4, 336)],
                [Task(4, 664), Task(5, 170)],
                [Task(5, 830)],
            ]
        ),
    ]
)
def test_create_tasks(
        incumbent_solutions: list[int], n_trials_per_iteration: int,
        paralleling_params: dict[str, Any], expected: list[list[Task]]
) -> None:
    """Test `create_tasks` function."""
    result = create_tasks(incumbent_solutions, n_trials_per_iteration, paralleling_params)
    assert result == expected


@pytest.mark.parametrize(
    "fragment, n_iterations, n_trials_per_iteration, beam_width, neighborhoods, perturbation, "
    "max_rotation, max_transposition, scoring_sets, scoring_sets_params",
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
            # `n_iterations`
            10,
            # `n_trials_per_iteration`
            10,
            # `beam_width`
            1,
            # `neighborhoods`
            [
                {
                    'n_transformations_per_trial': 1,
                    'transformation_probabilities': {
                        'inversion': 0.25,
                        'reversion': 0.25,
                        'transposition': 0.25,
                        'measure_durations_change': 0.25,
                    },
                },
                {
                    'n_transformations_per_trial': 2,
                    'transformation_probabilities': {
                        'inversion': 0.25,
                        'reversion': 0.25,
                        'rotation': 0.25,
                        'measure_durations_change': 0.25,
                    },
                },
            ],
            # `perturbation`
            {
                    'n_transformations': 1,
                    'transformation_probabilities': {
                        'line_durations_change': 1.0,
                    },
            },
            # `max_rotation`
            1,
            # `max_transposition`
            1,
            # `scoring_sets`
            ['default'],
            # `scoring_sets_params`
            [
                {
                    'name': 'default',
                    'scoring_functions': [
                        {
                            'name': 'smoothness_of_voice_leading',
                            'weights': {0.0: 1.0},
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
def test_optimize_with_variable_neighborhood_search(
        fragment: Fragment, n_iterations: int, n_trials_per_iteration: int,
        beam_width: int, neighborhoods: list[dict[str, Any]], perturbation: dict[str, Any],
        max_rotation: int, max_transposition: int,
        scoring_sets: list[str], scoring_sets_params: list[dict[str, Any]]
) -> None:
    """Test `optimize_with_variable_neighborhood_search` function."""
    transformations_registry = create_transformations_registry(max_rotation, max_transposition)
    scoring_sets_registry = parse_scoring_sets_registry(scoring_sets_params)
    override_calculated_attributes(fragment)
    optimize_with_variable_neighborhood_search(
        fragment, n_iterations, n_trials_per_iteration, beam_width, transformations_registry,
        neighborhoods, perturbation, scoring_sets, scoring_sets_registry
    )


@pytest.mark.parametrize(
    "records, n_records, expected",
    [
        ([], 5, []),
        ([Record('a', 0), Record('b', -1), Record('a', 0)], 2, [Record('a', 0), Record('b', -1)]),
    ]
)
def test_select_distinct_best_records(
        records: list[Record], n_records: int, expected: list[Record]
) -> None:
    """Test `select_distinct_best_records` function."""
    result = select_distinct_best_records(records, n_records)
    assert result == expected
