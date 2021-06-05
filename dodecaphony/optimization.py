"""
Iteratively improve fragment.

Author: Nikolay Lysenko
"""


import copy
import math
import os
from dataclasses import dataclass
from typing import Any, Optional

from .evaluation import SCORING_SETS_REGISTRY_TYPE, evaluate
from .fragment import Fragment
from .transformations import TRANSFORMATIONS_REGISTRY_TYPE, transform
from .utils import starmap_in_parallel


@dataclass
class Record:
    """An evaluated fragment and its score."""
    fragment: Fragment
    score: float


@dataclass
class Task:
    """Task for a single process."""
    incumbent_solution: Fragment
    n_trials: int


def select_distinct_best_records(records: list[Record], n_records: int) -> list[Record]:
    """
    Select records related to highest scores (without duplicates).

    :param records:
        original records
    :param n_records:
        number of unique records to select
    :return:
        best records
    """
    results = []
    for record in sorted(records, key=lambda x: -x.score):
        if record not in results:
            results.append(record)
        if len(results) == n_records:
            break
    return results


def generate_new_records(
        tasks: list[Task],
        n_records_to_return: int,
        n_transformations_per_trial: int,
        transformation_registry: TRANSFORMATIONS_REGISTRY_TYPE,
        transformation_names: list[str],
        transformation_probabilities: list[float],
        scoring_sets: list[str],
        scoring_sets_registry: SCORING_SETS_REGISTRY_TYPE
) -> list[Record]:
    """
    Generate records for given tasks.

    :param tasks:
        fragments such that their neighborhoods should be searched and numbers of transformed
        fragments to generate and evaluate for each of them
    :param n_records_to_return:
        number of best records to return; this argument is added to the function for the sake of
        performance, because it reduces amount of data returned by a child process to a
        parent process
    :param n_transformations_per_trial:
        number of transformations to be applied to generate a new fragment
    :param transformation_registry:
        mapping from names to corresponding transformations and their arguments
    :param transformation_names:
        names of transformations to choose from
    :param transformation_probabilities:
        probabilities of corresponding transformations; this argument must have the same length
        as `transformation_names`
    :param scoring_sets:
        names of scoring sets to be used for fragments evaluation
    :param scoring_sets_registry:
        mapping from a name of a scoring set to a list of triples of a scoring function,
        its weight, and its parameters
    :return:
        best generated records
    """
    new_records = []
    for task in tasks:
        incumbent_solution = task.incumbent_solution
        for trial_id in range(task.n_trials):
            candidate = Fragment(
                [copy.copy(x) for x in incumbent_solution.temporal_content],
                [copy.copy(x) for x in incumbent_solution.sonic_content],
                incumbent_solution.meter_numerator,
                incumbent_solution.meter_denominator,
                incumbent_solution.n_beats,
                incumbent_solution.line_ids,
                incumbent_solution.upper_line_highest_position,
                incumbent_solution.upper_line_lowest_position,
                incumbent_solution.n_melodic_lines_by_group,
                incumbent_solution.n_tone_row_instances_by_group,
                incumbent_solution.mutable_temporal_content_indices,
                incumbent_solution.mutable_sonic_content_indices
            )
            candidate = transform(
                candidate,
                n_transformations_per_trial,
                transformation_registry,
                transformation_names,
                transformation_probabilities
            )
            score = evaluate(candidate, scoring_sets, scoring_sets_registry)
            score = round(score, 10)  # Prevent overflow during records comparison.
            new_records.append(Record(candidate, score))
    new_records = select_distinct_best_records(new_records, n_records_to_return)
    return new_records


def create_tasks(
        incumbent_solutions: list[Fragment],
        n_trials_per_iteration: int,
        paralleling_params: dict[str, Any]
) -> list[list[Task]]:
    """
    Distribute incumbent solutions and trials to improve them between processes.

    :param incumbent_solutions:
        fragments such that their neighborhoods should be searched
    :param n_trials_per_iteration:
        number of transformed fragments to generate and evaluate per each incumbent solution
    :param paralleling_params:
        settings of parallel running of trials
    :return:
        tasks for each process
    """
    raw_tasks = []
    for incumbent_solution in incumbent_solutions:
        raw_tasks.append(Task(incumbent_solution, n_trials_per_iteration))
    n_trials = len(incumbent_solutions) * n_trials_per_iteration
    n_processes = paralleling_params.get('n_processes') or os.cpu_count()
    n_trials_per_process = math.ceil(n_trials / n_processes)
    remaining_n_trials_per_current_process = n_trials_per_process
    tasks = [[Task(incumbent_solutions[0], 0)]]
    i = 0
    while i < len(raw_tasks):
        if raw_tasks[i].incumbent_solution not in [task.incumbent_solution for task in tasks[-1]]:
            tasks[-1].append(Task(raw_tasks[i].incumbent_solution, 0))
        n_trials_to_add = min(remaining_n_trials_per_current_process, raw_tasks[i].n_trials)
        tasks[-1][-1].n_trials += n_trials_to_add
        remaining_n_trials_per_current_process -= n_trials_to_add
        raw_tasks[i].n_trials -= n_trials_to_add
        if remaining_n_trials_per_current_process == 0:
            tasks.append([])
            remaining_n_trials_per_current_process = n_trials_per_process
        if raw_tasks[i].n_trials == 0:
            i += 1
    if not tasks[-1]:
        tasks.pop()
    return tasks


def optimize_with_local_search(
        fragment: Fragment,
        n_iterations: int,
        n_trials_per_iteration: int,
        default_n_transformations_per_trial: int,
        n_transformations_increment: int,
        max_n_transformations_per_trial: int,
        beam_width: int,
        transformation_registry: TRANSFORMATIONS_REGISTRY_TYPE,
        transformation_probabilities: dict[str, float],
        scoring_sets: list[str],
        scoring_sets_registry: SCORING_SETS_REGISTRY_TYPE,
        paralleling_params: Optional[dict[str, Any]] = None
) -> list[Fragment]:
    """
    Optimize initial fragment with local beam search.

    :param fragment:
        initial fragment
    :param n_iterations:
        number of optimization steps where a step is generation of new incumbent solutions
        by transformation of previous incumbent solutions
    :param n_trials_per_iteration:
        number of transformed fragments to generate and evaluate per each incumbent solution
        at each iteration
    :param default_n_transformations_per_trial:
        default number of transformations to be applied to generate a new fragment
    :param n_transformations_increment:
        increment of number of transformations added after every iteration such that global
        best score has not been improved after it; default number is used again as soon as
        global best score improves
    :param max_n_transformations_per_trial:
        maximum number of transformations to be applied to generate a new fragment
    :param beam_width:
        number of incumbent solutions to consider at each iteration
    :param transformation_registry:
        mapping from names to corresponding transformations and their arguments
    :param transformation_probabilities:
        mapping from transformation name to its probability
    :param scoring_sets:
        names of scoring sets to be used for fragments evaluation
    :param scoring_sets_registry:
        mapping from a name of a scoring set to a list of triples of a scoring function,
        its weight, and its parameters
    :param paralleling_params:
        settings of parallel running of trials
    :return:
        optimized fragments
    """
    incumbent_solutions = [fragment]
    records = []
    previous_best_score = -1e9
    n_transformations_per_trial = default_n_transformations_per_trial
    transformation_names = list(transformation_probabilities.keys())
    transformation_probabilities = list(transformation_probabilities.values())
    paralleling_params = paralleling_params or {}

    for iteration_id in range(n_iterations):
        all_tasks = create_tasks(incumbent_solutions, n_trials_per_iteration, paralleling_params)
        args = [
            (
                tasks_for_process,
                beam_width,
                n_transformations_per_trial,
                transformation_registry,
                transformation_names,
                transformation_probabilities,
                scoring_sets,
                scoring_sets_registry
            )
            for tasks_for_process in all_tasks
        ]
        nested_new_records = starmap_in_parallel(generate_new_records, args, paralleling_params)
        new_records = [record for records in nested_new_records for record in records]
        best_new_records = select_distinct_best_records(new_records, beam_width)
        current_best_score = best_new_records[0].score
        incumbent_solutions = [record.fragment for record in best_new_records]

        if current_best_score > previous_best_score:
            n_transformations_per_trial = default_n_transformations_per_trial
        else:
            n_transformations_per_trial += n_transformations_increment
        if n_transformations_per_trial > max_n_transformations_per_trial:
            n_transformations_per_trial = default_n_transformations_per_trial

        records = select_distinct_best_records(records + new_records, beam_width)
        global_best_score = records[0].score
        previous_best_score = current_best_score
        print(
            f'Iteration #{iteration_id:>3}: '
            f'global_best_score = {global_best_score:.5f}, '
            f'current_best_score = {current_best_score:.5f}'
        )
    result = [record.fragment for record in records]
    return result
