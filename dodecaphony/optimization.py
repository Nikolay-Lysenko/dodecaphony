"""
Iteratively improve fragment.

Author: Nikolay Lysenko
"""


from copy import deepcopy
from typing import Union

from .evaluation import SCORING_SETS_REGISTRY_TYPE, evaluate
from .fragment import Fragment
from .transformations import TRANSFORMATIONS_REGISTRY_TYPE, transform


def generate_new_records(
        incumbent_solutions: list[Fragment],
        n_trials_per_iteration: int,
        n_transformations_per_trial: int,
        transformation_registry: TRANSFORMATIONS_REGISTRY_TYPE,
        transformation_names: list[str],
        transformation_probabilities: list[float],
        scoring_sets: list[str],
        scoring_sets_registry: SCORING_SETS_REGISTRY_TYPE
) -> list[dict[str, Union[Fragment, float]]]:
    """
    Generate records for current iteration.

    :param incumbent_solutions:
        fragments such that their neighborhoods should be searched
    :param n_trials_per_iteration:
        number of transformed fragments to generate and evaluate per each incumbent solution
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
        new records
    """
    new_records = []
    for incumbent_solution in incumbent_solutions:
        for trial_id in range(n_trials_per_iteration):
            candidate = deepcopy(incumbent_solution)
            candidate = transform(
                candidate,
                n_transformations_per_trial,
                transformation_registry,
                transformation_names,
                transformation_probabilities
            )
            score = evaluate(candidate, scoring_sets, scoring_sets_registry)
            new_records.append({'fragment': candidate, 'score': score})
    return new_records


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
        scoring_sets_registry: SCORING_SETS_REGISTRY_TYPE
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
    :return:
        optimized fragments
    """
    incumbent_solutions = [fragment]
    records = []
    best_score = -1e9
    n_transformations_per_trial = default_n_transformations_per_trial
    transformation_names = list(transformation_probabilities.keys())
    transformation_probabilities = list(transformation_probabilities.values())

    for iteration_id in range(n_iterations):
        new_records = generate_new_records(
            incumbent_solutions,
            n_trials_per_iteration,
            n_transformations_per_trial,
            transformation_registry,
            transformation_names,
            transformation_probabilities,
            scoring_sets,
            scoring_sets_registry
        )
        best_new_records = sorted(new_records, key=lambda x: -x['score'])[:beam_width]
        current_best_score = best_new_records[0]['score']
        incumbent_solutions = [record['fragment'] for record in best_new_records]

        if current_best_score <= best_score:
            n_transformations_per_trial += n_transformations_increment
            n_transformations_per_trial = min(n_transformations_per_trial, max_n_transformations_per_trial)
        else:
            n_transformations_per_trial = default_n_transformations_per_trial

        records = sorted(records + new_records, key=lambda x: -x['score'])[:beam_width]
        best_score = records[0]['score']
        print(
            f'Iteration #{iteration_id:>3}: '
            f'best_score = {best_score:.5f}, '
            f'current_best_score = {current_best_score:.5f}'
        )
    result = [record['fragment'] for record in records]
    return result
