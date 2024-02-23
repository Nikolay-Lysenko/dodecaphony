"""
Run specified tasks.

Author: Nikolay Lysenko
"""


import argparse
import importlib.resources
import os

import yaml

from .evaluation import evaluate, parse_scoring_sets_registry
from .fragment import FragmentParams, initialize_fragment
from .optimization import optimize_with_local_search
from .rendering import render
from .transformations import create_transformations_registry


def parse_cli_args() -> argparse.Namespace:
    """
    Parse arguments passed via Command Line Interface (CLI).

    :return:
        namespace with arguments
    """
    parser = argparse.ArgumentParser(description='Algorithmic composition of dodecaphonic music.')
    parser.add_argument(
        '-c', '--config_path', type=str, default=None, help='path to configuration file'
    )
    parser.add_argument(
        '-n', '--n_fragments', type=int, default=1, help='number of fragments to render'
    )
    cli_args = parser.parse_args()
    return cli_args


def main() -> None:
    """Parse CLI arguments and run requested tasks."""
    cli_args = parse_cli_args()

    default_config_path = importlib.resources.files("dodecaphony") / 'configs/default_config.yml'
    config_path = cli_args.config_path or default_config_path
    with open(config_path) as config_file:
        settings = yaml.load(config_file, Loader=yaml.FullLoader)

    fragment_params = FragmentParams(**settings['fragment'])
    initial_fragment = initialize_fragment(fragment_params)

    scoring_sets_registry = parse_scoring_sets_registry(settings['scoring_sets'])
    scoring_sets = settings['evaluation']['scoring_sets']
    transformations_registry = create_transformations_registry(
        settings['optimization'].pop('max_rotation'),
        settings['optimization'].pop('max_transposition_in_semitones')
    )
    fragments = optimize_with_local_search(
        initial_fragment,
        transformation_registry=transformations_registry,
        scoring_sets=scoring_sets,
        scoring_sets_registry=scoring_sets_registry,
        **settings['optimization']
    )

    reports = []
    for fragment in fragments:
        _, report = evaluate(fragment, scoring_sets, scoring_sets_registry, report=True)
        reports.append(report)
    print("\nEvaluation of selected fragments:\n")
    print(*reports, sep="\n\n")
    print()

    results_dir = settings['rendering']['dir']
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    for fragment, report in zip(fragments[:cli_args.n_fragments], reports):
        rendering_params = settings['rendering']
        rendering_params['meta_information'] = (
            f"Config path: {os.path.abspath(config_path)}\n\n"
            f"Evaluation scores:\n{report}"
        )
        render(fragment, rendering_params)


if __name__ == '__main__':
    main()
