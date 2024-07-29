"""
Transform a fragment in progress.

Note that intermediate functions from this module modify only `temporal_content` and
`tone_row_instances` attributes, but `sonic_content`, `melodic_lines`, and `sonorities` attributes
are left unchanged.
This is done for the sake of performance. It is cheaper to update all dependent attributes
just once after all transformations are applied. So, use `transform` function to get
consistent fragment.

Author: Nikolay Lysenko
"""


import random
from typing import Any, Callable

from .fragment import Fragment, ToneRowInstance, override_calculated_attributes, split_time_span
from .music_theory import invert_tone_row, revert_tone_row, rotate_tone_row, transpose_tone_row


TRANSFORMATION_REGISTRY_TYPE = dict[str, tuple[Callable, list[Any]]]


def draw_tone_row_instance(fragment: Fragment) -> ToneRowInstance:
    """
    Draw one random tone row instance.

    :param fragment:
        a fragment to be modified
    :return:
        random tone row instance
    """
    indices = fragment.mutable_independent_tone_row_instances_indices
    group_index, instance_index = random.choice(indices)
    tone_row_instance = fragment.grouped_tone_row_instances[group_index][instance_index]
    return tone_row_instance


def apply_inversion(fragment: Fragment) -> Fragment:
    """
    Invert one random series (transformed tone row instance).

    :param fragment:
        a fragment to be modified
    :return:
        modified fragment
    """
    tone_row_instance = draw_tone_row_instance(fragment)
    tone_row_instance.pitch_classes = invert_tone_row(tone_row_instance.pitch_classes)
    return fragment


def apply_reversion(fragment: Fragment) -> Fragment:
    """
    Revert one random series (transformed tone row instance).

    :param fragment:
        a fragment to be modified
    :return:
        modified fragment
    """
    tone_row_instance = draw_tone_row_instance(fragment)
    tone_row_instance.pitch_classes = revert_tone_row(tone_row_instance.pitch_classes)
    return fragment


def apply_rotation(fragment: Fragment, max_rotation: int) -> Fragment:
    """
    Rotate one random series (transformed tone row instance).

    :param fragment:
        a fragment to be modified
    :param max_rotation:
        maximum size of rotation (in elements)
    :return:
        modified fragment
    """
    tone_row_instance = draw_tone_row_instance(fragment)
    shift = random.randint(-max_rotation, max_rotation)
    tone_row_instance.pitch_classes = rotate_tone_row(tone_row_instance.pitch_classes, shift)
    return fragment


def apply_transposition(fragment: Fragment, max_transposition: int) -> Fragment:
    """
    Transpose one random series (transformed tone row instance).

    :param fragment:
        a fragment to be modified
    :param max_transposition:
        maximum interval of transposition (in semitones)
    :return:
        modified fragment
    """
    tone_row_instance = draw_tone_row_instance(fragment)
    shift = random.randint(-max_transposition, max_transposition)
    tone_row_instance.pitch_classes = transpose_tone_row(tone_row_instance.pitch_classes, shift)
    return fragment


def apply_measure_durations_change(fragment: Fragment) -> Fragment:
    """
    Change durations of events from a random measure of a single melodic line.

    :param fragment:
        a fragment to be modified
    :return:
        modified fragment
    """
    line_index = random.choice(fragment.mutable_temporal_content_indices)
    measure_index, measure_durations = random.choice(
        list(enumerate(fragment.temporal_content[line_index]))
    )
    candidate_durations = fragment.measure_durations_by_n_events[len(measure_durations)]
    if len(candidate_durations) > 1:
        candidate_durations = [x for x in candidate_durations if x != measure_durations]
    new_measure_durations = random.choice(candidate_durations)
    fragment.temporal_content[line_index][measure_index] = new_measure_durations
    return fragment


def apply_crossmeasure_event_transfer(fragment: Fragment) -> Fragment:
    """
    Move an event from one random measure to another random measure and change their splittings.

    :param fragment:
        a fragment to be modified
    :return:
        modified fragment
    """
    line_index = random.choice(fragment.mutable_temporal_content_indices)
    (first_index, first_durations), (second_index, second_durations) = random.sample(
        list(enumerate(fragment.temporal_content[line_index])), 2
    )
    if len(first_durations) > 1:
        first_key = len(first_durations) - 1
        second_key = len(second_durations) + 1
    else:
        first_key = len(first_durations) + 1
        second_key = len(first_durations) - 1
    if first_key not in fragment.measure_durations_by_n_events:
        return fragment  # pragma: no cover
    if second_key not in fragment.measure_durations_by_n_events:
        return fragment
    new_first_durations = random.choice(fragment.measure_durations_by_n_events[first_key])
    fragment.temporal_content[line_index][first_index] = new_first_durations
    new_second_durations = random.choice(fragment.measure_durations_by_n_events[second_key])
    fragment.temporal_content[line_index][second_index] = new_second_durations
    return fragment


def apply_line_durations_change(fragment: Fragment) -> Fragment:
    """
    Change durations of all events from a random melodic line.

    :param fragment:
        a fragment to be modified
    :return:
        modified fragment
    """
    line_index = random.choice(fragment.mutable_temporal_content_indices)
    line_durations = fragment.temporal_content[line_index]
    n_measures = len(line_durations)
    n_events = len([x for measure_durations in line_durations for x in measure_durations])
    new_line_durations = split_time_span(
        n_measures, n_events, fragment.measure_durations_by_n_events
    )
    fragment.temporal_content[line_index] = new_line_durations
    return fragment


def apply_pause_shift(fragment: Fragment) -> Fragment:
    """
    Shift a random pause one position to the left or to the right.

    :param fragment:
        a fragment to be modified
    :return:
        modified fragment
    """
    options = []
    for group_index, mutable_pauses_indices in enumerate(fragment.grouped_mutable_pauses_indices):
        max_index = len(fragment.sonic_content[group_index]) - 1
        immutable_pauses_indices = fragment.grouped_immutable_pauses_indices[group_index]
        pauses_indices = mutable_pauses_indices + immutable_pauses_indices
        for pause_index in mutable_pauses_indices:
            if pause_index > 0 and pause_index - 1 not in pauses_indices:
                options.append((group_index, pause_index, False))
            if pause_index < max_index and pause_index + 1 not in pauses_indices:
                options.append((group_index, pause_index, True))
    if not options:
        return fragment
    group_index, pause_index, to_the_right = random.choice(options)
    mutable_pauses_indices = fragment.grouped_mutable_pauses_indices[group_index]
    mutable_pauses_indices = [x for x in mutable_pauses_indices if x != pause_index]
    mutable_pauses_indices.append(pause_index + 1 if to_the_right else pause_index - 1)
    fragment.grouped_mutable_pauses_indices[group_index] = mutable_pauses_indices
    return fragment


def create_transformations_registry(
        max_rotation: int, max_transposition: int
) -> TRANSFORMATION_REGISTRY_TYPE:
    """
    Get mapping from names to corresponding transformations and their arguments.

    :param max_rotation:
        maximum size of rotation (in elements)
    :param max_transposition:
        maximum interval of transposition (in semitones)
    :return:
        registry of transformations
    """
    registry = {
        # Tone row transformations.
        'inversion': (apply_inversion, []),
        'reversion': (apply_reversion, []),
        'rotation': (apply_rotation, [max_rotation]),
        'transposition': (apply_transposition, [max_transposition]),
        # Rhythm transformations.
        'crossmeasure_event_transfer': (apply_crossmeasure_event_transfer, []),
        'line_durations_change': (apply_line_durations_change, []),
        'measure_durations_change': (apply_measure_durations_change, []),
        'pause_shift': (apply_pause_shift, []),
    }
    return registry


def transform(
        fragment: Fragment,
        n_transformations: int,
        transformation_registry: TRANSFORMATION_REGISTRY_TYPE,
        transformation_names: list[str],
        transformation_probabilities: list[float]
) -> Fragment:
    """
    Apply multiple random transformations to a fragment.

    :param fragment:
        a fragment to be modified
    :param n_transformations:
        number of transformations to be applied
    :param transformation_registry:
        mapping from names to corresponding transformations and their arguments
    :param transformation_names:
        names of transformations to choose from
    :param transformation_probabilities:
        probabilities of corresponding transformations; this argument must have the same length
        as `transformation_names`
    :return:
        modified fragment
    """
    names_of_transformations_to_be_applied = random.choices(
        transformation_names,
        transformation_probabilities,
        k=n_transformations
    )
    for transformation_name in names_of_transformations_to_be_applied:
        transformation_fn, args = transformation_registry[transformation_name]
        fragment = transformation_fn(fragment, *args)
    override_calculated_attributes(fragment)
    return fragment
