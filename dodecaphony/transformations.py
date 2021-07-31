"""
Transform a fragment in progress.

Note that intermediate functions from this module modify only `temporal_content` and
`sonic_content` attributes, but `melodic_lines` and `sonorities` attributes are left unchanged.
This is done for the sake of performance. It is cheaper to update all dependent attributes
just once after all transformations are applied. So use `transform` function to get
consistent fragment.

Author: Nikolay Lysenko
"""


import itertools
import random
from typing import Any, Callable

from .fragment import Fragment, SUPPORTED_DURATIONS, override_calculated_attributes
from .music_theory import (
    TONE_ROW_LEN, invert_tone_row, revert_tone_row, rotate_tone_row, transpose_tone_row
)


TRANSFORMATIONS_REGISTRY_TYPE = dict[str, tuple[Callable, list[Any]]]


def get_duration_changes() -> dict[tuple[float, float], list[tuple[float, float]]]:
    """
    Get mapping from durations of two events to list of pairs of durations of the same total sum.

    :return:
        mapping from durations of two events to list of pairs of durations of the same total sum
    """
    result = {}
    cartesian_product = itertools.product(SUPPORTED_DURATIONS, SUPPORTED_DURATIONS)
    for first_duration, second_duration in cartesian_product:
        if first_duration > second_duration:
            continue
        total_sum = first_duration + second_duration
        durations = []
        for duration in SUPPORTED_DURATIONS:
            complementary_duration = total_sum - duration
            if complementary_duration in SUPPORTED_DURATIONS:
                durations.append((duration, complementary_duration))
        result[(first_duration, second_duration)] = durations
    return result


def draw_random_indices(
        mutable_sonic_content_indices: list[int], n_tone_row_instances_by_group: list[int]
) -> tuple[int, int]:
    """
    Draw index of melodic lines group and index of tone row instance from it.

    :param mutable_sonic_content_indices:
        indices of groups such that their sonic content can be transformed
    :param n_tone_row_instances_by_group:
        list where number of tone row instances in the group is stored for each group of
        melodic lines sharing the same series (in terms of vertical distribution of series pitches)
    :return:
        index of group and index of tone row instance from it
    """
    group_index = random.choice(mutable_sonic_content_indices)
    n_instances = n_tone_row_instances_by_group[group_index]
    instance_index = random.randrange(0, n_instances)
    return group_index, instance_index


def find_instance_by_indices(
        fragment: Fragment, group_index: int, instance_index: int
) -> list[str]:
    """
    Find sequence of 12 pitch classes by its indices.

    :param fragment:
        a fragment
    :param group_index:
        index of group of melodic lines sharing the same series (in terms of vertical distribution
        of series pitches)
    :param instance_index:
        index of tone row instance within sonic content of the group
    :return:
        sequence of 12 pitch classes
    """
    line = fragment.sonic_content[group_index]
    start_event_index = instance_index * TONE_ROW_LEN
    end_event_index = (instance_index + 1) * TONE_ROW_LEN
    pitch_classes = []
    index = 0
    for pitch_class in line:
        if pitch_class == 'pause':
            continue
        if index >= end_event_index:
            break
        if index >= start_event_index:
            pitch_classes.append(pitch_class)
        index += 1
    return pitch_classes


def replace_instance(
        fragment: Fragment, group_index: int, instance_index: int, new_instance: list[str]
) -> Fragment:
    """
    Replace a particular sequence of 12 tones with another sequence of 12 tones.

    :param fragment:
        a fragment
    :param group_index:
        index of group of melodic lines sharing the same series (in terms of vertical distribution
        of series pitches)
    :param instance_index:
        index of tone row instance within sonic content of the group
    :param new_instance:
        new sequence of 12 pitch classes
    :return:
        modified fragment
    """
    line = fragment.sonic_content[group_index]
    start_event_index = instance_index * TONE_ROW_LEN
    end_event_index = (instance_index + 1) * TONE_ROW_LEN
    index_without_pauses = 0
    for index, pitch_class in enumerate(line):
        if pitch_class == 'pause':
            continue
        if index_without_pauses >= end_event_index:
            break
        if index_without_pauses >= start_event_index:
            line[index] = new_instance.pop(0)
        index_without_pauses += 1
    return fragment


def apply_duration_change(
        fragment: Fragment,
        duration_changes: dict[tuple[float, float], list[tuple[float, float]]]
) -> Fragment:
    """
    Change durations of two random events from the same melodic line.

    :param fragment:
        a fragment to be modified
    :param duration_changes:
        mapping from durations of two events to list of pairs of durations of the same total sum
    :return:
        modified fragment
    """
    line_index = random.choice(fragment.mutable_temporal_content_indices)
    line_durations = fragment.temporal_content[line_index]
    events_indices = random.sample(range(len(line_durations)), 2)
    key = tuple(sorted(line_durations[event_index] for event_index in events_indices))
    all_durations = duration_changes[key]
    durations = random.choice(all_durations)
    for event_index, duration in zip(events_indices, durations):
        line_durations[event_index] = duration
    return fragment


def apply_pause_shift(fragment: Fragment) -> Fragment:
    """
    Shift a random pause one position to the left or to the right.

    :param fragment:
        a fragment to be modified
    :return:
        modified fragment
    """
    line = random.choice(fragment.sonic_content)
    indices = []
    for index, (previous_pitch_class, pitch_class) in enumerate(zip(line, line[1:])):
        if pitch_class == 'pause' and previous_pitch_class != 'pause':
            indices.append(index)
        if pitch_class != 'pause' and previous_pitch_class == 'pause':
            indices.append(index)
    if not indices:
        return fragment
    index = random.choice(indices)
    line[index], line[index + 1] = line[index + 1], line[index]
    return fragment


def apply_inversion(fragment: Fragment) -> Fragment:
    """
    Invert one random series (transformed tone row instance).

    :param fragment:
        a fragment to be modified
    :return:
        modified fragment
    """
    group_index, instance_index = draw_random_indices(
        fragment.mutable_sonic_content_indices, fragment.n_tone_row_instances_by_group
    )
    tone_row_instance = find_instance_by_indices(fragment, group_index, instance_index)
    tone_row_instance = invert_tone_row(tone_row_instance)
    fragment = replace_instance(fragment, group_index, instance_index, tone_row_instance)
    return fragment


def apply_reversion(fragment: Fragment) -> Fragment:
    """
    Revert one random series (transformed tone row instance).

    :param fragment:
        a fragment to be modified
    :return:
        modified fragment
    """
    group_index, instance_index = draw_random_indices(
        fragment.mutable_sonic_content_indices, fragment.n_tone_row_instances_by_group
    )
    tone_row_instance = find_instance_by_indices(fragment, group_index, instance_index)
    tone_row_instance = revert_tone_row(tone_row_instance)
    fragment = replace_instance(fragment, group_index, instance_index, tone_row_instance)
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
    group_index, instance_index = draw_random_indices(
        fragment.mutable_sonic_content_indices, fragment.n_tone_row_instances_by_group
    )
    tone_row_instance = find_instance_by_indices(fragment, group_index, instance_index)
    shift = random.randint(-max_rotation, max_rotation)
    tone_row_instance = rotate_tone_row(tone_row_instance, shift)
    fragment = replace_instance(fragment, group_index, instance_index, tone_row_instance)
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
    group_index, instance_index = draw_random_indices(
        fragment.mutable_sonic_content_indices, fragment.n_tone_row_instances_by_group
    )
    tone_row_instance = find_instance_by_indices(fragment, group_index, instance_index)
    shift = random.randint(-max_transposition, max_transposition)
    tone_row_instance = transpose_tone_row(tone_row_instance, shift)
    fragment = replace_instance(fragment, group_index, instance_index, tone_row_instance)
    return fragment


def create_transformations_registry(
        max_rotation: int, max_transposition: int
) -> TRANSFORMATIONS_REGISTRY_TYPE:
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
        'duration_change': (apply_duration_change, [get_duration_changes()]),
        'pause_shift': (apply_pause_shift, []),
        'inversion': (apply_inversion, []),
        'reversion': (apply_reversion, []),
        'rotation': (apply_rotation, [max_rotation]),
        'transposition': (apply_transposition, [max_transposition]),
    }
    return registry


def transform(
        fragment: Fragment,
        n_transformations: int,
        transformation_registry: TRANSFORMATIONS_REGISTRY_TYPE,
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
    fragment = override_calculated_attributes(fragment)
    return fragment
