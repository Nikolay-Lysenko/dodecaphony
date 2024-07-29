"""
Define data structures representing a fragment of a musical piece.

Author: Nikolay Lysenko
"""


import itertools
import math
import random
from dataclasses import dataclass
from typing import Any, Optional

from sinethesizer.utils.music_theory import get_note_to_position_mapping

from .music_theory import (
    N_SEMITONES_PER_OCTAVE,
    get_smallest_intervals_between_pitch_classes,
    get_tone_row_transformations_registry,
    invert_tone_row,
    revert_tone_row
)


NOTE_TO_POSITION_MAPPING = get_note_to_position_mapping()
SMALLEST_INTERVALS_MAPPING = get_smallest_intervals_between_pitch_classes()


@dataclass
class Event:
    """An indivisible element of a musical piece."""
    line_index: int
    start_time: float
    duration: float
    pitch_class: Optional[str] = None
    position_in_semitones: Optional[int] = None


@dataclass
class Sonority:
    """Simultaneously sounding events."""
    events: list[Event]
    non_pause_events: list[Event]
    start_time: float
    end_time: float


@dataclass
class ToneRowInstance:
    """An instance of a tone row (i.e., single repetition of a series)."""
    pitch_classes: list[str]
    independent_instance_indices: Optional[tuple[int, int]] = None
    dependence_name: Optional[str] = None
    dependence_params: Optional[dict[str, Any]] = None


@dataclass
class FragmentParams:
    """Initial parameters of a fragment of a musical piece."""
    tone_row: list[str]
    groups: list[dict[str, Any]]
    n_measures: int
    meter_numerator: int
    meter_denominator: int
    measure_durations: list[list[float]]
    line_ids: list[int]
    upper_line_highest_note: str
    upper_line_lowest_note: str
    temporal_content: Optional[dict[int, dict[str, Any]]] = None


@dataclass
class Fragment:
    """A fragment of a musical piece."""
    # Core data structures.
    temporal_content: list[list[list[float]]]
    grouped_tone_row_instances: list[list[ToneRowInstance]]
    grouped_mutable_pauses_indices: list[list[int]]
    grouped_immutable_pauses_indices: list[list[int]]
    # Constants.
    n_beats: int
    meter_numerator: int
    meter_denominator: int
    measure_durations_by_n_events: dict[int, list[list[float]]]
    line_ids: list[int]
    upper_line_highest_position: int
    upper_line_lowest_position: int
    tone_row_len: int
    group_index_to_line_indices: dict[int, list[int]]
    mutable_temporal_content_indices: list[int]
    mutable_independent_tone_row_instances_indices: list[tuple[int, int]]
    mutable_dependent_tone_row_instances_indices: list[tuple[int, int]]
    # Calculated attributes.
    sonic_content: Optional[list[list[str]]] = None
    melodic_lines: Optional[list[list[Event]]] = None
    sonorities: Optional[list[Sonority]] = None

    def __eq__(self, other: Any):
        if not isinstance(other, Fragment):
            return NotImplemented
        return (
            self.temporal_content == other.temporal_content
            and self.sonic_content == other.sonic_content
        )


def validate_line_indices(params: FragmentParams) -> None:
    """
    Validate distribution of line indices among groups.

    :param params:
        parameters of a fragment to be created
    :return:
        None
    """
    line_indices = [index for group in params.groups for index in group['melodic_line_indices']]
    if sorted(line_indices) != sorted(list(set(line_indices))):
        raise ValueError("Line index can not be included in multiple groups.")
    if min(line_indices) < 0:
        raise ValueError("All line indices must be positive.")
    if max(line_indices) >= len(line_indices):
        raise ValueError("Each line index must be included in a group.")
    if len(line_indices) != len(params.line_ids):
        raise ValueError("Number of lines in `groups` is not equal to that in `line_ids`.")


def validate_pauses(params: FragmentParams) -> None:
    """
    Validate pauses.

    :param params:
        parameters of a fragment to be created
    :return:
        None
    """
    for group in params.groups:
        n_pauses = group.get('n_pauses', 0)
        n_immutable_pauses_indices = len(group.get('immutable_pauses_indices', []))
        if n_pauses < n_immutable_pauses_indices:
            raise ValueError("Number of immutable pauses exceeds total number of pauses.")


def validate_tone_row_instances(params: FragmentParams) -> None:
    """
    Validate parameters of tone row instances.

    :param params:
        parameters of a fragment to be created
    :return:
        None
    """
    for group in params.groups:
        for tone_row_instance in group['tone_row_instances']:
            pitch_classes = tone_row_instance.get('pitch_classes')
            if pitch_classes is not None and len(pitch_classes) != len(params.tone_row):
                raise ValueError("Tone row instance has wrong length.")
            if pitch_classes is not None and tone_row_instance.get('dependence') is not None:
                raise ValueError("Predefined tone row instance can not be dependent.")
            if pitch_classes is None and tone_row_instance.get('immutable', False):
                raise ValueError("Tone row instance can not be immutable if it is not predefined.")


def validate_measure_durations(params: FragmentParams) -> None:
    """
    Validate list of measure durations supported in the current fragment.

    :param params:
        parameters of a fragment to be created
    :return:
        None
    """
    for durations in params.measure_durations:
        if sum(durations) != params.meter_numerator:
            raise ValueError(
                "Measure durations must sum up to meter numerator. As of now, suspensions "
                f"are not supported. Invalid sequence: {durations}."
            )


def validate_initialized_temporal_content(params: FragmentParams) -> None:
    """
    Validate parameters related to initialized parts of temporal content.

    :param params:
        parameters of a fragment to be created
    :return:
        None
    """
    if params.temporal_content is None:
        return  # pragma: no cover
    total_duration_in_beats = params.meter_numerator * params.n_measures
    measure_ends_in_beats = [params.meter_numerator * i for i in range(1, params.n_measures + 1)]
    for line_index, line_params in params.temporal_content.items():
        duration_in_beats = sum(line_params['durations'])
        if duration_in_beats != total_duration_in_beats:
            raise ValueError(
                f"A line with index {line_index} has duration {duration_in_beats} beats, "
                f"whereas duration of the fragment is set to {total_duration_in_beats} beats."
            )
        if not line_params.get('immutable', False):
            events_ends = itertools.accumulate(line_params['durations'])
            uncovered_measure_ends = set(measure_ends_in_beats).difference(set(events_ends))
            if uncovered_measure_ends:
                raise ValueError(
                    "Suspensions over bar are not allowed in lines with mutable temporal content. "
                    f"Violations: line_index={line_index}, crossed_bars={uncovered_measure_ends}."
                )


def validate(params: FragmentParams) -> None:
    """
    Validate all parameters.

    :param params:
        parameters of a fragment to be created
    :return:
        None
    """
    if sorted(params.line_ids) != list(set(params.line_ids)):
        raise ValueError("IDs of melodic lines must be unique.")
    validate_line_indices(params)
    validate_pauses(params)
    validate_tone_row_instances(params)
    validate_measure_durations(params)
    validate_initialized_temporal_content(params)


def group_measure_durations_by_n_events(
        measure_durations: list[list[float]]
) -> dict[int, list[list[float]]]:
    """
    Map number of events to ways of splitting a measure into this number of events.

    :param measure_durations:
        list of all measure splits that can be used
    :return:
        mapping from number of events to ways of splitting a measure into this number of events
    """
    result = {}
    for durations in measure_durations:
        key = len(durations)
        new_value = result.get(key, []) + [durations]
        result[key] = new_value
    return result


def distribute_events_among_measures(
        n_measures: int, n_events: int, measure_durations_by_n_events: dict[int, list[list[float]]]
) -> list[int]:
    """
    Distribute events among measures.

    :param n_measures:
        number of measures
    :param n_events:
        number of events
    :param measure_durations_by_n_events:
        mapping from number of events to ways of splitting a measure into this number of events
    :return:
        list of numbers of events from a measure for all measures
    """
    while True:
        measures_cum_sum = sorted(random.sample(range(1, n_events), n_measures - 1))
        measures_cum_sum = [0] + measures_cum_sum + [n_events]
        result = []
        for former, latter in zip(measures_cum_sum, measures_cum_sum[1:]):
            result.append(latter - former)
        if all(x in measure_durations_by_n_events for x in result):  # pragma: no branch
            return result


def split_time_span(
        n_measures: int, n_events: int, measure_durations_by_n_events: dict[int, list[list[float]]]
) -> list[list[float]]:
    """
    Split time span of one melodic line into time spans of the specified number of events.

    :param n_measures:
        number of measures
    :param n_events:
        number of events
    :param measure_durations_by_n_events:
        mapping from number of events to ways of splitting a measure into this number of events
    :return:
        durations of events (in reference beats) grouped by measure
    """
    if n_events < n_measures:
        raise ValueError("Average duration of an event is longer than semibreve.")
    if n_events > n_measures * max(measure_durations_by_n_events.keys()):
        raise ValueError("The number of events is too high.")
    durations = []
    n_events_by_measure = distribute_events_among_measures(
        n_measures, n_events, measure_durations_by_n_events
    )
    for n_measure_events in n_events_by_measure:
        durations.append(random.choice(measure_durations_by_n_events[n_measure_events]))
    return durations


def group_durations_by_measures(durations: list[float], meter_numerator: int) -> list[list[float]]:
    """
    Group durations of events from a single melodic lines by measures.

    Output is a list where each measure is represented as list of durations of events forming it;
    suspended from the previous measure event is partially included
    and suspended to the next measure event is fully included.
    As of 2023-03-26, such suspensions may occur only in lines with immutable predefined
    temporal content.

    :param durations:
        durations of events (in reference beats)
    :param meter_numerator:
        number of reference beats per measure
    :return:
        durations of events (in reference beats) grouped by measure
    """
    grouped_durations = []
    current_measure_durations = []
    current_measure_total_time = 0
    for duration in durations:
        current_measure_durations.append(duration)
        current_measure_total_time += duration
        if current_measure_total_time > meter_numerator:
            grouped_durations.append(current_measure_durations)
            suspended_duration = current_measure_total_time - meter_numerator
            current_measure_durations = [suspended_duration]
            current_measure_total_time = suspended_duration
        elif current_measure_total_time == meter_numerator:
            grouped_durations.append(current_measure_durations)
            current_measure_durations = []
            current_measure_total_time = 0
    return grouped_durations


def distribute_events_among_lines(n_events: int, n_lines: int) -> list[int]:
    """
    Distribute evenly the specified number of events among the specified number of lines.

    :param n_events:
        number of events
    :param n_lines:
        number of lines
    :return:
        list of numbers of events in a particular line
    """
    results = [n_events // n_lines for _ in range(n_lines)]
    for i in range(n_events - sum(results)):
        results[i] += 1
    return results


def create_initial_temporal_content(
        params: FragmentParams, measure_durations_by_n_events: dict[int, list[list[float]]]
) -> list[list[list[float]]]:
    """
    Split time span of each melodic line into durations of individual events grouped by measures.

    :param params:
        parameters of a fragment to be created
    :param measure_durations_by_n_events:
        mapping from number of events to ways of splitting a measure into this number of events
    :return:
        lists of event durations (in reference beats) for each melodic line
    """
    flat_durations = []
    temporal_content = []
    for line_index, line_id in enumerate(params.line_ids):
        durations = params.temporal_content.get(line_index, {}).get('durations', [])
        flat_durations.append(durations)
        temporal_content.append(group_durations_by_measures(durations, params.meter_numerator))
    for group_index, group_params in enumerate(params.groups):
        n_pauses = group_params.get('n_pauses', 0)
        n_events = len(group_params['tone_row_instances']) * len(params.tone_row) + n_pauses
        line_indices = group_params['melodic_line_indices']
        n_defined_events = sum(len(flat_durations[line_index]) for line_index in line_indices)
        n_undefined_events = n_events - n_defined_events
        indices_of_undefined_lines = [i for i in line_indices if not temporal_content[i]]
        n_undefined_lines = len(indices_of_undefined_lines)
        n_events_per_line = distribute_events_among_lines(n_undefined_events, n_undefined_lines)
        for line_index, n_events in zip(indices_of_undefined_lines, n_events_per_line):
            durations = split_time_span(params.n_measures, n_events, measure_durations_by_n_events)
            temporal_content[line_index] = durations
    return temporal_content


def maybe_transform_pitch_classes(tone_row: list[str]) -> list[str]:
    """
    Get pitch classes of a tone row in its prime form or one of its altered forms.

    :param tone_row:
        tone row as list of pitch classes (like C or C#, flats are not allowed)
    :return:
        list of pitch classes from a form of the tone row
    """
    current_instance = [pitch_class for pitch_class in tone_row]
    if random.choice([True, False]):
        current_instance = invert_tone_row(current_instance)
    if random.choice([True, False]):
        current_instance = revert_tone_row(current_instance)
    return current_instance


def update_dependent_tone_row_instance(
        tone_row_instance: ToneRowInstance,
        pitch_classes: list[str]
) -> None:
    """
    Update dependent tone row instance given pitch classes of the instance on which it depends.

    :param tone_row_instance:
        tone row instance to be updated
    :param pitch_classes:
        pitch classes of the independent instance
    :return:
        function that defines dependence of pitch classes
    """
    transformations_registry = get_tone_row_transformations_registry()
    transformation_fn = transformations_registry[tone_row_instance.dependence_name]
    updated_pitch_classes = transformation_fn(pitch_classes, **tone_row_instance.dependence_params)
    tone_row_instance.pitch_classes = updated_pitch_classes


def create_initial_grouped_tone_row_instances(
        params: FragmentParams
) -> tuple[list[list[ToneRowInstance]], list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Create initial data structure that keeps track of tone row instances.

    :param params:
        parameters of a fragment to be created
    :return:
        a tuple of:
        1) grouped tone row instances with set pitch classes,
        2) list with a pair of group index and instance index within this group
           for each tone row instance that is mutable and independent,
        3) list with a pair of group index and instance index within this group
           for each tone row instance that depends on a mutable tone row instance
    """
    grouped_tone_row_instances = []
    mutable_independent_tone_row_instances_indices = []
    dependent_tone_row_instances_indices = []
    for group_index, group_params in enumerate(params.groups):
        tone_row_instances = []
        for instance_index, instance_params in enumerate(group_params['tone_row_instances']):
            indices = (group_index, instance_index)
            if instance_params.get('dependence') is None:
                drawn_pitch_classes = maybe_transform_pitch_classes(params.tone_row)
                pitch_classes = instance_params.get('pitch_classes', drawn_pitch_classes)
                tone_row_instance = ToneRowInstance(pitch_classes)
                immutable = instance_params.get('immutable', False)
                if not immutable:
                    mutable_independent_tone_row_instances_indices.append(indices)
            else:
                dependence_params = instance_params['dependence']
                independent_instance_indices = (
                    dependence_params['group_index'],
                    dependence_params['tone_row_instance_index']
                )
                tone_row_instance = ToneRowInstance(
                    [],
                    independent_instance_indices,
                    dependence_params['transformation'],
                    dependence_params.get('transformation_params', {})
                )
                dependent_tone_row_instances_indices.append(indices)
            tone_row_instances.append(tone_row_instance)
        grouped_tone_row_instances.append(tone_row_instances)

    mutable_dependent_tone_row_instances_indices = []
    for group_index, instance_index in dependent_tone_row_instances_indices:
        tone_row_instance = grouped_tone_row_instances[group_index][instance_index]
        independent_instance_indices = tone_row_instance.independent_instance_indices
        if independent_instance_indices in mutable_independent_tone_row_instances_indices:
            mutable_dependent_tone_row_instances_indices.append((group_index, instance_index))
        independent_group_index = independent_instance_indices[0]
        independent_instance_index = independent_instance_indices[1]
        pitch_classes = (
            grouped_tone_row_instances[independent_group_index][independent_instance_index]
            .pitch_classes
        )
        update_dependent_tone_row_instance(tone_row_instance, pitch_classes)

    return (
        grouped_tone_row_instances,
        mutable_independent_tone_row_instances_indices,
        mutable_dependent_tone_row_instances_indices
    )


def find_initial_pauses_indices(params: FragmentParams) -> tuple[list[list[int]], list[list[int]]]:
    """
    Find indices of pauses that might be shifted and indices of pauses that can not be moved.

    :param params:
        parameters of a fragment to be created
    :return:
        indices of pauses that might be shifted and indices of pauses that can not be moved
    """
    grouped_mutable_pauses_indices = []
    grouped_immutable_pauses_indices = []
    for group_params in params.groups:
        n_pauses = group_params.get('n_pauses', 0)
        n_events = len(group_params['tone_row_instances']) * len(params.tone_row) + n_pauses
        immutable_pauses_indices = group_params.get('immutable_pauses_indices', [])
        n_mutable_pauses_indices = n_pauses - len(immutable_pauses_indices)
        if n_mutable_pauses_indices > 0:
            free_indices = [i for i in range(n_events) if i not in immutable_pauses_indices]
            mutable_pauses_indices = random.sample(free_indices, n_mutable_pauses_indices)
        else:
            mutable_pauses_indices = []
        grouped_mutable_pauses_indices.append(mutable_pauses_indices)
        grouped_immutable_pauses_indices.append(immutable_pauses_indices)
    return grouped_mutable_pauses_indices, grouped_immutable_pauses_indices


def find_mutable_temporal_content_indices(params: FragmentParams) -> list[int]:
    """
    Find indices of melodic lines such that their temporal content can be modified.

    :param params:
        parameters of a fragment to be created
    :return:
        indices of melodic lines such that their temporal content can be modified
    """
    results = []
    for line_index, line_id in enumerate(params.line_ids):
        if not params.temporal_content.get(line_index, {}).get('immutable', False):
            results.append(line_index)
    return results


def update_dependent_tone_row_instances(fragment: Fragment) -> None:
    """
    Update pitch classes of dependent tone row instances.

    :param fragment:
        a fragment to be updated
    :return:
        None
    """
    grouped_tone_row_instances = fragment.grouped_tone_row_instances
    for group_index, instance_index in fragment.mutable_dependent_tone_row_instances_indices:
        tone_row_instance = fragment.grouped_tone_row_instances[group_index][instance_index]
        independent_group_index = tone_row_instance.independent_instance_indices[0]
        independent_instance_index = tone_row_instance.independent_instance_indices[1]
        pitch_classes = (
            grouped_tone_row_instances[independent_group_index][independent_instance_index]
            .pitch_classes
        )
        update_dependent_tone_row_instance(tone_row_instance, pitch_classes)


def set_sonic_content(fragment: Fragment) -> None:
    """
    Fill data structure that keeps track of pitch classes.

    This data structure is a list where for each group of melodic lines sharing the same series,
    sonic content of the series is stored (i.e., there is a sequence consisting of pitch classes
    and pauses)

    :param fragment:
        fragment with all non-optional attributes defined
    :return:
        None
    """
    sonic_content = []
    zipped = zip(
        fragment.grouped_tone_row_instances,
        fragment.grouped_mutable_pauses_indices,
        fragment.grouped_immutable_pauses_indices
    )
    for tone_row_instances, mutable_pauses_indices, immutable_pauses_indices in zipped:
        group_sonic_content = []
        for tone_row_instance in tone_row_instances:
            group_sonic_content.extend(tone_row_instance.pitch_classes)
        # Below, sorting is needed to place pauses exactly at the required places.
        pauses_indices = sorted(mutable_pauses_indices + immutable_pauses_indices)
        for pause_index in pauses_indices:
            group_sonic_content = (
                group_sonic_content[:pause_index]
                + ['pause']
                + group_sonic_content[pause_index:]
            )
        sonic_content.append(group_sonic_content)
    fragment.sonic_content = sonic_content


def create_grouped_rhythm_only_lines(fragment: Fragment) -> list[list[list[Event]]]:
    """
    Create rhythm-only lines grouped by sharing of the same series.

    :param fragment:
        fragment with non-empty temporal content
    :return:
        list where for each group of melodic lines sharing the same series for each melodic line
        there are events representing temporal (loosely speaking, rhythmic) structure of the line,
        but keeping no information about pitches
    """
    rhythm_only_lines = []
    for _, line_indices in fragment.group_index_to_line_indices.items():
        group_of_lines = []
        for line_index in line_indices:
            line = []
            start_time = 0
            first_note_is_suspended = False
            line_durations = fragment.temporal_content[line_index]
            for measure_durations in line_durations:
                for duration in measure_durations[int(first_note_is_suspended):]:
                    line.append(Event(line_index, start_time, duration))
                    start_time += duration
                first_note_is_suspended = sum(measure_durations) > fragment.meter_numerator
            group_of_lines.append(line)
        rhythm_only_lines.append(group_of_lines)
    return rhythm_only_lines


def set_melodic_lines_and_their_pitch_classes(fragment: Fragment) -> None:
    """
    Create melodic lines, distribute pitch classes among them, and store them to attribute.

    :param fragment:
        fragment with non-empty temporal content and sonic content
    :return:
        None
    """
    melodic_lines = [[] for _ in fragment.line_ids]
    grouped_rhythm_only_lines = create_grouped_rhythm_only_lines(fragment)
    zipped = zip(grouped_rhythm_only_lines, fragment.sonic_content)
    for group_of_rhythm_only_lines, group_sonic_content in zipped:
        timeline = [
            event
            for rhythm_only_line in group_of_rhythm_only_lines
            for event in rhythm_only_line
        ]
        timeline = sorted(timeline, key=lambda event: (event.start_time, event.line_index))
        for event, pitch_class in zip(timeline, group_sonic_content):
            new_event = Event(event.line_index, event.start_time, event.duration, pitch_class)
            melodic_lines[event.line_index].append(new_event)
    fragment.melodic_lines = melodic_lines


def set_sonorities(fragment: Fragment) -> None:
    """
    Find simultaneously sounding events and store them to the corresponding attribute.

    :param fragment:
        fragment with `melodic_lines` attribute
    :return:
        None
    """
    sonorities = []
    melodic_lines = fragment.melodic_lines
    timeline = [event for melodic_line in melodic_lines for event in melodic_line]
    timeline = sorted(timeline, key=lambda event: (event.start_time, event.line_index))
    indices = [-1 for _ in melodic_lines]
    current_times = [0 for _ in melodic_lines]
    previous_passed_time = 0
    for event in timeline:
        indices[event.line_index] += 1
        current_times[event.line_index] += event.duration
        passed_time = min(current_times)
        if passed_time > previous_passed_time:
            events = [melodic_line[index] for melodic_line, index in zip(melodic_lines, indices)]
            non_pause_events = [event for event in events if event.pitch_class != "pause"]
            sonority_start = max(event.start_time for event in events)
            sonority_end = min(event.start_time + event.duration for event in events)
            sonority = Sonority(events, non_pause_events, sonority_start, sonority_end)
            sonorities.append(sonority)
        previous_passed_time = passed_time
    fragment.sonorities = sonorities


def transpose_up(position: int, min_position: int) -> int:
    """
    Transpose pitch up by minimum number of octaves that is enough to place it above threshold.

    :param position:
        position of pitch (in semitones)
    :param min_position:
        minimum high enough position (in semitones)
    :return:
        transposed pitch
    """
    shortage = max(min_position - position, 0)
    position += int(math.ceil(shortage / N_SEMITONES_PER_OCTAVE)) * N_SEMITONES_PER_OCTAVE
    return position


def transpose_down(position: int, max_position: int) -> int:
    """
    Transpose pitch down by minimum number of octaves that is enough to place it below threshold.

    :param position:
        position of pitch (in semitones)
    :param max_position:
        maximum low enough position (in semitones)
    :return:
        transposed pitch
    """
    surplus = max(position - max_position, 0)
    position -= int(math.ceil(surplus / N_SEMITONES_PER_OCTAVE)) * N_SEMITONES_PER_OCTAVE
    return position


def set_pitches_of_upper_line(fragment: Fragment) -> None:
    """
    Set exact pitches of events from the upper line.

    :param fragment:
        fragment with `melodic_lines` attribute where pitch classes are set
    :return:
        None
    """
    upper_line = fragment.melodic_lines[0]
    for index, event in enumerate(upper_line):  # pragma: no branch
        if event.pitch_class != 'pause':
            break
    low_octave = '1'
    position = NOTE_TO_POSITION_MAPPING[event.pitch_class + low_octave]
    position = transpose_up(position, fragment.upper_line_lowest_position)
    event.position_in_semitones = position
    previous_event_pitch_class = event.pitch_class
    for event in upper_line[index + 1:]:
        if event.pitch_class == 'pause':
            continue
        interval = SMALLEST_INTERVALS_MAPPING[(previous_event_pitch_class, event.pitch_class)]
        position += interval
        position = transpose_up(position, fragment.upper_line_lowest_position)
        position = transpose_down(position, fragment.upper_line_highest_position)
        event.position_in_semitones = position
        previous_event_pitch_class = event.pitch_class


def set_pitches_of_lower_lines(
        fragment: Fragment, max_interval: int = 16, default_shift: int = 7
) -> None:
    """
    Set exact pitches of events from all melodic lines except the upper one.

    :param fragment:
        fragment with `melodic_lines` attribute where pitch classes are set
        and exact pitches are set in the upper line
    :param max_interval:
        maximum interval (in semitones) between two simultaneously sounding events
        from adjacent melodic lines
    :param default_shift:
        downward shift (in semitones) of upper threshold when upper event is a pause
    :return:
        None
    """
    upper_first_event = fragment.melodic_lines[0][0]
    threshold = upper_first_event.position_in_semitones or fragment.upper_line_lowest_position
    previous_positions = [threshold] + [None for _ in range(len(fragment.melodic_lines) - 1)]
    previous_pitch_classes = [None for _ in range(len(fragment.melodic_lines))]
    for sonority in fragment.sonorities:
        threshold = sonority.events[0].position_in_semitones or previous_positions[0]
        previous_positions[0] = threshold
        threshold -= 1  # It is inclusive, so subtraction prevents lines from overlapping.
        for event in sonority.events[1:]:
            if event.pitch_class == 'pause':
                threshold -= default_shift
                continue
            if event.start_time < sonority.start_time:
                threshold = event.position_in_semitones - 1
                continue
            previous_pitch_class = previous_pitch_classes[event.line_index]
            if previous_pitch_class is None:
                high_octave = '7'
                position = NOTE_TO_POSITION_MAPPING[event.pitch_class + high_octave]
            else:
                interval = SMALLEST_INTERVALS_MAPPING[(previous_pitch_class, event.pitch_class)]
                position = previous_positions[event.line_index] + interval
            position = transpose_down(position, threshold)
            position = transpose_up(position, 0)
            if threshold - position > max_interval:
                position += N_SEMITONES_PER_OCTAVE
            event.position_in_semitones = position
            previous_positions[event.line_index] = position
            previous_pitch_classes[event.line_index] = event.pitch_class
            threshold = position - 1


def override_calculated_attributes(fragment: Fragment) -> None:
    """
    Override calculated attributes with values computed based on core attributes.

    :param fragment:
        fragment to be updated
    :return:
        None
    """
    update_dependent_tone_row_instances(fragment)
    set_sonic_content(fragment)
    set_melodic_lines_and_their_pitch_classes(fragment)
    set_sonorities(fragment)
    set_pitches_of_upper_line(fragment)
    set_pitches_of_lower_lines(fragment)


def initialize_fragment(params: FragmentParams) -> Fragment:
    """
    Create fragment by its parameters.

    :param params:
        parameters of a fragment to be created
    :return:
        fragment
    """
    params.temporal_content = params.temporal_content or {}
    validate(params)
    measure_durations_by_n_events = group_measure_durations_by_n_events(params.measure_durations)
    temporal_content = create_initial_temporal_content(params, measure_durations_by_n_events)
    (
        grouped_tone_row_instances,
        mutable_independent_tone_row_instances_indices,
        mutable_dependent_tone_row_instances_indices
    ) = create_initial_grouped_tone_row_instances(params)
    (
        grouped_mutable_pauses_indices,
        grouped_immutable_pauses_indices
    ) = find_initial_pauses_indices(params)
    fragment = Fragment(
        temporal_content,
        grouped_tone_row_instances,
        grouped_mutable_pauses_indices,
        grouped_immutable_pauses_indices,
        params.n_measures * params.meter_numerator,
        params.meter_numerator,
        params.meter_denominator,
        measure_durations_by_n_events,
        params.line_ids,
        NOTE_TO_POSITION_MAPPING[params.upper_line_highest_note],
        NOTE_TO_POSITION_MAPPING[params.upper_line_lowest_note],
        len(params.tone_row),
        {i: group['melodic_line_indices'] for i, group in enumerate(params.groups)},
        find_mutable_temporal_content_indices(params),
        mutable_independent_tone_row_instances_indices,
        mutable_dependent_tone_row_instances_indices
    )
    override_calculated_attributes(fragment)
    return fragment
