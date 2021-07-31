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
    TONE_ROW_LEN,
    get_smallest_intervals_between_pitch_classes,
    invert_tone_row,
    revert_tone_row,
    validate_tone_row
)


NOTE_TO_POSITION_MAPPING = get_note_to_position_mapping()
SMALLEST_INTERVALS_MAPPING = get_smallest_intervals_between_pitch_classes()

SUPPORTED_DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
DURATION_SPLITS = {
    6.0: [[4.0, 2.0], [3.0, 3.0]],
    4.0: [[2.0, 2.0], [3.0, 1.0]],
    3.0: [[2.0, 1.0], [1.5, 1.5]],
    2.0: [[1.0, 1.0], [1.5, 0.5]],
    1.5: [[1.0, 0.5], [0.75, 0.75]],
    1.0: [[0.5, 0.5], [0.75, 0.25]],
    0.75: [[0.5, 0.25]],
    0.5: [[0.25, 0.25]],
    0.25: [[0.25]],
}


@dataclass
class Event:
    """An indivisible element of a musical piece."""
    line_index: int
    start_time: float
    duration: float
    pitch_class: Optional[str] = None
    position_in_semitones: Optional[int] = None


@dataclass
class FragmentParams:
    """Initial parameters of a fragment of a musical piece."""
    tone_row: list[str]
    groups: list[dict[str, int]]
    meter_numerator: int
    meter_denominator: int
    n_measures: int
    line_ids: list[int]
    upper_line_highest_note: str
    upper_line_lowest_note: str
    pauses_fraction: float
    temporal_content: Optional[dict[int, dict[str, Any]]] = None
    sonic_content: Optional[dict[int, dict[str, Any]]] = None


@dataclass
class Fragment:
    """A fragment of a musical piece."""
    temporal_content: list[list[float]]
    sonic_content: list[list[str]]
    meter_numerator: int
    meter_denominator: int
    n_beats: int
    line_ids: list[int]
    upper_line_highest_position: int
    upper_line_lowest_position: int
    n_melodic_lines_by_group: list[int]
    n_tone_row_instances_by_group: list[int]
    mutable_temporal_content_indices: list[int]
    mutable_sonic_content_indices: list[int]
    melodic_lines: Optional[list[list[Event]]] = None
    sonorities: Optional[list[list[Event]]] = None
    durations_of_measures: Optional[list[list[list[float]]]] = None

    def __eq__(self, other: Any):
        if not isinstance(other, Fragment):
            return NotImplemented
        return (
            self.temporal_content == other.temporal_content
            and self.sonic_content == other.sonic_content
        )


def validate_initialized_content(params: FragmentParams) -> None:
    """
    Validate parameters related to initialized parts of temporal content and sonic content.

    :param params:
        parameters of a fragment to be created
    :return:
        None
    """
    if params.temporal_content is not None:
        total_duration_in_beats = params.n_measures * params.meter_numerator
        for line_index, line_params in params.temporal_content.items():
            duration_in_beats = sum(line_params['durations'])
            if duration_in_beats != total_duration_in_beats:
                raise ValueError("A line has duration that is not equal to that of the fragment.")
    if params.sonic_content is not None:  # pragma: no branch
        for group_index, group_params in params.sonic_content.items():
            n_sound_events = len([x for x in group_params['pitch_classes'] if x != 'pause'])
            n_tone_row_instances = n_sound_events / TONE_ROW_LEN
            declared_n_instances = params.groups[group_index]['n_tone_row_instances']
            if n_tone_row_instances != declared_n_instances:
                raise ValueError("A group has wrong number of tone row instances.")


def validate(params: FragmentParams) -> None:
    """
    Validate all parameters.

    :param params:
        parameters of a fragment to be created
    :return:
        None
    """
    validate_tone_row(params.tone_row)
    if sum(group['n_melodic_lines'] for group in params.groups) != len(params.line_ids):
        raise ValueError("Number of lines in `groups` is not equal to that in `line_ids`.")
    if params.line_ids != list(set(params.line_ids)):
        raise ValueError("IDs of melodic lines must be unique.")
    if float(params.meter_numerator) not in SUPPORTED_DURATIONS:
        raise ValueError(f"Meter numerator = {params.meter_numerator} is not supported.")
    validate_initialized_content(params)


def split_time_span(n_measures: int, n_events: int, meter_numerator: float) -> list[float]:
    """
    Split time span of one melodic line into time spans of the specified number of events.

    :param n_measures:
        number of measures
    :param n_events:
        number of events
    :param meter_numerator:
        number of reference beats per measure
    :return:
        durations of events (in reference beats)
    """
    if n_events < n_measures:
        raise ValueError("Average duration of an event is longer than semibreve.")
    if n_events > n_measures * meter_numerator / min(SUPPORTED_DURATIONS):
        raise ValueError("The number of events is so high that some of them are too short.")
    durations = [meter_numerator for _ in range(n_measures)]
    current_n_events = n_measures
    index = 0
    while current_n_events < n_events:
        insertion = random.choice(DURATION_SPLITS[durations[index]])
        durations = durations[:index] + insertion + durations[index + 1:]
        index += len(insertion)
        index %= len(durations)
        current_n_events += len(insertion) - 1
    random.shuffle(durations)
    return durations


def calculate_number_of_undefined_events(
        group_index: int,
        temporal_content: list[list[float]],
        sonic_content: dict[int, dict[str, Any]],
        line_indices: list[int],
        n_tone_row_instances: int,
        pauses_fraction: float
) -> int:
    """
    Calculate total number of undefined events in a group of melodic lines sharing the same series.

    :param group_index:
        index of a group of melodic lines
    :param temporal_content:
        partially filled list of lists of event durations (in reference beats)
        for each melodic line
    :param sonic_content:
        mapping from group index to user-defined sequence of its pitch classes (including pauses)
    :param line_indices:
        indices of melodic lines forming the group
    :param n_tone_row_instances:
        number of tone row instances in the group
    :param pauses_fraction:
        fraction of pauses amongst all events from the group
    :return:
        total number of events in the group that are not initially defined by a user
    """
    if group_index in sonic_content:
        n_events = len(sonic_content[group_index]['pitch_classes'])
    else:
        n_sound_events = n_tone_row_instances * TONE_ROW_LEN
        n_events = int(round(n_sound_events / (1 - pauses_fraction)))
    n_events -= sum(len(temporal_content[line_index]) for line_index in line_indices)
    return n_events


def distribute_events_between_lines(n_events: int, n_lines: int) -> list[int]:
    """
    Distribute evenly the specified number of events between the specified number of lines.

    :param n_events:
        number of events
    :param n_lines:
        number of lines
    :return:
        list of numbers of events in a particular line
    """
    results = [n_events // n_lines for _ in range(n_lines)]
    i = 0
    while sum(results) < n_events:
        results[i] += 1
        i += 1
    return results


def create_initial_temporal_content(params: FragmentParams) -> list[list[float]]:
    """
    Split time span of each melodic line into durations of individual events.

    :param params:
        parameters of a fragment to be created
    :return:
        lists of event durations (in reference beats) for each melodic line
    """
    temporal_content = [
        params.temporal_content.get(line_index, {}).get('durations', [])
        for line_index, line_id in enumerate(params.line_ids)
    ]
    meter_numerator = float(params.meter_numerator)
    group_lines_start_index = 0
    for group_index, group_params in enumerate(params.groups):
        n_lines = group_params['n_melodic_lines']
        line_indices = list(range(group_lines_start_index, group_lines_start_index + n_lines))
        n_undefined_events = calculate_number_of_undefined_events(
            group_index,
            temporal_content,
            params.sonic_content,
            line_indices,
            group_params['n_tone_row_instances'],
            params.pauses_fraction
        )
        indices_of_undefined_lines = [x for x in line_indices if x not in params.temporal_content]
        n_undefined_lines = len(indices_of_undefined_lines)
        n_events_per_line = distribute_events_between_lines(n_undefined_events, n_undefined_lines)
        for line_index, n_events in zip(indices_of_undefined_lines, n_events_per_line):
            durations = split_time_span(params.n_measures, n_events, meter_numerator)
            temporal_content[line_index] = durations
        group_lines_start_index += n_lines
    return temporal_content


def replicate_tone_row(tone_row: list[str], n_instances: int) -> list[str]:
    """
    Create multiple instances of tone row.

    :param tone_row:
        tone row as list of pitch classes (like C or C#, flats are not allowed)
    :param n_instances:
        number of tone row instances
    :return:
        list of pitch classes consisting of instances of tone row (maybe, inverted or reverted)
    """
    pitch_classes = []
    for _ in range(n_instances):
        current_instance = [pitch_class for pitch_class in tone_row]
        if random.choice([True, False]):
            current_instance = invert_tone_row(current_instance)
        if random.choice([True, False]):
            current_instance = revert_tone_row(current_instance)
        for pitch_class in current_instance:
            pitch_classes.append(pitch_class)
    return pitch_classes


def create_initial_sonic_content(
        params: FragmentParams, temporal_content: list[list[float]]
) -> list[list[str]]:
    """
    Create initial data structure that keeps track of pitch classes.

    :param params:
        parameters of a fragment to be created
    :param temporal_content:
        lists of event durations (in reference beats) for each melodic line
    :return:
        list where for each group of melodic lines sharing the same series sonic content
        of the series is stored (i.e., there is a sequence consisting of pitch classes and pauses)
    """
    sonic_content = []
    group_lines_start_index = 0
    for group_index, group_params in enumerate(params.groups):
        n_lines = group_params['n_melodic_lines']
        if group_index in params.sonic_content:
            sonic_content.append(params.sonic_content[group_index]['pitch_classes'])
            group_lines_start_index += n_lines
            continue

        n_events = 0
        durations = temporal_content[group_lines_start_index:(group_lines_start_index + n_lines)]
        for line_durations in durations:
            n_events += len(line_durations)
        n_sound_events = group_params['n_tone_row_instances'] * TONE_ROW_LEN
        n_pauses = n_events - n_sound_events
        if n_pauses < 0:
            raise ValueError("User-defined temporal content has not enough events.")

        pauses_indices = random.sample(range(n_events), n_pauses)
        series = replicate_tone_row(params.tone_row, group_params['n_tone_row_instances'])
        group_sonic_content = []
        for index in range(n_events):
            group_sonic_content.append('pause' if index in pauses_indices else series.pop(0))
        sonic_content.append(group_sonic_content)

        group_lines_start_index += n_lines
    return sonic_content


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
    line_index = 0
    rhythm_only_lines = []
    end_indices = list(itertools.accumulate(fragment.n_melodic_lines_by_group))
    start_indices = [0] + end_indices[:-1]
    for start_index, end_index in zip(start_indices, end_indices):
        group_of_lines = []
        for line_durations in fragment.temporal_content[start_index:end_index]:
            start_time = 0
            line = []
            for duration in line_durations:
                line.append(Event(line_index, start_time, duration))
                start_time += duration
            group_of_lines.append(line)
            line_index += 1
        rhythm_only_lines.append(group_of_lines)
    return rhythm_only_lines


def distribute_pitch_classes(fragment: Fragment) -> list[list[Event]]:
    """
    Distribute pitch classes across melodic lines.

    :param fragment:
        fragment with non-empty temporal content and sonic content
    :return:
        melodic lines with pitch classes, but without exact pitches;
        these lines are derived from temporal content and sonic content of the fragment
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
    return melodic_lines


def find_sonorities(melodic_lines: list[list[Event]]) -> list[list[Event]]:
    """
    Find simultaneously sounding events.

    :param melodic_lines:
        melodic lines
    :return:
        list of simultaneously sounding events
    """
    timeline = [event for melodic_line in melodic_lines for event in melodic_line]
    timeline = sorted(timeline, key=lambda event: (event.start_time, event.line_index))
    indices = {i: -1 for i, _ in enumerate(melodic_lines)}
    current_times = {i: 0 for i, _ in enumerate(melodic_lines)}
    previous_passed_time = 0
    sonorities = []
    for event in timeline:
        indices[event.line_index] += 1
        current_times[event.line_index] += event.duration
        passed_time = min(v for k, v in current_times.items())
        if passed_time > previous_passed_time:
            sonorities.append([
                melodic_line[indices[line_number]]
                for line_number, melodic_line in enumerate(melodic_lines)
            ])
        previous_passed_time = passed_time
    return sonorities


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


def set_pitches_of_upper_line(fragment: Fragment) -> Fragment:
    """
    Set exact pitches of events from upper line.

    :param fragment:
        fragment with `melodic_lines` attribute where pitch classes are set
    :return:
        fragment with `melodic_lines` attribute where exact pitch of upper line events are set
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
    return fragment


def set_pitches_of_lower_lines(
        fragment: Fragment, max_interval: int = 16, default_shift: int = 7
) -> Fragment:
    """
    Set exact pitches of events from all melodic lines except the upper one.

    :param fragment:
        fragment with `melodic_lines` attribute where pitch classes are set and exact pitches
        are set in the upper line
    :param max_interval:
        maximum interval (in semitones) between two simultaneously sounding events from adjacent
        melodic lines
    :param default_shift:
        downward shift (in semitones) of upper threshold when upper event is pause
    :return:
        fragment where pitches are set for all events
    """
    upper_first_event = fragment.melodic_lines[0][0]
    threshold = upper_first_event.position_in_semitones or fragment.upper_line_lowest_position
    previous_positions = [threshold] + [None for _ in range(len(fragment.melodic_lines) - 1)]
    previous_pitch_classes = [None for _ in range(len(fragment.melodic_lines))]
    for sonority in fragment.sonorities:
        sonority_start = max(event.start_time for event in sonority)
        threshold = sonority[0].position_in_semitones or previous_positions[0]
        previous_positions[0] = threshold
        threshold -= 1  # It is inclusive, so subtraction prevents lines from overlapping.
        for event in sonority[1:]:
            if event.pitch_class == 'pause':
                threshold -= default_shift
                continue
            if event.start_time < sonority_start:
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
    return fragment


def calculate_durations_of_measures(fragment: Fragment) -> list[list[list[float]]]:
    """
    Calculate durations of measures.

    :param fragment:
        fragment with `melodic_lines` attribute
    :return:
        list where for each melodic line there is a list where each measure is represented as list
        of durations of events forming it; suspended from the previous measure event is partially
        included and suspended to the next measure event is fully included
    """
    durations_of_measures = []
    for melodic_line in fragment.melodic_lines:
        rhythm_data = []
        for event in melodic_line:
            record = {
                'measure_id': event.start_time // fragment.meter_numerator,
                'time_since_measure_start': event.start_time % fragment.meter_numerator,
                'duration': event.duration
            }
            rhythm_data.append(record)

        durations_of_measures_for_one_line = []
        for measure_id, values in itertools.groupby(rhythm_data, lambda x: x['measure_id']):
            first_value = values.__next__()
            if first_value['time_since_measure_start'] != 0:
                durations = [first_value['time_since_measure_start'], first_value['duration']]
            else:
                durations = [first_value['duration']]
            for value in values:
                durations.append(value['duration'])
            durations_of_measures_for_one_line.append(durations)
        durations_of_measures.append(durations_of_measures_for_one_line)
    return durations_of_measures


def override_calculated_attributes(fragment: Fragment) -> Fragment:
    """
    Override calculated attributes with values computed based on temporal and sonic content.

    :param fragment:
        fragment to be updated
    :return:
        updated fragment
    """
    fragment.melodic_lines = distribute_pitch_classes(fragment)
    fragment.sonorities = find_sonorities(fragment.melodic_lines)
    fragment = set_pitches_of_upper_line(fragment)
    fragment = set_pitches_of_lower_lines(fragment)
    fragment.durations_of_measures = calculate_durations_of_measures(fragment)
    return fragment


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


def find_mutable_sonic_content_indices(params: FragmentParams) -> list[int]:
    """
    Find indices of groups such that their sonic content can be modified.

    :param params:
        parameters of a fragment to be created
    :return:
        indices of groups such that their sonic content can be modified
    """
    results = []
    for group_index, group_params in enumerate(params.groups):
        if not params.sonic_content.get(group_index, {}).get('immutable', False):
            results.append(group_index)
    return results


def initialize_fragment(params: FragmentParams) -> Fragment:
    """
    Create fragment by its parameters.

    :param params:
        parameters of a fragment to be created
    :return:
        fragment
    """
    params.temporal_content = params.temporal_content or {}
    params.sonic_content = params.sonic_content or {}
    validate(params)
    temporal_content = create_initial_temporal_content(params)
    sonic_content = create_initial_sonic_content(params, temporal_content)
    fragment = Fragment(
        temporal_content,
        sonic_content,
        params.meter_numerator,
        params.meter_denominator,
        params.n_measures * params.meter_numerator,
        params.line_ids,
        NOTE_TO_POSITION_MAPPING[params.upper_line_highest_note],
        NOTE_TO_POSITION_MAPPING[params.upper_line_lowest_note],
        [group['n_melodic_lines'] for group in params.groups],
        [group['n_tone_row_instances'] for group in params.groups],
        find_mutable_temporal_content_indices(params),
        find_mutable_sonic_content_indices(params),
    )
    fragment = override_calculated_attributes(fragment)
    return fragment
