"""
Define data structures representing a fragment of a musical piece.

Author: Nikolay Lysenko
"""


import math
import random
from dataclasses import dataclass
from typing import Optional

from sinethesizer.utils.music_theory import get_note_to_position_mapping

from .utils import (
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


@dataclass
class Fragment:
    """A fragment of a musical piece."""
    temporal_content: list[list[list[Event]]]
    sonic_content: list[list[str]]
    meter_numerator: int
    meter_denominator: int
    n_beats: int
    line_ids: list[int]
    upper_line_highest_position: int
    upper_line_lowest_position: int
    n_tone_row_instances_by_group: list[int]
    melodic_lines: Optional[list[list[Event]]] = None
    sonorities: Optional[list[list[Event]]] = None


def validate(params: FragmentParams) -> None:
    """
    Validate parameters.

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


def find_initial_durations(params: FragmentParams) -> list[list[list[float]]]:
    """
    Split time span of each melodic line into durations of individual events.

    :param params:
        parameters of a fragment to be created
    :return:
        lists of event durations (in reference beats) for each melodic line from each group of
        melodic lines sharing the same series (in terms of vertical distribution of series pitches)
    """
    meter_numerator = float(params.meter_numerator)
    results = []
    for group_params in params.groups:
        nested_results = []
        n_sound_events = group_params['n_tone_row_instances'] * TONE_ROW_LEN
        n_events = int(round(n_sound_events / (1 - params.pauses_fraction)))
        n_lines = group_params['n_melodic_lines']
        n_events_per_line = [n_events // n_lines for _ in range(n_lines)]
        line_index = 0
        while sum(n_events_per_line) < n_events:
            n_events_per_line[line_index] += 1
            line_index += 1
        for current_n_events in n_events_per_line:
            durations = split_time_span(params.n_measures, current_n_events, meter_numerator)
            nested_results.append(durations)
        results.append(nested_results)
    return results


def find_initial_temporal_content(params: FragmentParams) -> list[list[list[Event]]]:
    """
    Find initial value of data structure that keeps track of event durations.

    :param params:
        parameters of a fragment to be created
    :return:
        list where for each group of melodic lines sharing the same series for each melodic line
        there are events representing temporal (loosely speaking, rhythmic) structure of the line
    """
    line_index = 0
    temporal_content = []
    groups_durations = find_initial_durations(params)
    for group_durations in groups_durations:
        group_content = []
        for line_durations in group_durations:
            start_time = 0
            line_content = []
            for duration in line_durations:
                line_content.append(Event(line_index, start_time, duration))
                start_time += duration
            group_content.append(line_content)
            line_index += 1
        temporal_content.append(group_content)
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


def find_initial_sonic_content(params: FragmentParams) -> list[list[str]]:
    """
    Find initial value of data structure that keeps track of pitch classes.

    :param params:
        parameters of a fragment to be created
    :return:
        list where for each group of melodic lines sharing the same series sonic content
        of the series is stored (i.e., there is a sequence consisting of pitch classes and pauses)
    """
    sonic_content = []
    for group_params in params.groups:
        group_content = []
        n_sound_events = group_params['n_tone_row_instances'] * TONE_ROW_LEN
        n_events = int(round(n_sound_events / (1 - params.pauses_fraction)))
        n_pauses = n_events - n_sound_events
        pauses_indices = random.sample(range(n_events), n_pauses)
        series = replicate_tone_row(params.tone_row, group_params['n_tone_row_instances'])
        for index in range(n_events):
            if index in pauses_indices:
                group_content.append('pause')
            else:
                group_content.append(series.pop(0))
        sonic_content.append(group_content)
    return sonic_content


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
    zipped = zip(fragment.temporal_content, fragment.sonic_content)
    for group_temporal_content, group_sonic_content in zipped:
        timeline = [
            event
            for line_temporal_content in group_temporal_content
            for event in line_temporal_content
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
    return fragment


def initialize_fragment(params: FragmentParams) -> Fragment:
    """
    Create fragment by its parameters.

    :param params:
        parameters of a fragment to be created
    :return:
        fragment
    """
    validate(params)
    fragment = Fragment(
        find_initial_temporal_content(params),
        find_initial_sonic_content(params),
        params.meter_numerator,
        params.meter_denominator,
        params.n_measures * params.meter_numerator,
        params.line_ids,
        NOTE_TO_POSITION_MAPPING[params.upper_line_highest_note],
        NOTE_TO_POSITION_MAPPING[params.upper_line_lowest_note],
        [group['n_tone_row_instances'] for group in params.groups]
    )
    fragment = override_calculated_attributes(fragment)
    return fragment
