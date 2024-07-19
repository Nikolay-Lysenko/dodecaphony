"""
Evaluate a fragment.

Author: Nikolay Lysenko
"""


import itertools
import math
import re
import string
from collections import Counter
from functools import cache
from typing import Any, Callable, Optional

from sinethesizer.utils.music_theory import get_note_to_position_mapping

from .fragment import Event, Fragment
from .music_theory import (
    IntervalTypes,
    N_SEMITONES_PER_OCTAVE,
    N_SEMITONES_TO_INTERVAL_TYPE_WITH_CONSONANT_P4,
    N_SEMITONES_TO_INTERVAL_TYPE_WITH_DISSONANT_P4,
    PITCH_CLASS_TO_POSITION,
    get_mapping_from_pitch_class_to_diatonic_scales,
    get_type_of_interval,
)
from .utils import compute_rolling_aggregate


NOTE_TO_POSITION_MAPPING = get_note_to_position_mapping()
SCORING_SETS_REGISTRY_TYPE = dict[
    str,
    list[tuple[Callable[..., float], dict[float, float], dict[str, Any]]]
]


def evaluate_absence_of_aimless_fluctuations(
        fragment: Fragment, penalties: dict[int, float], window_size: int
) -> float:
    """
    Evaluate melodic fluency based on absence of aimless fluctuations within narrow ranges.

    :param fragment:
        a fragment to be evaluated
    :param penalties:
        mapping from range (in semitones) covered within a window to penalty applicable to ranges
        of not greater size; it is recommended to set maximum penalty to 1
    :param window_size:
        size of rolling window (in events)
    :return:
        multiplied by -1 penalty for narrowness averaged over all windows (including those that
        are not penalized)
    """
    numerator = 0
    denominator = 0
    for melodic_line in fragment.melodic_lines:
        pitches = [event.position_in_semitones for event in melodic_line]
        pitches = [x for x in pitches if x is not None]
        rolling_mins = compute_rolling_aggregate(pitches, min, window_size)[window_size-1:]
        rolling_maxs = compute_rolling_aggregate(pitches, max, window_size)[window_size-1:]
        borders = zip(rolling_mins, rolling_maxs)
        for lower_border, upper_border in borders:
            range_width = upper_border - lower_border
            raw_penalties = [v for k, v in penalties.items() if k >= range_width]
            penalty = max(raw_penalties) if raw_penalties else 0
            numerator -= penalty
            denominator += 1
    score = numerator / denominator
    return score


def evaluate_absence_of_doubled_pitch_classes(fragment: Fragment) -> float:
    """
    Evaluate absence of vertical intervals that are whole multipliers of octave.

    If the same pitch class sounds in two or more melodic lines simultaneously,
    harmonic stability increases significantly and it interrupts musical flow associated
    with the twelve-tone technique.

    :param fragment:
        a fragment to be evaluated
    :return:
        minus one multiplied by number of pairs of simultaneously sounding events
        of the same pitch class and divided by total number of sonorities
    """
    score = 0
    for sonority in fragment.sonorities:
        for first, second in itertools.combinations(sonority, 2):
            if first.pitch_class == 'pause' or second.pitch_class == 'pause':
                continue
            interval = first.position_in_semitones - second.position_in_semitones
            if interval % N_SEMITONES_PER_OCTAVE == 0:
                score -= 1
    score /= len(fragment.sonorities)
    return score


def evaluate_absence_of_simultaneous_skips(
        fragment: Fragment, min_skip_in_semitones: int = 4, max_skips_share: float = 0.65
) -> float:
    """
    Evaluate absence of simultaneous large enough skips.

    :param fragment:
        a fragment to be evaluated
    :param min_skip_in_semitones:
        minimum size (in semitones) of a melodic interval to be considered a large enough skip
    :param max_skips_share:
        maximum share of melodic lines with skips between adjacent sonorities not to be penalized
        for this pair of sonorities
    :return:
        minus one multiplied by fraction of sonorities with enough number of simultaneous skips
    """
    score = 0
    for first_sonority, second_sonority in zip(fragment.sonorities, fragment.sonorities[1:]):
        n_melodic_intervals = 0
        n_skips = 0
        for first, second in zip(first_sonority, second_sonority):
            if first.pitch_class == 'pause' or second.pitch_class == 'pause':
                continue
            n_melodic_intervals += 1
            interval_size = abs(first.position_in_semitones - second.position_in_semitones)
            n_skips += int(interval_size >= min_skip_in_semitones)
        if n_melodic_intervals > 0 and n_skips / n_melodic_intervals >= max_skips_share:
            score -= 1
    score /= len(fragment.sonorities) - 1
    return score


def evaluate_absence_of_voice_crossing(
        fragment: Fragment, n_semitones_to_penalty: dict[int, float]
) -> float:
    """
    Evaluate absence of voice crossing.

    Voice crossing may result in wrong perception of tone row and in incoherence of the fragment.

    :param fragment:
        a fragment to be evaluated
    :param n_semitones_to_penalty:
        mapping from size of vertical interval between a pair of voices (this size is assumed to be
        non-positive) to penalty for this interval
    :return:
        minus one multiplied by average over all vertical intervals penalty
    """
    numerator = 0
    denominator = 0
    for sonority in fragment.sonorities:
        for first, second in itertools.combinations(sonority, 2):
            if first.pitch_class == 'pause' or second.pitch_class == 'pause':
                continue
            interval = first.position_in_semitones - second.position_in_semitones
            if interval <= 0:
                numerator -= n_semitones_to_penalty.get(interval, 1)
            denominator += 1
    score = numerator / denominator
    return score


def evaluate_cadence_duration(
        fragment: Fragment,
        max_duration: float,
        last_sonority_weight: float,
        last_notes_weight: float
) -> float:
    """
    Evaluate that cadence has enough duration.

    :param fragment:
        a fragment to be evaluated
    :param max_duration:
        maximum enough duration (in reference beats); higher durations do not increase score
    :param last_sonority_weight:
        weight that determines contribution of last sonority duration to final score
    :param last_notes_weight:
        weight that determines contribution of duration of last note of each melodic line
        to final score; this term is rather a mean than a goal, because it rewards intermediate
        steps when last sonority duration is unchanged, but one of the last notes extends its
        duration
    :return:
        minus one multiplied by missing in last sonority and last notes fraction of `max_duration`
    """
    clipped_durations = [min(event.duration, max_duration) for event in fragment.sonorities[-1]]
    last_sonority_duration = min(clipped_durations)
    avg_last_note_duration = sum(clipped_durations) / len(clipped_durations)
    total_weight = last_sonority_weight + last_notes_weight
    last_sonority_weight /= total_weight
    last_notes_weight /= total_weight
    last_sonority_term = last_sonority_weight * (last_sonority_duration / max_duration - 1)
    last_notes_term = last_notes_weight * (avg_last_note_duration / max_duration - 1)
    score = last_sonority_term + last_notes_term
    return score


def evaluate_climax_explicity(
        fragment: Fragment,
        height_penalties: dict[int, float],
        duplication_penalty: float
) -> float:
    """
    Evaluate goal-orientedness of melodic lines by explicity of their climax points.

    :param fragment:
        a fragment to be evaluated
    :param height_penalties:
        mapping from interval size (in semitones) between average pitch of a line and climax of
        this line to penalty for such interval; values for missing keys are forward-propagated
    :param duplication_penalty:
        penalty for each non-first occurrence of line's highest pitch within this line
    :return:
        summed over melodic lines penalty for not so high climax points and for duplications
        of climax points within the same line
    """
    score = 0
    for melodic_line in fragment.melodic_lines:
        pitches = [event.position_in_semitones for event in melodic_line]
        pitches = [x for x in pitches if x is not None]

        climax_pitch = max(pitches)
        average_pitch = sum(pitches) / len(pitches)
        interval_size = climax_pitch - average_pitch
        raw_height_penalties = [v for k, v in height_penalties.items() if k >= interval_size]
        height_penalty = max(raw_height_penalties) if raw_height_penalties else 0
        score -= height_penalty

        n_duplications = len([x for x in pitches if x == climax_pitch]) - 1
        score -= duplication_penalty * n_duplications
    score /= len(fragment.melodic_lines)
    return score


def evaluate_direction_change_after_large_skip(
        fragment: Fragment,
        min_skip_in_semitones: int = 5,
        max_opposite_move_in_semitones: int = 2,
        large_opposite_move_relative_penalty: float = 0.8
) -> float:
    """
    Evaluate presence of opposite stepwise motion after each large enough skip.

    :param fragment:
        a fragment to be evaluated
    :param min_skip_in_semitones:
        minimum size (in semitones) of a melodic interval to be considered a large enough skip
    :param max_opposite_move_in_semitones:
        maximum size (in semitones) of a melodic interval to be considered a stepwise motion
    :param large_opposite_move_relative_penalty:
        penalty for moving in opposite direction with a skip as a ratio to penalty for moving
        in the same direction
    :return:
        average over all melodic intervals penalty for improper motion after large enough skip
    """
    numerator = 0
    denominator = 0
    for melodic_line in fragment.melodic_lines:
        zipped = zip(melodic_line, melodic_line[1:], melodic_line[2:])
        for first_event, second_event, third_event in zipped:
            if first_event.pitch_class == "pause" or second_event.pitch_class == "pause":
                continue
            denominator += 1
            interval = second_event.position_in_semitones - first_event.position_in_semitones
            if abs(interval) < min_skip_in_semitones:
                continue
            if third_event.pitch_class == "pause":
                numerator += 1
                continue
            next_interval = third_event.position_in_semitones - second_event.position_in_semitones
            if interval * next_interval >= 0:
                numerator += 1
            elif abs(next_interval) >= max_opposite_move_in_semitones:
                numerator += large_opposite_move_relative_penalty
    score = -numerator / denominator
    return score


def find_indices_of_dissonating_events(
        sonority: list[Event], meter_numerator: int
) -> tuple[set[int], set[int]]:
    """
    Find indices of dissonating (i.e., dependent, non-free in terms of strict counterpoint) events.

    :param sonority:
        simultaneously sounding events
    :param meter_numerator:
        numerator in meter signature, i.e., number of reference beats per measure
    :return:
        indices of passing tones or neighbor dissonances and indices of suspended dissonances
    """
    passing_tones_and_neighbors = set()
    suspensions = set()
    sonority_start_time = max(event.start_time for event in sonority)
    pairs = itertools.combinations(sonority, 2)
    for first_event, second_event in pairs:
        if first_event.pitch_class == 'pause' or second_event.pitch_class == 'pause':
            continue
        n_semitones = first_event.position_in_semitones - second_event.position_in_semitones
        is_perfect_fourth_consonant = second_event.line_index != len(sonority) - 1
        interval_type = get_type_of_interval(n_semitones, is_perfect_fourth_consonant)
        if interval_type != IntervalTypes.DISSONANCE:
            continue
        first_event_continues = first_event.start_time < sonority_start_time
        second_event_continues = second_event.start_time < sonority_start_time
        if first_event_continues and second_event_continues:
            continue
        first_event_starts_on_downbeat = first_event.start_time % meter_numerator == 0
        second_event_starts_on_downbeat = second_event.start_time % meter_numerator == 0
        if first_event_continues and second_event_starts_on_downbeat:
            suspensions.add(first_event.line_index)
            continue
        if second_event_continues and first_event_starts_on_downbeat:
            suspensions.add(second_event.line_index)
            continue
        if not first_event_continues:
            passing_tones_and_neighbors.add(first_event.line_index)
        if not second_event_continues:
            passing_tones_and_neighbors.add(second_event.line_index)
    return passing_tones_and_neighbors, suspensions


def find_melodic_interval(
        event: Event, event_index: int, melodic_line: list[Event], shift: int
) -> Optional[int]:
    """
    Find melodic interval between the given event and an adjacent event.

    :param event:
        event (it is assumed that it is not pause)
    :param event_index:
        index of the event in its melodic line
    :param melodic_line:
        melodic line containing the event
    :param shift:
        -1 if interval of arrival in the event is needed or
        1 if interval of departure from the event is needed
    :return:
        size of melodic interval (in semitones)
    """
    try:
        adjacent_event = melodic_line[event_index + shift]
    except IndexError:
        return None
    if adjacent_event.pitch_class == 'pause':
        return None
    n_semitones = event.position_in_semitones - adjacent_event.position_in_semitones
    return n_semitones


def evaluate_dissonances_preparation_and_resolution(
        fragment: Fragment,
        n_semitones_to_pt_and_ngh_preparation_penalty: dict[int, float],
        n_semitones_to_pt_and_ngh_resolution_penalty: dict[int, float],
        n_semitones_to_suspension_resolution_penalty: dict[int, float]
) -> float:
    """
    Evaluate smoothness of dissonances preparation and resolution.

    :param fragment:
        a fragment to be evaluated
    :param n_semitones_to_pt_and_ngh_preparation_penalty:
        mapping from melodic interval size in semitones to a penalty for moving by this interval
        to a dissonance considered to be an analogue of passing tone or neighbor dissonance
    :param n_semitones_to_pt_and_ngh_resolution_penalty:
        mapping from melodic interval size in semitones to a penalty for moving by this interval
        from a dissonance considered to be an analogue of passing tone or neighbor dissonance
    :param n_semitones_to_suspension_resolution_penalty:
        mapping from melodic interval size in semitones to a penalty for moving by this interval
        from a dissonance considered to be an analogue of suspended dissonance
    :return:
        average over all vertical intervals penalty for their preparation and resolution
    """
    score = 0
    n_semitones_to_pt_and_ngh_preparation_penalty[None] = 0
    n_semitones_to_pt_and_ngh_resolution_penalty[None] = 0
    n_semitones_to_suspension_resolution_penalty[None] = 0
    event_indices = [0 for _ in fragment.melodic_lines]
    for sonority in fragment.sonorities:
        zipped = zip(sonority, fragment.melodic_lines, event_indices)
        for event, melodic_line, event_index in zipped:
            if event != melodic_line[event_index]:
                event_indices[event.line_index] += 1
        pt_and_ngh_line_indices, suspension_line_indices = find_indices_of_dissonating_events(
            sonority, fragment.meter_numerator
        )
        for line_index in pt_and_ngh_line_indices:
            event_index = event_indices[line_index]
            melodic_line = fragment.melodic_lines[line_index]
            event = melodic_line[event_index]
            arrival_interval = find_melodic_interval(event, event_index, melodic_line, shift=-1)
            score -= n_semitones_to_pt_and_ngh_preparation_penalty.get(arrival_interval, 1.0)
            departure_interval = find_melodic_interval(event, event_index, melodic_line, shift=1)
            score -= n_semitones_to_pt_and_ngh_resolution_penalty.get(departure_interval, 1.0)
        for line_index in suspension_line_indices:
            event_index = event_indices[line_index]
            melodic_line = fragment.melodic_lines[line_index]
            event = melodic_line[event_index]
            departure_interval = find_melodic_interval(event, event_index, melodic_line, shift=1)
            score -= n_semitones_to_suspension_resolution_penalty.get(departure_interval, 1.0)

    total_n_events = sum(len(melodic_line) for melodic_line in fragment.melodic_lines)
    n_first_events = len(fragment.melodic_lines)
    score /= total_n_events - n_first_events
    return score


def compute_harmonic_stability_of_sonority(
        sonority: list[Event], n_semitones_to_stability: dict[int, float]
) -> float:
    """
    Compute stability of sonority as average stability of intervals forming it.

    :param sonority:
        simultaneously sounding events
    :param n_semitones_to_stability:
        mapping from interval size in semitones to its harmonic stability
    :return:
        average stability of intervals forming the sonority
    """
    stability = 0
    sound_events = [event for event in sonority if event.pitch_class != 'pause']
    if len(sound_events) <= 1:
        return 1.0
    for first, second in itertools.combinations(sound_events, 2):
        interval_in_semitones = abs(first.position_in_semitones - second.position_in_semitones)
        interval_in_semitones %= N_SEMITONES_PER_OCTAVE
        stability += n_semitones_to_stability[interval_in_semitones]
    n_pairs = len(sound_events) * (len(sound_events) - 1) / 2
    stability /= n_pairs
    return stability


def find_sonority_type(
        sonority_start: float, sonority_end: float, regular_positions: list[dict[str, Any]],
        ad_hoc_positions: list[dict[str, Any]], n_beats: int
) -> str:
    """
    Find type of sonority based on its position in time.

    Note that collisions are resolved according to the two following rules:
    1) Ad hoc positions have higher precedence than regular positions;
    2) Precedence amongst either ad hoc positions or regular positions is based on order of
       the positions within the corresponding argument.

    :param sonority_start:
        start time of sonority (in reference beats)
    :param sonority_end:
        end time of sonority (in reference beats)
    :param regular_positions:
        parameters of regular positions (for example, downbeats or relatively strong beats)
    :param ad_hoc_positions:
        parameters of ad hoc positions which appear just once (for example, the beginning of
        the fragment or the 11th reference beat)
    :param n_beats:
        total duration of a fragment (in reference beats)
    :return:
        type of sonority based on its position in time
    """
    for ad_hoc_position in ad_hoc_positions:
        if ad_hoc_position['time'] < 0:
            ad_hoc_position['time'] += n_beats
        if sonority_start <= ad_hoc_position['time'] < sonority_end:
            return ad_hoc_position['name']
    for regular_position in regular_positions:
        denominator = regular_position['denominator']
        ratio = math.floor(sonority_start) // denominator
        processed_start = sonority_start - ratio * denominator
        processed_end = sonority_end - ratio * denominator
        current_time = regular_position['remainder']
        while current_time < processed_end:
            if current_time >= processed_start:
                return regular_position['name']
            current_time += denominator
    return 'default'


def evaluate_harmony_dynamic_by_positions(
        fragment: Fragment, regular_positions: list[dict[str, Any]],
        ad_hoc_positions: list[dict[str, Any]], ranges: dict[str, tuple[float, float]],
        n_semitones_to_stability: dict[int, float]
) -> float:
    """
    Evaluate values of harmonic stability/tension at particular positions.

    :param fragment:
        a fragment to be evaluated
    :param regular_positions:
        parameters of regular positions (for example, downbeats or relatively strong beats)
    :param ad_hoc_positions:
        parameters of ad hoc positions which appear just once (for example, the beginning of
        the fragment or the 11th reference beat)
    :param ranges:
        mapping from position type to minimum and maximum desired levels of harmonic stability
    :param n_semitones_to_stability:
        mapping from interval size in semitones to its harmonic stability
    :return:
        average over all sonorities deviation of harmonic stability from its ranges
    """
    score = 0
    for sonority in fragment.sonorities:
        sonority_start = max(event.start_time for event in sonority)
        sonority_end = min(event.start_time + event.duration for event in sonority)
        stability_of_current_sonority = compute_harmonic_stability_of_sonority(
            sonority, n_semitones_to_stability
        )
        sonority_type = find_sonority_type(
            sonority_start, sonority_end, regular_positions, ad_hoc_positions, fragment.n_beats
        )
        min_allowed_value = ranges[sonority_type][0]
        score += min(stability_of_current_sonority - min_allowed_value, 0)
        max_allowed_value = ranges[sonority_type][1]
        score += min(max_allowed_value - stability_of_current_sonority, 0)
    score /= len(fragment.sonorities)
    return score


def evaluate_harmony_dynamic_by_time_intervals(
        fragment: Fragment, intervals: list[tuple[float, float]],
        ranges: list[tuple[float, float]], n_semitones_to_stability: dict[int, float]
) -> float:
    """
    Evaluate values of harmonic stability/tension during particular time intervals.

    :param fragment:
        a fragment to be evaluated
    :param intervals:
        list of start times and end times of intervals for which harmonic stability is measured
    :param ranges:
        list of minimum and maximum desired levels of harmonic stability during each time interval
    :param n_semitones_to_stability:
        mapping from interval size in semitones to its harmonic stability
    :return:
        weighted (based on duration) average over all sonorities deviation of harmonic stability
        from its ranges
    """
    numerator = 0
    denominator = 0
    sonority_index = 0
    zipped = zip(intervals, ranges)
    for (interval_start, interval_end), (min_allowed_value, max_allowed_value) in zipped:
        while True:
            sonority = fragment.sonorities[sonority_index]
            sonority_start = max(event.start_time for event in sonority)
            sonority_end = min(event.start_time + event.duration for event in sonority)
            if sonority_end <= interval_start:
                sonority_index += 1
                continue
            intersection_start = max(sonority_start, interval_start)
            intersection_end = min(sonority_end, interval_end)
            intersection_duration = intersection_end - intersection_start

            stability_of_current_sonority = compute_harmonic_stability_of_sonority(
                sonority, n_semitones_to_stability
            )
            deviation = 0
            deviation += min(stability_of_current_sonority - min_allowed_value, 0)
            deviation += min(max_allowed_value - stability_of_current_sonority, 0)
            numerator += intersection_duration * deviation
            denominator += intersection_duration

            if sonority_end >= interval_end:
                break
            sonority_index += 1
    score = numerator / denominator
    return score


def evaluate_local_diatonicity_at_all_lines_level(
        fragment: Fragment, depth: int = 2, scale_types: Optional[tuple[str]] = None
) -> float:
    """
    Evaluate presence of diatonic scales at short periods of time.

    If for every short time interval there is a diatonic scale containing all its pitches,
    the fragment might be called pantonal with frequent modulations between diatonic scales.

    :param fragment:
        a fragment to be evaluated
    :param depth:
        duration of a time period as number of successive sonorities
    :param scale_types:
        types of diatonic scales to be tested; this tuple may include the following values:
        'major', 'natural_minor', 'harmonic_minor', 'dorian', 'phrygian', 'lydian', 'mixolydian',
        'locrian', and 'whole_tone'; however, keep in mind that due to relative modes there is no
        need to include all these values; if this argument is not passed, its value is set to
        `('major', 'harmonic_minor', 'whole_tone')` which covers all supported scales
    :return:
        minus one multiplied by average fraction of pitches that are out of the most fitting
        to the current short interval diatonic scale (with averaging over all short intervals)
    """
    score = 0
    scale_types = scale_types or ('major', 'harmonic_minor', 'whole_tone')
    pitch_class_to_diatonic_scales = get_mapping_from_pitch_class_to_diatonic_scales(scale_types)
    nested_pitch_classes = []
    for sonority in fragment.sonorities[:depth - 1]:
        nested_pitch_classes.append([event.pitch_class for event in sonority])
    for sonority in fragment.sonorities[depth - 1:]:
        nested_pitch_classes.append([event.pitch_class for event in sonority])
        pitch_classes = [x for y in nested_pitch_classes for x in y if x != 'pause']
        counter = Counter()
        for pitch_class in pitch_classes:
            counter.update(pitch_class_to_diatonic_scales[pitch_class])
        n_pitch_classes_from_best_scale = counter.most_common(1)[0][1]
        score -= 1 - n_pitch_classes_from_best_scale / len(pitch_classes)
        nested_pitch_classes.pop(0)
    n_periods = len(fragment.sonorities) - depth + 1
    score /= n_periods
    return score


def evaluate_local_diatonicity_at_line_level(
        fragment: Fragment, depth: int = 7, scale_types: Optional[tuple[str]] = None
) -> float:
    """
    Evaluate presence of diatonic scales at short periods of time for each line independently.

    If for every short time interval there is a diatonic scale containing all its pitches,
    the melodic line might be called pantonal with frequent modulations between diatonic scales.

    :param fragment:
        a fragment to be evaluated
    :param depth:
        duration of a time period as number of successive events
    :param scale_types:
        types of diatonic scales to be tested; this tuple may include the following values:
        'major', 'natural_minor', 'harmonic_minor', 'dorian', 'phrygian', 'lydian', 'mixolydian',
        'locrian', and 'whole_tone'; however, keep in mind that due to relative modes there is no
        need to include all these values; if this argument is not passed, its value is set to
        `('major', 'harmonic_minor', 'whole_tone')` which covers all supported scales
    :return:
        minus one multiplied by average fraction of pitches that are out of the most fitting
        to the current short interval diatonic scale (with averaging over all short intervals)
    """
    numerator = 0
    denominator = 0
    scale_types = scale_types or ('major', 'harmonic_minor', 'whole_tone')
    pitch_class_to_diatonic_scales = get_mapping_from_pitch_class_to_diatonic_scales(scale_types)
    for melodic_line in fragment.melodic_lines:
        pitch_classes = []
        for event in melodic_line[:depth - 1]:
            pitch_classes.append(event.pitch_class)
        for event in melodic_line[depth - 1:]:
            pitch_classes.append(event.pitch_class)
            non_pause_pitch_classes = [x for x in pitch_classes if x != 'pause']
            counter = Counter()
            for pitch_class in non_pause_pitch_classes:
                counter.update(pitch_class_to_diatonic_scales[pitch_class])
            n_pitch_classes_from_best_scale = counter.most_common(1)[0][1]
            numerator -= 1 - n_pitch_classes_from_best_scale / len(pitch_classes)
            denominator += 1
            pitch_classes.pop(0)
    score = numerator / denominator
    return score


def evaluate_motion_to_perfect_consonances(fragment: Fragment) -> float:
    """
    Evaluate absence of direct motion to perfect consonances and absence of successions of them.

    :param fragment:
        a fragment to be evaluated
    :return:
        minus one multiplied by fraction of sonorities with wrong motion to perfect consonances
    """
    score = 0
    previous_events = [None for _ in fragment.melodic_lines]
    for previous_sonority, sonority in zip(fragment.sonorities, fragment.sonorities[1:]):
        zipped = zip(previous_sonority, sonority)
        for line_index, (previous_event, current_event) in enumerate(zipped):
            if previous_event != current_event:
                previous_events[line_index] = previous_event
        sonority_start_time = max(event.start_time for event in sonority)
        pairs = itertools.combinations(sonority, 2)
        for first_event, second_event in pairs:
            if first_event.pitch_class == 'pause' or second_event.pitch_class == 'pause':
                continue
            n_semitones = first_event.position_in_semitones - second_event.position_in_semitones
            is_perfect_fourth_consonant = second_event.line_index != len(sonority) - 1
            interval_type = get_type_of_interval(n_semitones, is_perfect_fourth_consonant)
            if interval_type != IntervalTypes.PERFECT_CONSONANCE:
                continue

            first_previous_event = previous_events[first_event.line_index]
            first_event_continues = (
                first_previous_event is None
                or (
                    first_previous_event.start_time + first_previous_event.duration
                    < sonority_start_time
                )
            )
            second_previous_event = previous_events[second_event.line_index]
            second_event_continues = (
                second_previous_event is None
                or (
                    second_previous_event.start_time + second_previous_event.duration
                    < sonority_start_time
                )
            )
            if first_event_continues and second_event_continues:
                continue

            if first_event_continues:
                first_previous_event = first_event
            elif second_event_continues:
                second_previous_event = second_event

            any_previous_pauses = (
                first_previous_event.pitch_class == 'pause'
                or second_previous_event.pitch_class == 'pause'
            )
            if any_previous_pauses:
                continue
            n_semitones = (
                first_previous_event.position_in_semitones
                - second_previous_event.position_in_semitones
            )
            interval_type = get_type_of_interval(n_semitones, is_perfect_fourth_consonant)
            if interval_type == IntervalTypes.PERFECT_CONSONANCE:
                score -= 1
            if (
                    (
                        first_event.position_in_semitones
                        - first_previous_event.position_in_semitones
                    )
                    * (
                        second_event.position_in_semitones
                        - second_previous_event.position_in_semitones
                    ) > 0
            ):
                score -= 1
    score /= len(fragment.sonorities) - 1
    return score


def evaluate_movement_to_final_sonority(
        fragment: Fragment,
        contrary_motion_term: float = 0.4,
        conjunct_motion_term: float = 0.3,
        bass_downward_skip_term: float = 0.3
) -> float:
    """
    Evaluate sense of finality created by movement to the last sonority.

    :param fragment:
        a fragment to be evaluated
    :param contrary_motion_term:
        contribution of binary indicator whether there is a pair of melodic lines
        having contrary motion to a consonant interval
    :param conjunct_motion_term:
        contribution of binary indicator whether all lines except the bass lines have no skips
    :param bass_downward_skip_term:
        contribution of binary indicator whether bass line moves downward with a skip
    :return:
        minus one multiplied by total contribution of all unsatisfied conditions
    """
    score = 0
    final_pitches = []
    final_moves = []
    bass_indicators = []
    for melodic_line in fragment.melodic_lines:
        penultimate_pitch = melodic_line[-2].position_in_semitones
        final_pitch = melodic_line[-1].position_in_semitones
        if penultimate_pitch is None or final_pitch is None:
            continue
        final_pitches.append(final_pitch)
        final_moves.append(final_pitch - penultimate_pitch)
        bass_indicators.append(melodic_line[-1].line_index == len(fragment.melodic_lines) - 1)

    is_contrary_motion_to_consonance_absent = True
    zipped = zip(final_pitches, final_moves, bass_indicators)
    pairs = itertools.combinations(zipped, 2)
    for (first, first_move, _), (second, second_move, is_bass) in pairs:
        n_semitones = abs(first - second) % N_SEMITONES_PER_OCTAVE
        if is_bass:
            n_semitones_to_interval_type = N_SEMITONES_TO_INTERVAL_TYPE_WITH_DISSONANT_P4
        else:
            n_semitones_to_interval_type = N_SEMITONES_TO_INTERVAL_TYPE_WITH_CONSONANT_P4
        consonant_types = [IntervalTypes.PERFECT_CONSONANCE, IntervalTypes.IMPERFECT_CONSONANCE]
        if n_semitones_to_interval_type[n_semitones] not in consonant_types:
            continue
        if first_move * second_move < 0:
            is_contrary_motion_to_consonance_absent = False
    score -= int(is_contrary_motion_to_consonance_absent) * contrary_motion_term

    if bass_indicators[-1]:
        non_bass_final_moves = final_moves[:-1]
    else:
        non_bass_final_moves = final_moves
    score -= int(any(abs(x) > 2 for x in non_bass_final_moves)) * conjunct_motion_term

    bass_skips_downward = bass_indicators[-1] and final_moves[-1] < -2
    score -= int(not bass_skips_downward) * bass_downward_skip_term

    return score


def evaluate_pitch_class_distribution_among_lines(
        fragment: Fragment, line_id_to_banned_pitch_classes: dict[int, list[str]]
) -> float:
    """
    Evaluate that pitch classes are distributed among lines according to user specifications.

    For example, it is possible to use some pitch classes only in melody and the remaining
    pitch classes only in accompaniment.

    :param fragment:
        a fragment to be evaluated
    :param line_id_to_banned_pitch_classes:
        mapping from ID of a line to pitch classes that are banned for this line
    :return:
        minus one multiplied by fraction of events with improper pitch classes
    """
    score = 0
    for line_id, melodic_line in zip(fragment.line_ids, fragment.melodic_lines):
        current_score = 0
        banned_pitch_classes = line_id_to_banned_pitch_classes.get(line_id, [])
        for event in melodic_line:
            if event.pitch_class in banned_pitch_classes:
                current_score -= 1
        current_score /= len(melodic_line)
        score += current_score
    score /= len(fragment.melodic_lines)
    return score


def find_event_type(
        event: Event, regular_positions: list[dict[str, Any]],
        ad_hoc_positions: list[dict[str, Any]], n_beats: int
) -> str:
    """
    Find type of event based on its start time and, maybe, end time.

    Note that collisions are resolved according to the two following rules:
    1) Ad hoc positions have higher precedence than regular positions;
    2) Precedence amongst either ad hoc positions or regular positions is based on order of
       the positions within the corresponding argument.

    :param event:
        event
    :param regular_positions:
        parameters of regular positions (for example, downbeats or relatively strong beats)
    :param ad_hoc_positions:
        parameters of ad hoc positions which appear just once (for example, the beginning of
        the fragment or the 11th reference beat)
    :param n_beats:
        total duration of a fragment (in reference beats)
    :return:
        type of event based on its start time
    """
    for ad_hoc_position in ad_hoc_positions:
        if ad_hoc_position['time'] < 0:
            ad_hoc_position['time'] += n_beats
        if event.start_time <= ad_hoc_position['time'] <= event.start_time + event.duration:
            return ad_hoc_position['name']
    for regular_position in regular_positions:
        denominator = regular_position['denominator']
        quotient = event.start_time - regular_position['remainder']
        rounded_quotient = int(round(quotient))
        if quotient == rounded_quotient and rounded_quotient % denominator == 0:
            return regular_position['name']
    return 'default'


def evaluate_pitch_class_prominence(
        fragment: Fragment, pitch_class_to_prominence_range: dict[str, tuple[float, float]],
        regular_positions: list[dict[str, Any]], ad_hoc_positions: list[dict[str, Any]],
        event_type_to_weight: dict[str, float], default_weight: float = 1
) -> float:
    """
    Evaluate that accents are distributed among pitch classes according to user specifications.

    :param fragment:
        a fragment to be evaluated
    :param pitch_class_to_prominence_range:
        mapping from pitch class to a pair of minimum and maximum allowed prominences
    :param regular_positions:
        parameters of regular positions (for example, downbeats or relatively strong beats)
    :param ad_hoc_positions:
        parameters of ad hoc positions which appear just once (for example, the beginning of
        the fragment or the 11th reference beat)
    :param event_type_to_weight:
        mapping from type of event start time to multiplicative weight for its prominence
    :param default_weight:
        weight that is used for event types with unspecified weights
    :return:
        minus one multiplied by summed over pitch classes absolute deviation of their prominence
        from corresponding ranges
    """
    prominences = {pitch_class: 0 for pitch_class in PITCH_CLASS_TO_POSITION}
    n_beats = fragment.n_beats
    for melodic_line in fragment.melodic_lines:
        for event in melodic_line:
            if event.pitch_class == 'pause':
                continue
            event_type = find_event_type(event, regular_positions, ad_hoc_positions, n_beats)
            weight = event_type_to_weight.get(event_type, default_weight)
            prominence = weight * event.duration
            prominences[event.pitch_class] += prominence
    total_prominence = sum(v for k, v in prominences.items())
    prominences = {k: v / total_prominence for k, v in prominences.items()}
    score = 0
    for pitch_class, prominence in prominences.items():
        lower_bound, upper_bound = pitch_class_to_prominence_range.get(pitch_class, (0, 1))
        score -= max(lower_bound - prominence, 0) + max(prominence - upper_bound, 0)
    return score


@cache
def get_mapping_from_interval_size_to_character() -> dict[Optional[int], str]:
    """
    Get mapping from interval size (in semitones) to its single-char encoding.

    :return:
        mapping from size (in semitones) of a directed non-compound interval to
        its single-char representation; `None` as key relates to pauses
    """
    intervals = list(range(-N_SEMITONES_PER_OCTAVE, N_SEMITONES_PER_OCTAVE + 1)) + [None]
    result = {interval: letter for interval, letter in zip(intervals, string.ascii_lowercase)}
    return result


def encode_interval(interval: Optional[int]) -> str:
    """
    Encode an interval with a single character.

    :param interval:
        interval size (in semitones)
    :return:
        single-character encoding of the interval
    """
    mapping = get_mapping_from_interval_size_to_character()
    if interval is not None:
        interval = math.copysign(abs(interval) % N_SEMITONES_PER_OCTAVE, interval)
    return mapping[interval]


def encode_line_intervals(melodic_lines: list[list[Event]]) -> list[str]:
    """
    Encode intervals from all melodic lines

    :param melodic_lines:
        melodic lines
    :return:
        strings with encoded intervals for each melodic line
    """
    encoded_lines = []
    for melodic_line in melodic_lines:
        encoded_line = ''
        for previous_event, next_event in zip(melodic_line, melodic_line[1:]):
            if previous_event.pitch_class == 'pause' or next_event.pitch_class == 'pause':
                interval = None
            else:
                interval = next_event.position_in_semitones - previous_event.position_in_semitones
            encoded_line += encode_interval(interval)
        encoded_lines.append(encoded_line)
    return encoded_lines


def generate_elision_patterns(
        motif: tuple[int], original_pattern: str, run: bool = False
) -> list[str]:
    """
    Generate search patterns for modifications of the original motif with one pitch omitted.

    :param motif:
        sequence of directed interval sizes (in semitones) between successive pitches of a motif
    :param original_pattern:
        search pattern (in encoded lines) corresponding to the motif
    :param run:
        flag whether to generate any patterns
    :return:
        generated patterns corresponding to elisions of the motif
    """
    patterns = []
    if not run:
        return patterns
    not_first = f'[^{original_pattern[0]}]'
    first_omission_pattern = not_first + original_pattern[1:]
    patterns.append(first_omission_pattern)
    for i in range(len(motif) - 1):
        motif_with_omission = motif[:i] + (motif[i] + motif[i + 1],) + motif[i + 2:]
        omission_pattern = ''.join([encode_interval(x) for x in motif_with_omission])
        patterns.append(omission_pattern)
    not_last = f'[^{original_pattern[-1]}]'
    last_omission_pattern = original_pattern[:-1] + not_last
    patterns.append(last_omission_pattern)
    return patterns


@cache
def generate_regexps_for_intervallic_motif(
        motif: tuple[int],
        inversion: bool = True, reversion: bool = True,
        elision: bool = False, inverted_elision: bool = False, reverted_elision: bool = False
) -> list['_sre.SRE_Pattern']:
    """
    Generate regular expressions that match intervallic motif and its modifications.

    :param motif:
        sequence of directed interval sizes (in semitones) between successive pitches of a motif
    :param inversion:
        flag whether to include inversion of the original motif
    :param reversion:
        flag whether to include reversion of the original motif
    :param elision:
        flag whether to include modifications of the original motif with one pitch omitted
    :param inverted_elision:
        flag whether to include modifications of the inverted motif with one pitch omitted;
        if `inversion` is set to `False`, this flag affects nothing
    :param reverted_elision:
        flag whether to include modifications of the reverted motif with one pitch omitted;
        if `reversion` is set to `False`, this flag affects nothing
    :return:
        compiled regular expressions
    """
    original_pattern = ''.join([encode_interval(x) for x in motif])
    patterns = [original_pattern]
    patterns.extend(generate_elision_patterns(motif, original_pattern, elision))
    if inversion:
        inverted_motif = tuple(-x for x in motif)
        inverted_pattern = ''.join([encode_interval(x) for x in inverted_motif])
        patterns.append(inverted_pattern)
        patterns.extend(
            generate_elision_patterns(inverted_motif, inverted_pattern, inverted_elision)
        )
    if reversion:
        reverted_motif = motif[::-1]
        reverted_pattern = original_pattern[::-1]
        patterns.append(reverted_pattern)
        patterns.extend(
            generate_elision_patterns(reverted_motif, reverted_pattern, reverted_elision)
        )
    patterns = list(set(patterns))  # In particular, duplicates may occur due to symmetries.
    regexps = [re.compile(pattern) for pattern in patterns]
    return regexps


def evaluate_presence_of_intervallic_motif(
        fragment: Fragment, motif: list[int], min_n_occurrences: list[int],
        inversion: bool = True, reversion: bool = True,
        elision: bool = False, inverted_elision: bool = False, reverted_elision: bool = False
) -> float:
    """
    Evaluate presence of used-defined intervallic motif.

    :param fragment:
        a fragment to be evaluated
    :param motif:
        sequence of directed interval sizes (in semitones) between successive pitches of a motif
    :param min_n_occurrences:
        minimum numbers of motif occurrences for each melodic line
    :param inversion:
        flag whether to include inversion of the original motif
    :param reversion:
        flag whether to include reversion of the original motif
    :param elision:
        flag whether to include modifications of the original motif with one pitch omitted
    :param inverted_elision:
        flag whether to include modifications of the inverted motif with one pitch omitted;
        if `inversion` is set to `False`, this flag affects nothing
    :param reverted_elision:
        flag whether to include modifications of the reverted motif with one pitch omitted;
        if `reversion` is set to `False`, this flag affects nothing
    :return:
        minus one multiplied by relative lack of motif occurrences
    """
    regexps = generate_regexps_for_intervallic_motif(
        tuple(motif), inversion, reversion, elision, inverted_elision, reverted_elision
    )
    encoded_lines = encode_line_intervals(fragment.melodic_lines)
    numerator = 0
    denominator = 0
    for threshold, encoded_line in zip(min_n_occurrences, encoded_lines):
        n_occurrences = 0
        for regexp in regexps:
            n_occurrences += len(re.findall(regexp, encoded_line))
        numerator -= max(threshold - n_occurrences, 0)
        denominator += threshold
    score = numerator / denominator
    return score


def evaluate_presence_of_required_pauses(
        fragment: Fragment, pauses: list[tuple[float, float]]
) -> float:
    """
    Evaluate presence of pauses required by a user.

    :param fragment:
        a fragment to be evaluated
    :param pauses:
        list of pairs of start time and end time for user-defined pauses
    :return:
        minus one multiplied by share of declared pauses occupied with sounds
    """
    numerator = 0
    denominator = 0
    for melodic_line in fragment.melodic_lines:
        event_index = 0
        for pause_start_time, pause_end_time in pauses:
            while True:
                event = melodic_line[event_index]
                event_end_time = event.start_time + event.duration
                if event_end_time <= pause_start_time:
                    event_index += 1
                    continue
                intersection_start_time = max(event.start_time, pause_start_time)
                intersection_end_time = min(event_end_time, pause_end_time)
                intersection_duration = intersection_end_time - intersection_start_time
                if event.pitch_class != 'pause':
                    numerator += intersection_duration
                denominator += intersection_duration
                if event_end_time >= pause_end_time:
                    break
                event_index += 1
    score = -numerator / denominator
    return score


def evaluate_presence_of_vertical_intervals(
        fragment: Fragment, intervals: list[int], min_n_weighted_occurrences: float,
        regular_positions: list[dict[str, Any]], ad_hoc_positions: list[dict[str, Any]],
        position_weights: dict[str, float]
) -> float:
    """
    Evaluate presence of vertical intervallic sonorities responsible for global consistency.

    :param fragment:
        a fragment to be evaluated
    :param intervals:
        intervals (in semitones) from top to bottom
    :param min_n_weighted_occurrences:
        minimal sum of weights of intervallic sonorities occurrences
    :param regular_positions:
        parameters of regular positions (for example, downbeats or relatively strong beats)
    :param ad_hoc_positions:
        parameters of ad hoc positions which appear just once (for example, the beginning of
        the fragment or the 11th reference beat)
    :param position_weights:
        mapping from position name to its weight
    :return:
        minus one multiplied by fraction of lacking occurrences weight
    """
    weighted_n_occurrences = 0
    for sonority in fragment.sonorities:
        actual_intervals = []
        for upper_event, lower_event in zip(sonority, sonority[1:]):
            if upper_event.pitch_class == "pause" or lower_event.pitch_class == "pause":
                break
            interval = upper_event.position_in_semitones - lower_event.position_in_semitones
            actual_intervals.append(interval)
        if actual_intervals != intervals:
            continue
        sonority_start = max(event.start_time for event in sonority)
        sonority_end = min(event.start_time + event.duration for event in sonority)
        position_type = find_sonority_type(
            sonority_start, sonority_end, regular_positions, ad_hoc_positions, fragment.n_beats
        )
        weighted_n_occurrences += position_weights[position_type]
    score = min(weighted_n_occurrences - min_n_weighted_occurrences, 0)
    score /= min_n_weighted_occurrences
    return score


def evaluate_rhythmic_homogeneity(fragment: Fragment) -> float:
    """
    Evaluate rhythmic homogeneity among all measures except the last one.

    :param fragment:
        a fragment to be evaluated
    :return:
        a score between minis one and zero depending on rhythmic variation
    """
    score = 0
    for durations_of_measures_for_one_line in fragment.temporal_content:
        pairs = itertools.combinations(durations_of_measures_for_one_line[:-1], 2)
        for first_durations, second_durations in pairs:
            first_end_times = list(itertools.accumulate(first_durations))
            second_end_times = list(itertools.accumulate(second_durations))
            avg_n_events = (len(first_end_times) + len(second_end_times)) / 2
            n_unique_ends = len(set(first_end_times + second_end_times))
            score -= (n_unique_ends / avg_n_events - 1)
    n_non_last_measures = (fragment.n_beats // fragment.meter_numerator) - 1
    n_pairs = n_non_last_measures * (n_non_last_measures - 1) / 2
    score /= len(fragment.melodic_lines) * n_pairs
    return score


@cache
def find_normalization_coefficient_for_rhythmic_intensity(
        n_beats: float, half_life: float, max_intensity_factor: float
) -> float:
    """
    Find maximum possible value of exponentially decaying counter of sound events.

    :param n_beats:
        duration of a melodic line (in reference beats)
    :param half_life:
        half life (in reference beats) of exponentially decaying counter of sound events
    :param max_intensity_factor:
        factor that rescales maximum possible intensity (given high enough half life,
        its unscaled value might be too high, because it is calculated at the last event
        of the line consisting entirely of the shortest possible events)
    :return:
        normalization coefficient for a single melodic line
    """
    min_event_duration = 0.25  # TODO: Calculate this value as min value from `measure_durations`.
    max_possible_n_events = n_beats / min_event_duration
    decay_coef = 0.5 ** (min_event_duration / half_life)
    max_intensity = (1 - decay_coef ** max_possible_n_events) / (1 - decay_coef)
    normalization_coefficient = max_intensity_factor * max_intensity
    return normalization_coefficient


def evaluate_rhythmic_intensity_by_positions(
        fragment: Fragment, positions: list[float], ranges: list[list[tuple[float, float]]],
        half_life: float, max_intensity_factor: float = 1
) -> float:
    """
    Evaluate values of rhythmic intensity at particular time moments.

    :param fragment:
        a fragment to be evaluated
    :param positions:
        moments of time (in reference beats) at which rhythmic intensity is evaluated
    :param ranges:
        minimum and maximum desired levels of rhythmic intensity for each position and
        for each melodic line
    :param half_life:
        half life (in reference beats) of exponentially decaying counter of sound events
    :param max_intensity_factor:
        factor that rescales maximum possible intensity (given high enough half life,
        its unscaled value might be too high, because it is calculated at the last event
        of the line consisting entirely of the shortest possible events)
    :return:
        average over all positions deviation of rhythmic intensity from its ranges
    """
    score = 0
    normalization_coef = find_normalization_coefficient_for_rhythmic_intensity(
        fragment.n_beats, half_life, max_intensity_factor
    )
    for melodic_line, line_ranges in zip(fragment.melodic_lines, ranges):
        events_counter = 0
        previous_time_moment = 0
        annotated_positions = [
            (position, True, current_range)
            for position, current_range in zip(positions, line_ranges)
        ]
        event_starts = [
            (event.start_time, False, (0.0, 0.0))  # The last value of this tuple affects nothing.
            for event in melodic_line if event.pitch_class != 'pause'
        ]
        all_time_moments = sorted(annotated_positions + event_starts)
        for time_moment, evaluation_needed, current_range in all_time_moments:
            time_diff = time_moment - previous_time_moment
            events_counter *= 0.5 ** (time_diff / half_life)
            previous_time_moment = time_moment
            if evaluation_needed:
                normalized_counter = events_counter / normalization_coef
                score += min(normalized_counter - current_range[0], 0)
                score += min(current_range[1] - normalized_counter, 0)
            else:
                events_counter += 1
    score /= len(fragment.melodic_lines) * len(positions)
    return score


def evaluate_smoothness_of_voice_leading(
        fragment: Fragment,
        penalty_deduction_per_line: float,
        n_semitones_to_penalty: dict[int, float]
) -> float:
    """
    Evaluate presence of coherent melodic lines that move without large leaps.

    :param fragment:
        a fragment to be evaluated
    :param penalty_deduction_per_line:
        amount of averaged leaps penalty that is deducted for each melodic line
    :param n_semitones_to_penalty:
        mapping from size of melodic interval in semitones to penalty for it
    :return:
        average over melodic lines penalty
    """
    score = 0
    for line in fragment.melodic_lines:
        curr_score = 0
        line_without_pauses = [event for event in line if event.pitch_class != 'pause']
        for first, second in zip(line_without_pauses, line_without_pauses[1:]):
            melodic_interval = abs(first.position_in_semitones - second.position_in_semitones)
            curr_score -= n_semitones_to_penalty.get(melodic_interval, 1.0)
        curr_score /= (len(line_without_pauses) - 1)
        curr_score = min(curr_score + penalty_deduction_per_line, 0)
        score += curr_score
    score /= len(fragment.melodic_lines) * (1 - penalty_deduction_per_line)
    return score


def evaluate_sonic_intensity_by_positions(
        fragment: Fragment, positions: list[float], ranges: list[tuple[float, float]]
) -> float:
    """
    Evaluate sonic intensity (number of non-pause events) at particular time moments.

    :param fragment:
        a fragment to be evaluated
    :param positions:
        moments of time (in reference beats) at which sonic intensity is evaluated
    :param ranges:
        minimum and maximum number of non-pause events for each position
    :return:
        average over positions deviation of sonic intensity from its ranges
    """
    score = 0
    sonority_index = -1
    sonority_end = -1
    for position, (min_n_non_pause_events, max_n_non_pause_events) in zip(positions, ranges):
        while sonority_end <= position:
            sonority_index += 1
            sonority = fragment.sonorities[sonority_index]
            sonority_end = min(event.start_time + event.duration for event in sonority)
        n_non_pause_events = len([event for event in sonority if event.pitch_class != 'pause'])
        score -= max(0, min_n_non_pause_events - n_non_pause_events)
        score -= max(0, n_non_pause_events - max_n_non_pause_events)
    score /= len(positions)
    return score


def evaluate_stackability(fragment: Fragment, n_semitones_to_penalty: dict[int, float]) -> float:
    """
    Evaluate seamlessness of repeating a fragment after itself.

    Two copies of a fragment may be stacked together in ostinato or in a period consisting of
    two phrases each of which is the fragment.

    :param fragment:
        a fragment to be evaluated
    :param n_semitones_to_penalty:
        mapping from size of melodic interval (in semitones) between last and first events
        of a melodic line to penalty for it
    :return:
        average over melodic lines penalty
    """
    score = 0
    for line in fragment.melodic_lines:
        first_event = line[0]
        if first_event.pitch_class == 'pause':
            continue
        last_event = line[-1]
        if last_event.pitch_class == 'pause':
            continue
        interval = abs(first_event.position_in_semitones - last_event.position_in_semitones)
        score -= n_semitones_to_penalty.get(interval, 1.0)
    score /= len(fragment.melodic_lines)
    return score


def evaluate_transitions(
        fragment: Fragment, n_semitones_to_penalty: dict[int, float],
        left_end_notes: Optional[list[str]] = None, right_end_notes: Optional[list[str]] = None
) -> float:
    """
    Evaluate seamlessness of placing a fragment after and/or before the other fragments.

    :param fragment:
        a fragment to be evaluated
    :param n_semitones_to_penalty:
        mapping from size of melodic interval (in semitones) between last and first events
        of a melodic line to penalty for it
    :param left_end_notes:
        list of notes such that the previous fragment ends with sonority made of them
    :param right_end_notes:
        list of notes such that the next fragment starts with sonority made of them
    :return:
        average over melodic lines penalty
    """
    score = 0
    denominator_factor = 0
    if left_end_notes is not None:
        denominator_factor += 1
        for previous_note, line in zip(left_end_notes, fragment.melodic_lines):
            first_event = line[0]
            if previous_note == 'pause' or first_event.pitch_class == 'pause':
                continue
            previous_position_in_semitones = NOTE_TO_POSITION_MAPPING[previous_note]
            interval = abs(first_event.position_in_semitones - previous_position_in_semitones)
            score -= n_semitones_to_penalty.get(interval, 1.0)
    if right_end_notes is not None:
        denominator_factor += 1
        for next_note, line in zip(right_end_notes, fragment.melodic_lines):
            last_event = line[-1]
            if next_note == 'pause' or last_event.pitch_class == 'pause':
                continue
            next_position_in_semitones = NOTE_TO_POSITION_MAPPING[next_note]
            interval = abs(last_event.position_in_semitones - next_position_in_semitones)
            score -= n_semitones_to_penalty.get(interval, 1.0)
    score /= ((denominator_factor or 1) * len(fragment.melodic_lines))
    return score


def get_scoring_functions_registry() -> dict[str, Callable]:
    """
    Get mapping from names to corresponding scoring functions.

    :return:
        registry of scoring functions
    """
    registry = {
        'absence_of_aimless_fluctuations': evaluate_absence_of_aimless_fluctuations,
        'absence_of_doubled_pitch_classes': evaluate_absence_of_doubled_pitch_classes,
        'absence_of_simultaneous_skips': evaluate_absence_of_simultaneous_skips,
        'absence_of_voice_crossing': evaluate_absence_of_voice_crossing,
        'cadence_duration': evaluate_cadence_duration,
        'climax_explicity': evaluate_climax_explicity,
        'direction_change_after_large_skip': evaluate_direction_change_after_large_skip,
        'dissonances_preparation_and_resolution': evaluate_dissonances_preparation_and_resolution,
        'harmony_dynamic_by_positions': evaluate_harmony_dynamic_by_positions,
        'harmony_dynamic_by_time_intervals': evaluate_harmony_dynamic_by_time_intervals,
        'intervallic_motif': evaluate_presence_of_intervallic_motif,
        'local_diatonicity_at_all_lines_level': evaluate_local_diatonicity_at_all_lines_level,
        'local_diatonicity_at_line_level': evaluate_local_diatonicity_at_line_level,
        'motion_to_perfect_consonances': evaluate_motion_to_perfect_consonances,
        'movement_to_final_sonority': evaluate_movement_to_final_sonority,
        'pitch_class_distribution': evaluate_pitch_class_distribution_among_lines,
        'pitch_class_prominence': evaluate_pitch_class_prominence,
        'presence_of_required_pauses': evaluate_presence_of_required_pauses,
        'presence_of_vertical_intervals': evaluate_presence_of_vertical_intervals,
        'rhythmic_homogeneity': evaluate_rhythmic_homogeneity,
        'rhythmic_intensity': evaluate_rhythmic_intensity_by_positions,
        'smoothness_of_voice_leading': evaluate_smoothness_of_voice_leading,
        'sonic_intensity': evaluate_sonic_intensity_by_positions,
        'stackability': evaluate_stackability,
        'transitions': evaluate_transitions,
    }
    return registry


def parse_scoring_sets_registry(params: list[dict[str, Any]]) -> SCORING_SETS_REGISTRY_TYPE:
    """
    Parse mapping from names of scoring sets to scoring sets itself.

    :param params:
        raw parameters of scoring sets obtained from a config
    :return:
        mapping from a name of a scoring set to a list of triples of a scoring function,
        its weight, and its parameters
    """
    scoring_functions_registry = get_scoring_functions_registry()
    scoring_sets_registry = {}
    for scoring_set_params in params:
        scoring_set_name = scoring_set_params['name']
        scoring_fns = []
        for scoring_fn_info in scoring_set_params['scoring_functions']:
            scoring_fn = scoring_functions_registry[scoring_fn_info.pop('name')]
            weights = scoring_fn_info.pop('weights')
            scoring_fns.append((scoring_fn, weights, scoring_fn_info))
        scoring_sets_registry[scoring_set_name] = scoring_fns
    return scoring_sets_registry


def weight_score(unweighted_score: float, weights: dict[float, float]) -> float:
    """
    Transform original score by a piecewise linear function.

    :param unweighted_score:
        original score
    :param weights:
        mapping from a breakpoint to a slope coefficient for an interval from the breakpoint to
        the next-to-the-left breakpoint (or -1 if it is absent)
    :return:
        weighted score
    """
    weighted_score = 0
    breakpoints = list(weights.keys()) + [-1.0]
    for left_point, right_point in zip(breakpoints[1:], breakpoints):
        if right_point <= unweighted_score:
            break
        weighted_score -= (right_point - max(unweighted_score, left_point)) * weights[right_point]
    return weighted_score


def evaluate(
        fragment: Fragment,
        scoring_sets: list[str],
        scoring_sets_registry: SCORING_SETS_REGISTRY_TYPE,
        report: bool = False
) -> tuple[float, str]:
    """
    Evaluate fragment and report results.

    :param fragment:
        fragment to be evaluated
    :param scoring_sets:
        names of scoring sets to be used
    :param scoring_sets_registry:
        mapping from a name of a scoring set to a list of triples of a scoring function,
        its weight, and its parameters
    :param report:
        if it is set to `True`, detailed by functions scores are returned as a second output
    :return:
        weighted sum of scores returned by applicable scoring functions
    """
    score = 0
    report_lines = []
    for scoring_set_name in scoring_sets:
        scoring_set = scoring_sets_registry[scoring_set_name]
        for scoring_fn, weights, params in scoring_set:
            unweighted_score = scoring_fn(fragment, **params)
            curr_score = weight_score(unweighted_score, weights)
            if report:
                fn_name = scoring_fn.__name__.removeprefix('evaluate_')
                report_line = f'{fn_name:>40}: {curr_score}'
                report_lines.append(report_line)
            score += curr_score
    if report:
        report_lines.append(f'Overall score is: {score}')
        report_str = '\n'.join(report_lines)
    else:
        report_str = ""
    return score, report_str
