"""
Evaluate harmonic properties of a fragment.

Author: Nikolay Lysenko
"""


import itertools
import math
from collections import Counter
from typing import Any, Optional

from dodecaphony.fragment import Event, Fragment
from dodecaphony.music_theory import (
    IntervalTypes,
    N_SEMITONES_PER_OCTAVE,
    get_mapping_from_pitch_class_to_diatonic_scales,
    get_type_of_interval,
)


def evaluate_absence_of_doubled_pitch_classes(fragment: Fragment) -> float:
    """
    Evaluate absence of vertical intervals that are multiples of an octave (except unisons).

    If the same pitch class sounds in two or more simultaneous pitches,
    harmonic stability increases significantly and this interrupts musical flow
    associated with the twelve-tone technique.

    :param fragment:
        a fragment to be evaluated
    :return:
        minus one multiplied by number of pairs of simultaneously sounding events
        of the same pitch class and divided by total number of sonorities
    """
    score = 0
    for sonority in fragment.sonorities:
        for first, second in itertools.combinations(sonority.non_pause_events, 2):
            interval = first.position_in_semitones - second.position_in_semitones
            if interval != 0 and interval % N_SEMITONES_PER_OCTAVE == 0:
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
        for first, second in zip(first_sonority.events, second_sonority.events):
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
        for first, second in itertools.combinations(sonority.non_pause_events, 2):
            interval = first.position_in_semitones - second.position_in_semitones
            if interval <= 0:
                numerator -= n_semitones_to_penalty.get(interval, 1)
            denominator += 1
    score = numerator / denominator
    return score


def find_indices_of_dissonating_events(
        sonority_events: list[Event], sonority_start_time: float,
        n_melodic_lines: int, meter_numerator: int
) -> tuple[set[int], set[int]]:
    """
    Find indices of dissonating (i.e., dependent, non-free in terms of strict counterpoint) events.

    :param sonority_events:
        simultaneously sounding events (without pauses)
    :param sonority_start_time:
        start time of the sonority
    :param n_melodic_lines:
        total number of melodic lines in the fragment
    :param meter_numerator:
        numerator in meter signature, i.e., number of reference beats per measure
    :return:
        indices of passing tones or neighbor dissonances and indices of suspended dissonances
    """
    passing_tones_and_neighbors = set()
    suspensions = set()
    for first_event, second_event in itertools.combinations(sonority_events, 2):
        n_semitones = first_event.position_in_semitones - second_event.position_in_semitones
        is_perfect_fourth_consonant = second_event.line_index != n_melodic_lines - 1
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
        zipped = zip(sonority.events, fragment.melodic_lines, event_indices)
        for event, melodic_line, event_index in zipped:
            if event != melodic_line[event_index]:
                event_indices[event.line_index] += 1
        pt_and_ngh_line_indices, suspension_line_indices = find_indices_of_dissonating_events(
            sonority.non_pause_events, sonority.start_time,
            len(fragment.melodic_lines), fragment.meter_numerator
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
        sonority_events: list[Event], n_semitones_to_stability: dict[int, float]
) -> float:
    """
    Compute stability of sonority as average stability of intervals forming it.

    :param sonority_events:
        simultaneously sounding events (without pauses)
    :param n_semitones_to_stability:
        mapping from interval size in semitones to its harmonic stability
    :return:
        average stability of intervals forming the sonority
    """
    if len(sonority_events) <= 1:
        return 1.0
    stability = 0
    for first, second in itertools.combinations(sonority_events, 2):
        interval_in_semitones = abs(first.position_in_semitones - second.position_in_semitones)
        interval_in_semitones %= N_SEMITONES_PER_OCTAVE
        stability += n_semitones_to_stability[interval_in_semitones]
    n_pairs = len(sonority_events) * (len(sonority_events) - 1) / 2
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
        stability_of_current_sonority = compute_harmonic_stability_of_sonority(
            sonority.non_pause_events, n_semitones_to_stability
        )
        sonority_type = find_sonority_type(
            sonority.start_time, sonority.end_time, regular_positions, ad_hoc_positions,
            fragment.n_beats
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
            if sonority.end_time <= interval_start:
                sonority_index += 1
                continue
            intersection_start = max(sonority.start_time, interval_start)
            intersection_end = min(sonority.end_time, interval_end)
            intersection_duration = intersection_end - intersection_start

            stability_of_current_sonority = compute_harmonic_stability_of_sonority(
                sonority.non_pause_events, n_semitones_to_stability
            )
            deviation = 0
            deviation += min(stability_of_current_sonority - min_allowed_value, 0)
            deviation += min(max_allowed_value - stability_of_current_sonority, 0)
            numerator += intersection_duration * deviation
            denominator += intersection_duration

            if sonority.end_time >= interval_end:
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
        nested_pitch_classes.append([event.pitch_class for event in sonority.non_pause_events])
    for sonority in fragment.sonorities[depth - 1:]:
        nested_pitch_classes.append([event.pitch_class for event in sonority.non_pause_events])
        pitch_classes = [x for y in nested_pitch_classes for x in y]
        counter = Counter()
        for pitch_class in pitch_classes:
            counter.update(pitch_class_to_diatonic_scales[pitch_class])
        n_pitch_classes_from_best_scale = counter.most_common(1)[0][1]
        score -= 1 - n_pitch_classes_from_best_scale / len(pitch_classes)
        nested_pitch_classes.pop(0)
    n_periods = len(fragment.sonorities) - depth + 1
    score /= n_periods
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
        zipped = zip(previous_sonority.events, sonority.events)
        for line_index, (previous_event, current_event) in enumerate(zipped):
            if previous_event != current_event:
                previous_events[line_index] = previous_event
        pairs = itertools.combinations(sonority.non_pause_events, 2)
        for first_event, second_event in pairs:
            n_semitones = first_event.position_in_semitones - second_event.position_in_semitones
            is_perfect_fourth_consonant = second_event.line_index != len(sonority.events) - 1
            interval_type = get_type_of_interval(n_semitones, is_perfect_fourth_consonant)
            if interval_type != IntervalTypes.PERFECT_CONSONANCE:
                continue

            first_previous_event = previous_events[first_event.line_index]
            first_event_continues = (
                first_previous_event is None
                or (
                    first_previous_event.start_time + first_previous_event.duration
                    < sonority.start_time
                )
            )
            second_previous_event = previous_events[second_event.line_index]
            second_event_continues = (
                second_previous_event is None
                or (
                    second_previous_event.start_time + second_previous_event.duration
                    < sonority.start_time
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
        contribution of binary indicator whether all lines have no skips;
        the bass line is excluded if `bass_downward_skip_term` is greater than zero
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
    consonant_types = [IntervalTypes.PERFECT_CONSONANCE, IntervalTypes.IMPERFECT_CONSONANCE]
    zipped = zip(final_pitches, final_moves, bass_indicators)
    pairs = itertools.combinations(zipped, 2)
    for (first, first_move, _), (second, second_move, is_bass) in pairs:
        interval_type = get_type_of_interval(first - second, not is_bass)
        if interval_type not in consonant_types:
            continue
        if first_move * second_move < 0:
            is_contrary_motion_to_consonance_absent = False
            break
    score -= int(is_contrary_motion_to_consonance_absent) * contrary_motion_term

    if bass_indicators[-1] and bass_downward_skip_term > 0:
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


def evaluate_presence_of_vertical_intervals(
        fragment: Fragment, intervals: list[int], min_n_weighted_occurrences: float,
        regular_positions: list[dict[str, Any]], ad_hoc_positions: list[dict[str, Any]],
        position_weights: dict[str, float]
) -> float:
    """
    Evaluate presence of vertical intervallic sonorities responsible for global coherence.

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
        non_pause_events = sonority.non_pause_events
        for upper_event, lower_event in zip(non_pause_events, non_pause_events[1:]):
            interval = upper_event.position_in_semitones - lower_event.position_in_semitones
            actual_intervals.append(interval)
        if actual_intervals != intervals:
            continue
        position_type = find_sonority_type(
            sonority.start_time, sonority.end_time, regular_positions, ad_hoc_positions,
            fragment.n_beats
        )
        weighted_n_occurrences += position_weights[position_type]
    score = min(weighted_n_occurrences - min_n_weighted_occurrences, 0)
    score /= min_n_weighted_occurrences
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
    sonority_index = 0
    sonority = fragment.sonorities[sonority_index]
    for position, (min_n_non_pause_events, max_n_non_pause_events) in zip(positions, ranges):
        while sonority.end_time <= position:
            sonority_index += 1
            sonority = fragment.sonorities[sonority_index]
        n_non_pause_events = len(sonority.non_pause_events)
        score -= max(0, min_n_non_pause_events - n_non_pause_events)
        score -= max(0, n_non_pause_events - max_n_non_pause_events)
    score /= len(positions)
    return score
