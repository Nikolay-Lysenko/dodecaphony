"""
Evaluate rhythmic properties of a fragment.

Author: Nikolay Lysenko
"""


import itertools
from functools import cache

from dodecaphony.fragment import Fragment


def evaluate_cadence_duration(
        fragment: Fragment,
        min_desired_duration: float,
        last_sonority_weight: float,
        last_notes_weight: float
) -> float:
    """
    Evaluate that cadence has enough duration.

    :param fragment:
        a fragment to be evaluated
    :param min_desired_duration:
        minimum desired duration (in reference beats) of the last sonority;
        higher durations do not increase the score
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
    last_sonority_events = fragment.sonorities[-1].events
    clipped_durations = [min(event.duration, min_desired_duration) for event in last_sonority_events]
    last_sonority_duration = min(clipped_durations)
    avg_last_note_duration = sum(clipped_durations) / len(clipped_durations)
    total_weight = last_sonority_weight + last_notes_weight
    last_sonority_weight /= total_weight
    last_notes_weight /= total_weight
    last_sonority_term = last_sonority_weight * (last_sonority_duration / min_desired_duration - 1)
    last_notes_term = last_notes_weight * (avg_last_note_duration / min_desired_duration - 1)
    score = last_sonority_term + last_notes_term
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


def evaluate_rhythmic_homogeneity(fragment: Fragment) -> float:
    """
    Evaluate rhythmic homogeneity among all measures except the last one.

    :param fragment:
        a fragment to be evaluated
    :return:
        a score between minus one and zero depending on rhythmic variation
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
