"""
Evaluate a fragment.

Author: Nikolay Lysenko
"""


import itertools
import math
from typing import Any, Callable, Optional

from .fragment import Event, Fragment
from .music_theory import IntervalTypes, N_SEMITONES_PER_OCTAVE, get_type_of_interval


SCORING_SETS_REGISTRY_TYPE = dict[
    str,
    list[tuple[Callable[..., float], dict[float, float], dict[str, Any]]]
]


def evaluate_absence_of_doubled_pitch_classes(fragment: Fragment) -> float:
    """
    Evaluate absence of vertical intervals that are whole multipliers of octave.

    If the same pitch class sounds in two or more melodic lines simultaneously,
    this pitch class might be perceived as 'central' and so listeners may expect that the piece is
    tonal, whereas it is not.

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
        fragment: Fragment, min_skip_in_semitones: int = 5, max_skips_share: float = 0.65
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


def evaluate_absence_of_voice_crossing(fragment: Fragment) -> float:
    """
    Evaluate absence of voice crossing.

    Voice crossing may result in wrong perception of tone row and in incoherence of the fragment.

    :param fragment:
        a fragment to be evaluated
    :return:
        minus one multiplied by sum of sizes in semitones of wrong vertical intervals
        and divided by total number of vertical intervals
    """
    score = 0
    for sonority in fragment.sonorities:
        for first, second in itertools.combinations(sonority, 2):
            if first.pitch_class == 'pause' or second.pitch_class == 'pause':
                continue
            interval = first.position_in_semitones - second.position_in_semitones
            if interval < 0:
                score += interval
    score /= len(fragment.sonorities)
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


def evaluate_consistency_of_rhythm_with_meter(
        fragment: Fragment, consistent_patterns: list[list[float]]
) -> float:
    """
    Evaluate ease of deriving meter from rhythms of melodic lines.

    :param fragment:
        a fragment to be evaluated
    :param consistent_patterns:
        list of options to split a measure time span into time spans of individual notes
        considered to be consistent with meter
    :return:
        minus one multiplied by number of measures which are split in inconsistent manner
    """
    score = 0
    for durations_of_measures_for_one_line in fragment.durations_of_measures:
        for durations_of_measure in durations_of_measures_for_one_line:
            if durations_of_measure not in consistent_patterns:
                score -= 1
    score /= len(fragment.melodic_lines) * (fragment.n_beats // fragment.meter_numerator)
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
        -1 if interval of arrival to the event is needed or
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
        ratio = math.ceil(sonority_start) // denominator
        processed_start = sonority_start - ratio * denominator
        processed_end = sonority_end - ratio * denominator
        current_time = regular_position['remainder']
        while current_time < processed_end:
            if current_time >= processed_start:
                return regular_position['name']
            current_time += denominator
    return 'default'


def evaluate_harmony_dynamic(
        fragment: Fragment, regular_positions: list[dict[str, Any]],
        ad_hoc_positions: list[dict[str, Any]], ranges: dict[str, tuple[float, float]],
        n_semitones_to_stability: dict[int, float]
) -> float:
    """
    Evaluate dynamic of harmonic stability/tension relatively user-defined desired ranges.

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


def evaluate_rhythmic_homogeneity(fragment: Fragment) -> float:
    """
    Evaluate rhythmic homogeneity between all measures except the last one.

    :param fragment:
        a fragment to be evaluated
    :return:
        a score between minis one and zero depending on rhythmic variation
    """
    score = 0
    for durations_of_measures_for_one_line in fragment.durations_of_measures:
        pairs = itertools.combinations(durations_of_measures_for_one_line[:-1], 2)
        for first_durations, second_durations in pairs:
            first_start_times = list(itertools.accumulate(first_durations))
            second_start_times = list(itertools.accumulate(second_durations))
            avg_n_starts = (len(first_start_times) + len(second_start_times)) / 2
            total_n_starts = len(set(first_start_times + second_start_times))
            score -= (total_n_starts / avg_n_starts - 1)
    n_non_last_measures = (fragment.n_beats // fragment.meter_numerator) - 1
    n_pairs = n_non_last_measures * (n_non_last_measures - 1) / 2
    score /= len(fragment.melodic_lines) * n_pairs
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


def get_scoring_functions_registry() -> dict[str, Callable]:
    """
    Get mapping from names to corresponding scoring functions.

    :return:
        registry of scoring functions
    """
    registry = {
        'absence_of_doubled_pitch_classes': evaluate_absence_of_doubled_pitch_classes,
        'absence_of_simultaneous_skips': evaluate_absence_of_simultaneous_skips,
        'absence_of_voice_crossing': evaluate_absence_of_voice_crossing,
        'cadence_duration': evaluate_cadence_duration,
        'climax_explicity': evaluate_climax_explicity,
        'consistency_of_rhythm_with_meter': evaluate_consistency_of_rhythm_with_meter,
        'dissonances_preparation_and_resolution': evaluate_dissonances_preparation_and_resolution,
        'harmony_dynamic': evaluate_harmony_dynamic,
        'rhythmic_homogeneity': evaluate_rhythmic_homogeneity,
        'smoothness_of_voice_leading': evaluate_smoothness_of_voice_leading,
        'stackability': evaluate_stackability,
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
    for left_point, right_point in zip(breakpoints, breakpoints[1:]):
        if left_point <= unweighted_score:
            break
        weighted_score -= (left_point - max(unweighted_score, right_point)) * weights[left_point]
    return weighted_score


def evaluate(
        fragment: Fragment,
        scoring_sets: list[str],
        scoring_sets_registry: SCORING_SETS_REGISTRY_TYPE,
        verbose: bool = False
) -> float:
    """
    Evaluate fragment.

    :param fragment:
        fragment to be evaluated
    :param scoring_sets:
        names of scoring sets to be used
    :param scoring_sets_registry:
        mapping from a name of a scoring set to a list of triples of a scoring function,
        its weight, and its parameters
    :param verbose:
        if it is set to `True`, scores are printed with detailing by functions
    :return:
        weighted sum of scores returned by applicable scoring functions
    """
    score = 0
    for scoring_set_name in scoring_sets:
        scoring_set = scoring_sets_registry[scoring_set_name]
        for scoring_fn, weights, params in scoring_set:
            unweighted_score = scoring_fn(fragment, **params)
            curr_score = weight_score(unweighted_score, weights)
            if verbose:
                name = scoring_fn.__name__.removeprefix('evaluate_')
                print(f'{name:>40}: {curr_score}')
            score += curr_score
    if verbose:
        print(f'Overall score is: {score}')
    return score
