"""
Evaluate a fragment.

Author: Nikolay Lysenko
"""


import itertools
import math
from typing import Any, Callable

from .fragment import Fragment
from .music_theory import N_SEMITONES_PER_OCTAVE


SCORING_SETS_REGISTRY_TYPE = dict[str, list[tuple[Callable[..., float], float, dict[str, Any]]]]


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
    for melodic_line in fragment.melodic_lines:
        rhythm_data = []
        for event in melodic_line:
            record = {
                'measure_id': event.start_time // fragment.meter_numerator,
                'time_since_measure_start': event.start_time % fragment.meter_numerator,
                'duration': event.duration
            }
            rhythm_data.append(record)
        for measure_id, values in itertools.groupby(rhythm_data, lambda x: x['measure_id']):
            first_value = values.__next__()
            if first_value['time_since_measure_start'] != 0:
                durations = [first_value['time_since_measure_start'], first_value['duration']]
            else:
                durations = [first_value['duration']]
            for value in values:
                durations.append(value['duration'])
            if durations not in consistent_patterns:
                score -= 1
    score /= len(fragment.melodic_lines) * (fragment.n_beats // fragment.meter_numerator)
    return score


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

    def find_sonority_type() -> str:
        """Find type of current sonority."""
        for ad_hoc_position in ad_hoc_positions:
            if ad_hoc_position['time'] < 0:
                ad_hoc_position['time'] += fragment.n_beats
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

    def compute_harmonic_stability_of_sonority() -> float:
        """Compute stability of sonority as average stability of intervals forming it."""
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

    score = 0
    for sonority in fragment.sonorities:
        sonority_start = max(event.start_time for event in sonority)
        sonority_end = min(event.start_time + event.duration for event in sonority)
        stability_of_current_sonority = compute_harmonic_stability_of_sonority()
        sonority_type = find_sonority_type()
        min_allowed_value = ranges[sonority_type][0]
        score += min(stability_of_current_sonority - min_allowed_value, 0)
        max_allowed_value = ranges[sonority_type][1]
        score += min(max_allowed_value - stability_of_current_sonority, 0)
    score /= len(fragment.sonorities)
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
        'absence_of_voice_crossing': evaluate_absence_of_voice_crossing,
        'cadence_duration': evaluate_cadence_duration,
        'climax_explicity': evaluate_climax_explicity,
        'consistency_of_rhythm_with_meter': evaluate_consistency_of_rhythm_with_meter,
        'harmony_dynamic': evaluate_harmony_dynamic,
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
            weight = scoring_fn_info.pop('weight')
            scoring_fns.append((scoring_fn, weight, scoring_fn_info))
        scoring_sets_registry[scoring_set_name] = scoring_fns
    return scoring_sets_registry


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
        for scoring_fn, fn_weight, params in scoring_set:
            curr_score = fn_weight * scoring_fn(fragment, **params)
            if verbose:
                name = scoring_fn.__name__.removeprefix('evaluate_')
                print(f'{name:>35}: {curr_score}')
            score += curr_score
    if verbose:
        print(f'Overall score is: {score}')
    return score
