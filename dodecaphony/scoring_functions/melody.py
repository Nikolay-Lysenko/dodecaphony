"""
Evaluate melodic properties of a fragment.

Author: Nikolay Lysenko
"""


import math
import re
import string
from collections import Counter
from functools import cache
from typing import Any, Optional

from sinethesizer.utils.music_theory import get_note_to_position_mapping

from dodecaphony.fragment import Event, Fragment
from dodecaphony.music_theory import (
    N_SEMITONES_PER_OCTAVE,
    PITCH_CLASS_TO_POSITION,
    get_mapping_from_pitch_class_to_diatonic_scales,
)
from dodecaphony.utils import compute_rolling_aggregate


NOTE_TO_POSITION_MAPPING = get_note_to_position_mapping()


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
