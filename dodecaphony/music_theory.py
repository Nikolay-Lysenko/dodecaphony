"""
Provide utilities for working with pitches, pitch classes, and tone rows.

Author: Nikolay Lysenko
"""


import itertools
from enum import Enum


N_SEMITONES_PER_OCTAVE = 12
TONE_ROW_LEN = 12
PITCH_CLASS_TO_POSITION = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}
POSITION_TO_PITCH_CLASS = {v: k for k, v in PITCH_CLASS_TO_POSITION.items()}


class IntervalTypes(Enum):
    """Enumeration of interval types."""
    PERFECT_CONSONANCE = 1
    IMPERFECT_CONSONANCE = 2
    DISSONANCE = 3
    NOT_AN_INTERVAL = 4


N_SEMITONES_TO_INTERVAL_TYPE_WITH_CONSONANT_P4 = {
    0: IntervalTypes.PERFECT_CONSONANCE,
    1: IntervalTypes.DISSONANCE,
    2: IntervalTypes.DISSONANCE,
    3: IntervalTypes.IMPERFECT_CONSONANCE,
    4: IntervalTypes.IMPERFECT_CONSONANCE,
    5: IntervalTypes.IMPERFECT_CONSONANCE,
    6: IntervalTypes.DISSONANCE,
    7: IntervalTypes.PERFECT_CONSONANCE,
    8: IntervalTypes.IMPERFECT_CONSONANCE,
    9: IntervalTypes.IMPERFECT_CONSONANCE,
    10: IntervalTypes.DISSONANCE,
    11: IntervalTypes.DISSONANCE,
}
N_SEMITONES_TO_INTERVAL_TYPE_WITH_DISSONANT_P4 = {
    0: IntervalTypes.PERFECT_CONSONANCE,
    1: IntervalTypes.DISSONANCE,
    2: IntervalTypes.DISSONANCE,
    3: IntervalTypes.IMPERFECT_CONSONANCE,
    4: IntervalTypes.IMPERFECT_CONSONANCE,
    5: IntervalTypes.DISSONANCE,
    6: IntervalTypes.DISSONANCE,
    7: IntervalTypes.PERFECT_CONSONANCE,
    8: IntervalTypes.IMPERFECT_CONSONANCE,
    9: IntervalTypes.IMPERFECT_CONSONANCE,
    10: IntervalTypes.DISSONANCE,
    11: IntervalTypes.DISSONANCE,
}


def get_smallest_intervals_between_pitch_classes() -> dict[tuple[str, str], int]:
    """
    Get mapping from a pair of pitch classes to the smallest interval connecting them.

    Tritone intervals are stored as upward intervals.

    :return:
        mapping from a pair of pitch classes to the smallest interval (in semitones)
        connecting them
    """
    pitch_classes = PITCH_CLASS_TO_POSITION.keys()
    cartesian_product = itertools.product(pitch_classes, pitch_classes)
    result = {}
    for starting_pitch_class, destination_pitch_class in cartesian_product:
        destination_position = PITCH_CLASS_TO_POSITION[destination_pitch_class]
        starting_position = PITCH_CLASS_TO_POSITION[starting_pitch_class]
        shift = N_SEMITONES_PER_OCTAVE // 2 - 1
        value = (destination_position - starting_position + shift) % N_SEMITONES_PER_OCTAVE - shift
        result[(starting_pitch_class, destination_pitch_class)] = value
    return result


def get_type_of_interval(
        n_semitones: int,
        is_perfect_fourth_consonant: bool = True
) -> IntervalTypes:
    """
    Get type of a harmonic interval.

    :param n_semitones:
        interval size in semitones
    :param is_perfect_fourth_consonant:
        indicator whether perfect fourth is a consonant interval
    :return:
        type of interval
    """
    if is_perfect_fourth_consonant:
        n_semitones_to_consonance = N_SEMITONES_TO_INTERVAL_TYPE_WITH_CONSONANT_P4
    else:
        n_semitones_to_consonance = N_SEMITONES_TO_INTERVAL_TYPE_WITH_DISSONANT_P4
    n_semitones %= len(n_semitones_to_consonance)
    return n_semitones_to_consonance[n_semitones]


def validate_tone_row(tone_row: list[str]) -> None:
    """
    Validate tone row.

    :param tone_row:
        tone row as list of pitch classes (like C or C#, flats are not allowed)
    :return:
        None
    """
    if len(tone_row) != TONE_ROW_LEN:
        raise ValueError("Tone row must have 12 elements.")
    positions = [PITCH_CLASS_TO_POSITION[pitch_class] for pitch_class in tone_row]
    if sorted(positions) != list(range(TONE_ROW_LEN)):
        raise ValueError("All pitch classes must be included in a tone row.")


def invert_tone_row(tone_row: list[str]) -> list[str]:
    """
    Invert tone row preserving its first pitch class.

    :param tone_row:
        tone row as list of pitch classes (like C or C#, flats are not allowed)
    :return:
        inverted tone row
    """
    inverted_tone_row = [tone_row[0]]
    for former_pitch_class, latter_pitch_class in zip(tone_row, tone_row[1:]):
        latter_position = PITCH_CLASS_TO_POSITION[latter_pitch_class]
        former_position = PITCH_CLASS_TO_POSITION[former_pitch_class]
        interval = latter_position - former_position
        next_position = (PITCH_CLASS_TO_POSITION[inverted_tone_row[-1]] - interval) % TONE_ROW_LEN
        next_pitch_class = POSITION_TO_PITCH_CLASS[next_position]
        inverted_tone_row.append(next_pitch_class)
    return inverted_tone_row


def revert_tone_row(tone_row: list[str]) -> list[str]:
    """
    Revert tone row, i.e., apply retrograde inversion to it.

    :param tone_row:
        tone row as list of pitch classes (like C or C#, flats are not allowed)
    :return:
        reverted tone row
    """
    return tone_row[::-1]


def rotate_tone_row(tone_row: list[str], shift: int) -> list[str]:
    """
    Rotate tone row.

    :param tone_row:
        tone row as list of pitch classes (like C or C#, flats are not allowed)
    :param shift:
        shift; if positive; tone row is rotated forward, if negative, tone row is rotated backwards
    :return:
        rotated tone row
    """
    return tone_row[shift:] + tone_row[:shift]


def transpose_tone_row(tone_row: list[str], shift_in_semitones: int) -> list[str]:
    """
    Transpose tone row.

    :param tone_row:
        tone row as list of pitch classes (like C or C#, flats are not allowed)
    :param shift_in_semitones:
        transposition interval in semitones
    :return:
        transposed tone row
    """
    transposed_tone_row = []
    for pitch_class in tone_row:
        new_position = (PITCH_CLASS_TO_POSITION[pitch_class] + shift_in_semitones) % TONE_ROW_LEN
        new_pitch_class = POSITION_TO_PITCH_CLASS[new_position]
        transposed_tone_row.append(new_pitch_class)
    return transposed_tone_row
