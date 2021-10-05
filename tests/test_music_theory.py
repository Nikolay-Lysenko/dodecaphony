"""
Test `dodecaphony.music_theory` module.

Author: Nikolay Lysenko
"""


from typing import Optional

import pytest

from dodecaphony.music_theory import (
    get_mapping_from_pitch_class_to_diatonic_scales,
    get_smallest_intervals_between_pitch_classes,
    invert_tone_row,
    revert_tone_row,
    transpose_tone_row,
    validate_tone_row,
)


@pytest.mark.parametrize(
    "scale_types, key, value",
    [
        (
            # `scale_types`
            None,
            # `key`
            'C',
            # `value`
            [
                'C-major',
                'C#-major',
                'D#-major',
                'F-major',
                'G-major',
                'G#-major',
                'A#-major',
                'C-natural_minor',
                'D-natural_minor',
                'E-natural_minor',
                'F-natural_minor',
                'G-natural_minor',
                'A-natural_minor',
                'A#-natural_minor',
                'C-harmonic_minor',
                'C#-harmonic_minor',
                'E-harmonic_minor',
                'F-harmonic_minor',
                'G-harmonic_minor',
                'A-harmonic_minor',
                'A#-harmonic_minor',
                'C-dorian',
                'D-dorian',
                'D#-dorian',
                'F-dorian',
                'G-dorian',
                'A-dorian',
                'A#-dorian',
                'C-phrygian',
                'D-phrygian',
                'E-phrygian',
                'F-phrygian',
                'G-phrygian',
                'A-phrygian',
                'B-phrygian',
                'C-lydian',
                'C#-lydian',
                'D#-lydian',
                'F-lydian',
                'F#-lydian',
                'G#-lydian',
                'A#-lydian',
                'C-mixolydian',
                'D-mixolydian',
                'D#-mixolydian',
                'F-mixolydian',
                'G-mixolydian',
                'G#-mixolydian',
                'A#-mixolydian',
                'C-locrian',
                'D-locrian',
                'E-locrian',
                'F#-locrian',
                'G-locrian',
                'A-locrian',
                'B-locrian',
                'C-whole_tone',
                'D-whole_tone',
                'E-whole_tone',
                'F#-whole_tone',
                'G#-whole_tone',
                'A#-whole_tone',
            ]
        ),
        (
            # `scale_types`
            ('major', 'whole_tone'),
            # `key`
            'C#',
            # `value`
            [
                'C#-major',
                'D-major',
                'E-major',
                'F#-major',
                'G#-major',
                'A-major',
                'B-major',
                'C#-whole_tone',
                'D#-whole_tone',
                'F-whole_tone',
                'G-whole_tone',
                'A-whole_tone',
                'B-whole_tone',
            ]
        ),
    ]
)
def test_get_mapping_from_pitch_class_to_diatonic_scales(
        scale_types: Optional[list[str]], key: str, value: list[str]
) -> None:
    """Test `get_mapping_from_pitch_class_to_diatonic_scales` function."""
    mapping = get_mapping_from_pitch_class_to_diatonic_scales(scale_types)
    assert mapping[key] == value


@pytest.mark.parametrize(
    "key, value",
    [
        (('C', 'B'), -1),
        (('C', 'F#'), 6),
    ]
)
def test_get_smallest_intervals_between_pitch_classes(key: tuple[str, str], value: int) -> None:
    """Test `get_smallest_intervals_between_pitch_classes` function."""
    mapping = get_smallest_intervals_between_pitch_classes()
    assert mapping[key] == value


@pytest.mark.parametrize(
    "tone_row, expected",
    [
        (
            ['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
            ['B', 'C', 'D#', 'A', 'G', 'A#', 'G#', 'C#', 'E', 'F#', 'D', 'F'],
        ),
    ]
)
def test_invert_tone_row(tone_row: list[str], expected: list[str]) -> None:
    """Test `invert_tone_row` function."""
    result = invert_tone_row(tone_row)
    assert result == expected


@pytest.mark.parametrize(
    "tone_row, expected",
    [
        (
            ['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
            ['F', 'G#', 'E', 'F#', 'A', 'D', 'C', 'D#', 'C#', 'G', 'A#', 'B'],
        ),
    ]
)
def test_revert_tone_row(tone_row: list[str], expected: list[str]) -> None:
    """Test `revert_tone_row` function."""
    result = revert_tone_row(tone_row)
    assert result == expected


@pytest.mark.parametrize(
    "tone_row, shift_in_semitones, expected",
    [
        (
            ['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
            -1,
            ['A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E'],
        ),
    ]
)
def test_transpose_tone_row(
        tone_row: list[str], shift_in_semitones: int, expected: list[str]
) -> None:
    """Test `transpose_tone_row` function."""
    result = transpose_tone_row(tone_row, shift_in_semitones)
    assert result == expected


@pytest.mark.parametrize(
    "tone_row, match",
    [
        (
            ['A', 'B'],
            "Tone row must have 12 elements."
        ),
        (
            ['B', 'A', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
            "All pitch classes must be included in a tone row."
        ),
    ]
)
def test_validate_tone_row(tone_row: list[str], match: str) -> None:
    """Test `validate_tone_row` function."""
    with pytest.raises(ValueError, match=match):
        validate_tone_row(tone_row)
