"""
Test `dodecaphony.music_theory` module.

Author: Nikolay Lysenko
"""


import pytest

from dodecaphony.music_theory import (
    get_smallest_intervals_between_pitch_classes,
    invert_tone_row,
    revert_tone_row,
    transpose_tone_row,
    validate_tone_row,
)


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
