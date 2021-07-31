"""
Test `dodecaphony.transformations` module.

Author: Nikolay Lysenko
"""


import pytest

from dodecaphony.fragment import Event, Fragment
from dodecaphony.transformations import (
    apply_duration_change,
    apply_inversion,
    apply_pause_shift,
    apply_reversion,
    apply_transposition,
    create_transformations_registry,
    get_duration_changes,
    transform,
)


@pytest.mark.parametrize(
    "fragment, expected_options",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [2.0, 2.0],
                    [2.0, 1.0, 1.0],
                ],
                sonic_content=[
                    ['A', 'B', 'C', 'D', 'E']
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=4,
                line_ids=[1, 2],
                upper_line_highest_position=88,
                upper_line_lowest_position=1,
                n_melodic_lines_by_group=[2],
                n_tone_row_instances_by_group=[1],
                mutable_temporal_content_indices=[0, 1],
                mutable_sonic_content_indices=[0],
            ),
            # `expected_options`
            [
                [
                    [2.0, 2.0],
                    [2.0, 1.0, 1.0],
                ],
                [
                    [1.0, 3.0],
                    [2.0, 1.0, 1.0],
                ],
                [
                    [3.0, 1.0],
                    [2.0, 1.0, 1.0],
                ],
                [
                    [2.0, 2.0],
                    [1.0, 2.0, 1.0],
                ],
                [
                    [2.0, 2.0],
                    [1.0, 1.0, 2.0],
                ],
                [
                    [2.0, 2.0],
                    [1.5, 1.5, 1.0],
                ],
                [
                    [2.0, 2.0],
                    [1.5, 1.0, 1.5],
                ],
                [
                    [2.0, 2.0],
                    [2.0, 0.5, 1.5],
                ],
                [
                    [2.0, 2.0],
                    [2.0, 1.5, 0.5],
                ],
            ]
        )
    ]
)
def test_apply_duration_change(
        fragment: Fragment, expected_options: list[list[list[list[Event]]]]
) -> None:
    """Test `apply_duration_change` function."""
    fragment = apply_duration_change(fragment, get_duration_changes())
    assert fragment.temporal_content in expected_options


@pytest.mark.parametrize(
    "fragment, expected_options",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                ],
                sonic_content=[
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=24,
                line_ids=[1, 2],
                upper_line_highest_position=88,
                upper_line_lowest_position=1,
                n_melodic_lines_by_group=[1, 1],
                n_tone_row_instances_by_group=[2, 1],
                mutable_temporal_content_indices=[0, 1],
                mutable_sonic_content_indices=[0, 1],
            ),
            # `expected_options`
            [
                [
                    [
                        'B', 'C', 'D#', 'A', 'G', 'A#', 'G#', 'C#', 'E', 'F#', 'D', 'F', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'B', 'C', 'D#', 'A', 'G', 'A#', 'G#', 'C#', 'E', 'F#', 'D', 'F'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'B', 'C', 'D#', 'A', 'G', 'A#', 'G#', 'C#', 'E', 'F#', 'D', 'F'
                    ],
                ],
            ]
        ),
    ]
)
def test_apply_inversion(fragment: Fragment, expected_options: list[list[list[str]]]) -> None:
    """Test `apply_inversion` function."""
    fragment = apply_inversion(fragment)
    assert fragment.sonic_content in expected_options


@pytest.mark.parametrize(
    "fragment, expected_options",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ],
                sonic_content=[
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=24,
                line_ids=[1],
                upper_line_highest_position=88,
                upper_line_lowest_position=1,
                n_melodic_lines_by_group=[1],
                n_tone_row_instances_by_group=[2],
                mutable_temporal_content_indices=[0],
                mutable_sonic_content_indices=[0],
            ),
            # `expected_options`
            [
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'pause', 'F',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'B',
                        'pause', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
            ]
        ),
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                ],
                sonic_content=[
                    ['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=24,
                line_ids=[1],
                upper_line_highest_position=88,
                upper_line_lowest_position=1,
                n_melodic_lines_by_group=[1],
                n_tone_row_instances_by_group=[1],
                mutable_temporal_content_indices=[0],
                mutable_sonic_content_indices=[0],
            ),
            # `expected_options`
            [
                [
                    ['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                ],
            ]
        ),
    ]
)
def test_apply_pause_shift(fragment: Fragment, expected_options: list[list[list[str]]]) -> None:
    """Test `apply_pause_shift` function."""
    fragment = apply_pause_shift(fragment)
    assert fragment.sonic_content in expected_options


@pytest.mark.parametrize(
    "fragment, expected_options",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                ],
                sonic_content=[
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=24,
                line_ids=[1, 2],
                upper_line_highest_position=88,
                upper_line_lowest_position=1,
                n_melodic_lines_by_group=[1, 1],
                n_tone_row_instances_by_group=[2, 1],
                mutable_temporal_content_indices=[0, 1],
                mutable_sonic_content_indices=[0, 1],
            ),
            # `expected_options`
            [
                [
                    [
                        'F', 'G#', 'E', 'F#', 'A', 'D', 'C', 'D#', 'C#', 'G', 'A#', 'B', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'F', 'G#', 'E', 'F#', 'A', 'D', 'C', 'D#', 'C#', 'G', 'A#', 'B'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'F', 'G#', 'E', 'F#', 'A', 'D', 'C', 'D#', 'C#', 'G', 'A#', 'B'
                    ],
                ],
            ]
        ),
    ]
)
def test_apply_reversion(fragment: Fragment, expected_options: list[list[list[str]]]) -> None:
    """Test `apply_reversion` function."""
    fragment = apply_reversion(fragment)
    assert fragment.sonic_content in expected_options


@pytest.mark.parametrize(
    "fragment, max_transposition, expected_options",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                ],
                sonic_content=[
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=24,
                line_ids=[1, 2],
                upper_line_highest_position=88,
                upper_line_lowest_position=1,
                n_melodic_lines_by_group=[1, 1],
                n_tone_row_instances_by_group=[2, 1],
                mutable_temporal_content_indices=[0, 1],
                mutable_sonic_content_indices=[0, 1],
            ),
            # `max_transposition`
            1,
            # `expected_options`
            [
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'A#', 'A', 'F#', 'C', 'D', 'B', 'C#', 'G#', 'F', 'D#', 'G', 'E'
                    ],
                ],
                [
                    [
                        'C', 'B', 'G#', 'D', 'E', 'C#', 'D#', 'A#', 'G', 'F', 'A', 'F#', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'C', 'B', 'G#', 'D', 'E', 'C#', 'D#', 'A#', 'G', 'F', 'A', 'F#'
                    ],
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                ],
                [
                    [
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F', 'pause',
                        'B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'
                    ],
                    [
                        'C', 'B', 'G#', 'D', 'E', 'C#', 'D#', 'A#', 'G', 'F', 'A', 'F#'
                    ],
                ],
            ]
        ),
    ]
)
def test_apply_transposition(
        fragment: Fragment, max_transposition: int, expected_options: list[list[list[str]]]
) -> None:
    """Test `apply_transposition` function."""
    fragment = apply_transposition(fragment, max_transposition)
    assert fragment.sonic_content in expected_options


@pytest.mark.parametrize(
    "key, value",
    [
        ((0.5, 2.0), [(0.5, 2.0), (1.0, 1.5), (1.5, 1.0), (2.0, 0.5)]),
    ]
)
def test_get_duration_changes(key: tuple[float, float], value: list[tuple[float, float]]) -> None:
    """Test `get_duration_changes` function."""
    assert get_duration_changes()[key] == value


@pytest.mark.parametrize(
    "fragment, n_transformations, transformation_names, max_transposition, expected_options",
    [
        (
            # `fragment`
            Fragment(
                temporal_content=[
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                ],
                sonic_content=[
                    ['B', 'A#', 'G', 'C#', 'D#', 'C', 'D', 'A', 'F#', 'E', 'G#', 'F'],
                ],
                meter_numerator=4,
                meter_denominator=4,
                n_beats=12,
                line_ids=[1],
                upper_line_highest_position=55,
                upper_line_lowest_position=41,
                n_melodic_lines_by_group=[1],
                n_tone_row_instances_by_group=[1],
                mutable_temporal_content_indices=[0],
                mutable_sonic_content_indices=[0],
            ),
            # `n_transformations`
            1,
            # `transformation_names`
            ['inversion', 'reversion'],
            # `max_transposition`
            1,
            # `expected_options`
            [
                [
                    [
                        Event(line_index=0, start_time=0.0, duration=1.0, pitch_class='B', position_in_semitones=50),
                        Event(line_index=0, start_time=1.0, duration=1.0, pitch_class='C', position_in_semitones=51),
                        Event(line_index=0, start_time=2.0, duration=1.0, pitch_class='D#', position_in_semitones=54),
                        Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='A', position_in_semitones=48),
                        Event(line_index=0, start_time=4.0, duration=1.0, pitch_class='G', position_in_semitones=46),
                        Event(line_index=0, start_time=5.0, duration=1.0, pitch_class='A#', position_in_semitones=49),
                        Event(line_index=0, start_time=6.0, duration=1.0, pitch_class='G#', position_in_semitones=47),
                        Event(line_index=0, start_time=7.0, duration=1.0, pitch_class='C#', position_in_semitones=52),
                        Event(line_index=0, start_time=8.0, duration=1.0, pitch_class='E', position_in_semitones=55),
                        Event(line_index=0, start_time=9.0, duration=1.0, pitch_class='F#', position_in_semitones=45),
                        Event(line_index=0, start_time=10.0, duration=1.0, pitch_class='D', position_in_semitones=41),
                        Event(line_index=0, start_time=11.0, duration=1.0, pitch_class='F', position_in_semitones=44),
                    ]
                ],
                [
                    [
                        Event(line_index=0, start_time=0.0, duration=1.0, pitch_class='F', position_in_semitones=44),
                        Event(line_index=0, start_time=1.0, duration=1.0, pitch_class='G#', position_in_semitones=47),
                        Event(line_index=0, start_time=2.0, duration=1.0, pitch_class='E', position_in_semitones=43),
                        Event(line_index=0, start_time=3.0, duration=1.0, pitch_class='F#', position_in_semitones=45),
                        Event(line_index=0, start_time=4.0, duration=1.0, pitch_class='A', position_in_semitones=48),
                        Event(line_index=0, start_time=5.0, duration=1.0, pitch_class='D', position_in_semitones=53),
                        Event(line_index=0, start_time=6.0, duration=1.0, pitch_class='C', position_in_semitones=51),
                        Event(line_index=0, start_time=7.0, duration=1.0, pitch_class='D#', position_in_semitones=54),
                        Event(line_index=0, start_time=8.0, duration=1.0, pitch_class='C#', position_in_semitones=52),
                        Event(line_index=0, start_time=9.0, duration=1.0, pitch_class='G', position_in_semitones=46),
                        Event(line_index=0, start_time=10.0, duration=1.0, pitch_class='A#', position_in_semitones=49),
                        Event(line_index=0, start_time=11.0, duration=1.0, pitch_class='B', position_in_semitones=50),
                    ]
                ],
            ]
        ),
    ]
)
def test_transform(
        fragment: Fragment, n_transformations: int, transformation_names: list[str],
        max_transposition: int, expected_options: list[list[list[Event]]]
) -> None:
    """Test `transform` function."""
    registry = create_transformations_registry(max_transposition)
    transformation_probabilities = [1 / len(transformation_names) for _ in transformation_names]
    fragment = transform(
        fragment, n_transformations, registry, transformation_names, transformation_probabilities
    )
    assert fragment.melodic_lines in expected_options
