"""
Evaluate a fragment.

Author: Nikolay Lysenko
"""


from typing import Any, Callable

from dodecaphony.fragment import Fragment
from dodecaphony.scoring_functions.harmony import (
    evaluate_absence_of_doubled_pitch_classes,
    evaluate_absence_of_false_octaves,
    evaluate_absence_of_simultaneous_skips,
    evaluate_absence_of_voice_crossing,
    evaluate_dissonances_preparation_and_resolution,
    evaluate_harmony_dynamic_by_positions,
    evaluate_harmony_dynamic_by_time_intervals,
    evaluate_local_diatonicity_at_all_lines_level,
    evaluate_motion_to_perfect_consonances,
    evaluate_movement_to_final_sonority,
    evaluate_pitch_class_distribution_among_lines,
    evaluate_presence_of_vertical_intervals,
    evaluate_sonic_intensity_by_positions,
)
from dodecaphony.scoring_functions.melody import (
    evaluate_absence_of_aimless_fluctuations,
    evaluate_climax_explicity,
    evaluate_direction_change_after_large_skip,
    evaluate_local_diatonicity_at_line_level,
    evaluate_pitch_class_prominence,
    evaluate_presence_of_intervallic_motif,
    evaluate_smoothness_of_voice_leading,
    evaluate_stackability,
    evaluate_transitions,
)
from dodecaphony.scoring_functions.rhythm import (
    evaluate_cadence_duration,
    evaluate_presence_of_required_pauses,
    evaluate_rhythmic_homogeneity,
    evaluate_rhythmic_intensity_by_positions,
)


SCORING_SETS_REGISTRY_TYPE = dict[
    str,
    list[tuple[Callable[..., float], dict[float, float], dict[str, Any]]]
]


def get_scoring_functions_registry() -> dict[str, Callable]:
    """
    Get mapping from names to corresponding scoring functions.

    :return:
        registry of scoring functions
    """
    registry = {
        'absence_of_aimless_fluctuations': evaluate_absence_of_aimless_fluctuations,
        'absence_of_doubled_pitch_classes': evaluate_absence_of_doubled_pitch_classes,
        'absence_of_false_octaves': evaluate_absence_of_false_octaves,
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
    Parse mapping from names of scoring sets to the scoring sets itself.

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
    Transform original score with a piecewise linear function.

    :param unweighted_score:
        original score
    :param weights:
        mapping from a breakpoint to a slope coefficient for an interval
        from the breakpoint to the next-to-the-left breakpoint (or -1 if it is absent)
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
        if it is set to `True`, scores detailed to function level are returned as a second output
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
