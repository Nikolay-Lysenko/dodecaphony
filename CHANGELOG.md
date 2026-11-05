# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.2] - 2026-11-06

### Added:

* New scoring function: `pitch_ranges`.
* New transformation: `shuffle` (for free atonality and perturbations).
* When calculating sonority stability, more weight can be assigned to the most dissonant interval.

## Changed:

* Dependency management migrated from `requirements` to `uv`.
* Releases to PyPI are made with GitHub Actions.
* Maintainability evaluation migrated from CodeClimate to Qlty.

## [0.2.1] - 2024-08-06

### Added:

* Absence of false octaves is scored.
* 2/4 and 3/4 meters are supported for sheet music rendering.

### Changed:

* Non-pause events and start/end times are stored as pre-calculated attributes of `Sonority`.
* `measure_pair_durations_change` is renamed to `crossmeasure_event_transfer`.
* Scoring functions are extracted from `evaluation.py` module to separate subpackage.

### Fixed:

* Type of negative interval is returned without inverting the interval.
* Unisons are not penalized for doubled pitch classes.
* Bass line isn't penalized for lack of conjunct motion to the final sonority if its skip is requested.
* The lowest sounding line isn't considered the bass if the bass has pause.
* Pauses doesn't diminish numerator of line-level local diatonicity score.
* Right end of the time interval is exclusive for ad hoc positions.
* Elisiions in the very beginning or the very end of melodic line aren't missed.

## [0.2.0] - 2024-07-19

### Added:

* Dependent tone row instances are introduced (and `Fragment` class is seriously refactored to support this).
* Allowed measure durations made configurable.
* New scoring functions:
  - `direction_change_after_large_skip`
  - `harmony_dynamic_by_time_intervals`
  - `local_diatonicity_at_line_level`
  - `motion_to_perfect_consonances`
  - `movement_to_final_sonority`
  - `pitch_class_distribution`
  - `pitch_class_prominence`
  - `presence_of_required_pauses`
  - `presence_of_vertical_intervals`
  - `rhythmic_intensity`
  - `sonic_intensity`
  - `transitions`

### Changed:

* Optimization algorithm made more flexible with configurable neighborhoods and perturbations instead of expanding neighborhoods.
* `duration_change` transformation is replaced with `line_durations_change`, `measure_durations_change`, and `measure_pair_durations_change`.
* Some scoring functions keys are detailed in order to distinguish from newly introduced similar functions:
  - `harmony_dynamic` -> `harmony_dynamic_by_positions`
  - `local_diatonicity` -> `local_diatonicity_at_all_lines_level`

### Fixed:

* Calculation of Lilypond note duration is corrected.

## Removed:

* `consistency_of_rhythm_with_meter` scoring function isn't needed any more.

## [0.1.5] - 2024-02-23

### Added:

* Text file with meta-information (path to config and results of evaluation) is stored alongside rendered outputs.

### Changed

* Voice crossing penalization is normalized.

## [0.1.4] - 2021-12-14

### Fixed:

* Sonority types are inferred in a more correct fashion.

## [0.1.3] - 2021-11-13

## Added:

* New scoring functions:
  - `absence_of_aimless_fluctuations`
  - `intervallic_motif`
  - `local_diatonicity`

## [0.1.2] - 2021-08-10

## Added:

* New scoring functions:
  - `absence_of_simultaneous_skips`
  - `dissonances_preparation_and_resolution`
  - `rhythmic_homogeneity`
* New transformation: `rotation`.
* Weights of scores.

## Changed:

* `pause_swap` transformation is renamed to `pause_shift`.

## [0.1.1] - 2021-06-07

## Added:

* Users can configure initial values of temporal and sonic content and freeze immutable parts.
* YAML representation is saved (and can be re-used further in initial values).
* New scoring functions:
  - `climax_explicity`
  - `stackability`

## Changed:

* Optimization runs in parallel.

## Fixed:

* Beams are deduplicated.

## [0.1.0] - 2021-05-03

## Added:

* Initial version of the `Fragment` class.
* Scoring functions:
  - `absence_of_doubled_pitch_classes`
  - `absence_of_voice_crossing`
  - `cadence_duration`
  - `consistency_of_rhythm_with_meter`
  - `harmony_dynamic`,
  - `smoothness_of_voice_leading`
* Transformations:
  - `inversion`
  - `reversion`
  - `transposition`
  - `duration_change`
  - `pause_swap`
* Local search optimization.
* Rendering to MIDI, WAV, Lilypond, and PDF.
