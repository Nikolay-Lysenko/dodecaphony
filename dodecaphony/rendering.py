"""
Render fragment to some formats such as WAV and MIDI.

Author: Nikolay Lysenko
"""


import datetime
import os
import subprocess
import traceback
from math import ceil, floor
from pkg_resources import resource_filename
from typing import Any

import pretty_midi
import yaml
from sinethesizer.io import (
    convert_events_to_timeline,
    convert_tsv_to_events,
    create_instruments_registry,
    write_timeline_to_wav
)
from sinethesizer.synth.core import Instrument
from sinethesizer.utils.music_theory import get_list_of_notes, get_note_to_position_mapping

from .fragment import Event, Fragment


NOTE_TO_POSITION = get_note_to_position_mapping()
POSITION_TO_NOTE = {v: k for k, v in NOTE_TO_POSITION.items()}


def create_midi_from_fragment(
        fragment: Fragment,
        midi_path: str,
        beat_in_seconds: float,
        instruments: dict[int, int],
        velocity: int,
        opening_silence_in_seconds: float = 1.0,
        trailing_silence_in_seconds: float = 1.0
) -> None:
    """
    Create MIDI file from a fragment created by this package.

    :param fragment:
        musical fragment
    :param midi_path:
        path where resulting MIDI file is going to be saved
    :param beat_in_seconds:
        duration of one beat in seconds
    :param instruments:
        mapping from IDs of melodic lines to IDs of instruments (according to General MIDI
        specification) that play them
    :param velocity:
        one common velocity for all notes
    :param opening_silence_in_seconds:
        number of seconds with silence to add at the start of the composition
    :param trailing_silence_in_seconds:
        number of seconds with silence to add at the end of the composition
    :return:
        None
    """
    numeration_shift = pretty_midi.note_name_to_number('A0')
    pretty_midi_instruments = []
    for line_id, melodic_line in zip(fragment.line_ids, fragment.melodic_lines):
        pretty_midi_instrument = pretty_midi.Instrument(instruments[line_id], name=str(line_id))
        for event in melodic_line:
            if event.pitch_class in ['pause', 'skip']:
                continue
            start_time = event.start_time * beat_in_seconds
            start_time += opening_silence_in_seconds
            end_time = start_time + event.duration * beat_in_seconds
            pitch = event.position_in_semitones + numeration_shift
            note = pretty_midi.Note(
                start=start_time,
                end=end_time,
                pitch=pitch,
                velocity=velocity
            )
            pretty_midi_instrument.notes.append(note)
        pretty_midi_instrument.notes.sort(key=lambda x: (x.start, x.pitch))
        pretty_midi_instruments.append(pretty_midi_instrument)

    trailing_silence_start = fragment.n_beats * beat_in_seconds
    trailing_silence_start += opening_silence_in_seconds
    note = pretty_midi.Note(
        start=trailing_silence_start,
        end=trailing_silence_start + trailing_silence_in_seconds,
        velocity=0,
        pitch=1  # Arbitrary value that affects nothing.
    )
    pretty_midi_instruments[0].notes.append(note)

    composition = pretty_midi.PrettyMIDI()
    for pretty_midi_instrument in pretty_midi_instruments:
        composition.instruments.append(pretty_midi_instrument)
    composition.write(midi_path)


def create_tsv_events_from_fragment(
        fragment: Fragment,
        events_path: str,
        beat_in_seconds: float,
        instruments: dict[int, str],
        effects: dict[int, str],
        velocity: float,
        opening_silence_in_seconds: float = 1.0
) -> None:
    """
    Create TSV file with `sinethesizer` events from a fragment.

    :param fragment:
        musical fragment
    :param events_path:
        path to a file where result is going to be saved
    :param beat_in_seconds:
        duration of one beat in seconds
    :param instruments:
        mapping from IDs of melodic lines to names of instruments (from a `sinethesizer` preset)
        that play them
    :param effects:
        mapping from IDs of melodic lines to sound effects to be applied to all events
        from the corresponding lines
    :param velocity:
        one common velocity for all events
    :param opening_silence_in_seconds:
        number of seconds with silence to add at the start of the composition
    :return:
        None
    """
    all_notes = get_list_of_notes()
    events = []
    for line_id, melodic_line in zip(fragment.line_ids, fragment.melodic_lines):
        instrument = instruments[line_id]
        line_effects = effects.get(line_id, '')
        for event in melodic_line:
            if event.pitch_class in ['pause', 'skip']:
                continue
            start_time = event.start_time * beat_in_seconds
            start_time += opening_silence_in_seconds
            duration_in_seconds = event.duration * beat_in_seconds
            pitch_id = event.position_in_semitones
            note = all_notes[pitch_id]
            event = (
                instrument,
                start_time,
                duration_in_seconds,
                note,
                pitch_id,
                line_effects,
                line_id
            )
            events.append(event)
    events = sorted(events, key=lambda x: (x[1], x[4], x[2]))
    events = [f"{x[0]}\t{x[1]}\t{x[2]}\t{x[3]}\t{velocity}\t{x[5]}\t{x[6]}" for x in events]

    columns = [
        'instrument',
        'start_time',
        'duration',
        'frequency',
        'velocity',
        'effects',
        'line_id'
    ]
    header = '\t'.join(columns)
    results = [header] + events
    with open(events_path, 'w') as out_file:
        for line in results:
            out_file.write(line + '\n')


def create_sinethesizer_instruments(n_melodic_lines: int) -> dict[str, Instrument]:
    """
    Create registry of `sinethesizer` instruments with adjusted amplitudes.

    :param n_melodic_lines:
        number of melodic lines in a fragment
    :return:
        mapping from instrument names to instruments itself
    """
    presets_path = resource_filename('dodecaphony', 'configs/sinethesizer_presets.yml')
    instruments_registry = create_instruments_registry(presets_path)
    normalized_registry = {}
    for name, instrument in instruments_registry.items():
        amplitude_scaling = instrument.amplitude_scaling
        amplitude_scaling /= n_melodic_lines
        normalized_registry[name] = Instrument(
            instrument.partials, amplitude_scaling, instrument.effects
        )
    return normalized_registry


def create_wav_from_tsv_events(
        events_path: str, output_path: str,
        instruments_registry: dict[str, Instrument],
        trailing_silence_in_seconds: float
) -> None:
    """
    Create WAV file based on `sinethesizer` TSV file.

    :param events_path:
        path to TSV file with track represented as `sinethesizer` events
    :param output_path:
        path where resulting WAV file is going to be saved
    :param instruments_registry:
        mapping from instrument names to instruments itself
    :param trailing_silence_in_seconds:
        number of seconds with silence to add at the end of the composition
    :return:
        None
    """
    settings = {
        'frame_rate': 48000,
        'trailing_silence': trailing_silence_in_seconds,
        'instruments_registry': instruments_registry,
    }
    events = convert_tsv_to_events(events_path, settings)
    timeline = convert_events_to_timeline(events, settings)
    write_timeline_to_wav(output_path, timeline, settings['frame_rate'])


def create_yaml_from_fragment(fragment: Fragment, yaml_path: str) -> None:
    """
    Create YAML file that can be used for setting temporal and sonic content in a runtime config.

    :param fragment:
        musical fragment
    :param yaml_path:
        path to a file where result is going to be saved
    :return:
        None
    """
    temporal_content = {
        i: {'durations': durations}
        for i, durations in enumerate(fragment.temporal_content)
    }
    sonic_content = {
        i: {'pitch_classes': pitch_classes}
        for i, pitch_classes in enumerate(fragment.sonic_content)
    }
    result = {'temporal_content': temporal_content, 'sonic_content': sonic_content}
    with open(yaml_path, 'w') as out_file:
        yaml.dump(result, out_file)


def make_lilypond_template(
        n_voices: int,
        tonic: str,
        scale_type: str,
        meter_numerator: int,
        meter_denominator: int
) -> str:
    """
    Make template of Lilypond text file.

    :param n_voices:
        number of voices in a fragment to be rendered
    :param tonic:
        tonic pitch class represented by letter (like C or A#)
    :param scale_type:
        type of scale (e.g., 'major', 'natural_minor', 'harmonic_minor', 'dorian', and so on)
    :param meter_numerator:
        numerator in meter signature, i.e., number of reference beats per measure
    :param meter_denominator:
        denominator in meter signature, i.e., ratio of reference beat duration to measure duration
    :return:
        template
    """
    raw_template = (
        "\\version \"2.18.2\"\n"
        "\\layout {{{{\n"
        "    indent = #0\n"
        "}}}}\n"
        "\\new StaffGroup <<\n"
        "    \\new Staff <<\n"
        "        \\clef treble\n"
        "        \\time {}/{}\n"
        "        \\key {} \\{}\n"
        "{}"
        "    >>\n"
        "    \\new Staff <<\n"
        "        \\clef bass\n"
        "        \\time {}/{}\n"
        "        \\key {} \\{}\n"
        "{}"
        "    >>\n"
        ">>"
    )
    tonic = tonic.replace('#', 'is').replace('b', 'es').lower()
    scale_type = scale_type.split('_')[-1]
    voices = ["        {{{}}}\n" for _ in range(n_voices)]
    treble_bass_threshold = ceil(n_voices / 2)
    template = raw_template.format(
        meter_numerator, meter_denominator, tonic, scale_type,
        "        \\\\\n".join(voices[:treble_bass_threshold]),
        meter_numerator, meter_denominator, tonic, scale_type,
        "        \\\\\n".join(voices[treble_bass_threshold:])
    )
    return template


def get_lilypond_order_of_voices(n_voices: int) -> list[int]:
    """
    Enumerate voices (from highest to lowest) in Lilypond order.

    See more about Lilypond order ('Voice order' section):
    http://lilypond.org/doc/v2.18/Documentation/notation/multiple-voices

    :param n_voices:
        number of voices in a fragment to be rendered
    :return:
        indices of voices in Lilypond order
    """
    def enumerate_for_one_staff(n_voices_at_staff: int) -> list[int]:
        max_index = n_voices_at_staff - 1
        results = []
        for i in range(n_voices_at_staff):
            result = max_index - int(round(2 * abs(i - max_index / 2)))
            result += int(i < max_index / 2)
            results.append(result)
        return results

    n_voices_at_upper_staff = ceil(n_voices / 2)
    n_voices_at_lower_staff = floor(n_voices / 2)
    lower_voices_priorities = [
        x + n_voices_at_upper_staff
        for x in enumerate_for_one_staff(n_voices_at_lower_staff)
    ]
    upper_voices_priorities = enumerate_for_one_staff(n_voices_at_upper_staff)
    priorities = lower_voices_priorities + upper_voices_priorities
    ordering = sorted(
        list((index, priority) for index, priority in enumerate(priorities)),
        key=lambda x: x[1], reverse=True
    )
    ordering = [x[0] for x in ordering]
    return ordering


def find_lilypond_duration(
        duration: float,
        time_in_measure: float,
        meter_numerator: int,
        meter_denominator: int
) -> list[str]:
    """
    Find duration of a note in Lilypond reciprocal format.

    :param duration:
        duration of a note in beats
    :param time_in_measure:
        number of previous beats passed from the start of the current measure
    :param meter_numerator:
        numerator in meter signature, i.e., number of reference beats per measure
    :param meter_denominator:
        denominator in meter signature, i.e., ratio of reference beat duration to measure duration
    :return:
        strings representing Lilypond durations of notes (a note can be split to multiple notes
        if it crosses bar or if its duration is compound and requires ties)
    """
    if time_in_measure + duration <= meter_numerator:
        reciprocal_duration = meter_denominator / duration
        supported_reciprocal_durations = {
            16: ['16'],
            32 / 3: ['16.'],
            8: ['8'],
            16 / 3: ['8.'],
            4: ['4'],
            3.2: ['4~', '16'],
            1 / 0.34375: ['4~', '16.'],
            8 / 3: ['4.'],
            1 / 0.4375: ['4.~', '16'],
            1 / 0.46875: ['4.~', '16.'],
            2: ['2'],
            1 / 0.5625: ['2~', '16'],
            1 / 0.59375: ['2~', '16.'],
            1.6: ['2~', '8'],
            1 / 0.6875: ['2~', '8.'],
            4 / 3: ['2.'],
            1 / 0.8125: ['2.~', '16'],
            1 / 0.84375: ['2.~', '16.'],
            8 / 7: ['2.~', '8'],
            1 / 0.9375: ['2.~', '8.'],
            1: ['1'],
            2 / 3: ['1.'],
            0.5: ['\\breve'],
            1 / 3: ['\\breve.'],
            0.25: ['\\longa'],
            1 / 6: ['\\longa.'],
        }
        lilypond_duration = supported_reciprocal_durations.get(reciprocal_duration)
        if lilypond_duration is None:
            raise RuntimeError(f"Reciprocal duration {reciprocal_duration} is not supported yet.")
        return lilypond_duration
    else:
        remaining_duration = meter_numerator - time_in_measure
        remaining_results = find_lilypond_duration(
            remaining_duration,
            time_in_measure,
            meter_numerator,
            meter_denominator
        )
        remaining_results[-1] = remaining_results[-1] + '~'
        left_over_bar_duration = duration - meter_numerator + time_in_measure
        left_over_bar_results = find_lilypond_duration(
            left_over_bar_duration,
            0,
            meter_numerator,
            meter_denominator
        )
        return remaining_results + left_over_bar_results


def convert_to_lilypond_note(
        event: Event,
        meter_numerator: int,
        meter_denominator: int
) -> str:
    """
    Convert `Event` instance to note in Lilypond absolute notation.

    :param event:
        event
    :param meter_numerator:
        numerator in meter signature, i.e., number of reference beats per measure
    :param meter_denominator:
        denominator in meter signature, i.e., ratio of reference beat duration to measure duration
    :return:
        note in Lilypond absolute notation
    """
    if event.pitch_class == 'pause':
        note_without_duration = 'r'
    else:
        pitch_class = event.pitch_class
        pitch_class = pitch_class.replace('#', 'is').replace('b', 'es')
        pitch_class = pitch_class.lower()

        octave_id = int(POSITION_TO_NOTE[event.position_in_semitones][-1])
        lilypond_default_octave_id = 3
        octave_diff = octave_id - lilypond_default_octave_id
        octave_sign = "'" if octave_diff >= 0 else ','
        octave_info = "".join(octave_sign for _ in range(abs(octave_diff)))

        note_without_duration = pitch_class + octave_info

    start_time = event.start_time
    time_in_measure = start_time % meter_numerator
    durations = find_lilypond_duration(
        event.duration,
        time_in_measure,
        meter_numerator,
        meter_denominator
    )
    note = [f"{note_without_duration}{duration}" for duration in durations]
    note = " ".join(note)
    if event.pitch_class in ['pause', 'skip']:
        note = note.replace('~', '')  # Lilypond warns when pauses or skips are tied.
    return note


def create_lilypond_file_from_fragment(fragment: Fragment, output_path: str) -> None:
    """
    Create text file in format of Lilypond sheet music editor.

    :param fragment:
        musical fragment
    :param output_path:
        path where resulting file is going to be saved
    :return:
        None
    """
    n_voices = len(fragment.melodic_lines)
    template = make_lilypond_template(
        n_voices,
        # For a dodecaphonic piece, there is no tonic and no scale type
        # and the two below constants affect nothing.
        'C',
        'major',
        fragment.meter_numerator,
        fragment.meter_denominator
    )
    lilypond_voices = []
    indices = get_lilypond_order_of_voices(n_voices)
    for index in indices:
        melodic_line = fragment.melodic_lines[index]
        lilypond_voice = []
        for event in melodic_line:
            note = convert_to_lilypond_note(event, fragment.meter_numerator, fragment.meter_denominator)
            lilypond_voice.append(note)
        lilypond_voice = " ".join(lilypond_voice)
        lilypond_voices.append(lilypond_voice)
    result = template.format(*lilypond_voices)
    with open(output_path, 'w') as out_file:
        out_file.write(result)


def create_pdf_sheet_music_with_lilypond(lilypond_path: str) -> None:  # pragma: no cover
    """
    Create PDF file with sheet music.

    :param lilypond_path:
        path to a text file in Lilypond format
    :return:
        None:
    """
    dir_path, filename = os.path.split(lilypond_path)
    bash_command = f"lilypond {filename}"
    try:
        process = subprocess.Popen(
            bash_command.split(),
            cwd=dir_path,
            stdout=subprocess.PIPE
        )
        process.communicate()
    except Exception:
        print("Rendering sheet music to PDF failed. Do you have Lilypond?")
        print(traceback.format_exc())


def render(fragment: Fragment, rendering_params: dict[str, Any]) -> None:  # pragma: no cover
    """
    Save fragment to MIDI, WAV, TSV, PDF, and Lilypond files.

    :param fragment:
        musical fragment
    :param rendering_params:
        settings of fragment saving
    :return:
        None
    """
    top_level_dir = rendering_params['dir']
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S,%f")
    nested_dir = os.path.join(top_level_dir, f"result_{now}")
    os.mkdir(nested_dir)

    midi_path = os.path.join(nested_dir, 'music.mid')
    midi_params = rendering_params['midi']
    common_params = rendering_params['common'].copy()
    create_midi_from_fragment(fragment, midi_path, **midi_params, **common_params)

    events_path = os.path.join(nested_dir, 'sinethesizer_events.tsv')
    events_params = rendering_params['sinethesizer']
    trailing_silence_in_sec = common_params.pop('trailing_silence_in_seconds')
    create_tsv_events_from_fragment(fragment, events_path, **events_params, **common_params)

    wav_path = os.path.join(nested_dir, 'music.wav')
    n_melodic_lines = len(fragment.melodic_lines)
    instruments_registry = create_sinethesizer_instruments(n_melodic_lines)
    create_wav_from_tsv_events(events_path, wav_path, instruments_registry, trailing_silence_in_sec)

    yaml_path = os.path.join(nested_dir, 'content.yml')
    create_yaml_from_fragment(fragment, yaml_path)

    lilypond_path = os.path.join(nested_dir, 'sheet_music.ly')
    create_lilypond_file_from_fragment(fragment, lilypond_path)
    create_pdf_sheet_music_with_lilypond(lilypond_path)
