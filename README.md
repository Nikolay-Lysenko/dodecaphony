# Dodecaphony

## Overview

This is a configurable tool that generates twelve-tone fragments. The twelve-tone technique (also known as dodecaphony) is composition method that produces atonal music full of chromaticism and globally coherent due to deriving its content from the same sequence of 12 unique pitch classes.

Each run of the tool results in creation of a directory that contains:
* MIDI file;
* WAV file;
* Events file in [sinethesizer](https://github.com/Nikolay-Lysenko/sinethesizer) TSV format;
* PDF file with sheet music and its Lilypond source.

## Installation

To install a stable version, run:
```bash
pip install dodecaphony
```

## Usage

To create a new musical fragment, run:
```bash
python -m dodecaphony [-c path_to_config]
```

[Default config](https://github.com/Nikolay-Lysenko/dodecaphony/blob/master/dodecaphony/configs/default_config.yml) is used if `-c` argument is not passed. Advanced usage will be covered in a special guide.
