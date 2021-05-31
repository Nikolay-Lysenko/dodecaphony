[![Build Status](https://github.com/Nikolay-Lysenko/dodecaphony/actions/workflows/main.yml/badge.svg)](https://github.com/Nikolay-Lysenko/dodecaphony/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/Nikolay-Lysenko/dodecaphony/branch/master/graph/badge.svg)](https://codecov.io/gh/Nikolay-Lysenko/dodecaphony)
[![Maintainability](https://api.codeclimate.com/v1/badges/b83bc51361ac046bc7eb/maintainability)](https://codeclimate.com/github/Nikolay-Lysenko/dodecaphony/maintainability)
[![PyPI version](https://badge.fury.io/py/dodecaphony.svg)](https://pypi.org/project/dodecaphony/)

# Dodecaphony

## Overview

This is a configurable tool that generates twelve-tone fragments. The twelve-tone technique (also known as dodecaphony) is composition method that produces atonal music full of chromaticism and globally coherent due to deriving its content from the same sequence of 12 unique pitch classes.

Each run of the tool results in creation of a directory that contains:
* MIDI file;
* WAV file;
* Events file in [sinethesizer](https://github.com/Nikolay-Lysenko/sinethesizer) TSV format;
* PDF file with sheet music and its Lilypond source;
* YAML file that can be copied to a config for a derivative fragment.

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

If you are on Mac OS, please check that [parallelism is enabled](https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr).
