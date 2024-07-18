[![Build Status](https://github.com/Nikolay-Lysenko/dodecaphony/actions/workflows/main.yml/badge.svg)](https://github.com/Nikolay-Lysenko/dodecaphony/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/Nikolay-Lysenko/dodecaphony/branch/master/graph/badge.svg)](https://codecov.io/gh/Nikolay-Lysenko/dodecaphony)
[![Maintainability](https://api.codeclimate.com/v1/badges/b83bc51361ac046bc7eb/maintainability)](https://codeclimate.com/github/Nikolay-Lysenko/dodecaphony/maintainability)
[![PyPI version](https://badge.fury.io/py/dodecaphony.svg)](https://pypi.org/project/dodecaphony/)

# Dodecaphony

## Overview

This is a configurable tool that generates music written in [the twelve-tone technique](https://nikolay-lysenko.github.io/2024/04/30/introduction-to-twelve-tone-technique). In short, this means that the output pieces are based on different principles than the vast majority of classical and popular music. If you have never listened to the twelve-tone works, you may find them not ear-pleasant, but, given enough experience, you can change your mind. Twelve-tone music is logic-oriented and has many symmetries, so it is a good choice for combinatorial optimization.

Each run of the tool results in creation of a fragment which can be used as a part of a larger composition. These fragments are stored in separate directories containing:
* MIDI file
* WAV file
* Events file in [sinethesizer](https://github.com/Nikolay-Lysenko/sinethesizer) TSV format
* PDF file with sheet music and its Lilypond source
* YAML file that can be copied to a config for a derivative fragment
* Text file with meta information (such as path to config and evaluation scores)

With this tool, I released a 26-minutes album on Spotify and many other streaming platforms. However, main melodies for the album were written manually by myself and a lot of hard work was done with configs, but, nevertheless, the package really helped to generate background melodies. To read more about the album, please visit its [official page](https://nikolay-lysenko.github.io/2024/05/31/suite-for-virtual-pipe-organ-op1).

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

[Default config](https://github.com/Nikolay-Lysenko/dodecaphony/blob/master/dodecaphony/configs/default_config.yml) is used if `-c` argument is not passed. To create your own config, please look at [this example](https://github.com/Nikolay-Lysenko/dodecaphony/blob/master/docs/config_with_explanations.yml) with detailed explanations. Also, you can read the source code — it is structured and has built-in documentation.

If you are on macOS, please make sure that [parallelism is enabled](https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr).
