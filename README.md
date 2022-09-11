# Light propagation

Python app made to calculate field distribution transformation using the given propagation method based
on [article](https://www.researchgate.net/publication/357437782_Neural-network_based_approach_to_optimize_THz_computer_generated_holograms)

## Installation

Due to dependencies, it is suggested to run this app in a separate virtual environment like venv.

#### Creating virtual environment

Initialization

    python -m venv env

Activation (on Windows run activate.exe)

    source env/bin/activate.sh

#### Installing required libraries

    python -m pip install -r requirements.txt

## Usage:

****For a quick start, please refer to our examples:****

1. Exemplary usage of propagation with two methods - pure convolution and convolutional diffractive neural network.
```commandline
    python propagation_example.py
```

2. Exemplary usage of structure optimization through neural network optimization
```commandline
    python NN_training_example.py
```

3. Exemplary usage of structure optimization using Gerchberg-Saxton algorithm.
```commandline
    python GS_example.py
```

#### Components:

1. `lightprop/propagation/methods.py` - core set of classes defining propagation calculations (convolution, neural network)
2. `lightprop/lightfield.py` - field distribution representation using `A exp(i phi)`  notation
3. `lightprop/algorithms.py` - set of methods used for calculating phase distribution and 


## Contribution
If you are interested in using/improving/developing this project, don't hesitate to contact us using email: 
pawel.komorowski@wat.edu.pl

## License

[![license](https://img.shields.io/badge/license-MIT-green.svg)](hhttps://github.com/bbrozyna/light-propagation/blob/master/LICENSE)

This project is licensed under the terms of the [MIT license](/LICENSE).

## Running Unit tests

UT are written using pytest framework. To start tests use:

`python -m pytest`
