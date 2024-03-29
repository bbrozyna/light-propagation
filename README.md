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

## Linting

To support linting and unify styling we suggest to use [pre-commit](https://pre-commit.com)

#### pre-commit installation

    pip install pre-commit

#### pre-commit usage

**Running pre-commit automatically**

To run validation automatically before each commit, please use:

    pre-commit install

This will add pre-commit to git hooks and perform all the checks defined in `.pre-commit-config.yaml`

**Running pre-commit manually**

To check stying in all files, please use

    pre-commit run -a

#### pre-commit in CI

Every pull request should pass pre-commit stage to be merged

## Usage:

****For a quick start, please refer to our examples:****

1. Exemplary usage of structure optimization through neural network optimization
    ```commandline
        python doc/examples/nn_training.py
    ```
2. Exemplary usage of structure optimization using Gerchberg-Saxton algorithm.
    ```commandline
        python doc/examples/gerchberg_saxton.py
    ```

#### Components:

1. `lightprop/io` - input/output functionalities, loading/saving images/files
2. `lightprop/propagation` - core set of classes defining propagation calculations (convolution, neural network)
3. `lightprop/lightfield` - field distribution representation using `A exp(i phi)` notation
4. `lightprop/optimization` - set of methods used for calculating phase distribution 
5. `lightprop/structures` - optical structures, like lens

## Contribution
If you are interested in using/improving/developing this project, don't hesitate to contact us using email: 
pawel.komorowski@wat.edu.pl

## License

[![license](https://img.shields.io/badge/license-MIT-green.svg)](hhttps://github.com/bbrozyna/light-propagation/blob/master/LICENSE)

This project is licensed under the terms of the [MIT license](/LICENSE).

## Running Unit tests

UT are written using pytest framework. To start tests use:

`python -m pytest`
