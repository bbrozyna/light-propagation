# Light - propagation
Console app made to calculate field distribution transformation using the given propagation method based on [article](https://www.researchgate.net/publication/357437782_Neural-network_based_approach_to_optimize_THz_computer_generated_holograms)

### Currently supported method
1. FFT convolution
2. Neural Networks convolution

### Installation
Due to dependencies it is suggested to run this app in separate virtual environment

#### Installing required libraries

`python -m pip install -r requirements.txt`

#### CLI Usage:

`python console.py --help`

```commandline
Calculates propagation matrix with given method based on params in json input. Output is save as an image in given path [-h] [-m {conv,faith,sequential}] [--path PATH] json

positional arguments:
  json                  Json file to retrieve data

optional arguments:
  -h, --help      Shows this help message and exits
  -m, --method    Method used to calculate output matrix
  --path          Path to save output

```

#### Example:

To calculate simple propagation using convolution method and save the results in `outs/test.png` try:

`python console.py params.json --method conv --path outs/test.png`


### Linting

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

### Running Unit tests
UT are written using pytest framework. To start tests use:

`python -m pytest`
