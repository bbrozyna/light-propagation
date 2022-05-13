import sys
import argparse
import logging

from light_prop.propagation_params import PropagationParams
from light_prop.propagation import ConvolutionPropagation, ConvolutionFaithPropagation, ConvolutionPropagationSequentialNN, NNPropagation
from light_prop.visualisation import GeneratePropagationPlot


class PropagationArgParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def configure_logger():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def get_supported_propagations():
    propagations = {
        "conv": ConvolutionPropagation,
        "faith": ConvolutionFaithPropagation,
        "sequential": ConvolutionPropagationSequentialNN,
        "NN": NNPropagation,
    }
    return propagations


def get_parsed_input():
    propagation_choices = get_supported_propagations().keys()
    parser = PropagationArgParser('Calculates propagation matrix with given method based on params in json input. Output is save as an image in given path')
    parser.add_argument('json', type=str, help='Json file to retrieve data')
    parser.add_argument('-m', '--method', choices=propagation_choices, help='Method used to calculate output matrix')
    parser.add_argument('--path', type=str, default='outs/out.png', help='Path to save output')
    return parser.parse_args()


def build_generator_with_options(options):
    json_filename = options.json
    prop_params = PropagationParams.get_params_from_json_file(json_filename)
    prop_strat = get_supported_propagations().get(options.method)
    prop_strat = prop_strat(propagation_params=prop_params)
    return GeneratePropagationPlot(propagation_strategy=prop_strat)


def main():
    configure_logger()
    options = get_parsed_input()
    output_path = options.path or 'outs/out.png'
    logging.info(f"Starting propagation with params {options.__dict__}")
    plotter = build_generator_with_options(options)
    plotter.save_output_abs_figure(output_path)


if __name__ == '__main__':
    main()
