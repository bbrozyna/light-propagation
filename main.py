from light_prop.propagation_params import PropagationParams
from light_prop.propagation import ConvolutionPropagation
from light_prop.propagation_input import PropagationInput
from light_prop.visualisation import GeneratePropagationPlot


def propagate_with_function():
    params = PropagationParams.get_example_propagation_data()
    input = PropagationInput()
    input.calculate_standard_lens_from_params(params)
    cp = ConvolutionPropagation(params)
    result = cp.propagate(input)
    vis = GeneratePropagationPlot(result)
    vis.save_output_as_figure('test.png', output_type=GeneratePropagationPlot.PLOT_ABS)


if __name__ == "__main__":
    propagate_with_function()
