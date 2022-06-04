from light_prop.propagation_params import PropagationParams
from light_prop.propagation import ConvolutionPropagation, ConvolutionPropagationSequentialNN
from light_prop.visualisation import GeneratePropagationPlot


if __name__ == "__main__":
    params = PropagationParams.get_example_propagation_data()
    cp = ConvolutionPropagation(params)
    result = cp.propagate()
    vis = GeneratePropagationPlot(result)
    vis.save_output_as_figure('test.png')
