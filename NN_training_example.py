import numpy as np
from light_prop.algorithms import NNTrainer
from light_prop.calculations import get_gaussian_distribution
from light_prop.lightfield import LightField
from light_prop.propagation.params import PropagationParams
from light_prop.visualisation import GeneratePropagationPlot
from light_prop.calculations import gaussian

if __name__ == "__main__":
    params = PropagationParams.get_example_propagation_data()

#Choose proper propagation parameters
    params.sigma = 4
    params.matrix_size = 128
    params.pixel = 0.9

#Define target optical field and input amplitude
    x_shift1 = 50
    x_shift2 = 25
    target = np.array(
        [[gaussian(np.sqrt((x - x_shift1) ** 2 + y ** 2), params.sigma) for x in
          np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel] for y in
         np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel]) + np.array(
        [[gaussian(np.sqrt((x - x_shift2) ** 2 + y ** 2), params.sigma) for x in
          np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel] for y in
         np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel])
    params.sigma = 50
    amp = get_gaussian_distribution(params)
    phase = np.array(
        [[0 for x in
          np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel] for y in
         np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel])

#Build NNTrainer
    NN = NNTrainer(params)

# Run NN optimization
    trained_model = NN.optimize(LightField(amp, phase), LightField(target, phase), 10)

#Plot the result - optimized phase map
    plotter = GeneratePropagationPlot(LightField(amp, np.array(trained_model.layers[3].get_weights()[0])), output_type=GeneratePropagationPlot.PLOT_PHASE)
    plotter.save_output_as_figure("outs/NNstructure.png")
    plotter.show()
