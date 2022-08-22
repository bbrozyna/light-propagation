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
    params.sigma = 2
    params.matrix_size = 64
    params.pixel = 0.5


    #Define target optical field and input amplitude
    target = get_gaussian_distribution(params)
    params.sigma = 15
    amp = get_gaussian_distribution(params)
    phase = np.array(
        [[0 for x in
          np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel] for y in
         np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel])

    #Build NNTrainer
    NN = NNTrainer(params)

    # Run NN optimization
    trained_model = NN.optimize(LightField(amp, phase), LightField(target, phase), 1000)

    # Plot loss vs epochs
    NN.plot_loss()

    #Plot the result - optimized phase map
    plotter = GeneratePropagationPlot(LightField(amp, np.array(trained_model.layers[3].get_weights()[0])), output_type=GeneratePropagationPlot.PLOT_PHASE)
    plotter.save_output_as_figure("outs/NNstructure.png")

    plotter = GeneratePropagationPlot(LightField(target, phase), output_type=GeneratePropagationPlot.PLOT_ABS)
    plotter.save_output_as_figure("outs/NNtarget.png")

    plotter = GeneratePropagationPlot(LightField(amp, phase), output_type=GeneratePropagationPlot.PLOT_ABS)
    plotter.save_output_as_figure("outs/NNinput.png")

    field = np.array([amp, phase])
    field = field.reshape((1, 2, params.matrix_size, params.matrix_size,), order='F')
    result = trained_model(field).numpy()
    result = result.reshape(params.matrix_size, params.matrix_size, 2)[:, :, 0] + 1j * result.reshape(params.matrix_size, params.matrix_size, 2)[:, :, 1]
    plotter = GeneratePropagationPlot(LightField.from_complex_array(result), output_type=GeneratePropagationPlot.PLOT_ABS)
    plotter.save_output_as_figure("outs/NNresult.png")
