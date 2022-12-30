from datetime import datetime

import numpy as np

from lightprop.calculations import H_off_axis, get_gaussian_distribution
from lightprop.lightfield import LightField
from lightprop.optimization.nn import NN_FFTTrainer
from lightprop.propagation.params import PropagationParams
from lightprop.visualisation import Plotter, PlotTypes

if __name__ == "__main__":
    params = PropagationParams.get_example_propagation_data()

    # Choose proper propagation parameters
    params.beam_diameter = 2
    params.matrix_size = 128
    params.pixel_size = 0.8
    # Define target optical field and input amplitude
    # In this example a simple focusing from wider Gaussian beam to the thinner one
    x0 = 20
    y0 = 0
    target = get_gaussian_distribution(params, x0, y0) + get_gaussian_distribution(params, -x0, y0)
    params.beam_diameter = 15
    amp = get_gaussian_distribution(params, 0, 0)
    phase = np.array(
        [
            [0 for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size]
            for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
        ]
    )

    # Build NNTrainer or NNMultiTrainer
    NN = NN_FFTTrainer()

    # Run NN optimization
    # In case of NNMultiTrainer provide kernel as 3rd argument.
    # Please try running different numbers of iterations (last parameter)
    # Check the difference in the output for different amounts of training

    kernel = np.array(
        [
            [
                H_off_axis(
                    x / np.sqrt(x**2 + params.distance**2) / params.wavelength,
                    y / np.sqrt(params.distance**2 + y**2) / params.wavelength,
                    params.distance,
                    params.wavelength,
                )
                for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
            ]
            for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
        ]
    )

    trained_model = NN.optimize(
        LightField(amp, phase, params.wavelength, params.pixel_size),
        LightField(target, phase, params.wavelength, params.pixel_size),
        LightField.from_complex_array(kernel, params.wavelength, params.pixel_size),
        params.distance,
        iterations=2000,
    )

    # Plot loss vs epochs
    NN.plot_loss()

    # Extract the optimized phase map from the trainable layer
    optimized_phase = np.array(trained_model.layers[3].get_weights()[0])

    current_datetime = datetime.now()
    str_current_datetime = current_datetime.strftime("%d.%m.%Y-%H_%M_%S")

    # Plot the result - optimized phase map
    plotter = Plotter(
        LightField(amp, optimized_phase, params.wavelength, params.pixel_size), output_type=PlotTypes.PHASE
    )
    plotter.save_output_as_figure("outs/Structure_" + str_current_datetime + ".png")

    # # Plot the target amplitude
    # plotter = Plotter(LightField(target, phase, params.wavelength, params.pixel_size), output_type=PlotTypes.ABS)
    # plotter.save_output_as_figure("outs/Target_" + str_current_datetime + ".png")

    # # Plot the input amplitude
    # plotter = Plotter(LightField(amp, phase, params.wavelength, params.pixel_size), output_type=PlotTypes.ABS)
    # plotter.save_output_as_figure("outs/Input_" + str_current_datetime + ".png")
    # Plot the result - output amplitude

    # Prepare input field and kernel
    field = np.array([amp, phase])
    field = field.reshape(
        (
            1,
            2,
            params.matrix_size,
            params.matrix_size,
        ),
        order="F",
    )
    kernel = np.array([np.real(kernel), np.imag(kernel)])
    kernel = kernel.reshape(
        (
            1,
            2,
            params.matrix_size,
            params.matrix_size,
        ),
        order="F",
    )

    # Evaluate model on the input field
    result = trained_model([field, kernel]).numpy()
    result = result[0, 0, :, :] * np.exp(1j * result[0, 1, :, :])

    # Plot the result
    plotter = Plotter(
        LightField.from_complex_array(result, params.wavelength, params.pixel_size), output_type=PlotTypes.ABS
    )
    plotter.save_output_as_figure("outs/Result_" + str_current_datetime + ".png")