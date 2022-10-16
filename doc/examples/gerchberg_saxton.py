# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:44:04 2022

@author: PK
"""
import logging

import numpy as np

from lightprop.calculations import gaussian, get_gaussian_distribution
from lightprop.lightfield import LightField
from lightprop.optimization.gs import GerchbergSaxton
from lightprop.propagation.params import PropagationParams
from lightprop.visualisation import Plotter, PlotTypes

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    params = PropagationParams.get_example_propagation_data()

    # Choose proper propagation parameters
    params.beam_diameter = 4
    params.matrix_size = 256
    params.pixel_size = 0.9

    # Define target optical field and input amplitude
    # In this example two focal points placed outside the main optical axis
    x_shift1 = 50
    x_shift2 = 25
    target = np.array(
        [
            [
                gaussian(np.sqrt((x - x_shift1) ** 2 + y**2), params.beam_diameter)
                for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
            ]
            for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
        ]
    ) + np.array(
        [
            [
                gaussian(np.sqrt((x - x_shift2) ** 2 + y**2), params.beam_diameter)
                for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
            ]
            for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
        ]
    )
    params.beam_diameter = 50
    amp = get_gaussian_distribution(params)
    phase = np.array(
        [
            [0 for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size]
            for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
        ]
    )

    # Prepare optimizer
    GS = GerchbergSaxton(params)

    # Run optimizer
    input_plane, output_plane = GS.optimize(LightField(amp, phase), LightField(target, phase), iterations=3)

    # Plot the result - optimized phase map
    plotter = Plotter(input_plane, output_type=PlotTypes.PHASE)
    plotter.save_output_as_figure("outs/structure.png")
    plotter.show()

    # Plot the input amplitude
    plotter = Plotter(input_plane, output_type=PlotTypes.ABS)
    plotter.save_output_as_figure("outs/input_field.png")

    # Plot the result - output amplitude
    plotter = Plotter(output_plane, output_type=PlotTypes.ABS)
    plotter.save_output_as_figure("outs/result.png")
    plotter.show()

    # Plot the target amplitude
    plotter = Plotter(LightField(target, phase), output_type=PlotTypes.ABS)
    plotter.save_output_as_figure("outs/target.png")
    plotter.show()
