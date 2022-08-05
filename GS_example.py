# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:44:04 2022

@author: PK
"""

from light_prop.propagation.params import PropagationParams
from light_prop.lightfield import LightField
import light_prop.calculations as calc
import numpy as np
from light_prop.algorithms import GerchbergSaxton
from light_prop.visualisation import GeneratePropagationPlot


if __name__ == "__main__":
    params = PropagationParams.get_example_propagation_data()

    target_sigma = 2

    amp = np.array(
        [[calc.gaussian(np.sqrt(x ** 2 + y ** 2), params.sigma) for x in
          np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel] for y in
         np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel])

    target = amp = np.array(
        [[calc.gaussian(np.sqrt(x ** 2 + y ** 2), target_sigma) for x in
          np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel] for y in
         np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel])

    phase = np.array(
        [[0 for x in
          np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel] for y in
         np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel])

    GS = GerchbergSaxton(params)

    res = GS.optimize(LightField(amp, phase), LightField(target, phase), 5)

    plotter = GeneratePropagationPlot(res[1])
    plotter.save_output_as_figure("outs/structure.png", output_type=GeneratePropagationPlot.PLOT_PHASE)

    plotter2 = GeneratePropagationPlot(res[0])
    plotter2.save_output_as_figure("outs/result.png", output_type=GeneratePropagationPlot.PLOT_ABS)
