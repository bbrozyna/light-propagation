# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 14:09:21 2022

@author: dr PK & be BB with mse MD
"""

import light_prop.propagation.methods as prop
from light_prop.propagation.params import PropagationParams
from light_prop.lightfield import LightField


class GerchbergSaxton:
    def __init__(self, propagation_params: PropagationParams):
        self.params = propagation_params

    def optimize(self, input_field: LightField, target_field: LightField, iterations: int):
        temp_plane = target_field
        input_plane = None

        for i in range(iterations):
            temp_plane.amplitude = target_field.amplitude
            self.params.distance *= -1
            input_plane = prop.ConvolutionPropagation(self.params).propagate(temp_plane)
            input_plane.amplitude = input_field.amplitude
            self.params.distance *= -1
            temp_plane = prop.ConvolutionPropagation(self.params).propagate(temp_plane)

        return input_plane
