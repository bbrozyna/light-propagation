# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 14:09:21 2022

@author: PK
"""

import light_prop.propagation as prop
import light_prop.lightfield as lf
from light_prop.propagation_params import PropagationParams

class GerchbergSaxton:
    def __init__(self, propagation_params: PropagationParams):
        self.params = propagation_params
        
    def optimize(self, input_field: lf.LightField, target: lf.LightField, n: int):
        
        output_plane = target
        
        for i in range(n):
            output_plane.amplitude = target.amplitude
            self.params.distance*=-1;
            conv = prop.ConvolutionPropagation(self.params)
            input_plane = conv.propagate(output_plane)
            input_plane.amplitude = input_field.amplitude
            self.params.distance*=-1;
            output_plane = conv.propagate(input_plane)
            
        return (input_plane, output_plane)
    