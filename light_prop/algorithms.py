# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 14:09:21 2022

@author: dr PK & be BB with mse MD
"""

import light_prop.propagation.methods as prop
from light_prop.propagation.params import PropagationParams
from light_prop.lightfield import LightField
from tensorflow import keras
import numpy as np

class GerchbergSaxton:
    def __init__(self, propagation_params: PropagationParams):
        self.params = propagation_params

    def optimize(self, input_field: LightField, target_field: LightField, iterations: int):
        output_plane = target_field
        input_plane = None

        for i in range(iterations):
            output_plane.amplitude = target_field.amplitude
            self.params.distance *= -1
            input_plane = prop.ConvolutionPropagation(self.params).propagate(output_plane)
            input_plane.amplitude = input_field.amplitude
            self.params.distance *= -1
            output_plane = prop.ConvolutionPropagation(self.params).propagate(input_plane)

        return (input_plane, output_plane)

class NNTrainer:
    def __init__(self, propagation_params: PropagationParams):
        self.params = propagation_params
    
    def optimize(self, input_field: LightField, target_field: LightField, iterations: int):
        output_plane = target_field
        input_plane = None
        propagator = prop.NNPropagation(self.params)
        model = propagator.get_field_modifier()

        print("Model compiling")
        model.compile(
            optimizer = keras.optimizers.RMSprop(learning_rate=1e-3),
            loss = keras.losses.MeanSquaredError(),
            metrics = [keras.metrics.MeanSquaredError()],
        )

        print("Fitting model to data")
        history = model.fit(
            propagator.get_field_distribution(input_field),
            propagator.get_field_distribution(input_field).reshape((1, self.params.matrix_size, self.params.matrix_size, 2), order='F'),
            batch_size = 1,
            epochs = iterations,
        )
        
        print("Optimization statistics")
        print(history.history)

        return model

