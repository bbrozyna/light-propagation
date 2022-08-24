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
import matplotlib.pyplot as plt
import tensorflow as tf

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
        self.model = None
        self.history = None
    
    def amplitudeMSE(self, y_true, y_pred):
        squared_difference = tf.square(y_true[0, 0] - y_pred[0, 0])
        
        return tf.reduce_mean(squared_difference, axis=-1)

    def optimize(self, input_field: LightField, target_field: LightField, iterations: int):
        propagator = prop.NNPropagation(self.params)
        self.model = propagator.get_field_modifier()

        print("Model compiling")
        self.model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=3e-1),
            loss = self.amplitudeMSE,
        )

        checkpoint_filepath = './tmp/checkpoint'
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True
        )

        print("Fitting model to data")
        self.history = self.model.fit(
            propagator.get_field_distribution(input_field),
            propagator.get_field_distribution(target_field),
            batch_size = 1,
            epochs = iterations,
            callbacks = [model_checkpoint_callback]
        )

        self.model.load_weights(checkpoint_filepath)
        return self.model

    def plot_loss(self):
        plt.plot(self.history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.yscale('log')
        plt.show()
    
    
