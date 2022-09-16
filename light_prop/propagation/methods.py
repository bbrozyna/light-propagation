"""
Light propagation with convolution method

Models:
Paweł Komorowski
pawel.komorowski@wat.edu.pl
"""
import logging

import numpy as np
from keras.layers import Convolution2D
from scipy import signal
from tensorflow import keras

from light_prop.calculations import h
from light_prop.lightfield import LightField
from light_prop.propagation.keras_layers import Aexp, ReIm_convert, Structure
from light_prop.propagation.params import PropagationParams


class BasePropagation:
    def __int__(self, propagation_params: PropagationParams):
        self.params = propagation_params

    def propagate(self, propagation_input: LightField) -> LightField:
        logging.info("Calculating propagation")
        field_distribution = self.get_field_distribution(propagation_input)
        field_modifier = self.get_field_modifier()
        output = self.reshape_output(self.calculate_propagation(field_distribution, field_modifier))
        return LightField.from_complex_array(output, self.params.nu)

    def get_field_distribution(self, propagation_input):
        return propagation_input.to_complex()

    def get_field_modifier(self):
        raise NotImplementedError("Please implement field modifier")

    def calculate_propagation(self, field_distribution, field_modifier):
        return None

    def reshape_output(self, data):
        return data


class ConvolutionPropagation(BasePropagation):
    def __init__(self, propagation_params):
        super().__int__(propagation_params)

    def get_field_modifier(self):
        hkernel = np.array(
            [
                [
                    h(
                        np.sqrt(x**2 + y**2),
                        self.params.distance,
                        self.params.wavelength,
                    )
                    for x in np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2)
                    * self.params.pixel_size
                ]
                for y in np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2) * self.params.pixel_size
            ]
        )
        return hkernel

    def calculate_propagation(self, field_distribution, field_modifier):
        return signal.fftconvolve(field_distribution, field_modifier, mode="same")


class NNPropagation(ConvolutionPropagation):
    def __init__(self, propagation_params):
        super().__init__(propagation_params)

    def get_field_distribution(self, propagation_input):
        field = super().get_field_distribution(propagation_input)
        field = np.array([np.real(field), np.imag(field)])
        field = field.reshape(
            (
                1,
                2,
                self.params.matrix_size,
                self.params.matrix_size,
            ),
            order="F",
        )
        return field

    def custom_weights(self, shape, dtype=None, re=False):
        func = np.sin if re else np.cos
        kernel = np.array(
            [
                [
                    1
                    / (self.params.distance * self.params.wavelength)
                    * func(
                        np.pi * np.sqrt(x**2 + y**2) ** 2 / (self.params.distance * self.params.wavelength)
                        + 2 * np.pi * self.params.distance / self.params.wavelength
                    )
                    for x in np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2)
                    * self.params.pixel_size
                ]
                for y in np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2) * self.params.pixel_size
            ]
        )
        kernel = kernel.reshape(self.params.matrix_size, self.params.matrix_size, 1, 1)
        return kernel

    def custom_weights_Re(self, shape, dtype=None):
        return self.custom_weights(shape, dtype, re=True)

    def custom_weights_Im(self, shape, dtype=None):
        return self.custom_weights(shape, dtype, re=False)

    def get_field_modifier(self):
        inputs = keras.Input(shape=(2, self.params.matrix_size, self.params.matrix_size))
        x = Aexp()(inputs)
        x = keras.layers.Reshape((2, self.params.matrix_size, self.params.matrix_size))(x)

        x = Structure(kernel_initializer=keras.initializers.Zeros())(x)
        x = keras.layers.Reshape((2, self.params.matrix_size, self.params.matrix_size))(x)

        x = ReIm_convert()(x)
        x = keras.layers.Reshape((2, self.params.matrix_size, self.params.matrix_size))(x)

        Re = keras.layers.Cropping2D(cropping=((1, 0), (0, 0)))(x)
        Re = keras.layers.Reshape((self.params.matrix_size, self.params.matrix_size, 1))(Re)
        Im = keras.layers.Cropping2D(cropping=((0, 1), (0, 0)))(x)
        Im = keras.layers.Reshape((self.params.matrix_size, self.params.matrix_size, 1))(Im)

        ReRe = Convolution2D(
            1,
            self.params.matrix_size,
            padding="same",
            kernel_initializer=self.custom_weights_Re,
            use_bias=False,
        )(Re)
        ImRe = Convolution2D(
            1,
            self.params.matrix_size,
            padding="same",
            kernel_initializer=self.custom_weights_Im,
            use_bias=False,
        )(Re)
        ReIm = Convolution2D(
            1,
            self.params.matrix_size,
            padding="same",
            kernel_initializer=self.custom_weights_Re,
            use_bias=False,
        )(Im)
        ImIm = Convolution2D(
            1,
            self.params.matrix_size,
            padding="same",
            kernel_initializer=self.custom_weights_Im,
            use_bias=False,
        )(Im)

        Re = keras.layers.Subtract()([ReRe, ImIm])
        Im = keras.layers.Add()([ReIm, ImRe])
        x = keras.layers.Concatenate(axis=1)([Re, Im])
        x = keras.layers.Reshape((2, self.params.matrix_size, self.params.matrix_size))(x)
        x = Aexp()(x)
        outputs = keras.layers.Reshape((2, self.params.matrix_size, self.params.matrix_size))(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        for layer in model.layers[:]:
            layer.trainable = False
        model.layers[3].trainable = True

        return model

    def calculate_propagation(self, field_distribution, field_modifier):
        conv = field_modifier(field_distribution)
        return conv.numpy()

    def reshape_output(self, data):
        return data[0, 0, :, :] * np.exp(1j * data[0, 1, :, :])

class MultiparameterNNPropagation(ConvolutionPropagation):
    def __init__(self, propagation_params):
        super().__init__(propagation_params)

    def get_field_distribution(self, propagation_input):
        field = super().get_field_distribution(propagation_input)
        field = np.array([np.real(field), np.imag(field)])
        field = field.reshape(
            (
                1,
                2,
                self.params.matrix_size,
                self.params.matrix_size,
            ),
            order="F",
        )
        return field

    def custom_weights(self, shape, dtype=None, re=False):
        func = np.sin if re else np.cos
        kernel = np.array(
            [
                [
                    1
                    / (self.params.distance * self.params.wavelength)
                    * func(
                        np.pi * np.sqrt(x**2 + y**2) ** 2 / (self.params.distance * self.params.wavelength)
                        + 2 * np.pi * self.params.distance / self.params.wavelength
                    )
                    for x in np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2)
                    * self.params.pixel_size
                ]
                for y in np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2) * self.params.pixel_size
            ]
        )
        kernel = kernel.reshape(self.params.matrix_size, self.params.matrix_size, 1, 1)
        return kernel

    def custom_weights_Re(self, shape, dtype=None):
        return self.custom_weights(shape, dtype, re=True)

    def custom_weights_Im(self, shape, dtype=None):
        return self.custom_weights(shape, dtype, re=False)

    def get_field_modifier(self):
        inputField = keras.Input(shape=(2, self.params.matrix_size, self.params.matrix_size))
        inputParams = keras.Input(shape=(1,))

        y = inputParams

        x = Aexp()(inputField)
        x = keras.layers.Reshape((2, self.params.matrix_size, self.params.matrix_size))(x)

        x = Structure(kernel_initializer=keras.initializers.Zeros())(x)
        x = keras.layers.Reshape((2, self.params.matrix_size, self.params.matrix_size))(x)

        x = ReIm_convert()(x)
        x = keras.layers.Reshape((2, self.params.matrix_size, self.params.matrix_size))(x)

        Re = keras.layers.Cropping2D(cropping=((1, 0), (0, 0)))(x)
        Re = keras.layers.Reshape((self.params.matrix_size, self.params.matrix_size, 1))(Re)
        Im = keras.layers.Cropping2D(cropping=((0, 1), (0, 0)))(x)
        Im = keras.layers.Reshape((self.params.matrix_size, self.params.matrix_size, 1))(Im)

        ReRe = Convolution2D(
            1,
            self.params.matrix_size,
            padding="same",
            kernel_initializer=self.custom_weights_Re,
            use_bias=False,
        )(Re)
        ImRe = Convolution2D(
            1,
            self.params.matrix_size,
            padding="same",
            kernel_initializer=self.custom_weights_Im,
            use_bias=False,
        )(Re)
        ReIm = Convolution2D(
            1,
            self.params.matrix_size,
            padding="same",
            kernel_initializer=self.custom_weights_Re,
            use_bias=False,
        )(Im)
        ImIm = Convolution2D(
            1,
            self.params.matrix_size,
            padding="same",
            kernel_initializer=self.custom_weights_Im,
            use_bias=False,
        )(Im)

        Re = keras.layers.Subtract()([ReRe, ImIm])
        Im = keras.layers.Add()([ReIm, ImRe])
        x = keras.layers.Concatenate(axis=1)([Re, Im])
        x = keras.layers.Reshape((2, self.params.matrix_size, self.params.matrix_size))(x)
        x = Aexp()(x)
        outputs = keras.layers.Reshape((2, self.params.matrix_size, self.params.matrix_size))(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        for layer in model.layers[:]:
            layer.trainable = False
        model.layers[3].trainable = True

        return model

    def calculate_propagation(self, field_distribution, field_modifier):
        conv = field_modifier(field_distribution)
        return conv.numpy()

    def reshape_output(self, data):
        return data[0, 0, :, :] * np.exp(1j * data[0, 1, :, :])
