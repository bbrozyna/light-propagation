"""
Light propagation with convolution method

Paweł Komorowski
pawel.komorowski@wat.edu.pl
"""
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from calculations import h, gaussian, lens
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Convolution2D

from propagation_params import PropagationParams


class BasePropagation:
    def __int__(self, propagation_params: PropagationParams):
        self.params = propagation_params
        # self.size = propagation_params.matrix_size
        # self.wavelength = propagation_params.wavelength
        # self.pixel = propagation_params.pixel
        # self.sigma = propagation_params.sigma
        # self.focal_length = propagation_params.focal_length
        # self.z = propagation_params.distance

    def propagate(self):
        field_distribution = self.get_field_distribution()
        field_modifier = self.get_field_modifier()
        return self.calculate_propagation(field_distribution, field_modifier)

    def get_field_distribution(self):
        raise NotImplemented("Please implement field distribution")

    def get_field_modifier(self):
        raise NotImplemented("Please implement field modifier")

    def calculate_propagation(self, field_distribution, field_modifier):
        pass


class ConvolutionPropagation(BasePropagation):
    def __init__(self, propagation_params):
        super().__int__(propagation_params)

    def get_field_distribution(self):
        lens_distribution = np.array(
            [[lens(np.sqrt(x ** 2 + y ** 2), self.params.focal_length, self.params.wavelength) for x in
              np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2) * self.params.matrix_size] for y in
             np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2) * self.params.pixel])
        init_amplitude = np.array(
            [[gaussian(np.sqrt(x ** 2 + y ** 2), self.params.sigma) for x in
              np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2) * self.params.pixel] for y in
             np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2) * self.params.pixel])
        return init_amplitude * lens_distribution

    def get_field_modifier(self):
        hkernel = np.array(
            [[h(np.sqrt(x ** 2 + y ** 2), self.params.distance, self.params.wavelength) for x in
              np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2) * self.params.pixel] for y in
             np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2) * self.params.pixel])
        return hkernel

    def calculate_propagation(self, field_distribution, field_modifier):
        return signal.fftconvolve(field_distribution, field_modifier, mode='same')


class ConvolutionPropagationSequentialNN(ConvolutionPropagation):
    def custom_weights(self, shape, dtype=None):
        kernel = np.array(
            [[h(np.sqrt(x ** 2 + y ** 2), self.params.distance, self.params.wavelength) for x in
              np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2) * self.params.pixel] for y in
             np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2) * self.params.pixel])
        kernel = kernel.reshape(self.params.matrix_size, self.params.matrix_size, 1, 1)
        return kernel

    def get_field_distribution(self):
        field = super().get_field_distribution()
        field = field.reshape(1, self.params.matrix_size, self.params.matrix_size, 1)
        return field

    def get_field_modifier(self):
        model = Sequential()
        model.add(Convolution2D(1, kernel_size=(self.params.matrix_size, self.params.matrix_size), padding="same", use_bias=False,
                                kernel_initializer=self.custom_weights, input_shape=(self.params.matrix_size, self.params.matrix_size, 1)))
        return model

    def calculate_propagation(self, field_distribution, field_modifier):
        return field_modifier(field_distribution).numpy().reshape(self.params.matrix_size, self.params.matrix_size, 1)


class ConvolutionFaithPropagation(ConvolutionPropagation):
    def __init__(self, propagation_params):
        super().__init__(propagation_params)

    def get_field_distribution(self):
        field = super().get_field_distribution()
        field = np.array([[[np.real(field)[i, j], np.imag(field)[i, j]] for i in range(self.params.matrix_size)] for j in range(self.params.matrix_size)])
        return field.reshape((1, self.params.matrix_size, self.params.matrix_size, 2), order='F')

    def custom_weights(self, shape, dtype=None, re=False):
        func = np.sin if re else np.cos
        kernel = np.array([[1 / (self.params.distance * self.params.wavelength) * func(
            np.pi * np.sqrt(x ** 2 + y ** 2) ** 2 / (self.params.distance * self.params.wavelength) + 2 * np.pi * self.params.distance / self.params.wavelength)
                            for x in
                            np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2) * self.params.pixel] for y in
                           np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2) * self.params.pixel])
        kernel = kernel.reshape(self.params.matrix_size, self.params.matrix_size, 1, 1)
        return kernel

    def custom_weights_Re(self, shape, dtype=None):
        return self.custom_weights(shape, dtype, re=True)

    def custom_weights_Im(self, shape, dtype=None):
        return self.custom_weights(shape, dtype, re=False)

    def get_field_modifier(self):
        inputs = keras.Input(shape=(self.params.matrix_size, self.params.matrix_size, 2))
        x = keras.layers.Reshape((2, self.params.matrix_size, self.params.matrix_size))(inputs)

        Re = keras.layers.Cropping2D(cropping=((1, 0), 0))(x)
        Re = keras.layers.Reshape((self.params.matrix_size, self.params.matrix_size, 1))(Re)
        Im = keras.layers.Cropping2D(cropping=((0, 1), 0))(x)
        Im = keras.layers.Reshape((self.params.matrix_size, self.params.matrix_size, 1))(Im)

        ReRe = Convolution2D(1, self.params.matrix_size, padding="same", kernel_initializer=self.custom_weights_Re, use_bias=False)(Re)
        ImRe = Convolution2D(1, self.params.matrix_size, padding="same", kernel_initializer=self.custom_weights_Im, use_bias=False)(Re)
        ReIm = Convolution2D(1, self.params.matrix_size, padding="same", kernel_initializer=self.custom_weights_Re, use_bias=False)(Im)
        ImIm = Convolution2D(1, self.params.matrix_size, padding="same", kernel_initializer=self.custom_weights_Im, use_bias=False)(Im)

        Re = keras.layers.Subtract()([ReRe, ImIm])
        Im = keras.layers.Add()([ReIm, ImRe])
        outputs = keras.layers.Concatenate(axis=-1)([Re, Im])
        return keras.Model(inputs=inputs, outputs=outputs)

    def calculate_propagation(self, field_distribution, field_modifier):
        conv = field_modifier(field_distribution)
        return conv.numpy().reshape(self.params.matrix_size, self.params.matrix_size, 2)[:, :, 0] + 1j * conv.numpy().reshape(self.params.matrix_size, self.params.matrix_size, 2)[:, :, 1]


if __name__ == "__main__":
    params = PropagationParams()
    propagations = (ConvolutionPropagation, ConvolutionPropagationSequentialNN, ConvolutionFaithPropagation)
    for propagation in propagations:
        conv = propagation(params).propagate()
        plt.imshow(np.absolute(conv), interpolation='nearest')
        plt.show()
