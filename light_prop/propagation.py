"""
Light propagation with convolution method

Models:
Paweł Komorowski
pawel.komorowski@wat.edu.pl
"""
import numpy as np
import logging
from scipy import signal
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.models import backend as K
import tensorflow as tf

from light_prop.calculations import h, gaussian, lens
from light_prop.propagation_params import PropagationParams
from light_prop.propagation_results import PropagationResult


class BasePropagation:
    def __int__(self, propagation_params: PropagationParams):
        self.params = propagation_params

    def propagate(self):
        logging.info("Calculating propagation")
        field_distribution = self.get_field_distribution()
        field_modifier = self.get_field_modifier()
        output = self.reshape(self.calculate_propagation(field_distribution, field_modifier))
        return PropagationResult(output)

    def get_field_distribution(self):
        raise NotImplemented("Please implement field distribution")

    def get_field_modifier(self):
        raise NotImplemented("Please implement field modifier")

    def calculate_propagation(self, field_distribution, field_modifier):
        return None

    def reshape(self, data):
        return data


class ConvolutionPropagation(BasePropagation):
    def __init__(self, propagation_params):
        super().__int__(propagation_params)

    def get_field_distribution(self):
        lens_distribution = np.array(
            [[lens(np.sqrt(x ** 2 + y ** 2), self.params.focal_length, self.params.wavelength) for x in
              np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2) * self.params.pixel] for y in
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
        model.add(Convolution2D(1, kernel_size=(self.params.matrix_size, self.params.matrix_size), padding="same",
                                use_bias=False,
                                kernel_initializer=self.custom_weights,
                                input_shape=(self.params.matrix_size, self.params.matrix_size, 1)))
        return model

    def calculate_propagation(self, field_distribution, field_modifier):
        return field_modifier(field_distribution)

    def reshape(self, data):
        return data.numpy().reshape(self.params.matrix_size, self.params.matrix_size, 1)


class ConvolutionFaithPropagation(ConvolutionPropagation):
    def __init__(self, propagation_params):
        super().__init__(propagation_params)

    def get_field_distribution(self):
        field = super().get_field_distribution()
        field = np.array(
            [[[np.real(field)[i, j], np.imag(field)[i, j]] for i in range(self.params.matrix_size)] for j in
             range(self.params.matrix_size)])
        return field.reshape((1, self.params.matrix_size, self.params.matrix_size, 2), order='F')

    def custom_weights(self, shape, dtype=None, re=False):
        func = np.sin if re else np.cos
        kernel = np.array([[1 / (self.params.distance * self.params.wavelength) * func(
            np.pi * np.sqrt(x ** 2 + y ** 2) ** 2 / (
                    self.params.distance * self.params.wavelength) + 2 * np.pi * self.params.distance / self.params.wavelength)
                            for x in
                            np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2) * self.params.pixel]
                           for y in
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

        ReRe = Convolution2D(1, self.params.matrix_size, padding="same", kernel_initializer=self.custom_weights_Re,
                             use_bias=False)(Re)
        ImRe = Convolution2D(1, self.params.matrix_size, padding="same", kernel_initializer=self.custom_weights_Im,
                             use_bias=False)(Re)
        ReIm = Convolution2D(1, self.params.matrix_size, padding="same", kernel_initializer=self.custom_weights_Re,
                             use_bias=False)(Im)
        ImIm = Convolution2D(1, self.params.matrix_size, padding="same", kernel_initializer=self.custom_weights_Im,
                             use_bias=False)(Im)

        Re = keras.layers.Subtract()([ReRe, ImIm])
        Im = keras.layers.Add()([ReIm, ImRe])
        outputs = keras.layers.Concatenate(axis=-1)([Re, Im])
        return keras.Model(inputs=inputs, outputs=outputs)

    def calculate_propagation(self, field_distribution, field_modifier):
        conv = field_modifier(field_distribution)
        return conv.numpy()

    def reshape(self, data):
        return data.reshape(self.params.matrix_size, self.params.matrix_size, 2)[:, :, 0] + 1j * data.numpy().reshape(
            self.params.matrix_size, self.params.matrix_size, 2)[:, :, 1]


class Aexp(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Aexp, self).__init__(**kwargs)

    def call(self, inputs):
        self.A = K.sqrt(K.square(inputs[:, 0]) + K.square(inputs[:, 1]))
        self.phi = tf.math.atan2(inputs[:, 1], inputs[:, 0])
        return K.concatenate([self.A, self.phi], axis=1)


class ReIm_convert(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReIm_convert, self).__init__(**kwargs)

    def call(self, inputs):
        self.Re = inputs[:, 0] * K.cos(inputs[:, 1])
        self.Im = inputs[:, 0] * K.sin(inputs[:, 1])

        return K.concatenate([self.Re, self.Im], axis=1)


class Structure(keras.layers.Layer):
    def __init__(self, kernel_initializer, **kwargs):
        super(Structure, self).__init__(**kwargs)
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[2], input_shape[3]),
                                      initializer=self.kernel_initializer,  # TODO: Choose your initializer
                                      trainable=True)
        super(Structure, self).build(input_shape)

    def call(self, inputs):
        return K.concatenate([inputs[:, 0], inputs[:, 1] + self.kernel], axis=1)


class NNPropagation(ConvolutionPropagation):
    def __init__(self, propagation_params):
        super().__init__(propagation_params)

    def get_field_distribution(self):
        lens_distribution = np.array(
            [[lens(np.sqrt(x ** 2 + y ** 2), self.params.focal_length, self.params.wavelength) for x in
              np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2) * self.params.pixel] for y in
             np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2) * self.params.pixel])
        init_amplitude = np.array(
            [[gaussian(np.sqrt(x ** 2 + y ** 2), self.params.sigma) for x in
              np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2) * self.params.pixel] for y in
             np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2) * self.params.pixel])

        field = np.array([np.real(init_amplitude * lens_distribution), np.imag(init_amplitude * lens_distribution)])
        field = field.reshape((1, 2, self.params.matrix_size, self.params.matrix_size,), order='F')

        return field

    def custom_weights(self, shape, dtype=None, re=False):
        func = np.sin if re else np.cos
        kernel = np.array([[1 / (self.params.distance * self.params.wavelength) * func(
            np.pi * np.sqrt(x ** 2 + y ** 2) ** 2 / (
                    self.params.distance * self.params.wavelength) + 2 * np.pi * self.params.distance / self.params.wavelength)
                            for x in
                            np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2) * self.params.pixel]
                           for y in
                           np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2) * self.params.pixel])
        kernel = kernel.reshape(self.params.matrix_size, self.params.matrix_size, 1, 1)
        return kernel

    def custom_weights_Re(self, shape, dtype=None):
        return self.custom_weights(shape, dtype, re=True)

    def custom_weights_Im(self, shape, dtype=None):
        return self.custom_weights(shape, dtype, re=False)

    def custom_weights_lens(self, shape, dtype=None):
        kernel = np.array(
            [[(-2 * np.pi) / self.params.wavelength * np.sqrt(x ** 2 + y ** 2 + self.params.focal_length ** 2) for x in
              np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2) * self.params.pixel] for y in
             np.arange(-self.params.matrix_size / 2, self.params.matrix_size / 2) * self.params.pixel])
        kernel = kernel.reshape(self.params.matrix_size, self.params.matrix_size)
        return kernel

    def get_field_modifier(self):
        inputs = keras.Input(shape=(2, self.params.matrix_size, self.params.matrix_size))
        # x=keras.layers.Reshape((2,size,size))(inputs)
        x = Aexp()(inputs)
        x = keras.layers.Reshape((2, self.params.matrix_size, self.params.matrix_size))(x)

        x = Structure(kernel_initializer=keras.initializers.Zeros())(x)
        x = keras.layers.Reshape((2, self.params.matrix_size, self.params.matrix_size))(x)

        x = ReIm_convert()(x)
        x = keras.layers.Reshape((2, self.params.matrix_size, self.params.matrix_size))(x)

        Re = keras.layers.Cropping2D(cropping=((-1, 0), 0))(x)
        Re = keras.layers.Reshape((self.params.matrix_size, self.params.matrix_size, 1))(Re)
        Im = keras.layers.Cropping2D(cropping=((0, 1), 0))(x)
        Im = keras.layers.Reshape((self.params.matrix_size, self.params.matrix_size, 1))(Im)

        ReRe = Convolution2D(1, self.params.matrix_size, padding="same", kernel_initializer=self.custom_weights_Re,
                             use_bias=False)(Re)
        ImRe = Convolution2D(1, self.params.matrix_size, padding="same", kernel_initializer=self.custom_weights_Im,
                             use_bias=False)(Re)
        ReIm = Convolution2D(1, self.params.matrix_size, padding="same", kernel_initializer=self.custom_weights_Re,
                             use_bias=False)(Im)
        ImIm = Convolution2D(1, self.params.matrix_size, padding="same", kernel_initializer=self.custom_weights_Im,
                             use_bias=False)(Im)

        Re = keras.layers.Subtract()([ReRe, ImIm])
        Im = keras.layers.Add()([ReIm, ImRe])
        outputs = keras.layers.Concatenate(axis=-1)([Re, Im])
        return keras.Model(inputs=inputs, outputs=outputs)

    def calculate_propagation(self, field_distribution, field_modifier):
        conv = field_modifier(field_distribution)
        return conv.numpy()

    def reshape(self, data):
        return data.reshape(self.params.matrix_size, self.params.matrix_size, 2)[:, :, 0] + 1j * data.numpy().reshape(
            self.params.matrix_size, self.params.matrix_size, 2)[:, :, 1]
