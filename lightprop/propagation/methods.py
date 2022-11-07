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

from lightprop.calculations import h
from lightprop.lightfield import LightField
from lightprop.propagation.keras_layers import (
    Aexp,
    Convolve,
    ReIm_convert,
    Slice,
    Structure,
)
from lightprop.propagation.params import PropagationParams

class ConvolutionPropagation:
    def calculate_kernel(self, distance, wavelength, matrix_size, pixel_size):
        hkernel = np.array(
            [
                [
                    h(
                        np.sqrt(x**2 + y**2),
                        distance,
                        wavelength,
                    )
                    for x in np.arange(-matrix_size / 2, matrix_size / 2) * pixel_size
                ]
                for y in np.arange(-matrix_size / 2, matrix_size / 2) * pixel_size
            ]
        )
        return hkernel

    def propagate(self, propagation_input: LightField, distance: float) -> LightField:
        logging.info("Calculating propagation")
        field_distribution = propagation_input.get_complex_field()
        kernel = self.calculate_kernel(
            distance, propagation_input.wavelength, propagation_input.matrix_size, propagation_input.pixel
        )
        output = signal.fftconvolve(field_distribution, kernel, mode="same")
        return LightField.from_complex_array(output, propagation_input.wavelength, propagation_input.pixel)


class NNPropagation:
    def prepare_input_field(self, propagation_input: LightField):
        field = np.array([propagation_input.get_re(), propagation_input.get_im()])
        field = field.reshape(
            (
                1,
                2,
                propagation_input.matrix_size,
                propagation_input.matrix_size,
            ),
            order="F",
        )
        return field

    def calculate_kernel(self, distance, wavelength, matrix_size, pixel_size):
        kernel = np.array(
            [
                [
                    h(
                        np.sqrt(x**2 + y**2),
                        distance,
                        wavelength,
                    )
                    for x in np.arange(-matrix_size / 2, matrix_size / 2) * pixel_size
                ]
                for y in np.arange(-matrix_size / 2, matrix_size / 2) * pixel_size
            ]
        )
        kernel = kernel.reshape(matrix_size, matrix_size, 1, 1)
        return kernel

    def build_model(self, matrix_size: int):
        inputs = keras.Input(shape=(2, matrix_size, matrix_size))
        x = Aexp()(inputs)
        x = keras.layers.Reshape((2, matrix_size, matrix_size))(x)

        x = Structure(kernel_initializer=keras.initializers.Zeros())(x)
        x = keras.layers.Reshape((2, matrix_size, matrix_size))(x)

        x = ReIm_convert()(x)
        x = keras.layers.Reshape((2, matrix_size, matrix_size))(x)

        Re = keras.layers.Cropping2D(cropping=((1, 0), (0, 0)))(x)
        Re = keras.layers.Reshape((matrix_size, matrix_size, 1))(Re)
        Im = keras.layers.Cropping2D(cropping=((0, 1), (0, 0)))(x)
        Im = keras.layers.Reshape((matrix_size, matrix_size, 1))(Im)

        ReRe = Convolution2D(
            1,
            matrix_size,
            padding="same",
            kernel_initializer=keras.initializers.Zeros(),
            use_bias=False,
        )(Re)
        ImRe = Convolution2D(
            1,
            matrix_size,
            padding="same",
            kernel_initializer=keras.initializers.Zeros(),
            use_bias=False,
        )(Re)
        ReIm = Convolution2D(
            1,
            matrix_size,
            padding="same",
            kernel_initializer=keras.initializers.Zeros(),
            use_bias=False,
        )(Im)
        ImIm = Convolution2D(
            1,
            matrix_size,
            padding="same",
            kernel_initializer=keras.initializers.Zeros(),
            use_bias=False,
        )(Im)

        Re = keras.layers.Subtract()([ReRe, ImIm])
        Im = keras.layers.Add()([ReIm, ImRe])
        x = keras.layers.Concatenate(axis=1)([Re, Im])
        x = keras.layers.Reshape((2, matrix_size, matrix_size))(x)
        x = Aexp()(x)
        outputs = keras.layers.Reshape((2, matrix_size, matrix_size))(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        for layer in model.layers[:]:
            layer.trainable = False
        model.layers[3].trainable = True

        return model

    def set_kernels(self, model, propagation_input: LightField, distance: float):
        kernel = self.calculate_kernel(
            distance, propagation_input.wavelength, propagation_input.matrix_size, propagation_input.pixel
        )

        model.layers[11].set_weights([np.real(kernel)])
        model.layers[12].set_weights([np.imag(kernel)])
        model.layers[13].set_weights([np.real(kernel)])
        model.layers[14].set_weights([np.imag(kernel)])

        return model

    def propagate(self, propagation_input: LightField, distance: float) -> LightField:
        logging.info("Calculating propagation")
        field_distribution = self.prepare_input_field(propagation_input)

        model = self.build_model(propagation_input.matrix_size)

        model = self.set_kernels(model, propagation_input, distance)

        conv = model(field_distribution).numpy()

        return LightField.from_re_im(
            conv[0, 0, :, :], conv[0, 1, :, :], propagation_input.wavelength, propagation_input.pixel
        )

class MultiparameterNNPropagation(NNPropagation):
    
    def propagate(self, propagation_input: LightField, kernel: LightField) -> LightField:
        logging.info("Calculating propagation")
        field_distribution = self.prepare_input_field(propagation_input)
        model = self.build_model(propagation_input.matrix_size)
        conv = model(field_distribution, kernel).numpy()
        return LightField.from_re_im(
            conv[0, 0, :, :], conv[0, 1, :, :], propagation_input.wavelength, propagation_input.pixel
        )

    def build_model(self, matrix_size: int):
        inputField = keras.Input(shape=(2, matrix_size, matrix_size))
        Kernel = keras.Input(shape=(2, matrix_size, matrix_size), batch_size=1)

        x = Aexp()(inputField)
        x = keras.layers.Reshape((2, matrix_size, matrix_size))(x)

        x = Structure(kernel_initializer=keras.initializers.Zeros())(x)
        x = keras.layers.Reshape((2, matrix_size, matrix_size))(x)

        x = ReIm_convert()(x)
        x = keras.layers.Reshape((2, matrix_size, matrix_size))(x)

        Re = keras.layers.Cropping2D(cropping=((1, 0), (0, 0)))(x)
        Re = keras.layers.Reshape((matrix_size, matrix_size, 1))(Re)
        Im = keras.layers.Cropping2D(cropping=((0, 1), (0, 0)))(x)
        Im = keras.layers.Reshape((matrix_size, matrix_size, 1))(Im)

        KernelRe = keras.layers.Cropping2D(cropping=((1, 0), (0, 0)))(Kernel)
        KernelRe = keras.layers.Reshape((matrix_size, matrix_size, 1))(KernelRe)
        KernelRe = Slice()(KernelRe)
        KernelIm = keras.layers.Cropping2D(cropping=((0, 1), (0, 0)))(Kernel)
        KernelIm = keras.layers.Reshape((matrix_size, matrix_size, 1))(KernelIm)
        KernelIm = Slice()(KernelIm)

        ReRe = Convolve()([Re, KernelRe])
        ImRe = Convolve()([Re, KernelIm])
        ReIm = Convolve()([Im, KernelRe])
        ImIm = Convolve()([Im, KernelIm])

        Re = keras.layers.Subtract()([ReRe, ImIm])
        Im = keras.layers.Add()([ReIm, ImRe])
        x = keras.layers.Concatenate(axis=1)([Re, Im])
        x = keras.layers.Reshape((2, matrix_size, matrix_size))(x)
        x = Aexp()(x)
        outputs = keras.layers.Reshape((2, matrix_size, matrix_size))(x)

        model = keras.Model(inputs=[inputField, Kernel], outputs=outputs)

        for layer in model.layers[:]:
            layer.trainable = False
        model.layers[3].trainable = True

        return model

