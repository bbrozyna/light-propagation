import tensorflow as tf
from keras import backend as K
from tensorflow import keras


class Aexp(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        self.A = K.sqrt(K.square(inputs[:, 0]) + K.square(inputs[:, 1]))
        self.phi = tf.math.atan2(inputs[:, 1], inputs[:, 0])
        return K.concatenate([self.A, self.phi], axis=1)


class ReIm_convert(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        self.Re = inputs[:, 0] * K.cos(inputs[:, 1])
        self.Im = inputs[:, 0] * K.sin(inputs[:, 1])

        return K.concatenate([self.Re, self.Im], axis=1)


class Complex_from_Aexp(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        self.field = inputs[:, 0] * K.exp(1j * inputs[:, 1])

        return self.field


class Complex_to_Aexp(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        self.A = K.abs(inputs)
        self.phi = tf.math.angle(inputs)
        return K.concatenate([self.A, self.phi], axis=1)


class Structure(keras.layers.Layer):
    def __init__(self, kernel_initializer, **kwargs):
        super(Structure, self).__init__(**kwargs)
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[2], input_shape[3]),
            initializer=self.kernel_initializer,  # TODO: Choose your initializer
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return K.concatenate([inputs[:, 0], inputs[:, 1] + self.kernel], axis=1)


class Convolve(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, data):
        field = data[0]
        kernel = data[1]
        kernel = K.expand_dims(kernel, axis=-1)
        conv = K.conv2d(field, kernel, padding="same", strides=1, data_format="channels_last")
        return conv


class Slice(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, kernel):
        return kernel[0, :, :, :]


class FFTConvolve(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, data):
        field = tf.cast(data[0], tf.complex64)
        kernel = tf.cast(data[1], tf.complex64)

        self.ComplexField = field[:, 0] * K.exp(1j * field[:, 1])
        self.ComplexKernel = kernel[:, 0] * K.exp(1j * kernel[:, 1])

        self.ComplexField = tf.signal.fft2d(self.ComplexField)
        self.ComplexField *= self.ComplexKernel
        self.ComplexField = tf.signal.ifft2d(self.ComplexField)

        return K.concatenate([K.abs(self.ComplexField), tf.math.angle(self.ComplexField)], axis=1)
