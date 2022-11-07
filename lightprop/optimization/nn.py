import logging

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import lightprop.propagation.methods as prop
from lightprop.lightfield import LightField
from lightprop.propagation.params import PropagationParams


class NNTrainer:
    def __init__(self):
        self.model = None
        self.history = None
        self.log = logging.getLogger(type(self).__name__)

    def amplitudeMSE(self, y_true, y_pred):
        squared_difference = tf.square(y_true[0, 0] - y_pred[0, 0])

        return tf.reduce_mean(squared_difference, axis=-1)

    def optimize(self, input_field: LightField, target_field: LightField, distance, iterations: int = 100):
        propagator = prop.NNPropagation()
        self.model = propagator.build_model(input_field.matrix_size)

        self.model = propagator.set_kernels(self.model, input_field, distance)

        self.log.info("Compiling model...")
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-2),
            loss=keras.losses.MeanSquaredError(),
        )

        checkpoint_filepath = "./tmp/checkpoint"
        self.log.info(f"Setting up checkpoint at {checkpoint_filepath}...")
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, save_weights_only=True, monitor="loss", mode="min", save_best_only=True
        )

        self.log.info("Fitting model...")
        self.history = self.model.fit(
            propagator.prepare_input_field(input_field),
            propagator.prepare_input_field(target_field),
            batch_size=1,
            epochs=iterations,
            callbacks=[model_checkpoint_callback],
        )

        self.log.info("Loading best configuration...")
        self.model.load_weights(checkpoint_filepath)
        return self.model

    def plot_loss(self):
        plt.plot(self.history.history["loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.yscale("log")
        plt.show()

class NNMultiTrainer(NNTrainer):
    
    def optimize(self, input_field: LightField, target_field: LightField, kernel: LightField, distance, iterations: int = 100):
        propagator = prop.MultiparameterNNPropagation()
        self.model = propagator.build_model(input_field.matrix_size)
        
        self.log.info("Compiling model...")
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-2),
            loss=keras.losses.MeanSquaredError(),
        )

        checkpoint_filepath = "./tmp/checkpoint"
        self.log.info(f"Setting up checkpoint at {checkpoint_filepath}...")
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, save_weights_only=True, monitor="loss", mode="min", save_best_only=True
        )

        self.log.info("Fitting model...")
        self.history = self.model.fit(
            [propagator.prepare_input_field(input_field), propagator.prepare_input_field(kernel)],
            propagator.prepare_input_field(target_field),
            batch_size=1,
            epochs=iterations,
            callbacks=[model_checkpoint_callback],
        )

        self.log.info("Loading best configuration...")
        self.model.load_weights(checkpoint_filepath)
        return self.model

