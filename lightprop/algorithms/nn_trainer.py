import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import lightprop.propagation.methods as prop
from lightprop.lightfield import LightField
from lightprop.propagation.params import PropagationParams


class NNTrainer:
    def __init__(self, propagation_params: PropagationParams):
        self.params = propagation_params
        self.model = None
        self.history = None

    def amplitudeMSE(self, y_true, y_pred):
        squared_difference = tf.square(y_true[0, 0] - y_pred[0, 0])

        return tf.reduce_mean(squared_difference, axis=-1)

    def optimize(self, input_field: LightField, target_field: LightField, iterations: int = 100):
        propagator = prop.NNPropagation(self.params)
        self.model = propagator.get_field_modifier()

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-2),
            loss=self.amplitudeMSE,
        )

        checkpoint_filepath = "./tmp/checkpoint"
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, save_weights_only=True, monitor="loss", mode="min", save_best_only=True
        )

        self.history = self.model.fit(
            propagator.get_field_distribution(input_field),
            propagator.get_field_distribution(target_field),
            batch_size=1,
            epochs=iterations,
            callbacks=[model_checkpoint_callback],
        )

        self.model.load_weights(checkpoint_filepath)
        return self.model

    def plot_loss(self):
        plt.plot(self.history.history["loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.yscale("log")
        plt.show()
