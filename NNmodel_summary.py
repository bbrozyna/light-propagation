import logging

import tensorflow as tf

import light_prop.propagation.methods as prop
from light_prop.propagation.params import PropagationParams

# Prepare propagator
params = PropagationParams.get_example_propagation_data()
propagator = prop.MultiparameterNNPropagation(params)

# Extract network with dimensions and trainable weights
model = propagator.get_field_modifier()
logging.info(model._get_trainable_state())

# Print model summary
model.summary()

# Save network graph
dot_img_file = "outs/model.png"
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, dpi=1000)
