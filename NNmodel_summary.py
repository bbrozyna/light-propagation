import logging

import tensorflow as tf
import numpy as np

import lightprop.propagation.methods as prop
from lightprop.propagation.params import PropagationParams

# Prepare propagator
params = PropagationParams.get_example_propagation_data()
propagator = prop.NNPropagation()

# Extract network with dimensions and trainable weights
model = propagator.build_model(2)
logging.info(model._get_trainable_state())

# Print model summary
model.summary()

print(model.layers[11].get_weights())
print(propagator.calculate_kernel(1,1,2,1))
print(np.real(propagator.calculate_kernel(1,1,2,1)).numpy())

# # Save network graph
# dot_img_file = "outs/model.png"
# tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, dpi=1000)