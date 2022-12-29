import tensorflow as tf

import lightprop.propagation.methods as prop
from lightprop.propagation.params import PropagationParams

# Prepare propagator
params = PropagationParams.get_example_propagation_data()
propagator = prop.MultiparameterNNPropagation_FFTConv()

# Extract network with dimensions and trainable weights
model = propagator.build_model(params.matrix_size)
# print(model._get_trainable_state())

# Print model summary
model.summary()


# Save network graph
dot_img_file = "outs/model.png"
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True, dpi=1000)
