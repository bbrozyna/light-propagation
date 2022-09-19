import logging

import light_prop.propagation.methods as prop
from light_prop.propagation.params import PropagationParams

params = PropagationParams.get_example_propagation_data()
propagator = prop.MultiparameterNNPropagation(params)
model = propagator.get_field_modifier()
logging.info(model._get_trainable_state())
model.summary()
