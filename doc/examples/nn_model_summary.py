import lightprop.propagation.methods as prop
from lightprop.propagation.params import PropagationParams

params = PropagationParams.get_example_propagation_data()
propagator = prop.NNPropagation(params)
model = propagator.get_field_modifier()
print(model._get_trainable_state())
model.summary()
