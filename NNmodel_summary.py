import light_prop.propagation.methods as prop
from light_prop.propagation.params import PropagationParams
from light_prop.lightfield import LightField
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


params = PropagationParams.get_example_propagation_data()
propagator = prop.NNPropagation(params) 
model = propagator.get_field_modifier()
print(model._get_trainable_state())
model.summary()

