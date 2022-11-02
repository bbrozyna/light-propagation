import logging
import numpy as np

from lightprop.lightfield import LightField
from lightprop.propagation import methods as prop


class GerchbergSaxton:
    def __init__(self, distance: float):
        self.log = logging.getLogger(type(self).__name__)
        self.distance = distance

    def optimize(self, input_field: LightField, target_field: LightField, iterations: int = 5):
        output_plane = target_field
        input_plane = None

        for _i in range(iterations):
            self.log.info(f"iteration: {_i + 1}/{iterations}")
            output_plane.amp = target_field.get_amplitude()
            input_plane = prop.ConvolutionPropagation().propagate(output_plane, - self.distance)
            input_plane.amp = input_field.get_amplitude()
            output_plane = prop.ConvolutionPropagation().propagate(input_plane, self.distance)

        return input_plane, output_plane
