from lightprop.lightfield import LightField
from lightprop.propagation import methods as prop


class GerchbergSaxton:
    def __init__(self, propagation_params):
        self.params = propagation_params

    def optimize(self, input_field: LightField, target_field: LightField, iterations: int = 5):
        output_plane = target_field
        input_plane = None

        for _i in range(iterations):
            output_plane.amplitude = target_field.amplitude
            self.params.distance *= -1
            input_plane = prop.ConvolutionPropagation(self.params).propagate(output_plane)
            input_plane.amplitude = input_field.amplitude
            self.params.distance *= -1
            output_plane = prop.ConvolutionPropagation(self.params).propagate(input_plane)

        return input_plane, output_plane
