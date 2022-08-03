import numpy as np

from light_prop.calculations import lens, gaussian


class PropagationInput():
    def __init__(self):
        self.amplitude = None
        self.phase = None

    def from_image(self, amplitude_image, phase_image):
        pass

    def calculate_standard_lens_from_params(self, propagation_params):
        self.phase = np.array(
            [[lens(np.sqrt(x ** 2 + y ** 2), propagation_params.focal_length, propagation_params.wavelength) for x in
              np.arange(-propagation_params.matrix_size / 2, propagation_params.matrix_size / 2) * propagation_params.pixel] for y in
             np.arange(-propagation_params.matrix_size / 2, propagation_params.matrix_size / 2) * propagation_params.pixel])
        self.amplitude = np.array(
            [[gaussian(np.sqrt(x ** 2 + y ** 2), propagation_params.sigma) for x in
              np.arange(-propagation_params.matrix_size / 2, propagation_params.matrix_size / 2) * propagation_params.pixel] for y in
             np.arange(-propagation_params.matrix_size / 2, propagation_params.matrix_size / 2) * propagation_params.pixel])

    def from_function(self, amp_function,  params):
        return amp_function(**params)

    def get_value(self):
        if self.phase is None:
            raise Exception("Phase not defined")
        if self.amplitude is None:
            raise Exception("Amplitude not defined")
        return self.phase * self.amplitude
