import numpy
import numpy as np


class PropagationResult:
    def __init__(self, propagation_output):
        self.propagation_output = propagation_output

    @property
    def propagation_output(self):
        return self._data

    @propagation_output.setter
    def propagation_output(self, propagation_output):
        if not isinstance(propagation_output, numpy.ndarray):
            raise Exception(f"Invalid output, expected ndarray, got {type(propagation_output)}")
        self._data = propagation_output

    def to_abs(self):
        return np.absolute(self.propagation_output)

    def to_intensity(self):
        return self.to_abs() ** 2

    def to_phase(self):
        return np.angle(self.propagation_output)
