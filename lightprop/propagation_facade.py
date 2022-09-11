from collections import namedtuple

from lightprop.propagation.params import PropagationParams
from lightprop.lightfield import LightField
from lightprop.calculations import get_gaussian_distribution, get_lens_distribution


class PropagationFacade:
    def __init__(self, params: PropagationParams):
        self.params = params

    def generate_field(self):
        Field = namedtuple("Field", ["amp", "phase"])
        field = Field(amp=get_gaussian_distribution(self.params, 0, 0), phase=get_lens_distribution(self.params))
        return field

    def progagate(self, propagation_method) -> LightField:
        field = self.generate_field()
        lf = LightField(field.amp, field.phase)
        method = propagation_method(self.params)
        return method.propagate(lf)
