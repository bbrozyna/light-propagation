from light_prop.propagation import BasePropagation


class AiPropagation(BasePropagation):
    def __init__(self, matrix, propagation_params):
        self.matrix = matrix
        self.field_amplitude = None
        self.field_phase = None
        super().__init__(propagation_params)

    def propagate_matrix(self):
        self.do_stuff_with_matrix()
        self.propagate()

    def set_field_distribution(self, field_amplitude, field_phase):
        self.field_amplitude = field_amplitude
        self.field_phase = field_phase

    def do_stuff_with_matrix(self):
        pass
