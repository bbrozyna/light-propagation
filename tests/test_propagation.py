import pytest
from light_prop.propagation.params import PropagationParams
from light_prop.propagation.methods import ConvolutionPropagation, NNPropagation
from light_prop.lightfield import LightField
import light_prop.calculations as calc
import numpy as np
from light_prop.calculations import compare_np_arrays


class TestPropagation:

    @pytest.fixture
    def params(self):
        return PropagationParams.get_example_propagation_data()

    def test_conv_propagation(self, params):
        expected_result = np.array([[0.00093165, 0.00186445], [0.00186445, 0.00373122]])

        phase = np.array(
            [[calc.lens(np.sqrt(x ** 2 + y ** 2), params.focal_length, params.wavelength) for x in
              np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel] for y in
             np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel])
        amp = np.array(
            [[calc.gaussian(np.sqrt(x ** 2 + y ** 2), params.sigma) for x in
              np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel] for y in
             np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel])
        field = LightField(amp, phase)

        conv = ConvolutionPropagation(params)
        output_field = conv.propagate(field)

        assert compare_np_arrays(expected_result, output_field.to_abs())

    def test_NN_propagation(self, params):
        # todo add test to check if matrixes are swapping quarters or reversing
        expected_result = np.array([[0.00373122, 0.00186445], [0.00186445, 0.00093165]])

        phase = np.array(
            [[calc.lens(np.sqrt(x ** 2 + y ** 2), params.focal_length, params.wavelength) for x in
              np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel] for y in
             np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel])
        amp = np.array(
            [[calc.gaussian(np.sqrt(x ** 2 + y ** 2), params.sigma) for x in
              np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel] for y in
             np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel])
        field = LightField(amp, phase)

        conv = NNPropagation(params)
        output_field = conv.propagate(field)

        assert compare_np_arrays(expected_result, output_field.to_abs())
