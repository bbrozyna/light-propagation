import pytest
import numpy as np

from light_prop.propagation.params import PropagationParams
from light_prop.propagation.methods import ConvolutionPropagation, NNPropagation
from light_prop.lightfield import LightField
from light_prop.calculations import compare_np_arrays, get_lens_distribution, get_gaussian_distribution


class TestPropagation:

    @pytest.fixture
    def params(self):
        return PropagationParams.get_example_propagation_data()

    @pytest.fixture
    def amplitude(self, params):
        return get_gaussian_distribution(params)

    @pytest.fixture
    def phase(self, params):
        return get_lens_distribution(params)

    def test_conv_propagation(self, params, amplitude, phase):
        expected_result = np.array([[0.00093165, 0.00186445], [0.00186445, 0.00373122]])

        field = LightField(amplitude, phase)
        conv = ConvolutionPropagation(params)
        output_field = conv.propagate(field)

        assert compare_np_arrays(expected_result, output_field.to_abs())

    def test_NN_propagation(self, params, amplitude, phase):
        # todo add test to check if matrixes are swapping quarters or reversing
        expected_result = np.array([[0.00373122, 0.00186445], [0.00186445, 0.00093165]])

        field = LightField(amplitude, phase)
        conv = NNPropagation(params)
        output_field = conv.propagate(field)

        assert compare_np_arrays(expected_result, output_field.to_abs())
