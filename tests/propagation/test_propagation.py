import numpy as np
import pytest

from lightprop.calculations import (
    compare_np_arrays,
    get_gaussian_distribution,
    get_lens_distribution,
)
from lightprop.lightfield import LightField
from lightprop.propagation.methods import ConvolutionPropagation, NNPropagation
from lightprop.propagation.params import PropagationParams


class TestPropagation:
    @pytest.fixture()
    def params(self):
        params = PropagationParams.get_example_propagation_data()
        params.matrix_size = 2
        return params

    @pytest.fixture()
    def amplitude(self, params):
        return get_gaussian_distribution(params)

    @pytest.fixture()
    def phase(self, params):
        return get_lens_distribution(params)

    def test_conv_propagation(self, params, amplitude, phase):
        expected_result = np.array([[0.00093165, 0.00186445], [0.00186445, 0.00373122]])
        wavelength = 1
        pixel = 0.1
        field = LightField(amplitude, phase, wavelength, pixel)
        conv = ConvolutionPropagation()
        output_field = conv.propagate(field, 500)

        assert compare_np_arrays(expected_result, output_field.get_amplitude())

    def test_nn_propagation(self, params, amplitude, phase):
        expected_result = np.array([[0.00373122, 0.00186445], [0.00186445, 0.00093165]])
        wavelength = 1
        pixel = 0.1
        field = LightField(amplitude, phase, wavelength, pixel)
        conv = NNPropagation(params)
        output_field = conv.propagate(field)

        assert compare_np_arrays(expected_result, output_field.get_amplitude())
