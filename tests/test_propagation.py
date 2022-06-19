import pytest
from light_prop.propagation_params import PropagationParams
from light_prop.propagation import ConvolutionPropagation, ConvolutionPropagationSequentialNN, NNPropagation
from light_prop.propagation_input import PropagationInput


class TestPropagation:

    @pytest.fixture
    def params(self):
        return PropagationParams.get_example_propagation_data()

    def test_conv_field_shape(self, params):
        assert self.get_propagation_shape(params, ConvolutionPropagation) == (256, 256)

    def test_conv_sequential_field_shape(self, params):
        assert self.get_propagation_shape(params, ConvolutionPropagationSequentialNN) == (1, 256, 256, 1)

    def test_nn_propagation_field_shape(self, params):
        assert self.get_propagation_shape(params, NNPropagation) == (1, 2, 256, 256)

    def get_propagation_shape(self, params, propagation):
        propagation = propagation(params)
        input = PropagationInput()
        input.calculate_standard_lens_from_params(params)
        return propagation.get_field_distribution(input).shape
