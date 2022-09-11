import pytest
from lightprop.propagation.params import PropagationParams, ParamsValidationException


class TestPropagationParams:

    @pytest.fixture
    def params(self):
        return PropagationParams.get_example_propagation_data()

    def test_int_params(self, params):
        int_params = ["focal_length", "distance"]
        proper_value = 5
        proper_negative_value = -1
        proper_float_value = 0.1
        xfail_value_not_convertable = " - 1 2 test 3"

        for int_param in int_params:
            setattr(params, int_param, proper_value)
            assert getattr(params, int_param) == proper_value
            setattr(params, int_param, proper_negative_value)
            assert getattr(params, int_param) == proper_negative_value
            setattr(params, int_param, proper_float_value)
            assert getattr(params, int_param) == int(proper_float_value)
            with pytest.raises(ParamsValidationException, match=f"cannot be converted to {int}"):
                setattr(params, int_param, xfail_value_not_convertable)

    def test_positive_int_params(self, params):
        positive_int_params = ["matrix_size"]
        proper_value = 5
        xfail_zero = 0
        xfail_negative_value = -1

        for int_param in positive_int_params:
            setattr(params, int_param, proper_value)
            assert getattr(params, int_param) == proper_value
            with pytest.raises(ParamsValidationException, match=f"{int} greater than 0"):
                setattr(params, int_param, xfail_negative_value)
            with pytest.raises(ParamsValidationException, match=f"{int} greater than 0"):
                setattr(params, int_param, xfail_zero)

    def test_positive_float_params(self, params):
        positive_float_params = ["wavelength", "pixel_size", "nu", "beam_diameter"]
        proper_value = 5
        proper_value2 = "5.3"
        xfail_negative_value = -1

        for float_param in positive_float_params:
            setattr(params, float_param, proper_value)
            assert getattr(params, float_param) == proper_value
            setattr(params, float_param, proper_value2)
            assert getattr(params, float_param) == float(proper_value2)
            with pytest.raises(ParamsValidationException, match=f"{float} greater than 0"):
                setattr(params, float_param, xfail_negative_value)
