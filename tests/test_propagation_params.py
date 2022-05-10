import pytest
from light_prop.propagation_params import PropagationParams, ParamsValidationException


class TestPropagationParams:

    @pytest.fixture
    def params(self):
        return PropagationParams()

    def test_matrix_size(self, params):
        proper_value = 5
        xfail_value_negative = -1
        xfail_value_le_zero = 0
        xfail_value_not_convertable = " - 1 2 test 3"

        params.matrix_size = proper_value
        assert params.matrix_size == proper_value, "Couldn't assign proper value to matrix size"
        with pytest.raises(ParamsValidationException, match=f"{int} greater than 0"):
            params.matrix_size = xfail_value_negative
        with pytest.raises(ParamsValidationException, match=f"{int} greater than 0"):
            params.matrix_size = xfail_value_le_zero
        with pytest.raises(ParamsValidationException, match=f"cannot be converted to {int}"):
            params.matrix_size = xfail_value_not_convertable
