import pytest
import numpy as np
from light_prop.lightfield import LightField
from light_prop.propagation_params import PropagationParams


class TestLightField:

    @pytest.fixture
    def params(self):
        return PropagationParams.get_example_propagation_data()

    @pytest.fixture
    def sample_amp_phase(self):
        amp = np.array([[1, 2], [3, 4]])
        phase = np.array([[0, np.pi / 2], [np.pi, 3 * np.pi / 2]])
        return LightField(amp, phase)

    def test_light_field_conversion(self, sample_amp_phase):
        correct_abs = np.array([[1, 2], [3, 4]])
        correct_phase = np.array([[0, np.pi / 2], [np.pi, 3 * np.pi / 2]])
        correct_intensity = np.array([[1, 4], [9, 16]])
        correct_re = np.array([[1, 0], [-3, 0]])
        correct_im = np.array([[0, 2], [0, -4]])
        lf = sample_amp_phase
        assert self.compare_np_arrays(lf.to_abs(), correct_abs)
        assert self.compare_np_arrays(lf.to_phase(), correct_phase)
        assert self.compare_np_arrays(lf.to_intensity(), correct_intensity)
        assert self.compare_np_arrays(lf.to_re(), correct_re)
        assert self.compare_np_arrays(lf.to_im(), correct_im)

    def test_light_field_add(self, sample_amp_phase):
        lf1 = sample_amp_phase
        lf2 = sample_amp_phase
        # res = lf1 + lf2
        # assert res.amplitutude == np.array()
        # assert res.phase == np.array()

    def compare_np_arrays(self, array1, array2):
        return np.max(array1 - array2) < 10 ** -6
