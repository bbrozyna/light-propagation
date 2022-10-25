import numpy as np
import pytest

from lightprop.calculations import compare_np_arrays
from lightprop.lightfield import LightField
from lightprop.propagation.params import PropagationParams


class TestLightField:
    @pytest.fixture()
    def sample_amp_phase(self):
        amp = np.array([[1, 2], [3, 4]])
        phase = np.array([[0, np.pi / 2], [np.pi, 3 * np.pi / 2]])
        wavelength = 1
        pixel = 0.1
        return LightField.from_polar_coordinates(amp, phase, wavelength, pixel)

    def test_lightfield_from_polar(self):
        amp_correct = np.array([[1, 2], [3, 4]])
        phase_correct = np.array([[0, np.pi / 2], [np.pi, 3 * np.pi / 2]])
        wavelength = 1
        pixel = 0.1
        phase_incorrect = np.array(1)
        lf = LightField.from_polar_coordinates(amp_correct, phase_correct, wavelength, pixel)

        assert compare_np_arrays(lf.get_abs(), amp_correct)
        assert compare_np_arrays(lf.get_phase(), phase_correct)

        with pytest.raises(Exception, match="Dimensions do not match"):
            LightField.from_polar_coordinates(amp_correct, phase_incorrect, wavelength, pixel)

    def test_lightfield_conversion(self, sample_amp_phase):
        correct_abs = np.array([[1, 2], [3, 4]])
        correct_phase = np.array([[0, np.pi / 2], [np.pi, 3 * np.pi / 2]])
        correct_intensity = np.array([[1, 4], [9, 16]])
        correct_re = np.array([[1, 0], [-3, 0]])
        correct_im = np.array([[0, 2], [0, -4]])
        lf = sample_amp_phase
        assert compare_np_arrays(lf.get_abs(), correct_abs)
        assert compare_np_arrays(lf.get_phase(), correct_phase)
        assert compare_np_arrays(lf.get_intensity(), correct_intensity)
        assert compare_np_arrays(lf.get_re(), correct_re)
        assert compare_np_arrays(lf.get_im(), correct_im)

    def test_lightfield_add(self, sample_amp_phase):
        lf1 = sample_amp_phase
        lf2 = sample_amp_phase
        res = lf1 + lf2
        assert compare_np_arrays(res.get_abs(), lf1.get_abs() * 2)
        assert compare_np_arrays(res.get_phase(), lf1.get_phase())

    def test_lightfield_sub(self, sample_amp_phase):
        lf1 = sample_amp_phase
        lf2 = sample_amp_phase
        res = lf1 - lf2
        assert compare_np_arrays(res.get_abs(), np.array([[0, 0], [0, 0]]))

    def test_lightfield_mul(self, sample_amp_phase):
        lf1 = sample_amp_phase
        lf2 = sample_amp_phase
        lf2.re *= 2
        lf2.im *= 2
        res = lf1 * lf2
        res2 = lf2 * lf1
        assert compare_np_arrays(res.get_abs(), lf1.get_abs() * lf2.get_abs())
        
        assert compare_np_arrays(res2.get_abs(), lf1.get_abs() * lf2.get_abs())
        
