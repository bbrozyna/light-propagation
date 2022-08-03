import pytest
import numpy as np
from light_prop.lightfield import LightField
from light_prop.propagation_params import PropagationParams
from light_prop.calculations import compare_np_arrays


class TestLightField:

    @pytest.fixture
    def params(self):
        return PropagationParams.get_example_propagation_data()

    @pytest.fixture
    def sample_amp_phase(self):
        amp = np.array([[1, 2], [3, 4]])
        phase = np.array([[0, np.pi / 2], [np.pi, 3 * np.pi / 2]])
        return LightField(amp, phase)
    
    def test_lightfield_init(self):
        amp_correct = np.array([[1, 2], [3, 4]])
        phase_correct = np.array([[0, np.pi / 2], [np.pi, 3 * np.pi / 2]])
        phase_incorrect = np.array(1)
        bad_value1 = 5
        lf = LightField(amp_correct, phase_correct)
        
        assert compare_np_arrays(lf.amplitude, amp_correct)
        assert compare_np_arrays(lf.phase, phase_correct)
        
        
        with pytest.raises(Exception, match="Dimensions do not match"):
            LightField(amp_correct, phase_incorrect)
            
        with pytest.raises(Exception, match="Arguments must be np.array type"):
            LightField(amp_correct, bad_value1)

    def test_light_field_conversion(self, sample_amp_phase):
        correct_abs = np.array([[1, 2], [3, 4]])
        correct_phase = np.array([[0, np.pi / 2], [np.pi, 3 * np.pi / 2]])
        correct_intensity = np.array([[1, 4], [9, 16]])
        correct_re = np.array([[1, 0], [-3, 0]])
        correct_im = np.array([[0, 2], [0, -4]])
        lf = sample_amp_phase
        assert compare_np_arrays(lf.to_abs(), correct_abs)
        assert compare_np_arrays(lf.to_phase(), correct_phase)
        assert compare_np_arrays(lf.to_intensity(), correct_intensity)
        assert compare_np_arrays(lf.to_re(), correct_re)
        assert compare_np_arrays(lf.to_im(), correct_im)

    def test_light_field_add(self, sample_amp_phase):
        lf1 = sample_amp_phase
        lf2 = sample_amp_phase
        res = lf1 + lf2
        assert compare_np_arrays(res.amplitude, lf1.amplitude * 2)
        assert compare_np_arrays(res.phase, lf1.phase)
        
    def test_light_field_sub(self, sample_amp_phase):
        lf1 = sample_amp_phase
        lf2 = sample_amp_phase
        res = lf1 - lf2
        assert compare_np_arrays(res.amplitude, np.array([[0,0],[0,0]]))
        assert compare_np_arrays(res.phase, lf1.phase)

    def test_light_field_mul(self, sample_amp_phase):
        lf1 = sample_amp_phase
        lf2 = sample_amp_phase
        lf2.amplitude *= 2
        res = lf1 * lf2
        res2 = lf2 * lf1
        assert compare_np_arrays(res.amplitude, lf1.amplitude * lf2.amplitude)
        assert compare_np_arrays(res.phase, lf1.phase + lf2.phase)
        assert compare_np_arrays(res2.amplitude, lf1.amplitude * lf2.amplitude)
        assert compare_np_arrays(res2.phase, lf1.phase + lf2.phase)


