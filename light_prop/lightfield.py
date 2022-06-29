import numpy as np


class LightField:
    def __init__(self, amp: np.array, phase: np.array):
        if not (isinstance(amp, np.ndarray) and isinstance(phase, np.ndarray)):
            raise Exception("Arguments must be np.array type")
            
        if amp.size != phase.size:
            raise Exception("Dimensions do not match")

        self.amplitude = amp
        self.phase = phase
        

    def to_abs(self):
        return self.amplitude

    def to_intensity(self):
        return self.amplitude ** 2

    def to_phase(self):
        return self.phase

    def to_re(self):
        return self.amplitude * np.cos(self.phase)

    def to_im(self):
        return self.amplitude * np.sin(self.phase)

    def __add__(self, other):
        field = self.amplitude * np.exp(1j * self.phase) + other.amplitude * np.exp(1j * other.phase)
        amplitude = np.abs(field)
        phase = np.angle(field)
        return LightField(amplitude, phase)
    
    def __sub__(self, other):
        field = self.amplitude * np.exp(1j * self.phase) - other.amplitude * np.exp(1j * other.phase)
        amplitude = np.abs(field)
        phase = np.angle(field)
        return LightField(amplitude, phase)
    
    def __mul__(self, other):
        field = self.amplitude * np.exp(1j * self.phase) * other.amplitude * np.exp(1j * other.phase)
        amplitude = np.abs(field)
        phase = np.angle(field)
        return LightField(amplitude, phase)