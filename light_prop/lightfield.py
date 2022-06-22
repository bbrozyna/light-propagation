import numpy as np


class LightField:
    def __init__(self, amp, phase):
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
        if self.amplitude.all() and other.amplitude.all():
            field = self.amplitude * np.exp(1j * self.phase) + other.amplitude * np.exp(1j * other.phase)
            amplitude = np.abs(field)
            phase = np.angle(field)
        else:
            amplitude = self.amplitude if self.amplitude is not None else other.amplitude
            phase = self.phase if self.phase is not None else other.phase
        return LightField(amplitude, phase)
