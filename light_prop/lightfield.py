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

    # def __add__(self, other):
    #     if self.amplitude and other.amplitude:
    #         field = self.amplitude * np.exp(self.phase) + other.amplitude * np.exp(other.phase)
    #         self.amplitude = np.abs(field)
    #         self.phase = np.angle(field)
    #     else:
    #         self.amplitude = self.amplitude or other.amplitude
    #         self.phase = self.phase or other.phase
