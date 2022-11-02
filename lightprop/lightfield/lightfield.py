import numpy as np


class LightField:
    def __init__(self, amp: np.array, phase: np.array, wavelength: float, pixel_size: float):
        if amp.size != phase.size:
            raise Exception("Dimensions do not match")

        self.amp = amp
        self.phase = phase
        self.matrix_size = amp.shape[0]
        self.wavelength = wavelength
        self.pixel = pixel_size

    def get_amplitude(self):
        return self.amp

    def get_intensity(self):
        return self.amp**2

    def get_phase(self):
        return self.phase

    def get_re(self):
        return self.amp * np.cos(self.phase)

    def get_im(self):
        return self.amp * np.sin(self.phase)

    def get_complex_field(self):
        return self.amp * np.exp(1j * self.phase)

    def __add__(self, other):
        if self.amp.size != other.amp.size:
            raise Exception("Dimensions do not match")

        if self.wavelength != other.wavelength:
            raise Exception("Lightfields must have the same wavelength")

        if self.pixel != other.pixel:
            raise Exception("Lightfields must have the same pixel size")    

        re = self.get_re() + other.get_re()
        im = self.get_im() + other.get_im()
        return LightField.from_re_im(re, im, self.wavelength, self.pixel)

    def __sub__(self, other):
        if self.amp.size != other.amp.size:
            raise Exception("Dimensions do not match")

        if self.wavelength != other.wavelength:
            raise Exception("Lightfields must have the same wavelength")

        if self.pixel != other.pixel:
            raise Exception("Lightfields must have the same pixel size") 
        
        re = self.get_re() - other.get_re()
        im = self.get_im() - other.get_im()
        return LightField.from_re_im(re, im, self.wavelength, self.pixel)

    def __mul__(self, other):
        if self.amp.size != other.amp.size:
            raise Exception("Dimensions do not match")

        if self.wavelength != other.wavelength:
            raise Exception("Lightfields must have the same wavelength")

        if self.pixel != other.pixel:
            raise Exception("Lightfields must have the same pixel size") 

        amp = self.get_amplitude() * other.get_amplitude()
        phase = self.get_phase() + other.get_phase()
        return LightField(amp, phase, self.wavelength, self.pixel)

    @classmethod
    def from_complex_array(cls, complex_array, wavelength, pixel_size):
        amp = np.abs(complex_array)
        phase = np.angle(complex_array)
        return cls(amp, phase, wavelength, pixel_size)

    @classmethod
    def from_re_im(cls, re, im, wavelength, pixel_size):
        if re.size != im.size:
            raise Exception("Dimensions do not match")
        amp = np.sqrt(re**2 + im**2)
        phase = np.arctan(im / re)
        phase[np.isnan(phase)] = 0
        return cls(amp, phase, wavelength, pixel_size)
