import numpy as np


class LightField:
    def __init__(self, re: np.array, im: np.array, wavelength: float, pixel_size: float):
        if re.size != im.size:
            raise Exception("Dimensions do not match")

        self.re = re
        self.im = im
        self.matrix_size = re.shape[0]
        self.wavelength = wavelength
        self.pixel = pixel_size

    def get_abs(self):
        return np.sqrt(self.re**2 + self.im**2)

    def get_intensity(self):
        return self.re**2 + self.im**2

    def get_phase(self):
        # Errors may occur for the points on the axes. Avoid them.
        return 2 * np.arctan(self.im / (np.sqrt(self.re**2 + self.im**2) + self.re))

    def get_re(self):
        return self.re

    def get_im(self):
        return self.im

    def get_complex_field(self):
        return self.re + (1j * self.im)

    def __add__(self, other):
        if self.re.size != other.re.size:
            raise Exception("Dimensions do not match")

        if self.wavelength != other.wavelength:
            raise Exception("Lightfields must have the same wavelength")

        if self.pixel != other.pixel:
            raise Exception("Lightfields must have the same pixel size")    

        re = self.re + other.re
        im = self.im + other.im
        return LightField(re, im, self.wavelength, self.pixel)

    def __sub__(self, other):
        if self.re.size != other.re.size:
            raise Exception("Dimensions do not match")

        if self.wavelength != other.wavelength:
            raise Exception("Lightfields must have the same wavelength")

        if self.pixel != other.pixel:
            raise Exception("Lightfields must have the same pixel size") 
        
        re = self.re - other.re
        im = self.im - other.im
        return LightField(re, im, self.wavelength, self.pixel)

    def __mul__(self, other):
        if self.re.size != other.re.size:
            raise Exception("Dimensions do not match")

        if self.wavelength != other.wavelength:
            raise Exception("Lightfields must have the same wavelength")

        if self.pixel != other.pixel:
            raise Exception("Lightfields must have the same pixel size") 

        re = self.re * other.re - self.im * other.im
        im = self.re * other.im + self.im * other.re
        return LightField(re, im, self.wavelength, self.pixel)

    @classmethod
    def from_complex_array(cls, complex_array, wavelength, pixel_size):
        re = np.real(complex_array)
        im = np.imag(complex_array)
        return cls(re, im, wavelength, pixel_size)

    @classmethod
    def from_polar_coordinates(cls, amp, phase, wavelength, pixel_size):
        if amp.size != phase.size:
            raise Exception("Dimensions do not match")
        re = amp * np.cos(phase)
        im = amp * np.sin(phase)
        return cls(re, im, wavelength, pixel_size)
