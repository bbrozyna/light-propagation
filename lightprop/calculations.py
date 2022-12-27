import numpy as np

from lightprop.propagation.params import PropagationParams


# impulse response function in the on-axis approximation
def h(r, distance, wavelength):
    return (
        np.exp(1j * 2 * (np.pi / wavelength) * distance)
        / (1j * wavelength * distance)
        * np.exp(1j * np.pi / (wavelength * distance) * r * r)
    )


# transfer function in the on-axis approximation
def H_on_axis(vx, vy, distance, wavelength):
    return np.exp(1j * 2 * (np.pi / wavelength) * distance) * np.exp(
        -1j * np.pi * wavelength * distance * (vx**2 + vy**2)
    )


# full transfer function
def H_off_axis(vx, vy, distance, wavelength):
    return np.exp(
        1j * 2 * (np.pi / wavelength) * distance * np.sqrt(1 - (vx * wavelength) ** 2 - (vy * wavelength) ** 2)
    )


def gaussian(r, variance):
    return np.exp(-(r**2) / (2 * variance**2))


def lens(r, focal_length, wavelength):
    return (-2 * np.pi) / wavelength * np.sqrt(r**2 + focal_length**2)


def compare_np_arrays(array1, array2):
    return np.max(array1 - array2) < 10**-6


def get_lens_distribution(params: PropagationParams):
    return np.array(
        [
            [
                lens(np.sqrt(x**2 + y**2), params.focal_length, params.wavelength)
                for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
            ]
            for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
        ]
    )


def get_gaussian_distribution(params: PropagationParams, x0: float = 0, y0: float = 0):
    return np.array(
        [
            [
                gaussian(np.sqrt((x - x0) ** 2 + (y - y0) ** 2), params.beam_diameter)
                for x in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
            ]
            for y in np.arange(-params.matrix_size / 2, params.matrix_size / 2) * params.pixel_size
        ]
    )
