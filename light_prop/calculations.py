import numpy as np


def h(r, distance, wavelength):
    return np.exp(1j * 2 * np.pi / (wavelength * distance)) / (1j * wavelength * distance) * np.exp(1j * 2 * np.pi / (wavelength * 2 * distance) * r * r)


def gaussian(what_is_r, variance):
    return np.exp(-what_is_r ** 2 / (2 * variance ** 2))


def lens(r, focal_length, wavelength):
    return (-2 * np.pi) / wavelength * np.sqrt(r ** 2 + focal_length ** 2)


def compare_np_arrays(array1, array2):
    return np.max(array1 - array2) < 10 ** -6
