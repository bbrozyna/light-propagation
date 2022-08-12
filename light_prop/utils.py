import logging
import numpy as np
from PIL import Image


def image_to_amplitude(image_path):
    array = ndarray_to_image(image_path)
    return normalize_array(array, 1)


def image_to_phase(image_path):
    array = ndarray_to_image(image_path)
    return normalize_array(array, 2 * np.pi)


def amplitude_to_image(array: np.array, file_path="output", file_extension="bmp"):
    array = normalize_array(array, 255)
    ndarray_to_image(array, file_path, file_extension)


def phase_to_image(array: np.array, file_path="output", file_extension="bmp"):
    array = array % 2 * np.pi
    array = normalize_array(array, 255)
    ndarray_to_image(array, file_path, file_extension)


def ndarray_to_image(array: np.array, file_path="output", file_extension="bmp"):
    path = f"{file_path}.{file_extension}"
    logging.info("Reading array from ")
    im = Image.fromarray(array)

    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save(path)


def image_to_ndarray(image_path):
    array = np.asarray(Image.open(image_path))
    # convert rgb 3dimensional matrix to 1d bw
    return np.add.reduce(array, axis=2) / 3


def normalize_array(array: np.array, expected_max):
    logging.info("Normalizing array")
    if not array.size:
        raise Exception("Provided array is empty")
    array = array / array.max()
    return array * expected_max
