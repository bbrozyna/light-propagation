import json
import logging


class ParamsValidationException(Exception):
    pass


class PropagationParams:
    c = 299792458

    def __init__(self, matrix_size, nu, wavelength, sigma, focal_length, distance, pixel):
        logging.info("Loading propagation params")
        self.matrix_size = matrix_size
        self.nu = nu
        if not wavelength:
            wavelength = PropagationParams.get_wavelength_from_nu(self.nu)
        self.wavelength = wavelength
        self.sigma = sigma
        self.focal_length = focal_length
        self.distance = distance
        self.pixel = pixel

    def __str__(self):
        return self.__dict__

    @property
    def matrix_size(self):
        return self._matrix_size

    @matrix_size.setter
    def matrix_size(self, size):
        self._matrix_size = self._positive_integer_validator(size)

    @property
    def nu(self):
        return self._nu

    @nu.setter
    def nu(self, value):
        self._nu = self._positive_integer_validator(value)

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = self._positive_float_validator(value)

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = self._positive_integer_validator(value)

    @property
    def focal_length(self):
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value):
        self._focal_length = self._cast_to_type_validator(value, expected_type=int)

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, value):
        self._distance = self._cast_to_type_validator(value, expected_type=int)

    @property
    def pixel(self):
        return self._pixel

    @pixel.setter
    def pixel(self, value):
        self._pixel = self._positive_float_validator(value)

    def _positive_float_validator(self, value):
        return self._positive_value_validator(value, expected_type=float)

    def _positive_integer_validator(self, value):
        return self._positive_value_validator(value, expected_type=int)

    def _positive_value_validator(self, value, expected_type):
        value = self._cast_to_type_validator(value, expected_type)
        if expected_type(value) <= 0:
            raise ParamsValidationException(f"Value should be {expected_type} greater than 0")
        return value

    def _cast_to_type_validator(self, value, expected_type):
        try:
            return expected_type(value)
        except ValueError:
            raise ParamsValidationException(f"{value} cannot be converted to {expected_type}")

    @staticmethod
    def get_wavelength_from_nu(nu):
        return PropagationParams.c / nu * 10 ** -6

    @classmethod
    def get_example_propagation_data(cls):
        data = {
            "matrix_size": 256,
            "nu": 140,
            "wavelength": PropagationParams.get_wavelength_from_nu(140),
            "sigma": 20,
            "focal_length": 500,
            "distance": 500,
            "pixel": 1
        }
        return cls.get_params_from_dict(data)

    @classmethod
    def get_params_from_dict(cls, params_dict):
        return cls(**params_dict)

    @classmethod
    def get_params_from_json_file(cls, json_file):
        with open(json_file) as file:
            data = json.load(file)
        return PropagationParams.get_params_from_dict(data)
