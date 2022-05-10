class ParamsValidationException(Exception):
    pass


class PropagationParams:
    c = 299792458

    def __init__(self, matrix_size, nu, wavelength, sigma, focal_length, distance, pixel):
        self.matrix_size = matrix_size
        self.nu = nu
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
        self.positive_integer_validator(size)
        self._matrix_size = size

    @property
    def nu(self):
        return self._nu

    @nu.setter
    def nu(self, value):
        self.positive_integer_validator(value)
        self._nu = value

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        self.positive_float_validator(value)
        self._wavelength = value

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self.positive_integer_validator(value)
        self._sigma = value

    @property
    def focal_length(self):
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value):
        self.cast_to_type_validator(value, expected_type=int)
        self._focal_length = value

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, value):
        self.cast_to_type_validator(value, expected_type=int)
        self._distance = value

    @property
    def pixel(self):
        return self._pixel

    @pixel.setter
    def pixel(self, value):
        self.positive_float_validator(value)
        self._pixel = value

    def positive_float_validator(self, value):
        self.positive_value_validator(value, expected_type=float)

    def positive_integer_validator(self, value):
        self.positive_value_validator(value, expected_type=int)

    def positive_value_validator(self, value, expected_type):
        self.cast_to_type_validator(value, expected_type)
        if expected_type(value) <= 0:
            raise ParamsValidationException(f"Matrix size should be {expected_type} greater than 0")

    def cast_to_type_validator(self, value, expected_type):
        try:
            expected_type(value)
        except ValueError:
            raise ParamsValidationException(f"Matrix size: {value} cannot be converted to {expected_type}")

    @classmethod
    def get_example_propagation_data(cls):
        data = {
            "matrix_size": 256,
            "nu": 140,
            "wavelength": PropagationParams.c / 140 * 10 ** -6,
            "sigma": 20,
            "focal_length": 500,
            "distance": 500,
            "pixel": 1
        }
        return cls.get_params_from_dict(data)

    @classmethod
    def get_params_from_dict(cls, params_dict):
        return cls(**params_dict)
