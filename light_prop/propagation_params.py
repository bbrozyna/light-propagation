import datetime
import logging

from typing import List


class ParamsValidationException(Exception):
    pass


class PropagationParams:
    def __init__(self):
        c = 299792458
        self.size = 256
        self.Nu = 140
        self.wavelength = c / self.Nu * 10 ** -6
        self.sigma = 20
        self.focal_length = 500
        self.z = self.focal_length
        self.pixel = 1

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

    def positive_float_validator(self, value):
        self.positive_value_validator(value, expected_type=float)

    def positive_integer_validator(self, value):
        self.positive_value_validator(value, expected_type=int)

    def positive_value_validator(self, value, expected_type):
        try:
            if expected_type(value) <= 0:
                raise ParamsValidationException(f"Matrix size should be {expected_type} greater than 0")
        except ValueError:
            raise ParamsValidationException(f"Matrix size: {value} cannot be converted to {expected_type}")

class KimaiEntry:

    def __init__(self, datetime_from: str, project: str, activity: str, description: str, datetime_to=None, duration=None):
        if not (datetime_to or duration):
            raise ValueError('Please provide either duration or datetime_to')

        if datetime_to and duration:
            logging.info('Duration will overshadow the datetime_to argument')

        self.datetime_to = datetime_to or ''
        self.duration = duration or ''

        self.datetime_from = datetime_from
        self.project = project
        self.activity = activity
        self.description = description

    def __str__(self):
        return self.__dict__

    @property
    def datetime_from(self):
        return self._date_from

    @datetime_from.setter
    def datetime_from(self, datetime_text):
        self._date_from = self.__validate_datetime(datetime_text)

    @property
    def datetime_to(self):
        return self._datetime_to

    @datetime_to.setter
    def datetime_to(self, datetime_text):
        self._datetime_to = self.__validate_datetime(datetime_text) if datetime_text else ''

    def __validate_datetime(self, datetime_value):
        proper_datetime = ''
        try:
            proper_datetime = datetime.datetime.strptime(datetime_value, '%Y-%m-%d %H:%M')
        except ValueError:
            raise ValueError(f'Starting date {datetime_value} doesn\'t match format YYYY-MM-DD hh:mm')
        return proper_datetime.strftime('%Y-%m-%d %H:%M')  # avoiding processing single digit inputs

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, duration):
        self._duration = self.__validate_time(duration)

    def __validate_time(self, time):
        proper_time = ''
        try:
            proper_time = datetime.datetime.strptime(time, '%H:%M')
        except ValueError:
            raise ValueError(f'Starting date {time} doesn\'t match format HH:MM')
        return proper_time.strftime('%H:%M')  # avoiding processing single digit inputs

    @property
    def project(self):
        return self._project

    @project.setter
    def project(self, value):
        self._project = value

    @property
    def activity(self):
        return self._activity

    @activity.setter
    def activity(self, value):
        self._activity = value

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        if len(value) > 250:
            raise Exception('Please provide description with max 250 characters')
        self._description = value

    @classmethod
    def kimai_data_from_dict_list(cls, entries: List[dict]):
        kimai_data = []
        for kimai_entry in entries:
            ke = cls(**kimai_entry)
            kimai_data.append(ke)
        return kimai_data
