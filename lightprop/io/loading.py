from typing import TypedDict

from typing_extensions import Unpack

from lightprop.lightfield import LightField


class Fields(TypedDict):
    re: str
    im: str
    amplitude: str
    phase: str


def load(file: str = None, **kwargs: Unpack[Fields]) -> LightField:
    """

    :param file:
    :param kwargs:
    :return:
    """
    __validate_args(file, **kwargs)
    if file is None:
        for (key, value) in kwargs.items():
            lf = __loaders[key].load(value)
        return lf
    else:
        return __load_lf_file(file)


def __validate_args(file: str, **kwargs: Unpack[Fields]) -> None:
    if not file and not kwargs:
        raise ValueError(f"Either file  load both file and {kwargs.keys()}")

    if file and kwargs:
        raise ValueError(f"Cannot load both file and {kwargs.keys()}")


def __load_lf_file(file: str) -> LightField:
    pass


def __load_re(file: str) -> LightField:
    pass


def __load_im(file: str) -> LightField:
    pass


def __load_amplitude(file: str) -> LightField:
    pass


def __load_phase(file: str) -> LightField:
    pass


__loaders = {"re": __load_re, "im": __load_im, "amplitude": __load_amplitude, "phase": __load_phase}
