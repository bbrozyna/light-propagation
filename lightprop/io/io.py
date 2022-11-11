import numpy as np
from PIL import Image

from lightprop.lightfield import LightField


def load(
    *, full_file: str = None, re_file: str = None, im_file: str = None, amp_file: str = None, phase_file: str = None
) -> LightField:
    """
    Loads given files into LightField data structure. Either full_file, any single field file, or a pair of
    complementary fields is a valid argument combination. full_file must be of .lf type, field files should be images
    (any format supported by PIL library).

    Example usage:

    >>> lf = load(full_file="holo.lf")

    >>> lf = load(re_file="real.png")

    >>> lf = load(amp_file="amplitude.png", phase_file="/some/directory/phase.bmp")

    Invalid usages:

    >>> lf = load(re_file="real.png", phase_file="/some/directory/phase.bmp")

    :param full_file: path to the custom file (.lf) with full field info
    :param re_file: path to the image with REAL part of complex field
    :param im_file: path to the image with IMAGINARY part of complex field
    :param amp_file: path to the image with AMPLITUDE information of complex field
    :param phase_file: path to the image with PHASE information of complex field
    :return: fully initialized LightField instance
    """
    # load file with full field data
    if full_file:
        return __load_full(full_file)

    # get only non-None arguments
    fields = {
        k: v
        for k, v in {"re": re_file, "im": im_file, "amplitude": amp_file, "phase": phase_file}.items()
        if v is not None
    }

    # single field - simple, load and return as LightField
    if len(fields) == 1:
        ((field_name, field_file),) = fields.items()
        return __load_field(field_name, field_file)

    # complementary fields
    elif len(fields) == 2:
        if re_file and im_file:
            cplx = __load_image(re_file) + 1j * __load_image(im_file)
            amp = np.abs(cplx)
            phase = np.angle(cplx)
            return LightField(amp, phase)

        elif amp_file and phase_file:
            amp = __load_image(amp_file)
            # phase from -pi to pi
            phase = __load_image(phase_file) * np.pi * 2 - np.pi
            return LightField(amp, phase)

    # at this point we handled everything we could... so fail
    raise ValueError(f"Invalid arguments provided: {fields}")


def __load_field(field: str, file: str) -> LightField:
    data = __load_image(file)

    if field in ["re", "im", "amplitude"]:
        return LightField(data, np.zeros(data.shape))
    else:  # if field == "phase"
        return LightField(np.ones(data.shape), data)


def __load_image(file: str) -> np.ndarray:
    """
    loads image file into normalized array of floating point numbers, size = image.size
    :param file:
    :return: np.ndarray
    """
    img = Image.open(file).convert("L")
    return np.array(img.getdata(), dtype=np.uint8).astype("float32").reshape(img.size) / 255.0


def __load_full(file: str) -> LightField:
    # TODO implement when we decide upon file structure
    raise ValueError("not implemented")


def save(file, **kwargs):
    pass


def show(**kwargs):
    pass
