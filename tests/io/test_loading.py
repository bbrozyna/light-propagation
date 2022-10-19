import numpy as np
import pytest

import lightprop as lp

gray_png = "data/gray256.png"
gray_bmp = "data/gray256.bmp"
gray_shape = (16, 16)


@pytest.mark.parametrize("field", ["re", "im", "amp", "phase"])
@pytest.mark.parametrize("file", [gray_png, gray_bmp])
def test_should_load_file_as_field(field, file):
    field = f"{field}_file"
    lf = lp.load(**{field: file})
    assert lf
    assert lf.amplitude.shape == gray_shape
    assert lf.phase.shape == gray_shape
    assert np.count_nonzero(lf.amplitude) or np.count_nonzero(lf.phase)


@pytest.mark.parametrize("fields", [("re", "im"), ("amp", "phase")])
@pytest.mark.parametrize("file1", [gray_png, gray_bmp])
@pytest.mark.parametrize("file2", [gray_png, gray_bmp])
def test_should_load_two_complementary_fields(fields, file1, file2):
    field1 = f"{fields[0]}_file"
    field2 = f"{fields[1]}_file"
    assert lp.load(**{field1: file1, field2: file2})
    assert lp.load(**{field2: file2, field1: file1})


@pytest.mark.parametrize("fields", [("re", "amp"), ("re", "phase"), ("im", "amp"), ("im", "phase")])
@pytest.mark.parametrize("file1", [gray_png, gray_bmp])
@pytest.mark.parametrize("file2", [gray_png, gray_bmp])
def test_should_fail_on_non_complementary_fields(fields, file1, file2):
    field1 = f"{fields[0]}_file"
    field2 = f"{fields[1]}_file"
    with pytest.raises(ValueError, match="Invalid arguments provided.*"):
        lp.load(**{field1: file1, field2: file2})
    with pytest.raises(ValueError, match="Invalid arguments provided.*"):
        lp.load(**{field2: file2, field1: file1})
