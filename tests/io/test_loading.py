from itertools import combinations

import pytest

import lightprop as lp


def test_loading_existing_lf_file_succeeds():
    assert lp.load("sample.lf")


@pytest.mark.parametrize("field_type", ["re", "im", "amplitude", "phase"])
def test_loading_not_existing_file_raises_file_not_found_error(field_type):
    file_name = "not-existing.png"
    with pytest.raises(FileNotFoundError, match=f"File: {file_name} not found!"):
        lp.load(**{field_type: file_name})


@pytest.mark.parametrize("field_type", ["re", "im", "amplitude", "phase"])
def test_loading_existing_image_as_single_field_type_succeeds(field_type):
    assert lp.load(**{field_type: "sample.png"})
    assert lp.load(**{field_type: "sample.bmp"})


@pytest.mark.parametrize("f1,f2", [("re", "im"), ("amplitude", "phase")])
def test_loading_supplemental_field_types_succeeds(f1, f2):
    assert lp.load(**{f1: "sample.png", f2: "sample.png"})


@pytest.mark.parametrize(
    "f1,f2",
    [
        ("re", "amplitude"),
        ("amplitude", "re"),
        ("im", "amplitude"),
        ("amplitude", "im"),
        ("re", "phase"),
        ("phase", "re"),
        ("im", "phase"),
        ("phase", "im"),
    ],
)
def test_loading_not_supplemental_field_types_raises_value_error(f1, f2):
    with pytest.raises(ValueError, match=f"Cannot load fields [{f1},{f2}] simultaneously!"):
        assert lp.load(**{f1: "sample.png", f2: "sample.png"})


def test_loading_more_than_two_field_types_raises_value_error():
    for fields in combinations(["re", "im", "phase", "amplitude"], 3):
        with pytest.raises(ValueError, match=f"Cannot load fields: {fields} simultaneously!"):
            lp.load(**{key: "sample.png" for key in fields})
