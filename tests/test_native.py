import pyhank._pyhank_native as native


def test_native():
    assert native.sum_as_string(1, 2) == "3"
