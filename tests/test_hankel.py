from typing import Callable

import numpy as np
import pytest
from hankel import HankelTransform


@pytest.fixture()
def radius() -> np.ndarray:
    return np.linspace(0, 5e-2, 1024)


@pytest.fixture(params=range(0, 4))
def transformer(request, radius) -> HankelTransform:
    order = request.param
    return HankelTransform(order, radial_grid=radius)


def test_scaling():
    raise NotImplementedError


def test_one_shot():
    raise NotImplementedError


def test_round_trip(radius: np.ndarray, transformer: HankelTransform):
    func = np.random.random(radius.shape)
    ht = transformer.qdht(func)
    reconstructed = transformer.iqdht(ht)
    assert np.allclose(func, reconstructed)


@pytest.mark.parametrize('shape', [lambda r: np.exp(r**2),
                                   lambda r: r,
                                   lambda r: r**2,
                                   lambda r: 1 / (r+0.1)])
def test_round_trip_with_interpolation(shape: Callable,
                                       radius: np.ndarray,
                                       transformer: HankelTransform):
    # the function must be smoothish for interpolation
    # to work. Random every point doesn't work
    func = shape(radius)
    func_hr = transformer.to_transform_r(func)
    ht = transformer.qdht(func_hr)
    reconstructed_hr = transformer.iqdht(ht)
    reconstructed = transformer.to_original_r(reconstructed_hr)

    assert np.allclose(func, reconstructed)


def test_r_grid_errors():
    raise NotImplementedError


def test_r_grid_equivalence():
    raise NotImplementedError
