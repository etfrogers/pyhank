from typing import Callable

import numpy as np
import pytest
import scipy.special as scipybessel

from ..hankel import HankelTransform, HankelTransformMode


smooth_shapes = [lambda r: np.exp(-r ** 2),
                 lambda r: r,
                 lambda r: r ** 2,
                 lambda r: 1 / np.sqrt(r**2 + 0.1**2)]

all_shapes = smooth_shapes.copy()
all_shapes.append(lambda r: np.random.random(r.size))


def generalised_top_hat(r: np.ndarray, a: float, p: int) -> np.ndarray:
    top_hat = np.zeros_like(r)
    top_hat[r <= a] = 1
    return r ** p * top_hat


def generalised_jinc(v: np.ndarray, a: float, p: int):
    return a ** (p + 1) * scipybessel.jv(p + 1, 2 * np.pi * a * v) / v


@pytest.fixture()
def radius() -> np.ndarray:
    return np.linspace(0, 3, 1024)


@pytest.fixture(params=range(0, 5))
def transformer(request, radius) -> HankelTransform:
    order = request.param
    return HankelTransform(order, radial_grid=radius)


@pytest.mark.parametrize('scaling', HankelTransformMode)
def test_scaling(radius: np.ndarray, scaling: HankelTransformMode,
                 transformer: HankelTransform):
    func = np.random.random(radius.shape)
    if scaling in (HankelTransformMode.FR_SCALED, HankelTransformMode.BOTH_SCALED):
        scaled_func = func / transformer.JR
    else:
        scaled_func = func
    ht = transformer.qdht(scaled_func, scaling)
    if scaling in (HankelTransformMode.FV_SCALED, HankelTransformMode.BOTH_SCALED):
        ht = ht * transformer.JV
    assert np.allclose(ht, transformer.qdht(func))


@pytest.mark.parametrize('shape', all_shapes)
def test_parsevals_theorem(shape: Callable,
                           radius: np.ndarray,
                           transformer: HankelTransform):
    # As per equation 11 of Guizar-Sicairos, the UNSCALED transform is unitary,
    # i.e. if we pass in the unscaled fr (=Fr), the unscaled fv (=Fv)should have the
    # same sum of abs val^2
    func = shape(radius)
    intensity_before = np.abs(func)**2
    energy_before = np.sum(intensity_before)
    ht = transformer.qdht(func, HankelTransformMode.BOTH_SCALED)
    intensity_after = np.abs(ht)**2
    energy_after = np.sum(intensity_after)
    assert np.isclose(energy_before, energy_after)


@pytest.mark.parametrize('shape', [generalised_jinc, generalised_top_hat])
def test_energy_conservation(shape: Callable,
                             transformer: HankelTransform):
    transformer = HankelTransform(transformer.order, 10, transformer.n_points)
    func = shape(transformer.r, 0.5, transformer.order)
    intensity_before = np.abs(func)**2
    energy_before = np.trapz(y=intensity_before * 2 * np.pi * transformer.r,
                             x=transformer.r)

    ht = transformer.qdht(func)
    # ht *= transformer.n_points
    intensity_after = np.abs(ht)**2
    energy_after = np.sum(intensity_after)
    energy_after = np.trapz(y=intensity_after * 2 * np.pi * transformer.v,
                            x=transformer.v)
    assert np.isclose(energy_before, energy_after, rtol=0.01)


@pytest.mark.parametrize('scaling', HankelTransformMode)
def test_round_trip(radius: np.ndarray, scaling: HankelTransformMode,
                    transformer: HankelTransform):
    func = np.random.random(radius.shape)
    ht = transformer.qdht(func, scaling)
    reconstructed = transformer.iqdht(ht, scaling)
    assert np.allclose(func, reconstructed)


@pytest.mark.parametrize('shape', smooth_shapes)
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

    assert np.allclose(func, reconstructed, rtol=2e-4)


def test_r_grid_errors():
    r_1d = np.linspace(0, 1, 10)
    r_2d = np.repeat(r_1d[:, np.newaxis], repeats=5, axis=1)
    with pytest.raises(ValueError):
        # missing any radius info
        HankelTransform(order=0)
    with pytest.raises(ValueError):
        # missing n_points
        HankelTransform(order=0, max_radius=1)
    with pytest.raises(ValueError):
        # missing max_radius
        HankelTransform(order=0, n_points=10)
    with pytest.raises(ValueError):
        # radial_grid and n_points
        HankelTransform(order=0, radial_grid=r_1d, n_points=10)
    with pytest.raises(ValueError):
        # radial_grid and max_radius
        HankelTransform(order=0, radial_grid=r_1d, max_radius=1)

    with pytest.raises(AssertionError):
        HankelTransform(order=0, radial_grid=r_2d)

    # no error
    _ = HankelTransform(order=0, max_radius=1, n_points=10)
    _ = HankelTransform(order=0, radial_grid=r_1d)


@pytest.mark.xfail
@pytest.mark.parametrize('shape', smooth_shapes)
def test_r_grid_equivalence(shape: Callable, transformer: HankelTransform):
    # create a new transformer with original_radial_grid already correctly spaced
    original_transformer = transformer
    transformer = HankelTransform(order=original_transformer.order,
                                  radial_grid=original_transformer.r)

    assert np.allclose(transformer.r, transformer.original_radial_grid)
    function = shape(transformer.r)
    transformed = transformer.to_transform_r(function)
    back_transformed = transformer.to_original_r(transformed)
    inverse_transformed = transformer.to_original_r(function)

    assert np.allclose(function, transformed)
    assert np.allclose(function, back_transformed)
    assert np.allclose(function, inverse_transformed)


def test_2d_input():
    raise NotImplementedError


# Test known HT pairs
@pytest.mark.parametrize('a', [1, 0.7, 0.1])
def test_jinc(transformer: HankelTransform, a: float):
    f = generalised_jinc(transformer.r, a, transformer.order)
    expected_ht = generalised_top_hat(transformer.v, a, transformer.order)
    actual_ht = transformer.qdht(f)
    error = np.mean(np.abs(expected_ht-actual_ht))
    assert error < 1e-3


@pytest.mark.parametrize('a', [1, 1.5, 0.1])
def test_top_hat(transformer: HankelTransform, a: float):
    f = generalised_top_hat(transformer.r, a, transformer.order)
    expected_ht = generalised_jinc(transformer.v, a, transformer.order)
    actual_ht = transformer.qdht(f)
    error = np.mean(np.abs(expected_ht-actual_ht))
    assert error < 1e-3


def test_delta_bessel():
    raise NotImplementedError


def test_bessel_delta():
    raise NotImplementedError


def test_gaussian():
    raise NotImplementedError


def test_1_over_root_r2_plus_z2():
    raise NotImplementedError


def test_1_over_r2_plus_z2():
    raise NotImplementedError


def sinc(x):
    return np.sin(x) / x


@pytest.mark.parametrize('p', [1, 4])
def test_sinc(p):
    """Tests from figure 1 of
        *"Computation of quasi-discrete Hankel transforms of the integer
        order for propagating optical wave fields"*
        Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega
        J. Opt. Soc. Am. A **21** (1) 53-58 (2004)
        """
    transformer = HankelTransform(p, max_radius=3, n_points=256)
    v = transformer.v
    gamma = 5
    func = sinc(2 * np.pi * gamma * transformer.r)
    expected_ht = np.zeros_like(func)

    expected_ht[v < gamma] = (v[v < gamma]**p * np.cos(p * np.pi / 2)
                              / (2*np.pi*gamma * np.sqrt(gamma**2 - v[v < gamma]**2)
                                 * (gamma + np.sqrt(gamma**2 - v[v < gamma]**2))**p))
    expected_ht[v >= gamma] = (np.sin(p * np.arcsin(gamma/v[v >= gamma]))
                               / (2*np.pi*gamma * np.sqrt(v[v >= gamma]**2 - gamma**2)))
    ht = transformer.qdht(func)

    # use the same error measure as the paper
    dynamical_error = 20 * np.log10(np.abs(expected_ht-ht) / np.max(ht))
    not_near_gamma = np.logical_or(v > gamma*1.25,
                                   v < gamma*0.75)
    assert np.all(dynamical_error < -10)
    assert np.all(dynamical_error[not_near_gamma] < -35)
