from typing import Callable

import numpy as np
import pytest
import scipy.special as scipy_bessel

from pyhank import HankelTransform


smooth_shapes = [lambda r: np.exp(-r ** 2),
                 lambda r: r,
                 lambda r: r ** 2,
                 lambda r: 1 / np.sqrt(r**2 + 0.1**2)]

all_shapes = smooth_shapes.copy()
all_shapes.append(lambda r: np.random.random(r.size))

orders = list(range(0, 5))


def generalised_top_hat(r: np.ndarray, a: float, p: int) -> np.ndarray:
    top_hat = np.zeros_like(r)
    top_hat[r <= a] = 1
    return r ** p * top_hat


def generalised_jinc(v: np.ndarray, a: float, p: int):
    val = a ** (p + 1) * scipy_bessel.jv(p + 1, 2 * np.pi * a * v) / v
    if p == -1:
        val[v == 0] = np.inf
    elif p == -2:
        val[v == 0] = -np.pi
    elif p == 0:
        val[v == 0] = np.pi * a ** 2
    else:
        val[v == 0] = 0
    return val


@pytest.fixture(params=orders)
def transformer(request, radius) -> HankelTransform:
    order = request.param
    return HankelTransform(order, radial_grid=radius)


@pytest.mark.parametrize('shape', all_shapes)
def test_parsevals_theorem(shape: Callable,
                           radius: np.ndarray,
                           transformer: HankelTransform):
    # As per equation 11 of Guizar-Sicairos, the UNSCALED transform is unitary,
    # i.e. if we pass in the unscaled fr (=Fr), the unscaled fv (=Fv)should have the
    # same sum of abs val^2. Here the unscale transform is simply given by
    # ht = transformer.T @ func
    func = shape(radius)
    intensity_before = np.abs(func)**2
    energy_before = np.sum(intensity_before)
    ht = transformer.T @ func
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
    intensity_after = np.abs(ht)**2
    energy_after = np.trapz(y=intensity_after * 2 * np.pi * transformer.v,
                            x=transformer.v)
    assert np.isclose(energy_before, energy_after, rtol=0.01)


def test_round_trip(radius: np.ndarray, transformer: HankelTransform):
    func = np.random.random(radius.shape)
    ht = transformer.qdht(func)
    reconstructed = transformer.iqdht(ht)
    assert np.allclose(func, reconstructed)


@pytest.mark.parametrize('two_d_size', [1, 100, 27])
def test_round_trip_2d(two_d_size: int, radius: np.ndarray, transformer: HankelTransform):
    func = np.random.random((radius.size, two_d_size))
    ht = transformer.qdht(func)
    reconstructed = transformer.iqdht(ht)
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


@pytest.mark.parametrize('a', [1, 0.7, 0.1, 136., 1e-6])
@pytest.mark.parametrize('p', range(-10, 9))
def test_generalised_jinc_zero(a: float, p: int):
    if p == -1:
        pytest.skip('Skipping test for p=-11 as 1/eps does not go to inf correctly')
    v = np.array([0, 1e-200])
    val = generalised_jinc(v, a, p)
    assert np.isclose(val[0], val[1])


def test_original_r_k_grid():
    r_1d = np.linspace(0, 1, 10)
    k_1d = r_1d.copy()
    transformer = HankelTransform(order=0, max_radius=1, n_points=10)
    with pytest.raises(ValueError):
        _ = transformer.original_radial_grid
    with pytest.raises(ValueError):
        _ = transformer.original_k_grid

    transformer = HankelTransform(order=0, radial_grid=r_1d)
    # no error
    _ = transformer.original_radial_grid
    with pytest.raises(ValueError):
        _ = transformer.original_k_grid

    transformer = HankelTransform(order=0, k_grid=k_1d)
    # no error
    _ = transformer.original_k_grid
    with pytest.raises(ValueError):
        _ = transformer.original_radial_grid


def test_initialisation_errors():
    r_1d = np.linspace(0, 1, 10)
    k_1d = r_1d.copy()
    r_2d = np.repeat(r_1d[:, np.newaxis], repeats=5, axis=1)
    k_2d = r_2d.copy()
    with pytest.raises(ValueError):
        # missing any radius or k info
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

    with pytest.raises(ValueError):
        # k_grid and n_points
        HankelTransform(order=0, k_grid=k_1d, n_points=10)
    with pytest.raises(ValueError):
        # k_grid and max_radius
        HankelTransform(order=0, k_grid=k_1d, max_radius=1)
    with pytest.raises(ValueError):
        # k_grid and r_grid
        HankelTransform(order=0, k_grid=k_1d, radial_grid=r_1d)

    with pytest.raises(AssertionError):
        HankelTransform(order=0, radial_grid=r_2d)
    with pytest.raises(AssertionError):
        HankelTransform(order=0, radial_grid=k_2d)

    # no error
    _ = HankelTransform(order=0, max_radius=1, n_points=10)
    _ = HankelTransform(order=0, radial_grid=r_1d)
    _ = HankelTransform(order=0, k_grid=k_1d)


@pytest.mark.parametrize('n', [10, 100, 512, 1024])
@pytest.mark.parametrize('max_radius', [0.1, 10, 20, 1e6])
def test_r_creation_equivalence(n: int, max_radius: float):
    transformer1 = HankelTransform(order=0, n_points=1024, max_radius=50)
    r = np.linspace(0, 50, 1024)
    transformer2 = HankelTransform(order=0, radial_grid=r)

    for key, val in transformer1.__dict__.items():
        if key == '_original_radial_grid':
            continue
        val2 = getattr(transformer2, key)
        if val is None:
            assert val2 is None
        else:
            assert np.allclose(val, val2)


@pytest.mark.parametrize('shape', smooth_shapes)
@pytest.mark.parametrize('order', orders)
def test_round_trip_r_interpolation(radius: np.ndarray, order: int, shape: Callable):
    transformer = HankelTransform(order=order, radial_grid=radius)

    # the function must be smoothish for interpolation
    # to work. Random every point doesn't work
    func = shape(radius)
    transform_func = transformer.to_transform_r(func)
    reconstructed_func = transformer.to_original_r(transform_func)
    assert np.allclose(func, reconstructed_func, rtol=1e-4)


@pytest.mark.parametrize('shape', smooth_shapes)
@pytest.mark.parametrize('order', orders)
def test_round_trip_k_interpolation(radius: np.ndarray, order: int, shape: Callable):
    k_grid = radius/10
    transformer = HankelTransform(order=order, k_grid=k_grid)

    # the function must be smoothish for interpolation
    # to work. Random every point doesn't work
    func = shape(k_grid)
    transform_func = transformer.to_transform_k(func)
    reconstructed_func = transformer.to_original_k(transform_func)
    assert np.allclose(func, reconstructed_func, rtol=1e-4)


# -------------------
# Test known HT pairs
# -------------------

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


@pytest.mark.parametrize('a', [2, 5, 10])
def test_gaussian(a: float, radius: np.ndarray):
    # Note the definition in Guizar-Sicairos varies by 2*pi in
    # both scaling of the argument (so use kr rather than v) and
    # scaling of the magnitude.
    transformer = HankelTransform(order=0, radial_grid=radius)
    f = np.exp(-a ** 2 * transformer.r ** 2)
    expected_ht = 2*np.pi*(1 / (2 * a**2)) * np.exp(-transformer.kr**2 / (4 * a**2))
    actual_ht = transformer.qdht(f)
    assert np.allclose(expected_ht, actual_ht)


@pytest.mark.parametrize('a', [2, 5, 10])
def test_inverse_gaussian(a: float, radius: np.ndarray):
    # Note the definition in Guizar-Sicairos varies by 2*pi in
    # both scaling of the argument (so use kr rather than v) and
    # scaling of the magnitude.
    transformer = HankelTransform(order=0, radial_grid=radius)
    ht = 2*np.pi*(1 / (2 * a**2)) * np.exp(-transformer.kr**2 / (4 * a**2))
    actual_f = transformer.iqdht(ht)
    expected_f = np.exp(-a ** 2 * transformer.r ** 2)
    assert np.allclose(expected_f, actual_f)


@pytest.mark.parametrize('a', [2, 1, 0.1])
def test_1_over_r2_plus_z2(a: float):
    # Note the definition in Guizar-Sicairos varies by 2*pi in
    # both scaling of the argument (so use kr rather than v) and
    # scaling of the magnitude.
    transformer = HankelTransform(order=0, n_points=1024, max_radius=50)
    f = 1 / (transformer.r**2 + a**2)
    # kn cannot handle complex arguments, so a must be real
    expected_ht = 2 * np.pi * scipy_bessel.kn(0, a * transformer.kr)
    actual_ht = transformer.qdht(f)
    # These tolerances are pretty loose, but there seems to be large
    # error here
    assert np.allclose(expected_ht, actual_ht, rtol=0.1, atol=0.01)
    error = np.mean(np.abs(expected_ht - actual_ht))
    assert error < 4e-3


def sinc(x):
    return np.sin(x) / x


# noinspection DuplicatedCode
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
