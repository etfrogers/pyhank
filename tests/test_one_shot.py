import pytest
import numpy as np
import scipy.special as scipy_bessel

from pyhank import qdht, iqdht, HankelTransform
from tests.test_hankel import generalised_jinc, generalised_top_hat, orders


@pytest.mark.parametrize('a', [1, 0.7, 0.1])
@pytest.mark.parametrize('order', orders)
def test_jinc(radius: np.ndarray, a: float, order: int):
    f = generalised_jinc(radius, a, order)
    kr, actual_ht = qdht(radius, f, order=order)
    v = kr / (2*np.pi)
    expected_ht = generalised_top_hat(v, a, order)
    error = np.mean(np.abs(expected_ht-actual_ht))
    assert error < 1e-3


@pytest.mark.parametrize('two_d_size', [1, 35, 27])
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('a', [1, 0.7, 0.1])
@pytest.mark.parametrize('order', orders)
def test_jinc2d(radius: np.ndarray, a: float, order: int, axis: int, two_d_size: int):
    f = generalised_jinc(radius, a, order)
    second_axis = np.outer(np.linspace(0, 6, two_d_size), f)
    if axis == 0:
        f_array = np.outer(f, second_axis)
    else:
        f_array = np.outer(second_axis, f)
    kr, actual_ht = qdht(radius, f_array, axis=axis)
    v = kr / (2 * np.pi)
    expected_ht = generalised_top_hat(v, a, order)
    if axis == 0:
        expected_ht_array = np.outer(expected_ht, second_axis)
    else:
        expected_ht_array = np.outer(second_axis, expected_ht)
    error = np.mean(np.abs(expected_ht_array-actual_ht))
    # multiply tolerance to allow for the larger values caused
    # by second_axis having values greater than 1
    assert error < 1e-3 * 4


@pytest.mark.parametrize('order', orders)
@pytest.mark.parametrize('a', [1, 1.5, 0.1])
def test_top_hat(radius: np.ndarray, a: float, order: int):
    f = generalised_top_hat(radius, a, order)
    kr, actual_ht = qdht(radius, f, order)
    v = kr / (2 * np.pi)
    expected_ht = generalised_jinc(v, a, order)
    error = np.mean(np.abs(expected_ht-actual_ht))
    assert error < 1e-3


@pytest.mark.parametrize('a', [2, 5, 10])
def test_gaussian(a: float, radius: np.ndarray):
    # Note the definition in Guizar-Sicairos varies by 2*pi in
    # both scaling of the argument (so use kr rather than v) and
    # scaling of the magnitude.
    f = np.exp(-a ** 2 * radius ** 2)
    kr, actual_ht = qdht(radius, f)
    expected_ht = 2*np.pi*(1 / (2 * a**2)) * np.exp(-kr**2 / (4 * a**2))
    assert np.allclose(expected_ht, actual_ht)


@pytest.mark.parametrize('a', [2, 5, 10])
def test_inverse_gaussian(a: float):
    # Note the definition in Guizar-Sicairos varies by 2*pi in
    # both scaling of the argument (so use kr rather than v) and
    # scaling of the magnitude.
    kr = np.linspace(0, 200, 1024)
    ht = 2*np.pi*(1 / (2 * a**2)) * np.exp(-kr**2 / (4 * a**2))
    r, actual_f = iqdht(kr, ht)
    expected_f = np.exp(-a ** 2 * r ** 2)
    assert np.allclose(expected_f, actual_f)


@pytest.mark.parametrize('axis', [0, 1])
def test_gaussian_2d(axis: int, radius: np.ndarray):
    # Note the definition in Guizar-Sicairos varies by 2*pi in
    # both scaling of the argument (so use kr rather than v) and
    # scaling of the magnitude.
    a = np.linspace(2, 10)
    dims_a = np.ones(2, np.int)
    dims_a[1-axis] = len(a)
    dims_r = np.ones(2, np.int)
    dims_r[axis] = len(radius)
    a_reshaped = np.reshape(a, dims_a)
    r_reshaped = np.reshape(radius, dims_r)
    f = np.exp(-a_reshaped ** 2 * r_reshaped ** 2)
    kr, actual_ht = qdht(radius, f, axis=axis)
    kr_reshaped = np.reshape(kr, dims_r)
    expected_ht = 2*np.pi*(1 / (2 * a_reshaped**2)) * np.exp(-kr_reshaped**2 / (4 * a_reshaped**2))
    assert np.allclose(expected_ht, actual_ht)


@pytest.mark.parametrize('axis', [0, 1])
def test_inverse_gaussian_2d(axis: int):
    # Note the definition in Guizar-Sicairos varies by 2*pi in
    # both scaling of the argument (so use kr rather than v) and
    # scaling of the magnitude.
    kr = np.linspace(0, 200, 1024)
    a = np.linspace(2, 10)
    dims_a = np.ones(2, np.int)
    dims_a[1-axis] = len(a)
    dims_r = np.ones(2, np.int)
    dims_r[axis] = len(kr)
    a_reshaped = np.reshape(a, dims_a)
    kr_reshaped = np.reshape(kr, dims_r)
    ht = 2*np.pi*(1 / (2 * a_reshaped**2)) * np.exp(-kr_reshaped**2 / (4 * a_reshaped**2))
    r, actual_f = iqdht(kr, ht, axis=axis)
    r_reshaped = np.reshape(r, dims_r)
    expected_f = np.exp(-a_reshaped ** 2 * r_reshaped ** 2)
    assert np.allclose(expected_f, actual_f)


@pytest.mark.parametrize('a', [2, 1, 0.5])
def test_1_over_r2_plus_z2(a: float):
    # Note the definition in Guizar-Sicairos varies by 2*pi in
    # both scaling of the argument (so use kr rather than v) and
    # scaling of the magnitude.
    r = np.linspace(0, 50, 1024)
    f = 1 / (r**2 + a**2)
    # kn cannot handle complex arguments, so a must be real
    kr, actual_ht = qdht(r, f)
    expected_ht = 2 * np.pi * scipy_bessel.kn(0, a * kr)
    # as this diverges at zero, the first few entries have higher errors, so ignore them
    expected_ht = expected_ht[10:]
    actual_ht = actual_ht[10:]
    error = np.mean(np.abs(expected_ht - actual_ht))
    assert error < 1e-3


# -------------------
# Test equivalence of one-shot and standard
# -------------------
@pytest.mark.parametrize('a', [1, 0.7, 0.1])
@pytest.mark.parametrize('order', orders)
def test_jinc_equivalence(a: float, order: int, radius: np.ndarray):
    transformer = HankelTransform(order=order, radial_grid=radius)
    f = generalised_jinc(radius, a, order)
    kr, one_shot_ht = qdht(radius, f, order=order)

    f_t = generalised_jinc(transformer.r, a, transformer.order)
    standard_ht = transformer.qdht(f_t)
    assert np.allclose(one_shot_ht, standard_ht)


@pytest.mark.xfail(reason='generalised_top_hat has discontinuities, so deals badly with interpolation')
@pytest.mark.parametrize('a', [1, 0.7, 0.1])
@pytest.mark.parametrize('order', orders)
def test_top_hat_equivalence(a: float, order: int, radius: np.ndarray):
    transformer = HankelTransform(order=order, radial_grid=radius)
    f = generalised_top_hat(radius, a, order)
    kr, one_shot_ht = qdht(radius, f, order=order)

    f_t = generalised_top_hat(transformer.r, a, transformer.order)
    standard_ht = transformer.qdht(f_t)
    assert np.allclose(one_shot_ht, standard_ht)


@pytest.mark.parametrize('a', [2, 5, 10])
def test_gaussian_equivalence(a: float, radius: np.ndarray):
    # Note the definition in Guizar-Sicairos varies by 2*pi in
    # both scaling of the argument (so use kr rather than v) and
    # scaling of the magnitude.
    transformer = HankelTransform(order=0, radial_grid=radius)
    f = np.exp(-a ** 2 * radius ** 2)
    kr, one_shot_ht = qdht(radius, f)

    f_t = np.exp(-a ** 2 * transformer.r ** 2)
    standard_ht = transformer.qdht(f_t)
    assert np.allclose(one_shot_ht, standard_ht, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize('a', [2, 1, 0.1])
def test_1_over_r2_plus_z2_equivalence(a: float):
    r = np.linspace(0, 50, 1024)
    f = 1 / (r ** 2 + a ** 2)
    transformer = HankelTransform(order=0, radial_grid=r)
    f_transformer = 1 / (transformer.r**2 + a**2)
    assert np.allclose(transformer.to_transform_r(f), f_transformer, rtol=1e-2, atol=1e-6)

    kr, one_shot_ht = qdht(r, f)
    assert np.allclose(kr, transformer.kr)
    standard_ht = transformer.qdht(f_transformer)
    assert np.allclose(one_shot_ht, standard_ht, rtol=1e-3, atol=1e-2)
