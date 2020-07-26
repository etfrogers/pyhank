import os

import matlab
import numpy as np
import pytest
from matlab.engine import start_matlab

from hankel import BesselType, HankelTransformMode, bessel_zeros, HankelTransform, spline


@pytest.fixture(scope='session')
def engine():
    engine = start_matlab()

    hankel_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'matlab')
    engine.addpath(hankel_path)
    return engine


@pytest.mark.parametrize("type_", range(1, 5))
@pytest.mark.parametrize("order", [0, 1, 2, 3, 5, 6, 10])
@pytest.mark.parametrize("n_zeros", [5, 20])
def test_bessel_zeros(type_: int, order: int, n_zeros: int, engine):
    tolerance = 1e-5
    matlab_zeros = engine.bessel_zeros(float(type_), float(order), float(n_zeros), float(tolerance),
                                       nargout=1)
    matlab_zeros = np.asarray(matlab_zeros).transpose()[0, :]
    python_zeros = bessel_zeros(BesselType(type_), order, n_zeros)
    assert np.allclose(matlab_zeros, python_zeros)


def test_hankel_matrix(engine):
    matlab_to_python_mappings = {'N': 'n_points',
                                 'P': 'order'}
    r_max = 5e-3
    nr = 512
    h = HankelTransform(0, r_max, nr)
    hm = engine.hankel_matrix(0., r_max, float(nr), nargout=1)
    for key, matlab_value in hm.items():
        python_key = matlab_to_python_mappings.get(key, key)
        python_value = getattr(h, python_key)
        assert matlab_python_allclose(python_value, matlab_value), \
            f"Hankel matrix key {key} doesn't match"


def test_qdht(engine):
    r_max = 5e-3
    nr = 512
    h = HankelTransform(0, r_max, nr)
    hm = engine.hankel_matrix(0., r_max, float(nr), nargout=1)

    dr = r_max / (nr - 1)  # Radial spacing
    nr = np.arange(0, nr)  # Radial pixels
    r = nr * dr  # Radial positions

    er = r < 1e-3
    # noinspection PyUnresolvedReferences
    er_m = matlab.double(er[np.newaxis, :].transpose().tolist())

    for mode in range(0, 4):
        matlab_value = engine.qdht(er_m, hm, mode, nargout=1)
        python_value = h.qdht(er, HankelTransformMode(mode))
        assert matlab_python_allclose(python_value, matlab_value)


def test_iqdht(engine):
    r_max = 5e-3
    nr = 512
    h = HankelTransform(0, r_max, nr)
    hm = engine.hankel_matrix(0., r_max, float(nr), nargout=1)

    dr = r_max / (nr - 1)  # Radial spacing
    nr = np.arange(0, nr)  # Radial pixels
    r = nr * dr  # Radial positions

    er = r < 1e-3
    # noinspection PyUnresolvedReferences
    er_m = matlab.double(er[np.newaxis, :].transpose().tolist())

    for mode in range(0, 4):
        # Matlab QDHT and IQDHT use different mode number ordering, so we need to switch mode
        # around here. This allows sensibly named Enum values.
        matlab_mode = mode
        if mode == 1:
            matlab_mode = 2
        elif mode == 2:
            matlab_mode = 1

        matlab_value = engine.iqdht(er_m, hm, matlab_mode, nargout=1)
        python_value = h.iqdht(er, HankelTransformMode(mode))
        assert matlab_python_allclose(python_value, matlab_value)


def matlab_python_allclose(python_value, matlab_value):
    matlab_value = np.asarray(matlab_value)
    if matlab_value.ndim > 1 and matlab_value.shape[1] == 1:
        matlab_value = matlab_value[:, 0]
    return np.allclose(python_value, matlab_value)


def test_hankel_example(engine):
    engine.clear(nargout=0)
    engine.hankel_example(nargout=0)
    e_kr_h_matlab = engine.workspace['EkrH']
    irz_matlab = engine.workspace['Irz']
    e_kr_h_python, irz_python = example()
    assert matlab_python_allclose(e_kr_h_python, e_kr_h_matlab)
    assert matlab_python_allclose(irz_python, irz_matlab)


def example():
    # Gaussian function
    def gauss1d(x, x0, fwhm):
        return np.exp(-2 * np.log(2) * ((x - x0) / fwhm) ** 2)

    nr = 1024  # Number of sample points
    r_max = .05  # Maximum radius (5cm)
    dr = r_max / (nr - 1)  # Radial spacing
    ri = np.arange(0, nr)  # Radial pixels
    r = ri * dr  # Radial positions
    beam_radius = 5e-3  # 5mm
    propagation_direction = 5000
    Nz = 200  # Number of z positions
    z_max = .25  # Maximum propagation distance
    dz = z_max / (Nz - 1)
    z = np.arange(0, Nz) * dz  # Propagation axis

    ht = HankelTransform(0, r_max, nr)
    k_max = 2 * np.pi * ht.V  # Maximum K vector

    field = gauss1d(r, 0, beam_radius) * np.exp(1j * propagation_direction * r)  # Initial field
    ht_field = spline(r, field, ht.r)  # Resampled field

    transform = ht.qdht(ht_field, HankelTransformMode.UNSCALED)  # Convert from physical field to physical wavevector

    scaled_transform = transform / ht.JV  # Convert to scaled form for faster transform

    propagated_intensity = np.zeros((nr, Nz + 1))
    propagated_intensity[:, 0] = np.abs(field) ** 2
    for n in range(1, Nz):
        z_loop = z[n]
        propagation_phase = (np.sqrt(k_max ** 2 - ht.kr ** 2) - k_max) * z_loop  # Propagation phase
        propagated_transform = scaled_transform * np.exp(1j * propagation_phase)  # Apply propagation
        propagated_ht_field = ht.iqdht(propagated_transform, HankelTransformMode.BOTH_SCALED)  # iQDHT (no scaling)
        propagated_field = spline(ht.r, propagated_ht_field * ht.JR, r)  # Interpolate & scale output
        propagated_intensity[:, n] = np.abs(propagated_field) ** 2

    return transform, propagated_intensity
