"""
Speed of native vs pure Python implementations
==============================================

PyHank provides both a pure Python implementation and a compiled native Rust extension.
Here we will compare their speed using both the object approach and a large single transform.

The compiled native extension is expected to be much faster than the pure Python implementation
at creating the transformer, as the process of generating all the bessel functions, and their
zeros is computationally intensive.

For the actual transform (the ``qdht`` method of the ``HankTransform`` object),
the native extension is expected to be only slightly faster than the pure
Python implementation, as large parts of the transform are matrix multiplations, which is already
optimized by numpy, so the speed improvement is smaller.

"""

import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from helper import gauss1d, imagesc

from pyhank import _pure_python

try:
    # Remove the Sphinx docs mock so we can import the real native extension
    sys.modules.pop("pyhank._pyhank_native", None)
    from pyhank import _pyhank_native

    native_available = True
except ImportError:
    native_available = False


# %%
# First, we'll test the object approach using the same beam-propagation
# example as :ref:`sphx_glr_auto_examples_speed_usage_example.py`.
# We increase the grid sizes slightly to make the runtime difference more visible.

nr = 1024  # Number of sample points
r_max = 5e-3  # Maximum radius (5mm)
Nz = 200  # Number of z positions
z_max = 0.1  # Maximum propagation distance

r = np.linspace(0, r_max, nr)
z = np.linspace(0, z_max, Nz)

Dr = 100e-6  # Beam radius (100um)
lambda_ = 488e-9  # wavelength 488nm
k0 = 2 * np.pi / lambda_  # Vacuum k vector

field = gauss1d(r, 0, Dr)  # Initial field


# %%
# We define a function that propagates the beam, allowing us to specify
# whether to use the native or pure Python implementation.
def propagate_using_object(r: np.ndarray, field: np.ndarray, native: bool) -> np.ndarray:
    if native:
        transformer = _pyhank_native.HankelTransform(order=0, radial_grid=r)  # type: ignore
    else:
        transformer = _pure_python.HankelTransform(order=0, radial_grid=r)

    field_for_transform = transformer.to_transform_r(field)  # Resampled field
    hankel_transform = transformer.qdht(field_for_transform)

    propagated_field = np.zeros((nr, Nz), dtype=complex)
    kz = np.sqrt(k0**2 - transformer.kr**2)
    for n, z_loop in enumerate(z):
        phi_z = kz * z_loop  # Propagation phase
        hankel_transform_at_z = hankel_transform * np.exp(1j * phi_z)  # Apply propagation
        field_at_z_transform_grid = transformer.iqdht(hankel_transform_at_z)  # iQDHT
        propagated_field[:, n] = transformer.to_original_r(field_at_z_transform_grid)  # Interpolate output
    intensity = np.abs(propagated_field) ** 2
    return intensity


# %%
# Now run and time the propagation.

assert native_available, "Native code is not available"
assert _pyhank_native.is_release_build(), "Native code is not built in release mode"  # type: ignore


tic = time.time()
object_intensity_native = propagate_using_object(r, field, native=True)
toc = time.time()
print(f"Native object propagation took {toc - tic:.2f} s")

tic = time.time()
object_intensity_python = propagate_using_object(r, field, native=False)
toc = time.time()
print(f"Python object propagation took {toc - tic:.2f} s")


# %%
# Next, let's test a very large single transform using the functional interface.
# We will use a much larger grid so that a single forward and inverse transform takes
# a measureable amount of time.

nr_large = 4096  # Very large number of sample points
r_large = np.linspace(0, r_max, nr_large)
field_large = gauss1d(r_large, 0, Dr)


def large_single_transform(r: np.ndarray, field: np.ndarray, native: bool) -> np.ndarray:
    if native:
        kr, hankel_transform = _pyhank_native.qdht(r, field, order=0)  # type: ignore
        _, inverse_field = _pyhank_native.iqdht(kr, hankel_transform, order=0)  # type: ignore
    else:
        kr, hankel_transform = _pure_python.qdht(r, field, order=0)
        _, inverse_field = _pure_python.iqdht(kr, hankel_transform, order=0)
    return inverse_field


# %%
# The native code is about a factor of 2 faster, which is a significant improvement over the pure Python implementation, but
# not as much as the transformer creation below.
#
# Run and time the large single transform:

tic = time.time()
large_single_transform(r_large, field_large, native=True)
toc = time.time()
print(f"Native large single transform took {toc - tic:.2f} s")

tic = time.time()
large_single_transform(r_large, field_large, native=False)
toc = time.time()
print(f"Python large single transform took {toc - tic:.2f} s")


# %%
# Here, the native code is about 50x faster than the pure Python implementation.
#
# Finally, let's plot the object propagation results to verify they are identical:

plt.figure()
plt.subplot(2, 1, 1)
imagesc(z * 1e3, r * 1e3, object_intensity_native)
plt.title("Native Propagation")
plt.xlabel("Propagation distance ($z$) /mm")
plt.ylabel("Radial position ($r$) /mm")
plt.colorbar()
plt.ylim((0, 1))

plt.subplot(2, 1, 2)
imagesc(z * 1e3, r * 1e3, object_intensity_python)
plt.title("Pure Python Propagation")
plt.xlabel("Propagation distance ($z$) /mm")
plt.ylabel("Radial position ($r$) /mm")
plt.ylim((0, 1))
plt.colorbar()
plt.tight_layout()
