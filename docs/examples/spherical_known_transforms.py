"""
Known transforms — Spherical Hankel Transform
=============================================

The *spherical* Hankel transform (SQDHT) is a variant of the QDHT designed for
functions with spherical (3-D radial) symmetry.  It computes

.. math::

    \\mathcal{H}_\\text{sph}^{(n)}\\{f(k)\\}
        = \\int_0^\\infty f(r)\\; j_n(kr) \\; r^2 \\; dr

where :math:`j_n(x)` is the nth-order spherical Bessel function of the first
kind, defined as

.. math::

    j_n(x) = \\sqrt{\\frac{\\pi}{2x}} \\; J_{n+\\frac{1}{2}}(x)

Where :math:`J_n(x)` is the Bessel function of the first kind.
For :math:`n = 0` this reduces to :math:`j_0(x) = \\sin(x)/x` and hence

.. math::

    \\mathcal{H}_\\text{sph}^{(0)}\\{f(k)\\}
        = \\int_0^\\infty f(r) \\, \\frac{\\sin(kr)}{kr} \\, r^2 \\, dr

While the zeroth-order transform is demonstrated below, the
:class:`~pyhank.HankelTransform` object can be used to compute transforms of
any integer order by passing ``bessel_type="spherical"`` to the constructor.

Below we verify two well-known transform pairs against their analytical results.
"""

import matplotlib.pyplot as plt
import numpy as np

from pyhank import HankelTransform

# %%
# Gaussian Function
# -----------------
#
# The order-0 spherical Hankel transform of the Gaussian
# :math:`f(r) = e^{-ar^2}` is
#
# .. math::
#
#     \\mathcal{H}_\\text{sph}
#
# dummy
#
# .. math::
#
#     \\mathcal{H}_\\text{sph}\\{e^{-ar^2}\\}(k)
#         = \\frac{\\sqrt{\\pi}}{4\\, a^{3/2}}\\, e^{-k^2 / 4a}
#
# We demonstrate this for :math:`a = 2`.

a = 2.0
r_max = 20.0
n_points = 250

transformer = HankelTransform(order=0, max_radius=r_max, n_points=n_points, bessel_type="spherical")
function = np.exp(-a * transformer.r**2)

actual_transform = transformer.qdht(function)
kr = transformer.kr

# Dense analytical curve for comparison
kr_dense = np.linspace(0, kr[-1], 1024)
expected_transform_dense = (np.sqrt(np.pi) / (4 * a**1.5)) * np.exp(-(kr_dense**2) / (4 * a))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(transformer.r, function)
axes[0].set_xlim(0, 4)
axes[0].set_title(r"Gaussian function: $f(r) = e^{-ar^2}$, $a=2$")
axes[0].set_xlabel("Radius $r$")
axes[0].set_ylabel("Amplitude")

axes[1].plot(kr_dense, expected_transform_dense, label="Analytical")
axes[1].plot(kr, actual_transform, marker="x", linestyle="None", label="SQDHT")
axes[1].set_xlim(0, 10)
axes[1].set_title(r"Spherical Hankel transform of Gaussian")
axes[1].set_xlabel("Wavenumber $k$")
axes[1].set_ylabel("Amplitude")
axes[1].legend()

plt.tight_layout()

# Verify agreement
expected_at_grid = (np.sqrt(np.pi) / (4 * a**1.5)) * np.exp(-(kr**2) / (4 * a))
assert np.allclose(actual_transform, expected_at_grid, rtol=0.01, atol=1e-10)

# %%
# The SQDHT (crosses) closely follows the analytical result (solid line) across
# the entire wavenumber range.

# %%
# Top-Hat Function
# ----------------
#
# The order-0 spherical Hankel transform of the top-hat
# :math:`f(r) = 1` for :math:`r < a`, :math:`0` otherwise, is
#
# .. math::
#
#     \\mathcal{H}_\\text{sph}\\{f\\}(k)
#         = \\frac{\\sin(ka) - ka\\cos(ka)}{k^3}
#
# This is the 3-D analogue of the familiar jinc function that arises in the
# standard (2-D radially-symmetric) Hankel transform.
#
# We demonstrate this for :math:`a = 0.5`.  Because the Bessel-zero grid does
# not fall exactly on :math:`a`, we snap to the nearest grid sample to remove
# any spurious discretisation error from the comparison.

a_nominal = 0.5
r_max_hat = 20.0
n_points_hat = 1000

transformer_hat = HankelTransform(order=0, max_radius=r_max_hat, n_points=n_points_hat, bessel_type="spherical")

function_hat = (transformer_hat.r < a_nominal).astype(float)

# Snap a to the last grid point that is actually inside the hat
actual_a_index = np.where(function_hat > 0.5)[0][-1]
a = transformer_hat.r[actual_a_index]

actual_transform_hat = transformer_hat.qdht(function_hat)
kr_hat = transformer_hat.kr

# Dense analytical curve (avoid k=0 singularity)
kr_hat_dense = np.linspace(1e-6, kr_hat[-1], 1024)
expected_hat_dense = (np.sin(kr_hat_dense * a) - kr_hat_dense * a * np.cos(kr_hat_dense * a)) / kr_hat_dense**3

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

r_dense = np.linspace(0, 3, 1024)
axes[0].plot(r_dense, (r_dense < a_nominal).astype(float))
axes[0].set_xlim(0, 2)
axes[0].set_ylim(-0.1, 1.2)
axes[0].set_title(r"Top-hat function: $f(r) = 1$ for $r < a$, $a=0.5$")
axes[0].set_xlabel("Radius $r$")
axes[0].set_ylabel("Amplitude")

axes[1].plot(kr_hat_dense, expected_hat_dense, label="Analytical")
axes[1].plot(kr_hat, actual_transform_hat, marker="x", linestyle="None", label="SQDHT")
axes[1].set_xlim(0, 30)
axes[1].set_ylim(-0.02, 0.05)
axes[1].set_title(r"Spherical Hankel transform of top-hat")
axes[1].set_xlabel("Wavenumber $k$")
axes[1].set_ylabel("Amplitude")
axes[1].legend()

plt.tight_layout()

# Verify agreement (rtol is looser due to Gibbs-like ringing at the sharp edge)
expected_hat_at_grid = np.where(
    kr_hat == 0,
    a**3 / 3.0,
    (np.sin(kr_hat * a) - kr_hat * a * np.cos(kr_hat * a)) / kr_hat**3,
)
assert np.allclose(actual_transform_hat, expected_hat_at_grid, rtol=0.1, atol=0.001)

# %%
# The oscillatory decay of the transform reflects the sharp edge of the top-hat,
# in excellent agreement with the analytical formula.
