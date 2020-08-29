
"""
Example of single-shot transform
================================

In this example (as in the :ref:`sphx_glr_auto_examples_simple_example.py`)
we will check the band limit of a jinc function: :math:`f(r) = \\frac{J_1(r)}{r}`.
The (0 order) Hankel transform of this should be the top hat function.

In this case, we will use the simple, single shot functions. It should be noted
though, this simplicity comes at an increased overhead, and for multiple transforms
on the same grid, the approach in :ref:`sphx_glr_auto_examples_simple_example.py` is
recommended.

"""

# %%
# First import the ``qdht`` function and other packages.
from pyhank import qdht
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

# %%
# Create a grid for :math:`r` points and calculate the jinc function.
# The calculation fails at :math:`r = 0`, so we have to set that manually to the limit of :math:`1/2`.
r = np.linspace(0, 100, 1024)
f = np.zeros_like(r)
f[1:] = scipy.special.jv(1, r[1:]) / r[1:]
f[r == 0] = 0.5

plt.figure()
plt.plot(r, f)
plt.xlabel('Radius /m')

# %%
# Now take the Hankel transform using ``qdht``:
kr, ht = qdht(r, f)

plt.figure()
plt.plot(kr, ht)
plt.xlim([0, 5])
plt.xlabel('Radial wavevector /m$^{-1}$')

# %%
# As expected, this is a top-hat function bandlimited to :math:`k<1`, except for numerical error.
