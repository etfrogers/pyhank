"""
Simple example of PyHank usage
==============================

In this example (as in the :ref:`sphx_glr_auto_examples_one_shot_example.py`)
we will check the band limit of a jinc function: :math:`f(r) = \\frac{J_1(r)}{r}`.
The (0 order) Hankel transform of this should be the top hat function.

Here we create a :class:`.HankelTransform` object and use its
:meth:`~.HankelTransform.qdht` method.
In this simple case, the simpler, single shot functions used in
:ref:`sphx_glr_auto_examples_one_shot_example.py` may be simpler
to use. It should be noted, however, that they are not well suited for
multiple transforms on the same grid and the approach taken here is
recommended.

"""

# %%
# First import the :class:`.HankelTransform` class and other packages
from pyhank import HankelTransform
import scipy.special
import matplotlib.pyplot as plt

# %%
# Create a :class:`.HankelTransform` object which holds the grid for :math:`r` and
# :math:`k_r` points and calculate the jinc function.
#
# Note that although the calculation fails at :math:`r = 0`, ``transformer.r`` does
# not include :math:`r=0`.
transformer = HankelTransform(order=0, max_radius=100, n_points=1024)
f = scipy.special.jv(1, transformer.r) / transformer.r

plt.figure()
plt.plot(transformer.r, f)
plt.xlabel('Radius /m')

# %%
# Now take the Hankel transform using :meth:`.HankelTransform.qdht`
ht = transformer.qdht(f)

plt.figure()
plt.plot(transformer.kr, ht)
plt.xlim([0, 5])
plt.xlabel('Radial wavevector /m$^{-1}$')

# %%
# As expected, this is a top-hat function bandlimited to :math:`k<1`, except for numerical error.
