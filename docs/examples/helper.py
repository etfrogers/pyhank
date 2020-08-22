"""
Helper functions
----------------

Defines a couple of helper functions for the examples.
"""

import numpy as np
import matplotlib.pyplot as plt


# 1D Gaussian function
def gauss1d(x, x0, fwhm):
    return np.exp(-2 * np.log(2) * ((x - x0) / fwhm) ** 2)


# Plotting function equivalent to Matlab's imagesc
def imagesc(x: np.ndarray, y: np.ndarray, intensity: np.ndarray, axes=None, **kwargs):
    assert x.ndim == 1 and y.ndim == 1, "Both x and y must be 1d arrays"
    assert intensity.ndim == 2, "Intensity must be a 2d array"
    extent = (x[0], x[-1], y[-1], y[0])
    if axes is None:
        img = plt.imshow(intensity, extent=extent, **kwargs, aspect='auto')
    else:
        img = axes.imshow(intensity, extent=extent, **kwargs, aspect='auto')
    img.axes.invert_yaxis()
    return img
