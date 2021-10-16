
Introduction
============

PyHank is a Python implementation of the quasi-discrete Hankel transform as developed by Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega [#Guizar]_.

It was designed for use primarily in cases where a discrete Hankel transform is required, similar to the FFT for a Fourier transform. It operates on functions stored in NumPy arrays. If you want an Hankel transform that operates on a callable function, you may be interested in `hankel <https://github.com/steven-murray/hankel>`_ by Steven Murray.

I have used this code extensively for beam-propagation-method calculations of radially-symmetric beams (see :ref:`sphx_glr_auto_examples_usage_example.py` for an example of this). In the radially symmetric case, the 2D FFT over :math:`x` and :math:`y` that would be used in a non-symmetric system is replaced by a 1D QDHT over :math:`r`, making the computational load much lighter and allowing bigger simulations.


PyHank was inspired by Adam Wyatt's `Matlab version <https://uk.mathworks.com/matlabcentral/fileexchange/15623-hankel-transform>`_ which I used for many years, before moving to Python and needing my own implementation. It aims to simplify the interface (using Python's object-oriented approach) and utilise existing NumPy/SciPy functions wherever possible.

It has both a simple `single-shot interface <one_shot.html>`_, and also supplies a `transformer object <hankel.html>`_ which speeds up computation significantly if making multiple transforms on the same grid.

Contributions and comments are welcome using Github at:
http://github.com/etfrogers/pyhank

.. [#Guizar] *"Computation of quasi-discrete Hankel transforms of the integer order for propagating optical wave fields"*
  Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega
  J. Opt. Soc. Am. A **21** (1) 53-58 (2004)
