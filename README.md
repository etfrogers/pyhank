PyHank - Quasi-Discrete Hankel Transforms for Python
====================================================

##### Edward Rogers


[![Documentation Status](https://readthedocs.org/projects/pyhank/badge/?version=latest)](https://pyhank.readthedocs.io/en/latest/?badge=latest)


[![Test Status](https://img.shields.io/travis/com/etfrogers/pyhank/master.svg?label=tests)](https://www.travis-ci.com/etfrogers/pyhank)

[![Coverage](https://codecov.io/gh/etfrogers/pyhank/branch/master/graph/badge.svg)](https://codecov.io/gh/etfrogers/pyhank)

[![PyPI version](https://badge.fury.io/py/pyhank.svg)](https://badge.fury.io/py/pyhank)

[![Code style - flake 8](https://img.shields.io/badge/code%20style-flake8-brightgreen)](https://flake8.pycqa.org/en/latest/)

PyHank is a python implementation of the quasi-discrete Hankel transform as developed by Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega

> *"Computation of quasi-discrete Hankel transforms of the integer order for propagating optical wave fields"*
  Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega
  J. Opt. Soc. Am. A **21** (1) 53-58 (2004)

It was designed for use primarily in cases where a discrete Hankel transform is required, similar to the FFT for a Fourier transform. It operates on functions stored in NumPy arrays. If you want an Hankel transform that operates on a callable function, you may be interested in [hankel](<https://github.com/steven-murray/hankel>) by Steven Murray.

I have used this code extensively for beam-propagation-method calculations of radially-symmetric beams. In the radially symmetric case, the 2D FFT over x and y that would be used in a non-symmetric system is replaced by a 1D QDHT over r, making the computational load much lighter and allowing bigger simulations.

PyHank was inspired by Adam Wyatt's [Matlab version](https://uk.mathworks.com/matlabcentral/fileexchange/15623-hankel-transform) which I used for many years, before moving to Python and needing my own implementation. It aims to simplify the interface (using Python's object-oriented approach) and utilise existing NumPy/SciPy functions wherever possible.

It has both a simple single-shot interface, and a more advanced approach that speeds up computation significantly if making multiple transforms on the same grid.

Contributions and comments are welcome using Github at:
http://github.com/etfrogers/pyhank


Installation
------------

Installation can simply be done from pip.
PyHank requires numpy and scipy, but these will be installed by pip if necessary.

``pip install pyhank``

For building the docs, the following are required:

- sphinx-gallery >= 0.7
- matplotlib >= 3.2

For development, and running the tests, the following are recommended:

- pytest ~= 5.4.3
- flake8 ~= 3.8.3
- pytest-flake8 ~= 1.0.6
- pytest-cov ~= 2.10.0

Bugs & Contribution
-------------------

Please use Github to report bugs, feature requests and submit your code:
http://github.com/etfrogers/pyhank



Documentation
-------------

The documentation for PyHank can be found at [Read the docs](https://pyhank.readthedocs.io/en/latest/index.html)

Usage
-----

See the [Usage examples at ReadTheDocs](https://pyhank.readthedocs.io/en/latest/auto_examples/index.html)
