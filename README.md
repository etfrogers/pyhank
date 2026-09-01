PyHank - Quasi-Discrete Hankel Transforms for Python
====================================================

##### Edward Rogers


[![Documentation Status](https://readthedocs.org/projects/pyhank/badge/?version=latest)](https://pyhank.readthedocs.io/en/latest/?badge=latest)
[![CI](https://github.com/etfrogers/pyhank/actions/workflows/ci.yml/badge.svg)](https://github.com/etfrogers/pyhank/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/etfrogers/pyhank/branch/main/graph/badge.svg)](https://codecov.io/gh/etfrogers/pyhank)
[![PyPI version](https://badge.fury.io/py/pyhank.svg)](https://badge.fury.io/py/pyhank)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

PyHank is a python implementation of the quasi-discrete Hankel transform as developed by Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega:

> *"Computation of quasi-discrete Hankel transforms of the integer order for propagating optical wave fields"*
  Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega
  J. Opt. Soc. Am. A **21** (1) 53-58 (2004)

It operates on functions stored in NumPy arrays. If you want a Hankel transform that operates on a callable function, you may be interested in [hankel](https://github.com/steven-murray/hankel) by Steven Murray.

**PyHank 3.0** includes a high-performance compiled **Rust backend** (powered by `hankrs`), delivering **10x–50x speedups** for transform setup and computation across multi-dimensional arrays, with automatic fallback to pure Python where a compiler is unavailable.

I have used this code extensively for beam-propagation-method calculations of radially-symmetric beams. In the radially symmetric case, the 2D FFT over x and y that would be used in a non-symmetric system is replaced by a 1D QDHT over r, making the computational load much lighter and allowing bigger simulations.

PyHank was inspired by Adam Wyatt's [Matlab version](https://uk.mathworks.com/matlabcentral/fileexchange/15623-hankel-transform) which I used for many years, before moving to Python and needing my own implementation. It aims to simplify the interface (using Python's object-oriented approach) and utilise existing NumPy/SciPy functions wherever possible.

It has both a simple single-shot interface, and an object-oriented approach that precomputes transform matrices to speed up computation significantly when making multiple transforms on the same grid.

Contributions and comments are welcome using GitHub at:
https://github.com/etfrogers/pyhank


Installation
------------

Pre-compiled binary wheels (with compiled native Rust acceleration) are provided on PyPI for 64-bit Linux, macOS (Apple Silicon and Intel), and 64-bit Windows:

```bash
pip install pyhank
```

For development and running tests:

```bash
pip install -e .[dev]
```

For building documentation:

```bash
pip install -e .[docs]
```

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
