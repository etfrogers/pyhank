====================================================
PyHank - Quasi-Discrete Hankel Transforms for Python
====================================================

.. image:: https://readthedocs.org/projects/pyhank/badge/?version=latest
  :target: https://pyhank.readthedocs.io/en/latest/?badge=latest
 :alt: Documentation Status


.. image:: https://img.shields.io/travis/com/etfrogers/pyhank/master.svg?label=tests
  :alt: Test Status

.. image:: https://codecov.io/gh/etfrogers/pyhank/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/etfrogers/pyhank

PyHank is a python implementation of the quasi-discrete Hankel transform as developed by Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega [#Guizar]_

It was inspired by Adam Wyatt's `Matlab version <https://uk.mathworks.com/matlabcentral/fileexchange/15623-hankel-transform>`_ which I used for many years, before moving to Python and needing my own implementation. It aims to simplify the interface (using Python's object-oriented approach).

It has both a simple single-shot interface, and a more advanced approach that speeds up computation significantly if making multiple transforms on the same grid.

Contributions and comments are welcome using Github at:
http://github.com/etfrogers/pyhank

.. [#Guizar] *"Computation of quasi-discrete Hankel transforms of the integer order for propagating optical wave fields"*
  Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega
  J. Opt. Soc. Am. A **21** (1) 53-58 (2004)

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

The documentation for PyHank can be found at `Read the docs <https://pyhank.readthedocs.io/en/latest/index.html>`_

Usage
-----

See the `Usage examples at ReadTheDocs <https://pyhank.readthedocs.io/en/latest/auto_examples/index.html>`_


:author: Edward Rogers
:date: 18/08/2020
