
PyHank is a python implementation of the quasi-discrete Hankel transform as developed by Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega [#Guizar]_

It was inspired by Adam Wyatt's `Matlab version <https://uk.mathworks.com/matlabcentral/fileexchange/15623-hankel-transform>`_ which I used for many years, before moving to Python and needing my own implementation. It aims to simplify the interface (using Python's object-oriented approach).

It has both a simple single-shot interface, and a more advanced approach that speeds up computation significantly if making multiple transforms on the same grid.

Contributions and comments are welcome using Github at:
http://github.com/etfrogers/pyhank

.. [#Guizar] *"Computation of quasi-discrete Hankel transforms of the integer order for propagating optical wave fields"*
  Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega
  J. Opt. Soc. Am. A **21** (1) 53-58 (2004)
