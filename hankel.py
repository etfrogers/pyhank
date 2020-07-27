from enum import Enum

import numpy as np
import scipy.special as scipybessel
from scipy import interpolate


class BesselType(Enum):
    """TODO"""
    JN = 1  # Jn
    YN = 2  # Yn
    JNP = 3  # J_n'
    YNP = 4  # Y_n'


class HankelTransformMode(Enum):
    """TODO"""
    BOTH_SCALED = 0
    SCALED_OUTPUT = 1
    INPUT_SCALED = 2
    UNSCALED = 3


class HankelTransform:
    r"""The main class for performing Hankel Transforms
        
        :parameter order: Transform order :math:`p`
        :type order: :class:`int`
        :parameter max_radius: Radial extent of transform :math:`r_\textrm{max}`
        :type max_radius: :class:`float`
        :parameter n_points: Number of sample points :math:`N`
        :type n_points: :class:`int`

        :ivar alpha: The first :math:`N` Roots of the :math:`p` th order Bessel function.
        :ivar alpha_N1: (N+1)th root :math:`\alpha_{N1}`
        :ivar r: Radial co-ordinate vector
        :ivar v: frequency co-ordinate vector
        :ivar kr: Radial wave number co-ordinate vector
        :ivar V: Limiting frequency :math:`V = \alpha_{N1}/(2 \pi R)`
        :ivar S: RV product :math:`2\pi r_\textrm{max} V`
        :ivar T: Transform matrix
        :ivar JR: Radius transform vector :math:`J_{p+1}(\alpha) / r_\textrm{max}`
        :ivar JV: Frequency transform vector :math:`J_{p+1}(\alpha) / V`

        The algorithm used is that from:

            *"Computation of quasi-discrete Hankel transforms of the integer
            order for propagating optical wave fields"*
            Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega
            J. Opt. Soc. Am. A **21** (1) 53-58 (2004)
        
        The algorithm also calls the function:

        .. code-block:: python

            alpha = bessel_zeros(BesselType.JN, order, n_points+1,)

        where ``order`` and ``n_points`` are defined above, to calculate the roots of the bessel
        function. 
        """

    def __init__(self, order: int, max_radius: float, n_points: int):
        """Constructor"""
        self._order = order
        self._max_radius = max_radius
        self._n_points = n_points

        # Calculate N+1 roots:
        alpha = bessel_zeros(BesselType.JN, self.p, self.n_points + 1)
        self.alpha = alpha[0:-1]
        self.alpha_N1 = alpha[-1]

        # Calculate co-ordinate vectors
        self.r = self.alpha * self.max_radius / self.alpha_N1
        self.v = self.alpha / (2 * np.pi * self.max_radius)
        self.kr = 2 * np.pi * self.v
        self.V = self.alpha_N1 / (2 * np.pi * self.max_radius)
        self.S = self.alpha_N1

        # Calculate hankel matrix and vectors
        jp = scipybessel.jv(order, np.matmul(self.alpha[:, np.newaxis], self.alpha[np.newaxis, :]) / self.S)
        jp1 = np.abs(scipybessel.jv(order + 1, self.alpha))
        self.T = 2 * jp / (np.matmul(jp1[:, np.newaxis], jp1[np.newaxis, :]) * self.S)
        self.JR = jp1 / self.max_radius
        self.JV = jp1 / self.V

    @property
    def p(self):
        return self._order

    @property
    def max_radius(self):
        return self._max_radius

    @property
    def n_points(self):
        return self._n_points

    def qdht(self, input_function: np.ndarray,
             mode: HankelTransformMode = HankelTransformMode.UNSCALED):
        """ QDHT: Quasi Discrete Hankel Transform

        fv		            = HT(input_function) (sampled at self.v)
        input_function		= Radial function (sampled at self.r)
        mode		= Mode (optional)
        mode = BOTH_SCALED(0) :
            fv = self.T * input_function;
            input_function & fv are the scaled functions (i.e. input_function./self.JR & fv./self.JV)
        mode = SCALED_OUTPUT(1):
            fv = self.T * (input_function ./ self.JR);
            fv is the scaled function, input_function the real function.
        mode = INPUT_SCALED(2):
            fv = self.JV .* (self.T * input_function);
            fv is the real function, input_function is the scaled function
        mode = UNSCALED(3) (default):
            fv = self.JV .* (self.T * (input_function ./ self.JR));
            input_function, fv are the real distributions.
        """
        try:
            n2 = input_function.shape[1]
        except IndexError:
            n2 = 1
        if n2 > 1:
            jr = self.JR * np.ones((1, n2))
            jv = self.JV * np.ones((1, n2))
        else:
            jr = self.JR
            jv = self.JV

        if mode == HankelTransformMode.SCALED_OUTPUT:
            fv = np.matmul(self.T, (input_function / jr))
        elif mode == HankelTransformMode.INPUT_SCALED:
            fv = jv * np.matmul(self.T, input_function)
        elif mode == HankelTransformMode.UNSCALED:
            fv = jv * np.matmul(self.T, (input_function / jr))
        elif mode == HankelTransformMode.BOTH_SCALED:
            fv = np.matmul(self.T, input_function)
        else:
            raise NotImplementedError
        return fv

    def iqdht(self, input_function: np.ndarray,
              mode: HankelTransformMode = HankelTransformMode.UNSCALED):
        """IQDHT: Inverse Quasi Discrete Hankel Transform

        fr		= HT(input_function) (sampled at self.v)
        fr		= Radial function (sampled at self.r)
        mode		= Mode (optional)

        mode = BOTH_SCALED (0) :
            fr = self.T * input_function;
            fr & input_function are the scaled functions (i.e. fr./self.JR & input_function./self.JV)
        mode = INPUT_SCALED (2):
            fr = (self.T * input_function) .* self.JR;
            input_function is the scaled function, fr the real function.
        mode = SCALED_OUTPUT (1):
            fr = self.T * (input_function ./ self.JV);
            input_function is the real function, fr the scaled function
        mode = UNSCALED (3) (default):
            fr = self.JR .* (self.T * (input_function ./ self.JV));
            fr, input_function are the real distributions.
        """

        try:
            n2 = input_function.shape[1]
        except IndexError:
            n2 = 1
        if n2 > 1:
            jr = self.JR * np.ones((1, n2))
            jv = self.JV * np.ones((1, n2))
        else:
            jr = self.JR
            jv = self.JV

        if mode == HankelTransformMode.INPUT_SCALED:
            fr = np.matmul(self.T, input_function) * jr
        elif mode == HankelTransformMode.SCALED_OUTPUT:
            fr = np.matmul(self.T, (input_function / jv))
        elif mode == HankelTransformMode.UNSCALED:
            fr = jr * np.matmul(self.T, (input_function / jv))
        elif mode == HankelTransformMode.BOTH_SCALED:
            fr = np.matmul(self.T, input_function)
        else:
            raise NotImplementedError
        return fr


def bessel_zeros(bessel_function_type: BesselType, bessel_order: int, n_zeros: int):
    """Find the first :code:`n_zeros` zeros of a Bessel function of order :code:`bessel_order`.

    Bessel function type:
    JN (1):	    Jn
    YN (2):	    Yn
    JNP (3):	Jn'
    YNP (4):	Yn'

    This function is a convenience wrapper for :func:`scipy.special.jn_zeros`

    :parameter bessel_function_type:
    :type bessel_function_type: :class:`.BesselType`
    :parameter bessel_order: Bessel order The ordern the Bessel function :math:`n`
    :type bessel_order: :class:`int`
    :parameter n_zeros:	Number of zeros to find
    :type n_zeros: :class:`int`

    :return: Zeros of the Bessel function
    :rtype: :class:`numpy.ndarray`

    """
    if bessel_function_type == BesselType.JN:
        return scipybessel.jn_zeros(bessel_order, n_zeros)
    elif bessel_function_type == BesselType.YN:
        return scipybessel.yn_zeros(bessel_order, n_zeros)
    elif bessel_function_type == BesselType.JNP:
        zeros = scipybessel.jnp_zeros(bessel_order, n_zeros)
        if bessel_order == 0:
            # to match Matlab implementation
            zeros[1:] = zeros[:-1]
            zeros[0] = 0
        return zeros
    elif bessel_function_type == BesselType.YNP:
        return scipybessel.ynp_zeros(bessel_order, n_zeros)
    else:
        raise NotImplementedError


def spline(x0, y0, x):
    # tck = interpolate.splrep(x0, y0, s=0)
    # return interpolate.splev(x, tck)
    f = interpolate.interp1d(x0, y0, 'cubic', axis=0, fill_value='extrapolate')
    return f(x)
