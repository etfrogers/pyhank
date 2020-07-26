from enum import Enum

import numpy as np
import scipy.special as scipybessel
from scipy import interpolate


class BesselType(Enum):
    JN = 1  # Jn
    YN = 2  # Yn
    JNP = 3  # J_n'
    YNP = 4  # Y_n'


class HankelTransformMode(Enum):
    BOTH_SCALED = 0
    SCALED_OUTPUT = 1
    INPUT_SCALED = 2
    UNSCALED = 3


class HankelTransform:
    def __init__(self, order: int, max_radius: float, n_points: int):
        """HANKEL_MATRIX: Generates data to use for Hankel Transforms
        %
        %	self = hankel_matrix(order, max_radius, n_points)
        %
        %	order		    =	Transform order
        %	max_radius		=	Radial extent of transform
        %	n_points		=	Number of sample points
        %
        %	self:
        %		alpha		=	Roots of the pth order Bessel fn.
        %					=	[alpha_1, alpha_2, ... alpha_N]
        %		alpha_N1	=	(N+1)th root
        %		r			=	Radial co-ordinate vector
        %		v			=	frequency co-ordinate vector
        %		kr			=	Radial wave number co-ordinate vector
        %		V			=	Limiting frequency
        %					=	alpha_N1/(2piR)
        %		S			=	RV product
        %					=	2*pi*max_radius*V
        %		T			=	Transform matrix
        %		JR, JV		=	Transform vectors
        %					=	J_p1(alpha)/max_radius  or jp1(alpha)/V where p1 = order+1
        %
        %	The algorithm used is that from:
        %		"Computation of quasi-discrete Hankel transforms of the integer
        %		order for propagating optical wave fields"
        %		Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega
        %		J. Opt. Soc. Am. A 21(1) 53-58 (2004)
        %
        %	The algorithm also calls the function:
        %	zn = bessel_zeros(1, order, N+1, 1e-6),
        %	where order and N are defined above, to calculate the roots of the bessel
        %	function. This algorithm is taken from:
        %  		"An Algorithm with ALGOL 60 Program for the Computation of the
        %  		zeros of the Ordinary Bessel Functions and those of their
        %  		Derivatives".
        %  		N. M. Temme
        %  		Journal of Computational Physics, 32, 270-279 (1979)"""

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


def bessel_zeros(bessel_function_type: BesselType, bessel_order, n):
    """BESSEL_ZEROS: Finds the first n zeros of bessel_order bessel function
    z = bessel_zeros(bessel_function_type, bessel_order, n, e)

    z	=	zeros of the bessel function
    bessel_function_type	=	Bessel function type:
        JN (1):	    Jn
        YN (2):	    Yn
        JNP (3):	Jn'
        YNP (4):	Yn'
    bessel_order	=	Bessel order (bessel_order>=0)
    n	=	Number of zeros to find
    e	=	Relative error in root
    %
    This function uses the routine described in:
        "An Algorithm with ALGOL 60 Program for the Computation of the
        zeros of the Ordinary Bessel Functions and those of their
        Derivatives".
        N. M. Temme
        Journal of Computational Physics, 32, 270-279 (1979)"""

    if bessel_function_type == BesselType.JN:
        return scipybessel.jn_zeros(bessel_order, n)
    elif bessel_function_type == BesselType.YN:
        return scipybessel.yn_zeros(bessel_order, n)
    elif bessel_function_type == BesselType.JNP:
        zeros = scipybessel.jnp_zeros(bessel_order, n)
        if bessel_order == 0:
            # to match Matlab implementation
            zeros[1:] = zeros[:-1]
            zeros[0] = 0
        return zeros
    elif bessel_function_type == BesselType.YNP:
        return scipybessel.ynp_zeros(bessel_order, n)
    else:
        raise NotImplementedError


def spline(x0, y0, x):
    # tck = interpolate.splrep(x0, y0, s=0)
    # return interpolate.splev(x, tck)
    f = interpolate.interp1d(x0, y0, 'cubic', axis=0, fill_value='extrapolate')
    return f(x)

