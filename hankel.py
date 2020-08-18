from enum import Enum, IntEnum

import numpy as np
import scipy.special as scipybessel
from scipy import interpolate


class BesselType(Enum):
    """Enum class specifying the type of Bessel function to use in
    :func:`.bessel_zeros`"""
    JN = 1  #: Bessel function of the first kind :math:`J_n`
    YN = 2  #: Bessel function of the second kind :math:`Y_n`
    JNP = 3  #: Derivative of Bessel function of the first kind :math:`J'_n`
    YNP = 4  #: Derivative of Bessel function of the second kind :math:`Y'_n`


class HankelTransformMode(IntEnum):
    """Enum class specifying the scaling of the functions used in the Hankel
    transform. See :ref:`Scaling <scaling>` for details of usage"""
    BOTH_SCALED = 0  #:
    FV_SCALED = 1  #:
    FR_SCALED = 2  #:
    UNSCALED = 3  #:


class HankelTransform:
    r"""The main class for performing Hankel Transforms

        :parameter order: Transform order :math:`p`
        :type order: :class:`int`
        :parameter max_radius: Radial extent of transform :math:`r_\textrm{max}`
        :type max_radius: :class:`float`
        :parameter n_points: Number of sample points :math:`N`
        :type n_points: :class:`int`

        :ivar alpha: The first :math:`N` Roots of the :math:`p` th order Bessel function.
        :ivar alpha_n1: (N+1)th root :math:`\alpha_{N1}`
        :ivar r: Radial co-ordinate vector
        :ivar v: frequency co-ordinate vector
        :ivar kr: Radial wave number co-ordinate vector
        :ivar v_max: Limiting frequency :math:`v_\textrm{max} = \alpha_{N1}/(2 \pi R)`
        :ivar S: RV product :math:`2\pi r_\textrm{max} v_max`
        :ivar T: Transform matrix
        :ivar JR: Radius transform vector :math:`J_R = J_{p+1}(\alpha) / r_\textrm{max}`
        :ivar JV: Frequency transform vector :math:`J_V = J_{p+1}(\alpha) / v_\textrm{max}`

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

        .. _scaling:

        .. admonition:: Scaling

            The :meth:`.HankelTransform.qdht` and :meth:`~.HankelTransform.iqdht` functions can accept
            ``scaling`` argument (an instance of :class:`HankelTransformMode`) which allows
            skipping the scaling that is otherwise necessary in the
            algorithm. For a use case when the same function is transformed multiple times,
            this can increase the speed of the algorithm see :ref:`Test <sphx_glr_auto_examples_scaling_speed>`
            for an example of this.

            If the ``scaling`` argument is passed to :meth:`~.HankelTransform.qdht` and
            :meth:`~.HankelTransform.iqdht` then the following equations are used, where
            :math:`\mathbf{T}` is the transform matrix :attr:`.HankelTransform.T`

            :attr:`~.HankelTransformMode.BOTH_SCALED`
                :math:`f_r` & :math:`f_v` are the scaled functions (i.e. fr./self.JR & fv./self.JV)

                .. math::
                    f_v = \mathbf{T} \times f_r \quad f_r = \mathbf{T} \times f_v

            :attr:`~.HankelTransformMode.FV_SCALED`
                :math:`f_v` is the scaled function, :math:`f_r` is the real function

                .. math::
                    f_v = \mathbf{T} \times (f_r / J_R) \quad f_r = (\mathbf{T} \times f_v) \times J_R

            :attr:`~.HankelTransformMode.FR_SCALED`
                :math:`f_r` is the scaled function, :math:`f_v` is the real function

                .. math::
                    f_v = (\mathbf{T} \times f_r) \times J_V \quad f_r = \mathbf{T} \times (f_v / J_V)

            :attr:`~.HankelTransform.UNSCALED` (**default**)
                :math:`f_r`, :math:`f_v` are the real distributions.

                .. math::
                    f_v = (\mathbf{T} \times (f_r / J_R)) \times J_V \quad
                    f_r = (\mathbf{T} \times (f_v / J_V)) \times J_R
        """

    def __init__(self, order: int, max_radius: float = None, n_points: int = None,
                 radial_grid: np.ndarray = None):
        """Constructor"""

        usage = 'Either radial_grid or both max_radius and n_points must be supplied'
        if radial_grid is None:
            if max_radius is None or n_points is None:
                raise ValueError(usage)
        else:
            if max_radius is not None or n_points is not None:
                raise ValueError(usage)
            assert radial_grid.ndim == 1, 'Radial grid must be a 1d array'
            max_radius = np.max(radial_grid)
            n_points = radial_grid.size

        self._order = order
        self._max_radius = max_radius
        self._n_points = n_points
        self._original_radial_grid = radial_grid

        # Calculate N+1 roots:
        alpha = bessel_zeros(BesselType.JN, self.order, self.n_points + 1)
        self.alpha = alpha[0:-1]
        self.alpha_n1 = alpha[-1]

        # Calculate co-ordinate vectors
        self.r = self.alpha * self.max_radius / self.alpha_n1
        self.v = self.alpha / (2 * np.pi * self.max_radius)
        self.kr = 2 * np.pi * self.v
        self.v_max = self.alpha_n1 / (2 * np.pi * self.max_radius)
        self.S = self.alpha_n1

        # Calculate hankel matrix and vectors
        jp = scipybessel.jv(order, (self.alpha[:, np.newaxis] @ self.alpha[np.newaxis, :]) / self.S)
        jp1 = np.abs(scipybessel.jv(order + 1, self.alpha))
        self.T = 2 * jp / ((jp1[:, np.newaxis] @ jp1[np.newaxis, :]) * self.S)
        self.JR = jp1 / self.max_radius
        self.JV = jp1 / self.v_max

    @property
    def order(self):
        return self._order

    @property
    def max_radius(self):
        return self._max_radius

    @property
    def n_points(self):
        return self._n_points

    @property
    def original_radial_grid(self):
        if self._original_radial_grid is None:
            raise ValueError('Attempted to access original_radial_grid on HankelTransform '
                             'object that was not constructed with a r_grid')
        return self._original_radial_grid

    def to_transform_r(self, function):
        return _spline(self.original_radial_grid, function, self.r)

    def to_original_r(self, function):
        return _spline(self.r, function, self.original_radial_grid)

    def qdht(self, fr: np.ndarray,
             scaling: HankelTransformMode = HankelTransformMode.UNSCALED):
        r"""QDHT: Quasi Discrete Hankel Transform

        Performs the Hankel transform of a function of radius, returning
        a function of frequency.

        .. math::
            f_v(v) = \mathcal{H}^{-1}\{f_r(r)\}

        .. warning:
            The input function must be sampled at the points ``self.r``, and the output
            will be sampled at the points ``self.v`` (or equivalently ``self.kr``)

        See :ref:`Scaling <scaling>` above for a description of the effect of ``scaling``

        :parameter fr: Function in real space as a function of radius (sampled at ``self.r``)
        :type fr: :class:`numpy.ndarray`
        :parameter scaling: (optional) Parameter to control the scaling of input and output. See Scaling above
        :type scaling: :class:`.HankelTransformMode`

        :return fv: Function in frequency space (sampled at self.v)
        :rtype: :class:`numpy.ndarray`
        """
        jr, jv = self._get_scaling_factors(fr)

        if scaling == HankelTransformMode.FV_SCALED:
            fv = np.matmul(self.T, (fr / jr))
        elif scaling == HankelTransformMode.FR_SCALED:
            fv = jv * np.matmul(self.T, fr)
        elif scaling == HankelTransformMode.UNSCALED:
            fv = jv * np.matmul(self.T, (fr / jr))
        elif scaling == HankelTransformMode.BOTH_SCALED:
            fv = np.matmul(self.T, fr)
        else:
            raise NotImplementedError
        return fv

    def _get_scaling_factors(self, fr):
        try:
            n2 = fr.shape[1]
        except IndexError:
            n2 = 1
        if n2 > 1:
            jr = self.JR[:, np.newaxis] * np.ones((1, n2))
            jv = self.JV[:, np.newaxis] * np.ones((1, n2))
        else:
            jr = self.JR
            jv = self.JV
        return jr, jv

    def iqdht(self, fv: np.ndarray,
              scaling: HankelTransformMode = HankelTransformMode.UNSCALED):
        r"""IQDHT: Inverse Quasi Discrete Hankel Transform

        Performs the inverse Hankel transform of a function of frequency, returning
        a function of radius.

        .. math::
            f_r(r) = \mathcal{H}^{-1}\{f_v(v)\}

        See :ref:`Scaling <scaling>` above for a description of the effect of ``scaling``

        :parameter fv: Function in frequency space (sampled at self.v)
        :type fv: :class:`numpy.ndarray`
        :parameter scaling: (optional) Parameter to control the scaling of input and output. See Scaling above
        :type scaling: :class:`.HankelTransformMode`

        :return fr: Radial function (sampled at self.r) = IHT(fv)
        :rtype: :class:`numpy.ndarray`
        """
        jr, jv = self._get_scaling_factors(fv)

        if scaling == HankelTransformMode.FR_SCALED:
            fr = np.matmul(self.T, (fv / jv))
        elif scaling == HankelTransformMode.FV_SCALED:
            fr = np.matmul(self.T, fv) * jr
        elif scaling == HankelTransformMode.UNSCALED:
            fr = jr * np.matmul(self.T, (fv / jv))
        elif scaling == HankelTransformMode.BOTH_SCALED:
            fr = np.matmul(self.T, fv)
        else:
            raise NotImplementedError
        return fr


def bessel_zeros(bessel_function_type: BesselType, bessel_order: int, n_zeros: int):
    """Find the first :code:`n_zeros` zeros of a Bessel function of order :code:`bessel_order`.

    The type of the Bessel function can be selected using the ``bessel_function_type`` parameter.
    It can be :math:`J_n`, :math:`Y_n`, :math:`J'_n`, or :math:`Y'_n`.

    This function is a convenience wrapper for :func:`scipy.special.jn_zeros`,
    :func:`~scipy.special.yn_zeros`, :func:`~scipy.special.jnp_zeros`, and
    :func:`~scipy.special.ynp_zeros`. It calls those functions to the actual
    calculation.

    :parameter bessel_function_type: :class:`.BesselType` object specifying the
        type of Bessel function for which to find the zeros
    :type bessel_function_type: :class:`.BesselType`
    :parameter bessel_order: Bessel order The order of the Bessel function :math:`n`
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


def _spline(x0, y0, x, **kwargs):
    if 'kind' not in kwargs:
        kwargs['kind'] = 'cubic'
    f = interpolate.interp1d(x0, y0, axis=0, fill_value='extrapolate', **kwargs)
    return f(x)
