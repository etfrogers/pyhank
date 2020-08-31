from typing import Tuple

import numpy as np
import scipy.special as scipy_bessel
from scipy import interpolate


class HankelTransform:
    r"""The main class for performing Hankel Transforms

        For the QDHT to work, the function must be sampled a specific points, which this class generates
        and stores in :attr:`HankelTransform.r`. Any transform on this grid will be sampled at points
        :attr:`.HankelTransform.v` (frequency space) or equivalently :attr:`.HankelTransform.kr`
        (angular frequency or wavenumber space).

        The constructor has one required argument (``order``). The remaining four arguments offer
        three different ways of specifying the radial (and therefore implicitly the frequency) points:

        1. Supply both a maximum radius ``r_max`` and number of transform points ``n_points``
        2. Supply the original (often equally spaced) ``radial_grid`` on which you have currently
           have sample points. This approach allows easy conversion from the original grid using
           :meth:`.HankelTransform.to_transform_r()`. ``t = HankelTransform(order, radial_grid=r)``
           is effectively equivalent to ``t = HankelTransform(order, n_points=r.size, r_max=np.max(r))``
           except for the fact the the original radial grid is stored in the :class:`.HankelTransform`
           object for use in :meth:`~.HankelTransform.to_transform_r` and
           :meth:`~.HankelTransform.to_original_r`.
        3. Supply the original (often equally spaced) :math:`k`-space grid on which you
           have currently have sample points. This is most use if you intend to do inverse
           transforms. It allows easy conversion to and from the original grid using
           :meth:`~.HankelTransform.to_original_k()` and :meth:`~.HankelTransform.to_transform_k()`.
           As in option 2, :attr:`.HankelTransform.n_points` is determined by ``k_grid.size``.
           :attr:`HankelTransform.r_max` is determined in a more complex way from ``np.max(k_grid)``.

        :parameter order: Transform order :math:`p`
        :type order: :class:`int`
        :parameter max_radius: (Optional) Radial extent of transform :math:`r_\textrm{max}`
        :type max_radius: :class:`float`
        :parameter n_points: (Optional) Number of sample points :math:`N`
        :type n_points: :class:`int`
        :parameter radial_grid: (Optional) The radial grid that will be used to sample input functions
            it is used to set `N` and :math:`r_\textrm{max}` by ``n_points = radial_grid.size`` and
            ``r_max = np.max(radial_grid)``
        :type radial_grid: :class:`numpy.ndarray`
        :parameter k_grid: (Optional) Number of sample points :math:`N`
        :type k_grid: :class:`numpy.ndarray`

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

        The algorithm also calls the function :func:`scipy.special.jn_zeros` to calculate
        the roots of the bessel function.
        """

    def __init__(self, order: int, max_radius: float = None, n_points: int = None,
                 radial_grid: np.ndarray = None, k_grid: np.ndarray = None):
        """Constructor"""

        usage = 'Either radial_grid or k_grid or both max_radius and n_points must be supplied'
        if radial_grid is None and k_grid is None:
            if max_radius is None or n_points is None:
                raise ValueError(usage)
        elif k_grid is not None:
            if max_radius is not None or n_points is not None or radial_grid is not None:
                raise ValueError(usage)
            assert k_grid.ndim == 1, 'k grid must be a 1d array'
            n_points = k_grid.size
        elif radial_grid is not None:
            if max_radius is not None or n_points is not None:
                raise ValueError(usage)
            assert radial_grid.ndim == 1, 'Radial grid must be a 1d array'
            max_radius = np.max(radial_grid)
            n_points = radial_grid.size
        else:
            raise ValueError(usage)  # pragma: no cover - backup case: cannot currently be reached

        self._order = order
        self._n_points = n_points
        self._original_radial_grid = radial_grid
        self._original_k_grid = k_grid

        # Calculate N+1 roots must be calculated before max_radius can be derived from k_grid
        alpha = scipy_bessel.jn_zeros(self.order, self.n_points + 1)
        self.alpha = alpha[0:-1]
        self.alpha_n1 = alpha[-1]

        if k_grid is not None:
            v_max = np.max(k_grid) / (2 * np.pi)
            max_radius = self.alpha_n1 / (2 * np.pi * v_max)
        self._max_radius = max_radius

        # Calculate co-ordinate vectors
        self.r = self.alpha * self.max_radius / self.alpha_n1
        self.v = self.alpha / (2 * np.pi * self.max_radius)
        self.kr = 2 * np.pi * self.v
        self.v_max = self.alpha_n1 / (2 * np.pi * self.max_radius)
        self.S = self.alpha_n1

        # Calculate hankel matrix and vectors
        jp = scipy_bessel.jv(order, (self.alpha[:, np.newaxis] @ self.alpha[np.newaxis, :]) / self.S)
        jp1 = np.abs(scipy_bessel.jv(order + 1, self.alpha))
        self.T = 2 * jp / ((jp1[:, np.newaxis] @ jp1[np.newaxis, :]) * self.S)
        self.JR = jp1 / self.max_radius
        self.JV = jp1 / self.v_max

    @property
    def order(self) -> int:
        return self._order

    @property
    def max_radius(self) -> float:
        return self._max_radius

    @property
    def n_points(self) -> int:
        return self._n_points

    @property
    def original_radial_grid(self) -> np.ndarray:
        """ Return the original radial grid used to construct the object, or raise a :class:`ValueError`
        if the constructor was not called specifying a ``radial_grid`` parameter.

        :return: The original radial grid used to construct the object.
        :rtype: :class:`numpy.ndarray`
        """
        if self._original_radial_grid is None:
            raise ValueError('Attempted to access original_radial_grid on HankelTransform '
                             'object that was not constructed with a radial_grid')
        return self._original_radial_grid

    @property
    def original_k_grid(self) -> np.ndarray:
        """ Return the original k grid used to construct the object, or raise a :class:`ValueError`
        if the constructor was not called specifying a ``k_grid`` parameter.

        :return: The original k grid used to construct the object.
        :rtype: :class:`numpy.ndarray`
        """
        if self._original_k_grid is None:
            raise ValueError('Attempted to access original_k_grid on HankelTransform '
                             'object that was not constructed with a k_grid')
        return self._original_k_grid

    def to_transform_r(self, function: np.ndarray) -> np.ndarray:
        """Interpolate a function, assumed to have been given at the original radial
        grid points used to construct the ``HankelTransform`` object onto the grid required
        of use in the QDHT algorithm.

        If the the ``HankelTransform`` object was constructed with a (say) equally-spaced
        grid in radius, then it needs the function to transform to be sampled at a specific
        grid before it can be passed to :meth:`.HankelTransform.qdht`. This method provides
        a convenient way of doing this.

        :parameter function: The function to be interpolated. Specified at the radial points
            :attr:`~.HankelTransform.original_radial_grid`.
        :type function: :class:`numpy.ndarray`

        :return: Interpolated function suitable to passing to
            :meth:`HankelTransform.qdht` (sampled at ``self.r``)
        :rtype: :class:`numpy.ndarray`
        """
        return _spline(self.original_radial_grid, function, self.r)

    def to_original_r(self, function: np.ndarray) -> np.ndarray:
        """Interpolate a function, assumed to have been given at the Hankel transform points
        ``self.r`` (as returned by :meth:`HankelTransform.iqdht`) back onto the original grid
        used to construct the ``HankelTransform`` object.

        If the the ``HankelTransform`` object was constructed with a (say) equally-spaced
        grid in radius, it may be useful to convert back to this grid after a IQDHT.
        This method provides a convenient way of doing this.

        :parameter function: The function to be interpolated. Specified at the radial points
            ``self.r``.
        :type function: :class:`numpy.ndarray`

        :return: Interpolated function at the points held in :attr:`~.HankelTransform.original_radial_grid`.
        :rtype: :class:`numpy.ndarray`
        """
        return _spline(self.r, function, self.original_radial_grid)

    def to_transform_k(self, function: np.ndarray) -> np.ndarray:
        """Interpolate a function, assumed to have been given at the original k
        grid points used to construct the ``HankelTransform`` object onto the grid required
        of use in the IQDHT algorithm.

        If the the ``HankelTransform`` object was constructed with a (say) equally-spaced
        grid in :math:`k`, then it needs the function to transform to be sampled at a specific
        grid before it can be passed to :meth:`.HankelTransform.iqdht`. This method provides
        a convenient way of doing this.

        :parameter function: The function to be interpolated. Specified at the k points
            :attr:`~.HankelTransform.original_k_grid`.
        :type function: :class:`numpy.ndarray`

        :return: Interpolated function suitable to passing to
            :meth:`HankelTransform.qdht` (sampled at ``self.kr``)
        :rtype: :class:`numpy.ndarray`
        """

        return _spline(self.original_k_grid, function, self.kr)

    def to_original_k(self, function: np.ndarray) -> np.ndarray:
        """Interpolate a function, assumed to have been given at the Hankel transform points
        ``self.k`` (as returned by :meth:`HankelTransform.qdht`) back onto the original grid
        used to construct the ``HankelTransform`` object.

        If the the ``HankelTransform`` object was constructed with a (say) equally-spaced
        grid in :math:`k`, it may be useful to convert back to this grid after a QDHT.
        This method provides a convenient way of doing this.

        :parameter function: The function to be interpolated. Specified at the radial points
            ``self.k``.
        :type function: :class:`numpy.ndarray`

        :return: Interpolated function at the points held in :attr:`~.HankelTransform.original_k_grid`.
        :rtype: :class:`numpy.ndarray`
        """
        return _spline(self.kr, function, self.original_k_grid)

    def qdht(self, fr: np.ndarray) -> np.ndarray:
        r"""QDHT: Quasi Discrete Hankel Transform

        Performs the Hankel transform of a function of radius, returning
        a function of frequency.

        .. math::
            f_v(v) = \mathcal{H}^{-1}\{f_r(r)\}

        .. warning:
            The input function must be sampled at the points ``self.r``, and the output
            will be sampled at the points ``self.v`` (or equivalently ``self.kr``)

        :parameter fr: Function in real space as a function of radius (sampled at ``self.r``)
        :type fr: :class:`numpy.ndarray`

        :return: Function in frequency space (sampled at ``self.v``)
        :rtype: :class:`numpy.ndarray`
        """
        jr, jv = self._get_scaling_factors(fr)

        fv = jv * np.matmul(self.T, (fr / jr))
        return fv

    def iqdht(self, fv: np.ndarray) -> np.ndarray:
        r"""IQDHT: Inverse Quasi Discrete Hankel Transform

        Performs the inverse Hankel transform of a function of frequency, returning
        a function of radius.

        .. math::
            f_r(r) = \mathcal{H}^{-1}\{f_v(v)\}

        :parameter fv: Function in frequency space (sampled at self.v)
        :type fv: :class:`numpy.ndarray`

        :return: Radial function (sampled at self.r) = IHT(fv)
        :rtype: :class:`numpy.ndarray`
        """
        jr, jv = self._get_scaling_factors(fv)
        fr = jr * np.matmul(self.T, (fv / jv))
        return fr

    def _get_scaling_factors(self, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            n2 = f.shape[1]
            jr = self.JR[:, np.newaxis] * np.ones((1, n2))
            jv = self.JV[:, np.newaxis] * np.ones((1, n2))
        except IndexError:
            jr = self.JR
            jv = self.JV
        return jr, jv


def _spline(x0: np.ndarray, y0: np.ndarray, x: np.ndarray) -> np.ndarray:
    f = interpolate.interp1d(x0, y0, axis=0, fill_value='extrapolate', kind='cubic')
    return f(x)
