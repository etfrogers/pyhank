import numpy as np
import scipy.special as scipy_bessel
from scipy import interpolate
from scipy.optimize import brentq


class HankelTransform:
    r"""The main class for performing Hankel Transforms

    For the QDHT to work, the function must be sampled a specific points, which this class generates
    and stores in :attr:`HankelTransform.r`. Any transform on this grid will be sampled at points
    :attr:`.HankelTransform.v` (frequency space) or equivalently :attr:`.HankelTransform.kr`
    (angular frequency or wavenumber space).

    The constructor has one required argument (``order``). The remaining five arguments offer
    three different ways of specifying the radial (and therefore implicitly the frequency) points
    and type of Bessel functions:

    1. Supply both a maximum radius ``r_max`` and number of transform points ``n_points``
    2. Supply the original (often equally spaced) ``radial_grid`` on which you currently
       have sample points. This approach allows easy conversion from the original grid using
       :meth:`.HankelTransform.to_transform_r()`. ``t = HankelTransform(order, radial_grid=r)``
       is effectively equivalent to ``t = HankelTransform(order, n_points=r.size, r_max=np.max(r))``
       except for the fact the original radial grid is stored in the :class:`.HankelTransform`
       object for use in :meth:`~.HankelTransform.to_transform_r` and
       :meth:`~.HankelTransform.to_original_r`.
    3. Supply the original (often equally spaced) :math:`k`-space grid on which you
       currently have sample points. This is most use if you intend to do inverse
       transforms. It allows easy conversion to and from the original grid using
       :meth:`~.HankelTransform.to_original_k()` and :meth:`~.HankelTransform.to_transform_k()`.
       As in option 2, :attr:`.HankelTransform.n_points` is determined by ``k_grid.size``.
       :attr:`HankelTransform.r_max` is determined in a more complex way from ``np.max(k_grid)``.

    By setting the argument ``bessel_type`` to either ``"polar"`` or ``"spherical"`` it is possible
    to choose between :math:`J_n` and :math:`j_n` Bessel functions (default is ``"polar"``).

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
    :parameter k_grid: (Optional) The :math:`k`-space grid that will be used to sample input functions
    :type k_grid: :class:`numpy.ndarray`
    :parameter bessel_type: (Optional) Type of Bessel functions used to compute the transform
    :type bessel_type: :class:`str`

    :ivar order: Transform order :math:`p`
    :ivar n_points: Number of sample points :math:`N`
    :ivar max_radius: Radial extent of transform :math:`r_\textrm{max}`
    :ivar bessel_type: Type of Bessel functions used (``"polar"`` or ``"spherical"``)
    :ivar r: Radial coordinate vector
    :ivar v: Frequency coordinate vector
    :ivar kr: Radial wavenumber coordinate vector (:math:`2\pi v`)
    :ivar T: Unitary transform matrix
    :ivar original_radial_grid: Original radial grid used to construct the transform, if provided
    :ivar original_k_grid: Original :math:`k`-space grid used to construct the transform, if provided

    The algorithm used is that from:

        *"Computation of quasi-discrete Hankel transforms of the integer
        order for propagating optical wave fields"*
        Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega
        J. Opt. Soc. Am. A **21** (1) 53-58 (2004)

    The algorithm also calls the function :func:`scipy.special.jn_zeros` to calculate
    the roots of the bessel function.
    """

    __module__ = "pyhank"

    def __init__(
        self,
        order: int,
        max_radius: float | None = None,
        n_points: int | None = None,
        radial_grid: np.ndarray | None = None,
        k_grid: np.ndarray | None = None,
        bessel_type: str = "polar",
    ):
        """Constructor"""

        usage = "Either radial_grid or k_grid or both max_radius and n_points must be supplied"
        if radial_grid is None and k_grid is None:
            if max_radius is None or n_points is None:
                raise ValueError(usage)
        elif k_grid is not None:
            if max_radius is not None or n_points is not None or radial_grid is not None:
                raise ValueError(usage)
            if k_grid.ndim != 1:
                raise TypeError("k grid must be a 1d array")
            n_points = k_grid.size
        elif radial_grid is not None:
            if max_radius is not None or n_points is not None:
                raise ValueError(usage)
            if radial_grid.ndim != 1:
                raise TypeError("Radial grid must be a 1d array")
            max_radius = np.max(radial_grid)
            n_points = radial_grid.size
        else:
            raise ValueError(usage)  # pragma: no cover - backup case: cannot currently be reached

        self._order = order
        self._n_points = n_points
        self._original_radial_grid = radial_grid
        self._original_k_grid = k_grid
        self._bessel_type = bessel_type

        # Calculate N+1 roots must be calculated before max_radius can be derived from k_grid
        usage = f"Invalid transform type: '{bessel_type}'. Expected 'polar' or 'spherical'."
        alpha = None
        if bessel_type == "polar":
            alpha = scipy_bessel.jn_zeros(self._order, self._n_points + 1)
        elif bessel_type == "spherical":
            alpha = _Jn_spherical_zeros(self._order, self._n_points + 1)
        else:
            raise ValueError(usage)

        self._alpha = alpha[0:-1]
        self._alpha_n1 = alpha[-1]

        if k_grid is not None:
            v_max = np.max(k_grid) / (2 * np.pi)
            max_radius = self._alpha_n1 / (2 * np.pi * v_max)
        self._max_radius: float = max_radius  # pyright: ignore[reportAttributeAccessIssue]

        # Calculate co-ordinate vectors
        self._r = self._alpha * self._max_radius / self._alpha_n1
        self._v = self._alpha / (2 * np.pi * self._max_radius)
        self._kr = 2 * np.pi * self._v
        self._v_max = self._alpha_n1 / (2 * np.pi * self._max_radius)
        self._S = self._alpha_n1

        # Calculate hankel matrix and vectors
        if bessel_type == "polar":
            jp = scipy_bessel.jv(order, (self._alpha[:, np.newaxis] @ self._alpha[np.newaxis, :]) / self._S)
            jp1 = np.abs(scipy_bessel.jv(order + 1, self._alpha))
            self._T = 2 * jp / ((jp1[:, np.newaxis] @ jp1[np.newaxis, :]) * self._S)
            self._JR = jp1 / self._max_radius
            self._JV = jp1 / self._v_max
        elif bessel_type == "spherical":
            jp = scipy_bessel.spherical_jn(order, (self._alpha[:, np.newaxis] @ self._alpha[np.newaxis, :]) / self._S)
            jp1 = np.abs(scipy_bessel.spherical_jn(order + 1, self._alpha))
            self._T = np.sqrt(2 * np.pi / self._S**3) * jp / (jp1[:, np.newaxis] @ jp1[np.newaxis, :])
            self._JR = jp1 / self._max_radius
            self._JV = jp1 * (self._max_radius**2) * np.sqrt(np.pi / (2 * self._S**3))
        else:
            raise ValueError(usage)  # pragma: no cover - backup case: cannot currently be reached

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
    def bessel_type(self) -> str:
        return self._bessel_type

    @property
    def r(self) -> np.ndarray:
        return self._r

    @property
    def v(self) -> np.ndarray:
        return self._v

    @property
    def kr(self) -> np.ndarray:
        return self._kr

    @property
    def T(self) -> np.ndarray:
        return self._T

    @property
    def original_radial_grid(self) -> np.ndarray:
        """Return the original radial grid used to construct the object, or raise a :class:`ValueError`
        if the constructor was not called specifying a ``radial_grid`` parameter.

        :return: The original radial grid used to construct the object.
        :rtype: :class:`numpy.ndarray`
        """
        if self._original_radial_grid is None:
            raise ValueError(
                "Attempted to access original_radial_grid on HankelTransform "
                "object that was not constructed with a radial_grid"
            )
        return self._original_radial_grid

    @property
    def original_k_grid(self) -> np.ndarray:
        """Return the original k grid used to construct the object, or raise a :class:`ValueError`
        if the constructor was not called specifying a ``k_grid`` parameter.

        :return: The original k grid used to construct the object.
        :rtype: :class:`numpy.ndarray`
        """
        if self._original_k_grid is None:
            raise ValueError(
                "Attempted to access original_k_grid on HankelTransform object that was not constructed with a k_grid"
            )
        return self._original_k_grid

    def to_transform_r(self, function: np.ndarray, axis: int = 0) -> np.ndarray:
        """Interpolate a function, assumed to have been given at the original radial
        grid points used to construct the ``HankelTransform`` object onto the grid required
        of use in the QDHT algorithm.

        If the ``HankelTransform`` object was constructed with a (say) equally-spaced
        grid in radius, then it needs the function to transform to be sampled at a specific
        grid before it can be passed to :meth:`.HankelTransform.qdht`. This method provides
        a convenient way of doing this.

        :parameter function: The function to be interpolated. Specified at the radial points
            :attr:`~.HankelTransform.original_radial_grid`.
        :type function: :class:`numpy.ndarray`
        :parameter axis: Axis representing the radial dependence of `function`.
        :type axis: :class:`int`

        :return: Interpolated function suitable to passing to
            :meth:`HankelTransform.qdht` (sampled at ``self.r``)
        :rtype: :class:`numpy.ndarray`
        """
        if function.ndim == 1:
            axis = 0
        return _spline(self.original_radial_grid, function, self.r, axis)

    def to_original_r(self, function: np.ndarray, axis: int = 0) -> np.ndarray:
        """Interpolate a function, assumed to have been given at the Hankel transform points
        ``self.r`` (as returned by :meth:`HankelTransform.iqdht`) back onto the original grid
        used to construct the ``HankelTransform`` object.

        If the ``HankelTransform`` object was constructed with a (say) equally-spaced
        grid in radius, it may be useful to convert back to this grid after a IQDHT.
        This method provides a convenient way of doing this.

        :parameter function: The function to be interpolated. Specified at the radial points
            ``self.r``.
        :type function: :class:`numpy.ndarray`
        :parameter axis: Axis representing the radial dependence of `function`.
        :type axis: :class:`int`

        :return: Interpolated function at the points held in :attr:`~.HankelTransform.original_radial_grid`.
        :rtype: :class:`numpy.ndarray`
        """
        if function.ndim == 1:
            axis = 0
        return _spline(self.r, function, self.original_radial_grid, axis)

    def to_transform_k(self, function: np.ndarray, axis: int = 0) -> np.ndarray:
        """Interpolate a function, assumed to have been given at the original k
        grid points used to construct the ``HankelTransform`` object onto the grid required
        of use in the IQDHT algorithm.

        If the ``HankelTransform`` object was constructed with a (say) equally-spaced
        grid in :math:`k`, then it needs the function to transform to be sampled at a specific
        grid before it can be passed to :meth:`.HankelTransform.iqdht`. This method provides
        a convenient way of doing this.

        :parameter function: The function to be interpolated. Specified at the k points
            :attr:`~.HankelTransform.original_k_grid`.
        :type function: :class:`numpy.ndarray`
        :parameter axis: Axis representing the frequency dependence of `function`.
        :type axis: :class:`int`

        :return: Interpolated function suitable to passing to
            :meth:`HankelTransform.qdht` (sampled at ``self.kr``)
        :rtype: :class:`numpy.ndarray`
        """
        if function.ndim == 1:
            axis = 0
        return _spline(self.original_k_grid, function, self.kr, axis)

    def to_original_k(self, function: np.ndarray, axis: int = 0) -> np.ndarray:
        """Interpolate a function, assumed to have been given at the Hankel transform points
        ``self.k`` (as returned by :meth:`HankelTransform.qdht`) back onto the original grid
        used to construct the ``HankelTransform`` object.

        If the ``HankelTransform`` object was constructed with a (say) equally-spaced
        grid in :math:`k`, it may be useful to convert back to this grid after a QDHT.
        This method provides a convenient way of doing this.

        :parameter function: The function to be interpolated. Specified at the radial points
            ``self.k``.
        :type function: :class:`numpy.ndarray`
        :parameter axis: Axis representing the frequency dependence of `function`.
        :type axis: :class:`int`

        :return: Interpolated function at the points held in :attr:`~.HankelTransform.original_k_grid`.
        :rtype: :class:`numpy.ndarray`
        """
        if function.ndim == 1:
            axis = 0
        return _spline(self.kr, function, self.original_k_grid, axis)

    def qdht(self, fr: np.ndarray, axis: int = -2) -> np.ndarray:
        r"""QDHT: Quasi Discrete Hankel Transform

        Performs the Hankel transform of a function of radius, returning
        a function of frequency.

        .. math::
            f_v(v) = \mathcal{H}\{f_r(r)\}

        .. warning::
            The input function must be sampled at the points ``self.r``, and the output
            will be sampled at the points ``self.v`` (or equivalently ``self.kr``)

        :parameter fr: Function in real space as a function of radius (sampled at ``self.r``)
        :type fr: :class:`numpy.ndarray`
        :parameter axis: Axis over which to compute the Hankel transform.
        :type axis: :class:`int`

        :return: Function in frequency space (sampled at ``self.v``)
        :rtype: :class:`numpy.ndarray`
        """
        if (fr.ndim == 1) or (axis == -2):
            jr, jv = self._get_scaling_factors(fr)

            fv = jv * np.matmul(self.T, (fr / jr))
            return fv
        else:
            _fr = np.swapaxes(fr, axis, -2)
            jr, jv = self._get_scaling_factors(_fr)
            fv = jv * np.matmul(self.T, (_fr / jr))
            return np.swapaxes(fv, axis, -2)

    def iqdht(self, fv: np.ndarray, axis: int = -2) -> np.ndarray:
        r"""IQDHT: Inverse Quasi Discrete Hankel Transform

        Performs the inverse Hankel transform of a function of frequency, returning
        a function of radius.

        .. math::
            f_r(r) = \mathcal{H}^{-1}\{f_v(v)\}

        :parameter fv: Function in frequency space (sampled at self.v)
        :type fv: :class:`numpy.ndarray`
        :parameter axis: Axis over which to compute the Hankel transform.
        :type axis: :class:`int`

        :return: Radial function (sampled at self.r) = IHT(fv)
        :rtype: :class:`numpy.ndarray`
        """
        if (fv.ndim == 1) or (axis == -2):
            jr, jv = self._get_scaling_factors(fv)
            fr = jr * np.matmul(self.T, (fv / jv))
            return fr
        else:
            _fv = np.swapaxes(fv, axis, -2)
            jr, jv = self._get_scaling_factors(_fv)
            fr = jr * np.matmul(self.T, (_fv / jv))
            return np.swapaxes(fr, axis, -2)

    def _get_scaling_factors(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if f.ndim > 1:
            n2 = list(f.shape)
            n2[-2] = 1
            _shape = np.ones_like(n2)
            _shape[-2] = len(self._JR)
            jr = np.reshape(self._JR, _shape) * np.ones(n2)
            jv = np.reshape(self._JV, _shape) * np.ones(n2)
        else:
            jr = self._JR
            jv = self._JV
        return jr, jv

    def _approx_equal(self, other: "HankelTransform"):
        for attr in ["order", "n_points", "max_radius", "bessel_type", "r", "v", "kr", "T"]:
            try:
                val1 = getattr(self, attr)
                val2 = getattr(other, attr)
            except AttributeError:
                return False
            if isinstance(val1, (int, str)):
                if val1 != val2:
                    return False
            elif isinstance(val1, float):
                if not np.isclose(val1, val2):
                    return False
            else:
                if not np.allclose(val1, val2):
                    return False
        return True


def _spline(x0: np.ndarray, y0: np.ndarray, x: np.ndarray, axis: int) -> np.ndarray:
    f = interpolate.interp1d(x0, y0, axis=axis, fill_value="extrapolate", kind="cubic")
    return f(x)


# adapted from SciPy Cookbook https://scipy-cookbook.readthedocs.io/items/SphericalBesselZeros.html
def _Jn_spherical_zeros(n, nt):
    zerosj = np.zeros((n + 1, nt), dtype=float)
    zerosj[0] = np.arange(1, nt + 1) * np.pi
    if n == 0:
        return zerosj[0]
    points = np.arange(1, nt + n + 1) * np.pi
    racines = np.zeros(nt + n, dtype=float)

    def Jn(r, n):
        return scipy_bessel.spherical_jn(n, r)

    for i in range(1, n + 1):
        for j in range(nt + n - i):
            foo = brentq(Jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:nt] = racines[:nt]
    return zerosj[-1]
