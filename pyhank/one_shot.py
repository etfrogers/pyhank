from typing import Tuple

import numpy as np

from pyhank import HankelTransform


def qdht(r: np.ndarray, f: np.ndarray, order: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Perform a quasi-discrete Hankel transform of the function ``f`` (sampled at points
    ``r``) and return the transformed function and its sample points in :math:`k`-space.

    If you requires the transform on a frequency axis (as opposed to the :math:`k`-axis), the
    frequency axis :math:`v` can be calculated using :math:`v = \\frac{k}{2\\pi}`.

    .. warning::
        This method is a convenience wrapper for :meth:`.HankelTransform.qdht`, but incurs a
        significant overhead in calculating the :class:`.HankelTransform` object. If you
        are performing multiple transforms on the same grid, it will be much quicker to
        construct a single :class:`.HankelTransform` object and call
        :meth:`.HankelTransform.qdht` multiple times.

    :param r: The radial coordinates at which the function is sampled
    :type r: :class:`numpy.ndarray`
    :param f: The value of the function to be transformed.
    :type f: :class:`numpy.ndarray`
    :param order: The order of the Hankel Transform to perform. Defaults to 0.
    :return: A tuple containing the k coordinates of the transformed function and its values
    :rtype: (:class:`numpy.ndarray`, :class:`numpy.ndarray`)
    """
    transformer = HankelTransform(order=order, radial_grid=r)
    f_transform = transformer.to_transform_r(f)
    ht = transformer.qdht(f_transform)
    return transformer.kr, ht


def iqdht(k: np.ndarray, f: np.ndarray, order: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Perform a inverse quasi-discrete Hankel transform of the function ``f`` (sampled at points
    ``k``) and return the transformed function and its sample points in radial space.

    If you have the transform on a frequency axis (as opposed to a :math:`k`-axis), the
    :math:`k`-axis can be calculated using :math:`k = 2\\pi{}f`.

    .. warning::
        This method is a convenience wrapper for :meth:`.HankelTransform.iqdht`, but incurs a
        significant overhead in calculating the :class:`.HankelTransform` object. If you
        are performing multiple transforms on the same grid, it will be much quicker to
        construct a single :class:`.HankelTransform` object and call
        :meth:`.HankelTransform.iqdht` multiple times.

    :param k: The :math:`k` coordinates at which the function is sampled
    :type k: :class:`numpy.ndarray`
    :param f: The value of the function to be transformed.
    :type f: :class:`numpy.ndarray`
    :param order: The order of the Hankel Transform to perform. Defaults to 0.
    :return: A tuple containing the radial coordinates of the transformed function and its values
    :rtype: (:class:`numpy.ndarray`, :class:`numpy.ndarray`)
    """
    transformer = HankelTransform(order=order, k_grid=k)
    f_transform = transformer.to_transform_k(f)
    ht = transformer.iqdht(f_transform)
    return transformer.r, ht
