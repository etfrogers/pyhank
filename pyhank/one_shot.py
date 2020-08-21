from typing import Tuple

import numpy as np

from pyhank import HankelTransform


def qdht(r: np.ndarray, f: np.ndarray, order: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    transformer = HankelTransform(order=order, radial_grid=r)
    f_transform = transformer.to_transform_r(f)
    ht = transformer.qdht(f_transform)
    return transformer.kr, ht


def iqdht(k: np.ndarray, f: np.ndarray, order: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    transformer = HankelTransform(order=order, k_grid=k)
    f_transform = transformer.to_transform_k(f)
    ht = transformer.iqdht(f_transform)
    return transformer.r, ht
