import numpy as np
import pyhank._pyhank_native as native  # type: ignore
import pytest

import pyhank._pure_python as pure_python


@pytest.mark.parametrize("order", [1, 2, 5])
@pytest.mark.parametrize(
    "inputs",
    [{"radial_grid": np.linspace(0, 3, 50)}, {"n_points": 50, "max_radius": 100}, {"k_grid": np.linspace(0, 10, 50)}],
)
def test_native_vs_pure_python(order, inputs):
    r = np.linspace(0, 3, 50)
    f = np.sin(r)

    py_transformer = pure_python.HankelTransform(order=order, **inputs)
    native_transformer = native.HankelTransform(order=order, **inputs)
    assert py_transformer._approx_equal(native_transformer)
    assert native_transformer._approx_equal(py_transformer)

    if "radial_grid" in inputs:
        resampled_py = py_transformer.to_transform_r(f)
        resampled_native = native_transformer.to_transform_r(f)
        assert np.allclose(resampled_py, resampled_native)
    elif "k_grid" in inputs:
        resampled_py = py_transformer.to_transform_k(f)
        resampled_native = native_transformer.to_transform_k(f)
        assert np.allclose(resampled_py, resampled_native)
    else:
        resampled_py = f
        resampled_native = f
        assert np.allclose(resampled_py, resampled_native)

    py_transform = py_transformer.qdht(resampled_py)
    native_transform = native_transformer.qdht(resampled_native)
    assert np.allclose(py_transform, native_transform)
