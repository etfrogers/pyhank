import numpy as np
import pytest

import pyhank._pure_python as pure_python

native = pytest.importorskip("pyhank._pyhank_native")  # type: ignore


@pytest.mark.parametrize("order", [1, 2, 5])
@pytest.mark.parametrize(
    "inputs",
    [{"radial_grid": np.linspace(0, 3, 50)}, {"n_points": 50, "max_radius": 100}, {"k_grid": np.linspace(0, 10, 50)}],
)
@pytest.mark.parametrize("bessel_type", ["polar", "spherical"])
def test_native_vs_pure_python(order, inputs, bessel_type):
    r = np.linspace(0, 3, 50)
    f = np.sin(r)

    py_transformer = pure_python.HankelTransform(order=order, bessel_type=bessel_type, **inputs)
    native_transformer = native.HankelTransform(order=order, bessel_type=bessel_type, **inputs)

    assert py_transformer.bessel_type == native_transformer.bessel_type == bessel_type
    assert np.isclose(py_transformer.max_radius, native_transformer.max_radius)
    assert py_transformer.order == native_transformer.order
    assert py_transformer.n_points == native_transformer.n_points
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


@pytest.mark.parametrize(
    "prop",
    ["r", "v", "kr", "T", "order", "n_points", "max_radius", "bessel_type"],
)
def test_properties_readonly(prop):
    py_t = pure_python.HankelTransform(0, max_radius=1.0, n_points=32)
    native_t = native.HankelTransform(0, max_radius=1.0, n_points=32)

    with pytest.raises((AttributeError, TypeError)):
        setattr(py_t, prop, 42)

    with pytest.raises((AttributeError, TypeError)):
        setattr(native_t, prop, 42)

