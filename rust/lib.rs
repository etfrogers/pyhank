use hankrs::HankelTransform;
use numpy::{
    ndarray::Axis, IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1,
    PyReadonlyArrayDyn, PyUntypedArrayMethods, ToPyArray,
};
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyclass(name = "HankelTransform")]
pub struct PyHankelTransform {
    inner: HankelTransform,
}

#[pyfunction]
fn is_release_build() -> bool {
    // cfg!(debug_assertions) is true in dev mode, false in release mode
    !cfg!(debug_assertions)
}

// 2. Expose the methods to Python
#[pymethods]
impl PyHankelTransform {
    #[new]
    #[pyo3(signature = (order, max_radius=None, n_points=None, radial_grid=None, k_grid=None, bessel_type="polar"))]
    fn new<'py>(
        order: i32,
        max_radius: Option<f64>,
        n_points: Option<usize>,
        radial_grid: Option<PyReadonlyArray1<'py, f64>>,
        k_grid: Option<PyReadonlyArray1<'py, f64>>,
        bessel_type: &str,
    ) -> PyResult<Self> {
        let ht = match (max_radius, n_points, radial_grid, k_grid) {
            (None, None, Some(radial_grid), None) => {
                let radial_grid = radial_grid.as_array().to_owned();
                HankelTransform::new_from_r_grid(order, radial_grid)
            }
            (Some(max_radius), Some(n_points), None, None) => {
                HankelTransform::new(order, max_radius, n_points)
            }
            (None, None, None, Some(k_grid)) => {
                let k_grid = k_grid.as_array().to_owned();
                HankelTransform::new_from_k_grid(order, k_grid)
            }
            _ => {
                return Err(PyValueError::new_err(
                    "Either radial_grid or k_grid or both max_radius and n_points must be supplied",
                ));
            }
        };
        Ok(PyHankelTransform { inner: ht })
    }

    #[getter]
    #[allow(non_snake_case)]
    fn T<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let matrix = self.inner.transform_matrix();
        // Note this copies the data into a new PyArray2
        matrix.to_pyarray(py)
    }

    #[getter]
    fn r<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let r = self.inner.radius();
        // Note this copies the data into a new PyArray1
        r.to_pyarray(py)
    }

    #[getter]
    fn v<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let v = self.inner.frequency();
        // Note this copies the data into a new PyArray1
        v.to_pyarray(py)
    }

    #[getter]
    fn order<'py>(&self) -> i32 {
        let order = self.inner.order();
        order
    }

    #[pyo3(signature = (data, axis=None))]
    fn qdht<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArrayDyn<'py, f64>,
        axis: Option<usize>,
    ) -> Bound<'py, PyArrayDyn<f64>> {
        // Call the underlying pure Rust method
        let axis = axis.unwrap_or(data.ndim().saturating_sub(2));
        let data_view = data.as_array();
        let result = py.detach(|| self.inner.qdht(&data_view, Axis(axis)));
        result.into_pyarray(py)
    }

    #[pyo3(signature = (data, axis=None))]
    fn iqdht<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArrayDyn<'py, f64>,
        axis: Option<usize>,
    ) -> Bound<'py, PyArrayDyn<f64>> {
        // Call the underlying pure Rust method
        let axis = axis.unwrap_or(data.ndim().saturating_sub(2));
        let data_view = data.as_array();
        let result = py.detach(|| self.inner.iqdht(&data_view, Axis(axis)));
        result.into_pyarray(py)
    }

    #[pyo3(signature = (function, axis=0))]
    fn to_transform_r<'py>(
        &self,
        py: Python<'py>,
        function: PyReadonlyArrayDyn<'py, f64>,
        axis: usize,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let fun_view = function.as_array();
        // let result = py.allow_threads(|| self.inner.to_transform_r_nd(&fun_view, Axis(axis)));
        let result = self.inner.to_transform_r_nd(&fun_view, Axis(axis));
        let result = result.map_err(|string| PyValueError::new_err(string.to_owned()))?;
        Ok(result.into_pyarray(py))
    }
}

#[pymodule]
fn _pyhank_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_release_build, m)?)?;
    m.add_class::<PyHankelTransform>()?;
    Ok(())
}
