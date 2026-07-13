use approx::relative_eq;
use hankrs::{
    one_shot::{iqdht as rust_iqdht, qdht as rust_qdht},
    HankelTransform, InterpError,
};
use numpy::{
    ndarray::Axis, IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArrayDyn, PyUntypedArrayMethods, ToPyArray,
};
use pyo3::{
    exceptions::{PyNotImplementedError, PyValueError},
    prelude::*,
};

#[pyclass(name = "HankelTransform")]
pub struct PyHankelTransform {
    inner: HankelTransform,
}

#[pyfunction]
fn is_release_build() -> bool {
    // cfg!(debug_assertions) is true in dev mode, false in release mode
    !cfg!(debug_assertions)
}

#[pyfunction]
#[pyo3(signature = (r, f, order=0, axis=None, bessel_type="polar"))]
fn qdht<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    f: PyReadonlyArrayDyn<'py, f64>,
    order: i32,
    axis: Option<usize>,
    bessel_type: &str,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArrayDyn<f64>>)> {
    if bessel_type != "polar" {
        return Err(PyNotImplementedError::new_err(
            "Only polar bessel type is implemented",
        ));
    }
    let r = r.to_owned_array();
    let f_view = f.as_array();
    let axis = default_axis(axis, &f);
    let result = py.detach(|| rust_qdht(r, &f_view, order, axis));
    let kr = result.0.into_pyarray(py);
    let ht = result.1.into_pyarray(py);
    Ok((kr, ht))
}

#[pyfunction]
#[pyo3(signature = (kr, f, order=0, axis=None, bessel_type="polar"))]
fn iqdht<'py>(
    py: Python<'py>,
    kr: PyReadonlyArray1<'py, f64>,
    f: PyReadonlyArrayDyn<'py, f64>,
    order: i32,
    axis: Option<usize>,
    bessel_type: &str,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArrayDyn<f64>>)> {
    if bessel_type != "polar" {
        return Err(PyNotImplementedError::new_err(
            "Only polar bessel type is implemented",
        ));
    }
    let kr = kr.to_owned_array();
    let f_view = &f.as_array();
    let axis = default_axis(axis, &f);
    let result = py.detach(|| rust_iqdht(kr, f_view, order, axis));
    let r = result.0.into_pyarray(py);
    let ht = result.1.into_pyarray(py);
    Ok((r, ht))
}

fn default_axis(axis: Option<usize>, data: &PyReadonlyArrayDyn<f64>) -> Axis {
    Axis(axis.unwrap_or(data.ndim().saturating_sub(2)))
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
        if bessel_type != "polar" {
            return Err(PyNotImplementedError::new_err(
                "Only polar bessel type is implemented",
            ));
        }
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
    fn kr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let v = self.inner.kr();
        // Note this copies the data into a new PyArray1
        v.to_pyarray(py)
    }

    #[getter]
    fn order<'py>(&self) -> i32 {
        let order = self.inner.order();
        order
    }

    #[getter]
    fn n_points<'py>(&self) -> usize {
        let n = self.inner.n_points();
        n
    }

    #[getter]
    fn original_radial_grid<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let v = self.inner.original_radial_grid();
        // Note this copies the data into a new PyArray1
        match v {
            Some(val) => Ok(val.to_pyarray(py)),
            None => Err(PyValueError::new_err(
                "Attempted to access original_radial_grid on HankelTransform \
                object that was not constructed with a radial_grid",
            )),
        }
    }

    #[getter]
    fn original_k_grid<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let v = self.inner.original_k_grid();
        // Note this copies the data into a new PyArray1
        match v {
            Some(val) => Ok(val.to_pyarray(py)),
            None => Err(PyValueError::new_err(
                "Attempted to access original_k_grid on HankelTransform \
                object that was not constructed with a k_grid",
            )),
        }
    }

    #[pyo3(signature = (data, axis=None))]
    fn qdht<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArrayDyn<'py, f64>,
        axis: Option<usize>,
    ) -> Bound<'py, PyArrayDyn<f64>> {
        // Call the underlying pure Rust method
        let axis = default_axis(axis, &data);
        let data_view = data.as_array();
        let result = py.detach(|| self.inner.qdht(&data_view, axis));
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
    ) -> Result<Bound<'py, PyArrayDyn<f64>>, PyHankError> {
        let fun_view = function.as_array();
        let result = py.detach(|| self.inner.to_transform_r_nd(&fun_view, Axis(axis)));
        Ok(result?.into_pyarray(py))
    }

    #[pyo3(signature = (function, axis=0))]
    fn to_original_r<'py>(
        &self,
        py: Python<'py>,
        function: PyReadonlyArrayDyn<'py, f64>,
        axis: usize,
    ) -> Result<Bound<'py, PyArrayDyn<f64>>, PyHankError> {
        let fun_view = function.as_array();
        let result = py.detach(|| self.inner.to_original_r_nd(&fun_view, Axis(axis)));
        Ok(result?.into_pyarray(py))
    }

    #[pyo3(signature = (function, axis=0))]
    fn to_transform_k<'py>(
        &self,
        py: Python<'py>,
        function: PyReadonlyArrayDyn<'py, f64>,
        axis: usize,
    ) -> Result<Bound<'py, PyArrayDyn<f64>>, PyHankError> {
        let fun_view = function.as_array();
        let result = py.detach(|| self.inner.to_transform_k_nd(&fun_view, Axis(axis)));
        Ok(result?.into_pyarray(py))
    }

    #[pyo3(signature = (function, axis=0))]
    fn to_original_k<'py>(
        &self,
        py: Python<'py>,
        function: PyReadonlyArrayDyn<'py, f64>,
        axis: usize,
    ) -> Result<Bound<'py, PyArrayDyn<f64>>, PyHankError> {
        let fun_view = function.as_array();
        let result = py.detach(|| self.inner.to_original_k_nd(&fun_view, Axis(axis)));
        Ok(result?.into_pyarray(py))
    }

    fn _approx_equal(&self, other: &Self) -> bool {
        relative_eq!(self.inner, other.inner)
    }
}

#[pymodule]
fn _pyhank_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(is_release_build, m)?)?;
    m.add_function(wrap_pyfunction!(qdht, m)?)?;
    m.add_function(wrap_pyfunction!(iqdht, m)?)?;
    m.add_class::<PyHankelTransform>()?;
    Ok(())
}

pub enum PyHankError {
    Interp(InterpError),
}

impl From<InterpError> for PyHankError {
    fn from(err: InterpError) -> Self {
        PyHankError::Interp(err)
    }
}

impl From<PyHankError> for PyErr {
    fn from(err: PyHankError) -> Self {
        match err {
            PyHankError::Interp(e) => PyValueError::new_err(e.to_string()),
            // PyHankError::Core(e) => ... map your core errors here ...
        }
    }
}
