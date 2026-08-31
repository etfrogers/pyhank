use approx::relative_eq;
use hankrs::{
    one_shot::{
        iqdht as rust_iqdht, iqdht_spherical as rust_iqdht_spherical, qdht as rust_qdht,
        qdht_spherical as rust_qdht_spherical,
    },
    HankelError, HankelScalar, HankelTransform,
};
use num::complex::Complex64;
use numpy::{
    ndarray::Axis, IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArrayDyn, PyUntypedArray, PyUntypedArrayMethods, ToPyArray,
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

macro_rules! dynamic_dispatch_to {
    ($fun:ident, $py:ident, $kr:ident, $f:ident, $order:expr, $axis:expr, $bessel_type:expr) => {
        match $f {
            PyDataArray::Float(f_array) => {
                $fun::<f64>($py, $kr, f_array, $order, $axis, $bessel_type)
            }
            PyDataArray::Complex(f_array) => {
                $fun::<Complex64>($py, $kr, f_array, $order, $axis, $bessel_type)
            }
        }
    };
    ($fun:ident, $py:ident, $transformer:ident, $data:ident, $axis:expr) => {
        match $data {
            PyDataArray::Float(f_array) => $fun::<f64>($transformer, $py, f_array, $axis),
            PyDataArray::Complex(f_array) => $fun::<Complex64>($transformer, $py, f_array, $axis),
        }
    };
}

#[pyfunction]
#[pyo3(signature = (r, f, order=0, axis=None, bessel_type=TransformType::Polar))]
fn qdht<'py>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    f: PyDataArray<'py>,
    order: i32,
    axis: Option<usize>,
    bessel_type: TransformType,
) -> Result<(Bound<'py, PyArray1<f64>>, Bound<'py, PyUntypedArray>), PyHankError> {
    dynamic_dispatch_to!(internal_qdht, py, r, f, order, axis, bessel_type)
}

fn internal_qdht<'py, T: HankelScalarLocal>(
    py: Python<'py>,
    r: PyReadonlyArray1<'py, f64>,
    f: PyReadonlyArrayDyn<'py, T>,
    order: i32,
    axis: Option<usize>,
    bessel_type: TransformType,
) -> Result<(Bound<'py, PyArray1<f64>>, Bound<'py, PyUntypedArray>), PyHankError> {
    let r = r.to_owned_array();
    let f_view = &f.as_array();
    let axis = default_axis(axis, &f);
    let result = py.detach(|| match bessel_type {
        TransformType::Polar => rust_qdht(r, f_view, order, axis),
        TransformType::Spherical => rust_qdht_spherical(r, f_view, order, axis),
    })?;
    let kr = result.0.into_pyarray(py);
    let ht = result.1.into_pyarray(py);
    Ok((kr, ht.as_untyped().clone()))
}

#[pyfunction]
#[pyo3(signature = (kr, f, order=0, axis=None, bessel_type=TransformType::Polar))]
fn iqdht<'py>(
    py: Python<'py>,
    kr: PyReadonlyArray1<'py, f64>,
    f: PyDataArray<'py>,
    order: i32,
    axis: Option<usize>,
    bessel_type: TransformType,
) -> Result<(Bound<'py, PyArray1<f64>>, Bound<'py, PyUntypedArray>), PyHankError> {
    dynamic_dispatch_to!(internal_iqdht, py, kr, f, order, axis, bessel_type)
}

trait HankelScalarLocal: HankelScalar + numpy::Element {}
impl HankelScalarLocal for f64 {}
impl HankelScalarLocal for Complex64 {}

fn internal_iqdht<'py, T: HankelScalarLocal + 'py>(
    py: Python<'py>,
    kr: PyReadonlyArray1<'py, f64>,
    f: PyReadonlyArrayDyn<'py, T>,
    order: i32,
    axis: Option<usize>,
    bessel_type: TransformType,
) -> Result<(Bound<'py, PyArray1<f64>>, Bound<'py, PyUntypedArray>), PyHankError> {
    let kr = kr.to_owned_array();
    let f_view = &f.as_array();
    let axis = default_axis(axis, &f);
    let result = py.detach(|| match bessel_type {
        TransformType::Polar => rust_iqdht(kr, f_view, order, axis),
        TransformType::Spherical => rust_iqdht_spherical(kr, f_view, order, axis),
    })?;
    let r = result.0.into_pyarray(py);
    let ht = result.1.into_pyarray(py);
    Ok((r, ht.as_untyped().clone()))
}

fn default_axis<T: HankelScalarLocal>(axis: Option<usize>, data: &PyReadonlyArrayDyn<T>) -> Axis {
    Axis(axis.unwrap_or(data.ndim().saturating_sub(2)))
}

fn transformer_qdht<'py, T: HankelScalarLocal>(
    transformer: &HankelTransform,
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, T>,
    axis: Option<usize>,
) -> Bound<'py, PyUntypedArray> {
    let axis = default_axis(axis, &data);
    let data_view = data.as_array();
    let result = py.detach(|| transformer.qdht(&data_view, axis));
    result.into_pyarray(py).as_untyped().clone()
}

fn transformer_iqdht<'py, T: HankelScalarLocal>(
    transformer: &HankelTransform,
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, T>,
    axis: Option<usize>,
) -> Bound<'py, PyUntypedArray> {
    let axis = axis.unwrap_or(data.ndim().saturating_sub(2));
    let data_view = data.as_array();
    let result = py.detach(|| transformer.iqdht(&data_view, Axis(axis)));
    result.into_pyarray(py).as_untyped().clone()
}

macro_rules! generic_interpolator {
    ($fun:path, $py:ident, $transformer:expr, $data:ident, $axis:ident) => {
        match $data {
            PyDataArray::Float($data) => {
                let data_view = $data.as_array();
                let result = $py.detach(|| $fun(&$transformer, &data_view, Axis($axis)));
                Ok(result?.into_pyarray($py).as_untyped().clone())
            }
            PyDataArray::Complex($data) => {
                let data_view = $data.as_array();
                let result = $py.detach(|| $fun(&$transformer, &data_view, Axis($axis)));
                Ok(result?.into_pyarray($py).as_untyped().clone())
            }
        }
    };
}

#[pymethods]
impl PyHankelTransform {
    #[new]
    #[pyo3(signature = (order, max_radius=None, n_points=None, radial_grid=None, k_grid=None, bessel_type=TransformType::Polar))]
    fn new<'py>(
        order: i32,
        max_radius: Option<f64>,
        n_points: Option<usize>,
        radial_grid: Option<PyReadonlyArray1<'py, f64>>,
        k_grid: Option<PyReadonlyArray1<'py, f64>>,
        bessel_type: TransformType,
    ) -> Result<Self, PyHankError> {
        let ht = match (max_radius, n_points, radial_grid, k_grid) {
            (None, None, Some(radial_grid), None) => {
                let radial_grid = radial_grid.as_array().to_owned();
                if bessel_type == TransformType::Spherical {
                    HankelTransform::new_spherical_from_r_grid(order, radial_grid)
                } else {
                    HankelTransform::new_from_r_grid(order, radial_grid)
                }
            }
            (Some(max_radius), Some(n_points), None, None) => {
                if bessel_type == TransformType::Spherical {
                    HankelTransform::new_spherical(order, max_radius, n_points)
                } else {
                    HankelTransform::new(order, max_radius, n_points)
                }
            }
            (None, None, None, Some(k_grid)) => {
                let k_grid = k_grid.as_array().to_owned();
                if bessel_type == TransformType::Spherical {
                    HankelTransform::new_spherical_from_k_grid(order, k_grid)
                } else {
                    HankelTransform::new_from_k_grid(order, k_grid)
                }
            }
            _ => {
                return Err(PyHankError::new_err(
                    "Either radial_grid or k_grid or both max_radius and n_points must be supplied"
                        .to_string(),
                ));
            }
        }?;
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
    fn max_radius(&self) -> f64 {
        self.inner.max_radius()
    }

    #[getter]
    fn bessel_type(&self) -> &str {
        self.inner.transform_type().as_str()
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
        data: PyDataArray<'py>,
        axis: Option<usize>,
    ) -> Bound<'py, PyUntypedArray> {
        let transformer = &self.inner;
        dynamic_dispatch_to!(transformer_qdht, py, transformer, data, axis)
    }

    #[pyo3(signature = (data, axis=None))]
    fn iqdht<'py>(
        &self,
        py: Python<'py>,
        data: PyDataArray<'py>,
        axis: Option<usize>,
    ) -> Bound<'py, PyUntypedArray> {
        let transformer = &self.inner;
        dynamic_dispatch_to!(transformer_iqdht, py, transformer, data, axis)
    }

    #[pyo3(signature = (function, axis=0))]
    fn to_transform_r<'py>(
        &self,
        py: Python<'py>,
        function: PyDataArray<'py>,
        axis: usize,
    ) -> Result<Bound<'py, PyUntypedArray>, PyHankError> {
        generic_interpolator!(
            HankelTransform::to_transform_r_nd,
            py,
            &self.inner,
            function,
            axis
        )
    }

    #[pyo3(signature = (function, axis=0))]
    fn to_original_r<'py>(
        &self,
        py: Python<'py>,
        function: PyDataArray<'py>,
        axis: usize,
    ) -> Result<Bound<'py, PyUntypedArray>, PyHankError> {
        generic_interpolator!(
            HankelTransform::to_original_r_nd,
            py,
            &self.inner,
            function,
            axis
        )
    }

    #[pyo3(signature = (function, axis=0))]
    fn to_transform_k<'py>(
        &self,
        py: Python<'py>,
        function: PyDataArray<'py>,
        axis: usize,
    ) -> Result<Bound<'py, PyUntypedArray>, PyHankError> {
        generic_interpolator!(
            HankelTransform::to_transform_k_nd,
            py,
            &self.inner,
            function,
            axis
        )
    }

    #[pyo3(signature = (function, axis=0))]
    fn to_original_k<'py>(
        &self,
        py: Python<'py>,
        function: PyDataArray<'py>,
        axis: usize,
    ) -> Result<Bound<'py, PyUntypedArray>, PyHankError> {
        generic_interpolator!(
            HankelTransform::to_original_k_nd,
            py,
            &self.inner,
            function,
            axis
        )
    }

    fn _approx_equal(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other_native) = other.extract::<pyo3::PyRef<Self>>() {
            Ok(relative_eq!(slf.borrow().inner, other_native.inner))
        } else {
            let res = other.call_method1("_approx_equal", (slf,))?;
            res.extract()
        }
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
    Hankel(HankelError),
    Interface(String),
}

impl From<HankelError> for PyHankError {
    fn from(err: HankelError) -> Self {
        PyHankError::Hankel(err)
    }
}

impl From<PyHankError> for PyErr {
    fn from(err: PyHankError) -> Self {
        match err {
            PyHankError::Hankel(e) => PyValueError::new_err(e.to_string()),
            PyHankError::Interface(e) => PyValueError::new_err(e),
        }
    }
}

impl PyHankError {
    pub fn new_err(msg: String) -> Self {
        PyHankError::Interface(msg)
    }
}

#[derive(FromPyObject)]
pub enum PyDataArray<'py> {
    Float(PyReadonlyArrayDyn<'py, f64>),
    Complex(PyReadonlyArrayDyn<'py, Complex64>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TransformType {
    Polar,
    Spherical,
}

// Teach PyO3 how to extract this enum from a Python object
impl<'a, 'py> FromPyObject<'a, 'py> for TransformType {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let s: &str = ob.extract()?;

        match s.to_lowercase().as_str() {
            "polar" => Ok(TransformType::Polar),
            "spherical" => Ok(TransformType::Spherical),
            // 3. Automatically throw a clean Python ValueError if they misspell it!
            _ => Err(PyValueError::new_err(format!(
                "Invalid transform type: '{s}'. Expected 'polar' or 'spherical'.",
            ))),
        }
    }
}
