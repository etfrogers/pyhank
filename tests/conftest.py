import numpy as np
import pytest

from pyhank import (
    _pure_python,
)


@pytest.fixture()
def radius() -> np.ndarray:
    return np.linspace(0, 3, 1024)


# Build our list of backends to test
BACKENDS = [pytest.param(_pure_python, id="pure_python")]

# Try to import the Rust extension and add it to the test matrix if available
try:
    from pyhank import _pyhank_native  # type: ignore

    BACKENDS.append(pytest.param(_pyhank_native, id="rust_native"))
except ImportError:
    # Optional: You can emit a warning or just silently skip
    pass


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param
