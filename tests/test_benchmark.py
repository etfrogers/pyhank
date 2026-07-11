import numpy as np
import pytest

from pyhank import _pure_python, _pyhank_native  # type: ignore

from .test_hankel import generalised_jinc


# Test a tiny array (shows Python/Rust boundary overhead)
# Test a large array (shows pure math performance)
@pytest.mark.parametrize("size", [10, 256, 512])
@pytest.mark.parametrize("backend", [pytest.param(_pure_python, id="python"), pytest.param(_pyhank_native, id="rust")])
def test_qdht_performance(benchmark, size, backend):
    # Generate random test data once per test
    # data = np.random.rand(size)
    def process(size):
        transformer = backend.HankelTransform(order=1, radial_grid=np.linspace(0, 3, size))
        f = generalised_jinc(transformer.r, 1.0, transformer.order)
        return transformer.qdht(f)

    # benchmark() runs the target function repeatedly to get accurate timings
    result = benchmark(process, size)

    assert result is not None
