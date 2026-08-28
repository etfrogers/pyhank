import numpy as np
import pytest


@pytest.fixture()
def radius() -> np.ndarray:
    return np.linspace(0, 3, 1024)
