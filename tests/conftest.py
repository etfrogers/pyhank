import pytest
import numpy as np


@pytest.fixture()
def radius() -> np.ndarray:
    return np.linspace(0, 3, 1024)
