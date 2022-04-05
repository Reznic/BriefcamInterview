import numpy as np
from unittest.mock import MagicMock
import pytest

from estimator import Estimator, DimensionNotSupportedError
from samples_generator import SamplesSuit, ShapeSamples


def test_unsupported_dimension():
    very_high_dimension = 10
    samples = np.zeros((100, very_high_dimension))
    algo = MagicMock()
    estimator = Estimator(algo)
    with pytest.raises(DimensionNotSupportedError):
        estimator.estimate_shape(samples, 0)

