import numpy as np
from pytest import approx

from shapes import Line2D
from ransac import Regressor


def test_regression_line2d():
    line = Line2D()
    points = np.array([[0.01, 0], [1, 1.01], [2, 1.99]])
    expected_slope = 1
    expected_intercept = 0

    regressor = Regressor()
    line.accept(regressor, points)
    slope = line.get_slope()
    assert slope == approx(expected_slope, abs=0.1)
    intercept = line.y1 - line.x1 * slope
    assert intercept == approx(expected_intercept, abs=0.1)



