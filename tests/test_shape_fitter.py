from shapes import Circle2D
from ransac import ShapeFitter

import numpy as np


def test_circle2d_fit():
    circle = Circle2D()
    samples = np.array([[1,0], [0,1], [2,1]])
    expected_center = np.array([1, 1])
    expected_radius = 1
    shape_fitter = ShapeFitter()
    status_success = circle.accept(shape_fitter, samples)
    assert status_success, \
        "Shape fitter unexpectedly failed to fit sample circle"

    assert circle.radius == expected_radius, \
        "Shape fitter fitted a circle with wrong radius"

    assert np.allclose(circle.center, expected_center), \
        "Shape fitter fitted a circle with wrong center"




