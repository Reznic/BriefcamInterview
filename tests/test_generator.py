import pytest
import numpy as np

from shapes import Line2D
from generator import Generator, GroundTruthSamplesGenerator


def test_ground_truth_samples_of_line2d():
    """Test that ground truth generator, can generate sample points on a 2d line.

    * Initialize a Line2D shape, representing the line function f(x) = 3x + 4
    * Initialize a GroundTruthSamplesGenerator instance
    * Invoke the generator on the 2d-line shape
    * Assert the generator returned a 2-dimensional point sample
    * Assert the returned point sample is on the line equation
    """
    line = Line2D()
    line_func = lambda x: 3*x + 4
    line.p1 = np.array([1, line_func(1)])
    line.p2 = np.array([2, line_func(2)])

    ground_truth_generator = GroundTruthSamplesGenerator()
    gt_sample = line.accept(ground_truth_generator)
    assert len(gt_sample) == 2, \
        "Ground truth generator returned sample of wrong dimension"
    sample_x = gt_sample[0]
    sample_y = gt_sample[1]

    assert sample_y == line_func(sample_x), \
        "Ground truth generator returned a point not on the line function"

