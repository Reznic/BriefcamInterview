"""Unit tests for generator module."""
import numpy as np
from pytest import approx

from shapes import Line2D
from configs import Configurations
from samples import Generator, GroundTruthSamplesGenerator


# Make tests deterministic
np.random.seed(3246)

NUMERIC_PERCISION = 0.00001


def test_ground_truth_samples_of_line2d():
    """Test that ground truth generator can generate sample points on a 2d line.

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

    num_of_sampels = 5

    ground_truth_generator = GroundTruthSamplesGenerator()
    gt_samples = line.accept(ground_truth_generator,
                             num_of_samples=num_of_sampels,
                             samples_range=100)

    assert len(gt_samples) == num_of_sampels, \
        "Ground truth generator generated wrong number of samples"
    for sample in gt_samples:
        assert len(sample) == 2, \
            "Ground truth generator returned sample of wrong dimension"
        sample_x = sample[0]
        sample_y = sample[1]

        assert line_func(sample_x) == approx(sample_y, NUMERIC_PERCISION), \
            f"Ground truth generator returned a point " \
            f"{sample} not on the line function"


def test_generator():
    """Test that Generator, generates expected samples.

    * Initialize a Configuration for generation of 2 lines, 10 samples each.
    * Initialize a Generator instance, with the config, and generate sample-suit
    * Assert the correct total number of shapes were generated in the suit
    * For each shape samples in the suit:
        - Assert correct type of shape generated
        - Assert sampels of correct dimension and size were created
        - Assert all samples did not exceed the expected max range,
          plus a small margin (considering the case of added noise)
    """
    config = Configurations()
    config.randomness = 0.1
    config.shapes = {"Line2D": 2}
    config.num_points = 10

    generator = Generator(config)
    suit = generator.generate_samples_suit()

    expected_total_shapes = sum(config.shapes.values())
    assert suit.len == expected_total_shapes, \
        f"Generator generated {suit.len} " \
        f"shapes instead of {expected_total_shapes}"

    expected_samples_range = max(generator.INLIERS_RANGE,
                                 generator.OUTLIERS_RANGE) + 10

    for shape_samples in suit:
        assert shape_samples.shape.__class__ == Line2D, \
            "Generator generated shape of wrong type"

        expected_samples_dimension = (config.num_points,
                                      shape_samples.shape.DIMENSION)
        assert shape_samples.samples.shape == expected_samples_dimension, \
            "Samples with wrong dimension generated"

        samples_range = np.max(np.abs(shape_samples.samples))
        assert samples_range <= expected_samples_range, \
            "Generated samples exceeded the expected range"

