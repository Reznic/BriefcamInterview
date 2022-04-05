import numpy as np

from shapes import Line2D
from generator import Generator, GroundTruthSamplesGenerator
from configs import Configurations


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

        assert sample_y == line_func(sample_x), \
            f"Ground truth generator returned a point " \
            f"{sample} not on the line function"


def test_generator():
    config = Configurations()
    config.randomness = 0
    config.shapes = {"Line2D": 2}
    config.num_points = 10

    generator = Generator(config)
    suit = generator.generate_samples_suit()

    expected_total_shapes = sum(config.shapes.values())
    assert suit.len == expected_total_shapes, \
        f"Generator generated {suit.len} " \
        f"shapes instead of {expected_total_shapes}"

    for shape_samples in suit:
        assert len(shape_samples.samples) == config.num_points, \
            "Wrong number of generated samples"

