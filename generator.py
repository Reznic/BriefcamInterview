"""Generation of noisy sample data for random shapes .

Usage:
  generator.py <config_path> <output_path> [--debug]
  generator.py (-h | --help)

Options:
  -h --help     Show this screen.
  -d --debug       plot generated shapes and data
"""
from docopt import docopt
from os import path
import numpy as np

from utils import validate_file_path
from shapes import ShapeFactory, ShapeOperation, Line2D, ShapeSamples, SamplesSuit
from configs import Configurations


class Generator:
    INLIERS_RATIO = 0.8
    OUTLIERS_RANGE = 1000
    INLIERS_RANGE = 1000

    def __init__(self, config):
        self.shape_factory = ShapeFactory()
        self.config = config
        self.ground_truth_generator = GroundTruthSamplesGenerator()
        self.inliers_num = round(self.config.num_points * self.INLIERS_RATIO)
        outliers_ratio = 1 - self.INLIERS_RATIO
        self.outliers_num = round(self.config.num_points * outliers_ratio)

    def _get_gaussian_noise(self, dimension, num, stddev):
        return np.random.randn(num, dimension) * stddev

    def _get_inlier_samples(self, shape):
        inlier_samples = shape.accept(self.ground_truth_generator,
                                      num_of_samples=self.inliers_num,
                                      samples_range=self.INLIERS_RANGE)
        noise = self._get_gaussian_noise(shape.DIMENSION, self.inliers_num,
                                         stddev=self.config.randomness)
        noisy_inlier_samples = inlier_samples + noise
        return noisy_inlier_samples

    def _get_outlier_samples(self, dimension):
        size = (self.outliers_num, dimension)
        samples_range = (-self.OUTLIERS_RANGE, self.OUTLIERS_RANGE)
        outlier_samples = np.random.uniform(*samples_range, size)
        return outlier_samples

    def _get_shape_samples(self, shape):
        outlier_samples = self._get_outlier_samples(shape.DIMENSION)
        inlier_samples = self._get_inlier_samples(shape)
        samples = np.concatenate([outlier_samples, inlier_samples])
        return ShapeSamples(shape, samples)

    def generate_samples_suit(self):
        shapes = self.shape_factory.get_random_shapes(self.config.shapes)
        suit = SamplesSuit()
        for shape in shapes:
            samples = self._get_shape_samples(shape)
            suit.add(samples)

        return suit


class GroundTruthSamplesGenerator(ShapeOperation):
    """Generate random sample points on the perimeter of the given shape."""

    def visit_line2d(self, line: Line2D, num_of_samples, samples_range):
        line_x1, line_y1 = line.p1[0], line.p1[1]
        line_x2, line_y2 = line.p2[0], line.p2[1]
        line_slope = (line_y1 - line_y2) / (line_x1 - line_x2)
        random_shifts = np.random.uniform(-samples_range, samples_range,
                                          num_of_samples)
        samples_x = line_x1 + random_shifts
        samples_y = line_y1 + line_slope * random_shifts
        return np.stack([samples_x, samples_y], axis=1)


def parse_args():
    arguments = docopt(__doc__)
    config_path = arguments["<config_path>"]
    output_path = arguments["<output_path>"]
    debug = arguments["--debug"]
    config_path = path.abspath(config_path)
    validate_file_path(config_path)
    output_path = path.abspath(output_path)

    return config_path, output_path, debug


def generate_shapes_and_samples(config_path):
    # Load configuration file
    config = Configurations()
    with open(config_path, "rb") as config_file:
        config.load_from_file(config_file)

    # Generate random shapes, and noisy point samples of them
    gen = Generator(config)
    samples_suit = gen.generate_samples_suit()
    return samples_suit


def save_shapes_and_samples(samples_suit, output_path):
    # Save shapes and noisy data samples to file
    with open(output_path, "wb") as output_file:
        samples_suit.save_to_file(output_file)


def main():
    config_path, output_path, debug = parse_args()
    samples_suit = generate_shapes_and_samples(config_path)
    save_shapes_and_samples(samples_suit, output_path)

    if debug:
        samples_suit.plot()


if __name__ == '__main__':
    main()
