"""Generation of noisy sample data for random shapes .

Usage:
  generator.py <config_path> <output_path>
  generator.py (-h | --help)

Options:
  -h --help     Show this screen.
  -d --debug       plot generated shapes and data
"""
from docopt import docopt
from os import path
import numpy as np

from utils import validate_file_path
from shapes import ShapeFactory, ShapeVisitor, Line2D, ShapeSamples, SamplesSuit
from configs import Configurations


class Generator:
    INLIERS_RATIO = 0.8
    OUTLIERS_RANGE = 100

    def __init__(self, config):
        self.config = config
        self.shape_factory = ShapeFactory()

    def _get_gaussian_noise(self):
        pass

    def _get_inlier_samples(self, shape):
        inliers_num = self.config.num_points * self.INLIERS_RATIO

        pass

    def _get_outlier_samples(self, dimension):
        outliers_ratio = 1 - self.INLIERS_RATIO
        outliers_num = self.config.num_points * outliers_ratio
        outlier_samples = \
            self.OUTLIERS_RANGE * np.random.rand(outliers_num, dimension)
        return outlier_samples

    def _get_shape_samples(self, shape):
        outlier_samples = self._get_outlier_samples(shape.DIMENSION)
        inlier_samples = self._get_inlier_samples(shape)
        samples = outlier_samples + inlier_samples
        return ShapeSamples(shape, samples)

    def generate_samples_suit(self):
        shapes = self.shape_factory.get_random_shapes(self.config.shapes)
        suit = SamplesSuit()
        for shape in shapes:
            samples = self._get_shape_samples(shape)
            suit.add(samples)

        return suit


class GroundTruthSamplesGenerator(ShapeVisitor):
    def visit_line2d(self, shape: Line2D, *args, **kwargs):
        return None


def parse_args():
    arguments = docopt(__doc__)
    config_path = arguments["<config_path>"]
    output_path = arguments["<output_path>"]
    debug = arguments.get("debug", False)
    config_path = path.abspath(config_path)
    output_path = path.abspath(output_path)

    validate_file_path(config_path)
    validate_file_path(output_path)

    return config_path, output_path, debug


def generate_shapes_and_samples(config_path):
    # Load configuration file
    with open(config_path, "rb") as config_file:
        config = Configurations(config_file)

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


if __name__ == '__main__':
    main()

