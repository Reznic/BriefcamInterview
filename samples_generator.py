import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt

from shapes import ShapeFactory, ShapeOperation


class ShapeSamples:
    def __init__(self, shape, samples):
        self.shape = shape
        self.samples = samples

    def plot(self, canvas=None):
        self.shape.plot(plt)
        plt.scatter(x=self.samples.T[0], y=self.samples.T[1], s=1, c="red")
        plt.show()


class SamplesSuit:
    def __init__(self):
        self.len = 0
        self._suit = {}

    def add(self, samples: ShapeSamples):
        self._suit[self.len] = samples
        self.len += 1

    def __len__(self):
        return self.len

    def get_shapes(self):
        return self._suit.values()

    def delete_shape_params(self):
        """Clear shapes data from samples suit, and leave only sample points."""
        for shape_samples in self:
            shape_samples.shape = None

    def save_to_file(self, out_file):
        pickle.dump(self, out_file)

    @staticmethod
    def load_from_file(in_file):
        return pickle.load(in_file)

    def __iter__(self):
        return iter(self._suit.values())

    def plot(self):
        for samples in self:
            samples.plot()


class Generator:
    """Generates random shape-samples: Both noisy-inlier and outlier
    sample points on perimeters of random shapes.
    """
    INLIERS_RATIO = 0.8
    OUTLIERS_RANGE = 1000
    INLIERS_RANGE = 1000

    def __init__(self, config):
        self.log = logging.getLogger("Generator")
        self.shape_factory = ShapeFactory()
        self.config = config
        self.ground_truth_generator = GroundTruthSamplesGenerator()
        self.inliers_num = round(self.config.num_points * self.INLIERS_RATIO)
        outliers_ratio = 1 - self.INLIERS_RATIO
        self.outliers_num = round(self.config.num_points * outliers_ratio)

    def generate_samples_suit(self):
        self.log.info(f"Generating random shapes: {self.config.shapes}")
        self.log.info(f"{self.outliers_num} outlier and "
                      f"{self.inliers_num} inlier samples, "
                      f"with {self.config.randomness} noise deviation")
        shapes = self.shape_factory.generate_random_shapes(self.config.shapes)
        suit = SamplesSuit()
        for shape in shapes:
            samples = self._get_shape_samples(shape)
            suit.add(samples)

        return suit

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


class GroundTruthSamplesGenerator(ShapeOperation):
    """Generate random sample points on the perimeter of the given shape.

    Implements shape operation - Visitor of Shape type.

    Arguments:
        shape: Shape. to generate samples on it's perimeter.
        num_of_sampels: int.
        samples_range: int. maximal absolute value for each sample coordinate.
    """
    def visit_line2d(self, line, num_of_samples, samples_range):
        line_slope = self._get_line2d_slope(line)
        samples_range = self._calc_max_x_range_for_line_samples(samples_range,
                                                                line_slope)

        rand_x_shifts = np.random.uniform(-samples_range, samples_range,
                                          num_of_samples)
        samples_x = line.x1 + rand_x_shifts
        samples_y = line.y1 + line_slope * rand_x_shifts
        return np.stack([samples_x, samples_y], axis=1)

    @staticmethod
    def _get_line2d_slope(line):
        line_slope = (line.y1 - line.y2) / (line.x1 - line.x2)
        return line_slope

    @staticmethod
    def _calc_max_x_range_for_line_samples(max_range, line_slope):
        if abs(line_slope) > 1:
            return max_range / abs(line_slope)
        else:
            return max_range

