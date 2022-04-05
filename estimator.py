"""Estimation of shapes from noisy sample data, using RANSAC algorithm.

Usage:
  estimator.py <input_path> <output_path> [--debug]
  estimator.py (-h | --help)

Options:
  -h --help     Show this screen.
  -d --debug       plot estimated shapes
"""
from os import path

import logging
from docopt import docopt

from shapes import ShapeFactory
from ransac import RansacEstimator
from samples_generator import SamplesSuit
from utils import validate_file_path, init_logger


class Estimator:
    """Estimate shapes from suit of noisy data points."""
    def __init__(self, estimation_algorithm):
        self.log = logging.getLogger("Estimator")
        self.algorithm = estimation_algorithm
        self.shape_factory = ShapeFactory()

    def estimate_suit(self, samples_suit):
        """Run estimation algorithm on every shape sampels in the suit.

        Returns: SampleSuit. all samples, with their estimated shapes.
            Estimation is done in-place.
            The given SamplesSuit instance is modified.
        """
        for id, shape_samples in enumerate(samples_suit):
            try:
                estimation = self.estimate_shape(shape_samples.samples, id)
                shape_samples.shape = estimation
            except BaseException as e:
                self.log.error(f"Failed to estimate samples_id={id}. Skipping")
                self.log.exception(e)

    def estimate_shape(self, samples, samples_id):
        candidate_shapes = self._get_candidate_shapes(samples)
        best_candidate = None
        best_score = 0

        for candidate_class in candidate_shapes:
            try:
                estimated_shape, score = \
                    self.algorithm.estimate(samples, candidate_class)
                if score > best_score:
                    best_score = score
                    best_candidate = estimated_shape

            except BaseException as e:
                self.log.error(f"Failed to try estimation of "
                               f"{candidate_class.__name__} for "
                               f"samples_id={samples_id}")
                self.log.exception(e)

        self.log.info(f"Estimated {best_candidate} shape "
                      f"with score={best_score} for samples_id={samples_id}")

        return best_candidate

    def _get_candidate_shapes(self, samples):
        dimension = samples.shape[1]
        candidates = self.shape_factory.get_all_shapes_of_dimension(dimension)
        if len(candidates) == 0:
            raise DimensionNotSupportedError(f"No supported estimations of "
                                             f"{dimension}-dimension shapes. "
                                             f"Could not estimate given sample "
                                             f"data")
        return candidates


class DimensionNotSupportedError(BaseException):
    pass


def parse_args():
    arguments = docopt(__doc__)
    input_path = arguments["<input_path>"]
    output_path = arguments["<output_path>"]
    debug = arguments["--debug"]
    validate_file_path(input_path)
    output_path = path.abspath(output_path)

    return input_path, output_path, debug


def load_samples(log, input_path):
    log.info(f"Loading samples suit from {input_path}")
    samples_suit = None
    try:
        with open(input_path, "rb") as input_file:
            samples_suit = SamplesSuit.load_from_file(input_file)

    except BaseException as e:
        log.error(f"Failed to load samples suit from {input_path}")
        log.exception(e)
        exit()

    return samples_suit


def run_estimations(samples_suit):
    ransac = RansacEstimator()
    estimator = Estimator(ransac)
    estimator.estimate_suit(samples_suit)


def save_estimations(log, suit, output_path):
    log.info(f"Saving estimations to {output_path}")
    try:
        with open(output_path, "wb") as output_file:
            suit.save_to_file(output_file)
    except BaseException as e:
        log.error(f"Failed to save samples suit to {output_path}")
        log.exception(e)


def main():
    input_path, output_path, debug_mode = parse_args()
    log = init_logger("Estimator", debug_mode)

    samples_suit = load_samples(log, input_path)

    # Do not pass the correct shapes data to the estimator.
    # Keep only the noisy samples
    samples_suit.delete_shape_params()

    log.info(f"Estimating samples suit")
    run_estimations(samples_suit)
    save_estimations(log, samples_suit, output_path)

    if debug_mode:
        log.debug("Plotting estimated shapes")
        samples_suit.plot()


if __name__ == '__main__':
    main()
