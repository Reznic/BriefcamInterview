"""Estimation of shapes from noisy sample data, using RANSAC algorithm.

Usage:
  estimator.py <input_path> <output_path> [--debug]
  estimator.py (-h | --help)

Options:
  -h --help     Show this screen.
  -d --debug       plot estimated shapes
"""
from os import path

from docopt import docopt

from ransac import RansacEstimator
from samples_generator import SamplesSuit
from utils import validate_file_path, init_logger


class Estimator:
    def __init__(self, samples_suit, estimation_algorithm):
        self.samples_suit = samples_suit
        self.algorithm = estimation_algorithm

    def estimate(self):
        """Run estimation algorithm on every shape sampels in the suit.

        Returns: SampleSuit. all samples, with their estimated shapes.
        """
        for shape_samples in self.samples_suit:
            estimated_shape = self.algorithm.estimate(shape_samples.samples)
            shape_samples.shape = estimated_shape

        return self.samples_suit


def parse_args():
    arguments = docopt(__doc__)
    input_path = arguments["<input_path>"]
    output_path = arguments["<output_path>"]
    debug = arguments["--debug"]
    validate_file_path(input_path)
    output_path = path.abspath(output_path)

    return input_path, output_path, debug


def main():
    input_path, output_path, debug_mode = parse_args()
    log = init_logger("Estimator", debug_mode)

    log.info(f"Loading samples suit from {input_path}")
    samples_suit = None
    with open(input_path, "rb") as input_file:
        samples_suit = SamplesSuit.load_from_file(input_file)

    samples_suit.delete_shape_params()

    log.info(f"Estimating samples suit")
    ransac = RansacEstimator()
    estimator = Estimator(samples_suit, ransac)
    estimator.estimate()

    log.info(f"Saving estimations to {output_path}")
    with open(output_path, "wb") as output_file:
        samples_suit.save_to_file(output_file)

    if debug_mode:
        log.debug("Plotting estimated shapes")
        samples_suit.plot()


if __name__ == '__main__':
    main()
