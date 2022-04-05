"""Generation of noisy sample data for random shapes .

Usage:
  generator.py <config_path> <output_path> [--debug]
  generator.py (-h | --help)

Options:
  -h --help     Show this screen.
  -d --debug       plot generated shapes and data
"""
from os import path

from docopt import docopt

from samples import Generator
from configs import Configurations
from utils import validate_file_path, init_logger


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
    config_path, output_path, debug_mode = parse_args()
    log = init_logger("Generator", debug_mode)

    samples_suit = generate_shapes_and_samples(config_path)
    save_shapes_and_samples(samples_suit, output_path)

    if debug_mode:
        log.debug("Plotting sample suit")
        samples_suit.plot()


if __name__ == '__main__':
    main()
