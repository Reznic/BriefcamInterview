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
import json

from utils import validate_file_path


class Configurations:
    def __init__(self, config_file):
        config_dict = json.load(config_file)
        self.randomness = config_dict.get("randomness", 0)
        self.shapes = config_dict.get("shapes", {})
        self.num_points = config_dict.get("num_points", 0)


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


def main():
    config_path, output_path, debug = parse_args()
    with open(config_path, "rb") as config_file:
        config = Configurations(config_file)


if __name__ == '__main__':
    main()

