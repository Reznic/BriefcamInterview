"""Utility functions"""
from os.path import isfile

import logging
import coloredlogs


def validate_file_path(path):
    if not isfile(path):
        raise FileNotFoundError(f"Could not find file at: {path}")


def init_logger(name, debug_mode):
    coloredlogs.install()
    logger = logging.getLogger(name)
    if debug_mode:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    return logger

def get_subclasses(base_class):
    for subclass in base_class.__subclasses__():
        yield subclass
        yield from get_subclasses(subclass)

