"""Utility functions"""
from os.path import isfile


def validate_file_path(path):
    if not isfile(path):
        raise FileNotFoundError(f"Could not find file at: {path}")
