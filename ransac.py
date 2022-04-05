"""Random sample concensus algorithm implementation."""
import numpy as np

from shapes import Shape


class RansacEstimator:
    """Random sample concensus shape estimator."""
    def __init__(self):
        pass

    def estimate(self, samples: np.ndarray, shape: Shape):
        sub_sample_size = shape.MIN_POINTS_FOR_FITTING
        # for chosen iterations:
        # pick random sub sample
        # call fit on shape
        # count inliers
        # update best score
        return shape, 1
