"""Random sample concensus algorithm implementation."""
import numpy as np
from math import cos, sin, atan

from shapes import ShapeOperation


class RansacEstimator:
    """Random sample concensus shape estimator."""
    ITERATIONS = 10000
    INLIERS_THRESHOLD = 1

    def __init__(self):
        self.shape_fitter = ShapeFitter()
        self.regressor = Regressor()
        self.distance_to_point_measure = DistanceToPoint()

    def estimate(self, samples: np.array, shape: type):
        sub_sample_size = shape.MIN_POINTS_FOR_FITTING

        best_shape = None
        best_score = 0

        for _ in range(self.ITERATIONS):
            current_shape = shape()
            sub_sample = self.get_sub_sample(samples, sub_sample_size)
            is_fitted = current_shape.accept(self.shape_fitter, sub_sample)
            if not is_fitted:
                continue

            inliers = self.find_inliers(current_shape, samples)
            # Improve shape estimation by Regression on found inliers
            current_shape.accept(self.regressor, inliers)

            inliers_after_regression = self.find_inliers(current_shape, samples)
            score = len(inliers_after_regression)

            if score >= best_score:
                best_score = score
                best_shape = current_shape

        return best_shape, best_score

    def find_inliers(self, shape, samples):
        distances = shape.accept(self.distance_to_point_measure, samples)
        inliers = samples[distances < self.INLIERS_THRESHOLD]
        return inliers

    @staticmethod
    def get_sub_sample(samples, size):
        indices = np.random.choice(len(samples), size, replace=False)
        sub_sample = np.take(samples, indices, axis=0)
        return sub_sample


class ShapeFitter(ShapeOperation):
    """Generate random sample points on the perimeter of the given shape.

    Implements shape operation - Visitor of Shape type.

    Arguments:
        shape: Shape.
        samples: ndarray.
    """
    def visit_line2d(self, line, samples):
        if samples.shape != (2, 2) or np.array_equal(samples[0], samples[1]):
            return False

        else:
            line.p1 = samples[0]
            line.p2 = samples[1]
            return True


class DistanceToPoint(ShapeOperation):
    """Generate random sample points on the perimeter of the given shape.

    Implements shape operation - Visitor of Shape type.

    Arguments:
        shape: Shape.
        samples: ndarray.
    """
    def visit_line2d(self, line, points):
        """Measure distances between given points to given line."""
        translated_points = points - line.p1

        line_slope = line.get_slope()
        if line_slope is not None:
            line_angle = atan(line_slope)
            rotation_angle = -line_angle
            rotation_mat = self._rotation_matrix_2d(rotation_angle)
            rotated_points = np.matmul(rotation_mat, translated_points.T)
            y_column = rotated_points[1]
            distances_from_line = np.abs(y_column)
        else:
            # Line perpendicular to y axis, distance is x coord of points
            x_column = translated_points.T[0]
            distances_from_line = np.abs(x_column)

        return distances_from_line


    @staticmethod
    def _rotation_matrix_2d(angle):
        return np.array([[cos(angle), -sin(angle)],
                         [sin(angle), cos(angle)]])


class Regressor(ShapeOperation):
    """Generate random sample points on the perimeter of the given shape.

    Implements shape operation - Visitor of Shape type.

    Arguments:
        shape: Shape.
        samples: ndarray.
    """
    def visit_line2d(self, line, points):
        """Perform Least-Squares on given points, and save result in given line."""
        points_x = points.T[0]
        points_y = points.T[1]
        A = np.vstack([points_x, np.ones(len(points_x))]).T
        m, c = np.linalg.lstsq(A, points_y, rcond=None)[0]

        line.p1 = np.array([1, m+c])
        line.p2 = np.array([2, 2*m+c])

