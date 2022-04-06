"""Random sample consensus algorithm implementation."""
import numpy as np
from math import cos, sin, atan

from shapes import ShapeOperation
from regression import Regressor


class RansacEstimator:
    """Random sample consensus shape estimator."""
    ITERATIONS = 100
    INLIERS_THRESHOLD = 1

    def __init__(self):
        self.shape_fitter = ShapeFitter()
        self.regressor = Regressor()
        self.distance_to_point_measure = DistanceToPoint()

    def estimate(self, samples: np.array, shape: type):
        """Run RANSAC on given sample points and estimate shape of given type.

        Algorithm:

        run in a loop for constant number of iterations :
            1. choose a random sub sample of points from all sample points.
                The sub sample size is the minimal points required to determine
                the estimated shape type. for example: 2 points for 2d-Line
            2. Try to fit shape of the estimated type, to the chosen sub sample
                If fitting cannot be done, pick a new random sub-sample
            3. Calculate the distance between each point, to the fitted shape.
            4. Filter out points with a distance larger than a
                threshold parameter. The remaining close points are the Inliers.
            5. Find a better estimation of the fitted shape by running
                regression on the close inliers points.
                For example: Least-Squares algorithm on inlier points of 2d-Line.
            6. Repeat steps 3 and 4, on the better shape estimation, and update
                the inlier points.
            7. Give a score to the estimated shape - the number of inlier points
            8. Choose the shape estimation with the largest score,
                across all iterations.

        Parameters:
            ITERATIONS: int. number of iterations to perform
            INLIERS_THRESHOLD: float. Maximal distance range between inlier
                                      points and the shape.

        Returns: tuple. (Shape, int) the estimated shape, and estimation score.
        """
        sub_sample_size = shape.MIN_POINTS_FOR_FITTING
        best_shape = None
        best_score = 0

        for _ in range(self.ITERATIONS):
            current_shape = shape()
            sub_sample = self._get_sub_sample(samples, sub_sample_size)
            is_fitted = current_shape.accept(self.shape_fitter, sub_sample)

            if not is_fitted:
                # Could not fit shape to sub-sample. retry sub sampling
                continue

            inliers = self._find_inliers(current_shape, samples)
            # Improve shape estimation by Regression on found inliers
            current_shape.accept(self.regressor, inliers)

            inliers_after_regression = self._find_inliers(current_shape,
                                                          samples)
            score = len(inliers_after_regression)

            if score >= best_score:
                best_score = score
                best_shape = current_shape

        return best_shape, best_score

    def _find_inliers(self, shape, samples):
        distances = shape.accept(self.distance_to_point_measure, samples)
        inliers = samples[distances < self.INLIERS_THRESHOLD]
        return inliers

    @staticmethod
    def _get_sub_sample(samples, size):
        indices = np.random.choice(len(samples), size, replace=False)
        sub_sample = np.take(samples, indices, axis=0)
        return sub_sample


class ShapeFitter(ShapeOperation):
    """Find percise parameters of shape that pass through points.

    Implements shape operation - Visitor of Shape type.

    Arguments:
        shape: Shape. Shape to fit. result parameters saved to this instance.
        samples: ndarray. sample points to fit shape to, percisely
    """
    def visit_line2d(self, line, samples):
        if samples.shape != (2, 2) or np.array_equal(samples[0], samples[1]):
            return False

        else:
            line.p1 = samples[0]
            line.p2 = samples[1]
            return True

    def visit_circle2d(self, circle, samples):
        p1, p2, p3 = samples[0], samples[1], samples[2]

        temp = p2[0] * p2[0] + p2[1] * p2[1]
        bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
        cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
        det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - \
              (p2[0] - p3[0]) * (p1[1] - p2[1])

        if abs(det) < 1.0e-6:
            return False

        # Center of circle
        cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
        cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

        circle.radius = np.sqrt((cx - p1[0]) ** 2 + (cy - p1[1]) ** 2)
        circle.center = np.array([cx, cy])
        return True


class DistanceToPoint(ShapeOperation):
    """Measure the shortest distance between given point/s and a shape.

    Implements shape operation - Visitor of Shape type.

    Arguments:
        shape: Shape.
        samples: ndarray.
    """
    def visit_line2d(self, line, points):
        """Measure distances between given points and given line."""
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

    def visit_circle2d(self, circle, points):
        """Measure distances between given points and given circle perimeter."""
        translated_points = points - circle.center
        distances_from_center = np.linalg.norm(translated_points, axis=1)
        distances_from_perimeter = np.abs(distances_from_center - circle.radius)
        return distances_from_perimeter

