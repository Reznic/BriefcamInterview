import numpy as np

from shapes import ShapeOperation


class Regressor(ShapeOperation):
    """Find best fitting shape to given noisy datapoints.

    Implements shape operation - Visitor of Shape type.

    Arguments:
        shape: Shape. out - fitted shape is saved to this shape instance.
            Also, defines the type of shape to be fitted.
        points: ndarray. sample data to fit shape to.
    """
    def visit_line2d(self, line, points):
        """Perform Least-Squares on given points, and save result in given line."""
        points_x = points.T[0]
        points_y = points.T[1]
        xs = np.vstack([points_x, np.ones(len(points_x))]).T
        m, c = np.linalg.lstsq(xs, points_y, rcond=None)[0]

        line.p1 = np.array([1, m+c])
        line.p2 = np.array([2, 2*m+c])

    def visit_circle2d(self, circle, points):
        """Perform Least-Squares on given points, and save result in given circle."""
        # Currently not implemented. Circle remains unchanged.
        return

