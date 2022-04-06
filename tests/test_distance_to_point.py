from shapes import Line2D, Circle2D
from ransac import DistanceToPoint
from math import sqrt

import numpy as np


def validate_correct_distances(shape, points, expected_distances):
    distance_to_point = DistanceToPoint()
    distances = shape.accept(distance_to_point, points)
    assert np.allclose(distances, expected_distances), \
        f"DistanceToPoint operation returned " \
        f"wrong distances between {shape.__class__.__name__} and test points"


def test_distance_to_point_line2d():
    line = Line2D()
    line.p2 = np.array([2, 2])
    line.p1 = np.array([1, 1])
    points = np.array([[0, 0], [0, 1], [1, 0], [2, 0]])
    expected_distances = np.array([0, sqrt(0.5), sqrt(0.5), sqrt(2)])
    validate_correct_distances(line, points, expected_distances)


def test_distance_to_points_on_line2d():
    line = Line2D()
    line.p1 = np.array([-27.06051402, 231.66345089])
    line.p2 = np.array([78.06731383, -662.76207316])
    points = np.array([line.p1, line.p2])
    expected_distances = np.array([0, 0])
    validate_correct_distances(line, points, expected_distances)


def test_distance_to_point_line2d_perpendicular():
    line = Line2D()
    line.p1 = np.array([1, 1])
    line.p2 = np.array([1, 2])
    points = np.array([[0, 0], [0, 1], [1, 0], [2, 0]])
    expected_distances = np.array([1, 1, 0, 1])
    validate_correct_distances(line, points, expected_distances)


def test_distance_to_point_circle2d():
    circle = Circle2D()
    circle.center = np.array([1, 0])
    circle.radius = 3
    points = np.array([[4, 0], [3, 0], [1, 5]])
    expected_distances = np.array([0, 1, 2])
    validate_correct_distances(circle, points, expected_distances)

