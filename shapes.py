from abc import ABC, abstractmethod
import numpy as np

from utils import get_subclasses


class Shape(ABC):
    """Abstract Shape interface. All concrete shapes should implement it."""
    DIMENSION = NotImplemented
    MIN_POINTS_FOR_FITTING = NotImplemented

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit(self, *args, **kwargs)

    def randomize(self):
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__

    def plot(self, canvas):
        raise NotImplementedError()


class Line2D(Shape):
    """2-dimensional line, represented by 2 points on the line."""
    DIMENSION = 2
    MIN_POINTS_FOR_FITTING = 2

    def __init__(self):
        self.p1 = None
        self.p2 = None

    def randomize(self):
        """Randomly choose the line parameters."""
        self.p1 = np.random.rand(self.DIMENSION)
        self.p2 = np.random.rand(self.DIMENSION)

    def plot(self, canvas):
        canvas.axline(self.p1, self.p2)

    @property
    def x1(self):
        return self.p1[0]

    @property
    def x2(self):
        return self.p2[0]

    @property
    def y1(self):
        return self.p1[1]

    @property
    def y2(self):
        return self.p2[1]

    def get_slope(self):
        """Return slope of the line. Return None in case of a vertical line."""
        dx = self.x1 - self.x2
        if dx == 0:
            return None
        else:
            dy = self.y1 - self.y2
            return dy / dx


class Circle2D(Shape):
    """2-dimensional circle, represented by center point and radius."""
    DIMENSION = 2
    MIN_POINTS_FOR_FITTING = 3
    RANGE = 100

    def __init__(self):
        self.center = None
        self.radius = None

    def randomize(self):
        """Randomly choose the line parameters."""
        self.center = np.random.uniform(low=-self.RANGE, high=self.RANGE,
                                        size=2)
        self.radius = np.random.uniform(low=0, high=self.RANGE)

    def plot(self, canvas):
        cc = canvas.Circle(self.center, self.radius, fill=False)
        axes = canvas.gca()
        axes.set_aspect(1)
        axes.add_artist(cc)


class ShapeFactory:
    """Initialization of Shape instances of all concrete shape classes."""
    def __init__(self):
        self.shapes_name_to_class = self._find_all_shape_classes()

    def _find_all_shape_classes(self):
        shapes = list(get_subclasses(Shape))
        return {shape.__name__: shape for shape in shapes}

    def _get_shape(self, shape_name: str):
        if shape_name not in self.shapes_name_to_class:
            raise InvalidShapeError(f"Could not find {shape_name} shape")
        shape_class = self.shapes_name_to_class[shape_name]
        return shape_class()

    def generate_random_shapes(self, shape_amounts: dict):
        """Get list of shapes, according to given amounts,
        with randomized parameters.

        Args:
            shape_amounts: dictionary with Shape type as keys, and amounts to
                generate from each type (int) as values.

        Returns: list<Shape>: generated shapes, with randomized parameters.
        """
        shapes = []
        for shape_name, amount in shape_amounts.items():
            for _ in range(amount):
                shape = self._get_shape(shape_name)
                shape.randomize()
                shapes.append(shape)

        return shapes

    def get_all_shapes_of_dimension(self, dimension: int):
        """Get list of all implemented shape classes, of given dimension"""
        return [shape for shape in self.shapes_name_to_class.values()
                if shape.DIMENSION == dimension]


class ShapeOperation(ABC):
    """Represent abstract operation on shapes. (Abstract visitor for Shape).

    Implementing classes, should implement the visit_<shape> methods,
    for every shape type, the operation support.
    """
    def visit(self, shape: Shape, *args, **kwargs):
        visit_method = self._fetch_visit_method(shape)
        return visit_method(shape, *args, **kwargs)

    def _fetch_visit_method(self, shape):
        shape_class = shape.__class__.__name__.lower()
        method_name = f"visit_{shape_class}"
        if not hasattr(self, method_name):
            raise ShapeOperationNotImplementedError(self, shape)
        else:
            return self.__getattribute__(method_name)

    @abstractmethod
    def visit_line2d(self, shape: Line2D, *args, **kwargs):
        raise ShapeOperationNotImplementedError(self, shape)

    @abstractmethod
    def visit_circle2d(self, shape: Circle2D, *args, **kwargs):
        raise ShapeOperationNotImplementedError(self, shape)

class InvalidShapeError(BaseException):
    pass


class ShapeOperationNotImplementedError(NotImplementedError):
    def __init__(self, visitor, shape):
        self.message = f"{visitor.__class__.__name__} operation not " \
                       f"implemented for {shape.__class__.__name__} shape."
