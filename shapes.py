from abc import ABC, abstractmethod
import numpy as np
import pickle
import matplotlib.pyplot as plt


class Shape(ABC):
    DIMENSION = NotImplemented
    MIN_POINTS_FOR_FITTING = NotImplemented

    def accept(self, visitor, *args, **kwargs):
        return visitor.visit(self, *args, **kwargs)

    def randomize(self):
        raise NotImplementedError()

    def plot(self, canvas):
        raise NotImplementedError()


class Line2D(Shape):
    DIMENSION = 2
    MIN_POINTS_FOR_FITTING = 2

    def __init__(self):
        self.p1 = None
        self.p2 = None

    def randomize(self):
        self.p1 = np.random.rand(self.DIMENSION)
        self.p2 = np.random.rand(self.DIMENSION)

    def plot(self, canvas):
        canvas.axline(self.p1, self.p2)


class ShapeFactory:
    SHAPE_NAME_TO_CLASS = {
        "Line2D": Line2D,
    }

    def _get_shape(self, shape_name: str):
        if shape_name not in self.SHAPE_NAME_TO_CLASS:
            raise InvalidShapeError(f"Could not find {shape_name} shape")
        shape_class = self.SHAPE_NAME_TO_CLASS[shape_name]
        return shape_class()

    def get_random_shapes(self, shape_amounts: dict):
        shapes = []
        for shape_name, amount in shape_amounts.items():
            for _ in range(amount):
                shape = self._get_shape(shape_name)
                shape.randomize()
                shapes.append(shape)

        return shapes


class ShapeSamples:
    def __init__(self, shape, samples):
        self.shape = shape
        self.samples = samples

    def plot(self, canvas=None):
        self.shape.plot(plt)
        plt.scatter(x=self.samples.T[0], y=self.samples.T[1], s=1, c="red")
        plt.show()


class SamplesSuit:
    def __init__(self):
        self.len = 0
        self._suit = {}

    def add(self, samples: ShapeSamples):
        self._suit[self.len] = samples
        self.len += 1

    def __len__(self):
        return self.len

    def get_shapes(self):
        return self._suit.values()

    def save_to_file(self, out_file):
        pickle.dump(self, out_file)

    @staticmethod
    def load_from_file(in_file):
        return pickle.load(in_file)

    def __iter__(self):
        return iter(self._suit.values())

    def plot(self):
        for samples in self:
            samples.plot()


class ShapeOperation(ABC):
    """Represent abstract operation on shapes. (Abstract visitor for Shape)."""
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


class InvalidShapeError(BaseException):
    pass


class ShapeOperationNotImplementedError(NotImplementedError):
    def __init__(self, visitor, shape):
        self.message = f"{visitor.__class__.__name__} operation not implemented " \
                       f"for {shape.__class__.__name__} shape."
