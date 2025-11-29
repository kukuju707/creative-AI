from SeedCore.Maths.Curve import Curve

class LinearCurve(Curve):
    """
    A class representing a linear curve defined by two points.

    Inherits from the base Curve class.
    """

    def __init__(self, point1, point2):
        """
        Initialize the LinearCurve with two points.

        Args:
            point1: The first point (x1, y1).
            point2: The second point (x2, y2).
        """
        super().__init__()
        self.point1 = point1
        self.point2 = point2

    def normalize(self) -> 'LinearCurve':
        """
        Normalize the linear curve points to a 0-1 range for both x and y axes.

        Returns:
            A new LinearCurve instance with normalized points.
        """
        x1, y1 = self.point1
        x2, y2 = self.point2

        min_x = min(x1, x2)
        max_x = max(x1, x2)
        min_y = min(y1, y2)
        max_y = max(y1, y2)

        norm_point1 = ((x1 - min_x) / (max_x - min_x), (y1 - min_y) / (max_y - min_y))
        norm_point2 = ((x2 - min_x) / (max_x - min_x), (y2 - min_y) / (max_y - min_y))

        return LinearCurve(norm_point1, norm_point2)

    def get_value_at(self, t):
        """
        Get the value of the linear curve at a specific parameter t.

        Args:
            t: Parameter along the curve (0 <= t <= 1).

        Returns:
            Value of the curve at parameter t.
        """
        x1, y1 = self.point1
        x2, y2 = self.point2

        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        return (x, y)

    def visualize(self, **kwargs):
        """
        Visualize the linear curve using a plotting library (e.g., Matplotlib).
        """
        import matplotlib.pyplot as plt

        x_values = [self.point1[0], self.point2[0]]
        y_values = [self.point1[1], self.point2[1]]

        plt.plot(x_values, y_values, **kwargs)
        plt.scatter([self.point1[0], self.point2[0]], [self.point1[1], self.point2[1]], color='red')