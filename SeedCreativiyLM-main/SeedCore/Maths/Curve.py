
class Curve:
    """
    A base class for the Curve structure.

    Represents a mathematical curve defined by a set of points, or functions.

    It must support normalization and value retrieval at specific parameters, and should be extended for specific curve types (e.g., Bezier, Spline, function-based).
    """

    def __init__(self):
        """
        Initialize the Curve with default parameters.
        """
        pass

    def normalize(self) -> 'Curve':
        """
        Normalize the curve points to a 0-1 range, both for x, y axes.
        """
        pass

    def get_value_at(self, t):
        """
        Get the value of the curve at a specific parameter t.
        't' represents a normalized position along the curve (0 <= t <= 1), which map the curve's domain (first point to last point).

        Args:
            t: Parameter along the curve.

        Returns:
            Value of the curve at parameter t.
        """
        pass

    def visualize(self, **kwargs):
        """
        Visualize the curve using a plotting library (e.g., Matplotlib).
        """
        pass
