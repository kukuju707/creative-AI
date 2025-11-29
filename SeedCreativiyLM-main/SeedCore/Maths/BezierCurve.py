from SeedCore.Maths.Curve import Curve

class BezierCurve(Curve):
    """
    A class representing a (Rational) Bezier curve defined by weighted control points.
    """

    def __init__(self, control_points):
        """
        Initialize the BezierCurve with control points.

        Args:
            control_points: A list of control points.
                            Either [(x, y), ...] or [(x, y, w), ...].
                            If w is omitted, weight=1.0 is assumed.
        """
        super().__init__()

        # Normalize input into (x, y, w)
        normalized = []
        for p in control_points:
            if len(p) == 2:
                normalized.append((p[0], p[1], 1.0))  # default weight
            elif len(p) == 3:
                normalized.append(p)
            else:
                raise ValueError("Control point must have (x, y) or (x, y, w).")
        self.control_points = normalized

    def normalize(self) -> 'BezierCurve':
        """
        Normalize control points so x,y span 0–1.
        Weight w is preserved.
        """
        xs = [p[0] for p in self.control_points]
        ys = [p[1] for p in self.control_points]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        norm = []
        for (x, y, w) in self.control_points:
            nx = (x - min_x) / (max_x - min_x) if max_x != min_x else 0
            ny = (y - min_y) / (max_y - min_y) if max_y != min_y else 0
            norm.append((nx, ny, w))

        return BezierCurve(norm)

    def get_value_at(self, t):
        """
        Evaluate the Rational Bezier curve at parameter t.
        Uses weighted De Casteljau's algorithm.
        """
        # Copy control points: we will mutate them
        pts = [(x, y, w) for (x, y, w) in self.control_points]
        n = len(pts)

        # Rational De Casteljau
        for r in range(1, n):
            for i in range(n - r):
                x0, y0, w0 = pts[i]
                x1, y1, w1 = pts[i + 1]

                # Weighted interpolation (projective coords)
                w = (1 - t) * w0 + t * w1
                x = ((1 - t) * x0 * w0 + t * x1 * w1) / w
                y = ((1 - t) * y0 * w0 + t * y1 * w1) / w

                pts[i] = (x, y, w)

        return pts[0][:2]  # return (x, y)

    def visualize(self, **kwargs):
        import matplotlib.pyplot as plt
        import numpy as np

        t_values = np.linspace(0, 1, 100)
        curve_points = [self.get_value_at(t) for t in t_values]
        xs, ys = zip(*curve_points)

        ctrl_xs = [p[0] for p in self.control_points]
        ctrl_ys = [p[1] for p in self.control_points]

        plt.plot(xs, ys, label='Rational Bezier Curve')
        plt.plot(ctrl_xs, ctrl_ys, 'ro--', label='Control Points')

        plt.legend()
        plt.title('Rational Bezier Curve (Weighted)')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid()
        plt.show()


    @classmethod
    def make_from_points(cls, points):
        # support things like this:

        """
        "temperature_curve_points": [
            [0.0, 1.0, 0.1],
            [0.2, 0.1, 0.7],
            [0.5, 0.3, 0.2],
            [0.6, 0.6, 1.0],
            [1.0, 0.0, 0.1]
        ]

        self.temperature_curve = BezierCurve([tuple(p) for p in ConfigSubsystem.get_config("config/LLM/SG_Core.json").get('temperature_curve_points')]).normalize()
        """

        return cls([tuple(p) for p in points])

