import numpy as np
from numba import jit


@jit(nopython=True)
def cross_test(x, y, x1, y1, x2, y2):
    """
    Cross product test for determining whether left of line.
    Can switch inequality if prefer ccw hull.
    """
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    return cross < 0


@jit(nopython=True)
def giftwrapping_kernel(pts, hull):
    for row in range(pts.shape[0]):
        # Initialize hull as left-most coordinate
        pointOnHull = 0
        min_x = pts[row, 0, 0]
        for j in range(pts.shape[1]):
            x = pts[row, j, 0]
            if x < min_x:
                min_x = x
                pointOnHull = j

        # Enter giftwrapping main loop
        startpoint = pointOnHull
        count = 0
        while True:
            hull[row, count] = pointOnHull
            count += 1
            endpoint = 0
            for j in range(pts.shape[1]):
                if endpoint == pointOnHull:
                    endpoint = j
                elif cross_test(
                    pts[row, j, 0],
                    pts[row, j, 1],
                    pts[row, pointOnHull, 0],
                    pts[row, pointOnHull, 1],
                    pts[row, endpoint, 0],
                    pts[row, endpoint, 1],
                ):
                    endpoint = j
            pointOnHull = endpoint
            if endpoint == startpoint:
                break
        for j in range(count, pts.shape[1], 1):
            hull[row, j] = -1


def giftwrapping(pts):
    """Launcher for numba kernel."""
    n, m, _ = pts.shape
    hull = np.full((n, m), -1, dtype=np.int32)
    giftwrapping_kernel(pts, hull)
    return hull
