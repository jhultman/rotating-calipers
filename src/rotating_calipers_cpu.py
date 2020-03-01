import numpy as np
from numba import jit


@jit(nopython=True)
def get_bbox_vertices(pts, angle):
    mean = np.float32([pts[:, 0].mean(), pts[:, 0].mean()])
    c, s = np.cos(angle), np.sin(angle)
    R = np.float32([c, -s, s, c]).reshape(2, 2)
    pts = (pts.astype(np.float32) - mean) @ R
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    corners = np.float32([x0, y0, x0, y1, x1, y1, x1, y0])
    corners = corners.reshape(-1, 2) @ R.T + mean
    return corners


@jit(nopython=True)
def compute_area(pts, caliper_angles):
    """Uses fact that inv(R) = R.T."""
    c = np.cos(caliper_angles[0])
    s = np.sin(caliper_angles[0])
    R = np.float32([c, -s, s, c]).reshape(2, 2)
    pts = pts @ R
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    return (x1 - x0) * (y1 - y0)


@jit(nopython=True)
def rotating_calipers(pts):
    """
    Assumes incoming pts are on convex hull in clock-wise order.
    Reference:
        Toussaint, Godfried T. "Solving geometric problems with
        the rotating calipers." Proc. IEEE Melecon. Vol. 83. 1983.
    """
    N = pts.shape[0]
    i, l = [pts[:, i].argmin() for i in range(2)]
    k, j = [pts[:, i].argmax() for i in range(2)]

    calipers = np.int32([i, j, k, l]) # left, top, right, bottom
    caliper_angles = np.float32([0.5, 0, -0.5, 1]) * np.pi
    best_area = np.inf

    for _ in range(N):
        calipers_advanced = (calipers + 1) % N        # Roll vertices clockwise
        vec = pts[calipers_advanced] - pts[calipers]  # Vectors from previous calipers to candidates
        angles = np.arctan2(vec[:, 1], vec[:, 0])     # Find angles of candidate edgelines
        angle_deltas = caliper_angles - angles        # Find candidate angle deltas
        pivot = np.abs(angle_deltas).argmin()         # Select pivot with smallest rotation
        calipers[pivot] = calipers_advanced[pivot]    # Advance selected pivot caliper
        caliper_angles -= angle_deltas[pivot]         # Rotate all supporting lines by angle delta

        # Check if found better set of calipers
        area = compute_area(pts[calipers], caliper_angles)
        if area < best_area:
            best_area = area
            best_calipers = calipers.copy()
            best_caliper_angles = caliper_angles.copy()

    return best_calipers, best_caliper_angles, best_area
