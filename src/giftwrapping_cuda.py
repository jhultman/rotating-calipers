from numba import cuda
import numpy as np
NUMBA_ENABLE_CUDASIM = 1


@cuda.jit(device=True)
def cross_test(x, y, x1, y1, x2, y2):
    """Cross product test for determining whether left of line."""
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    return cross < 0


@cuda.jit
def giftwrapping_kernel(pts, hull):
    row = cuda.grid(1)
    if row >= pts.shape[0]:
        return

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
    """Launcher for CUDA kernel."""
    n, m, _ = pts.shape
    tpb = 128
    bpg = (n + (tpb - 1)) // tpb
    pts_dev = cuda.to_device(pts)
    hull_dev = cuda.device_array((n, m), dtype=np.int32)
    giftwrapping_kernel[bpg, tpb](pts_dev, hull_dev)
    hull = hull_dev.copy_to_host()
    return hull
