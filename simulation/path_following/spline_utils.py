# nav_mpc/simulation/path_following/spline_utils.py

import numpy as np
from scipy.interpolate import splprep, splev

def smooth_and_resample_path(path_xy: np.ndarray, *, ds: float = 0.05, smoothing: float = 0.01, k: int = 3, dense_factor: int = 30) -> np.ndarray:
    """
    Fit a parametric spline to path_xy and resample points approximately equidistant
    in arc-length with spacing ds.

    Args:
        path_xy: (M,2) polyline waypoints
        ds: desired spacing [m] between consecutive resampled points (geometry-based)
        smoothing: splprep smoothing factor s
        k: spline degree
        dense_factor: internal oversampling factor to build arc-length map

    Returns:
        path_resampled: (K,2) points with ~ds spacing along arc-length.
    """
    path_xy = np.asarray(path_xy, dtype=float)
    if path_xy.ndim != 2 or path_xy.shape[1] != 2:
        raise ValueError(f"path_xy must be (M,2), got {path_xy.shape}")
    if ds <= 0:
        raise ValueError("ds must be > 0")

    # Remove consecutive duplicates
    diffs = np.diff(path_xy, axis=0)
    keep = np.ones(path_xy.shape[0], dtype=bool)
    keep[1:] = np.linalg.norm(diffs, axis=1) > 1e-9
    path_xy = path_xy[keep]

    if path_xy.shape[0] < 2:
        raise ValueError("Need at least 2 distinct points to fit a spline.")

    # spline degree constraints: splprep requires m > k
    k = int(np.clip(k, 1, 5))
    k = min(k, path_xy.shape[0] - 1)

    x = path_xy[:, 0]
    y = path_xy[:, 1]

    # Fit parametric spline: x(u), y(u)
    tck, _ = splprep([x, y], s=float(smoothing), k=k)

    # Dense sampling in u to build an arc-length map
    M = path_xy.shape[0]
    n_dense = max(300, dense_factor * M)
    u_dense = np.linspace(0.0, 1.0, n_dense)
    x_dense, y_dense = splev(u_dense, tck)
    dense = np.column_stack([x_dense, y_dense])

    seg = np.diff(dense, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    s_cum = np.concatenate([[0.0], np.cumsum(seglen)])
    L = float(s_cum[-1])

    if L < 1e-9:
        return dense[:1].copy()

    # Target arc-length positions: 0, ds, 2ds, ...
    s_targets = np.arange(0.0, L, ds)
    if s_targets.size == 0 or s_targets[-1] < L:
        s_targets = np.append(s_targets, L)

    # Invert s(u) via interpolation on dense samples
    u_targets = np.interp(s_targets, s_cum, u_dense)

    # Evaluate spline at u_targets
    x_out, y_out = splev(u_targets, tck)
    out = np.column_stack([x_out, y_out])

    return out