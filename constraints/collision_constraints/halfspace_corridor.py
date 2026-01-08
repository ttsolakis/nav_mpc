# nav_mpc/constraints/collision_constraints/halfspace_corridor.py

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class HalfspaceCorridorCollisionConfig:
    M: int = 16
    pos_idx: tuple[int, int] = (0, 1)     # indices of (x,y) in state
    r_robot: float = 0.10
    r_buffer: float = 0.05
    roi: float = 1.5
    angle_offset: float = 0.0
    b_loose: float = 1e6                  # inactive constraints upper bound

    @property
    def r_safe(self) -> float:
        return float(self.r_robot + self.r_buffer)

    def normals(self) -> np.ndarray:
        angles = self.angle_offset + np.linspace(0.0, 2.0 * np.pi, self.M, endpoint=False)
        return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(float)


def compute_collision_bounds_horizon(
    obstacles_xy: np.ndarray | None,
    centers_xy: np.ndarray,      # (N,2)
    normals: np.ndarray,         # (M,2)
    *,
    r_safe: float,
    roi: float,
    b_loose: float,
) -> np.ndarray:
    """
    Returns B of shape (N, M) with:
      b[k,i] = min_j ( n_i^T o_j - r_safe )  for o_j within ROI of center c_k
    If no points in ROI: b[k,i] = b_loose.
    """
    N = centers_xy.shape[0]
    M = normals.shape[0]
    B = np.full((N, M), b_loose, dtype=float)

    if obstacles_xy is None or len(obstacles_xy) == 0:
        return B

    obs = np.asarray(obstacles_xy, dtype=float).reshape(-1, 2)

    roi2 = float(roi) * float(roi)
    for k in range(N):
        c = centers_xy[k]
        d = obs - c[None, :]
        mask = (d[:, 0] ** 2 + d[:, 1] ** 2) <= roi2
        pts = obs[mask]
        if len(pts) == 0:
            continue
        proj_rel = normals @ (pts - c[None, :]).T     # (M, P)
        alpha = np.max(proj_rel, axis=1) - float(r_safe)
        B[k, :] = alpha + (normals @ c)              # make it absolute for n^T p <= B


    return B