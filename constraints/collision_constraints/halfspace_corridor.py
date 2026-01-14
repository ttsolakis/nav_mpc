# constraints/collision_constraints/halfspace_corridor.py

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class HalfspaceCorridorCollisionConfig:
    """
    Per stage k we build up to M halfspaces from lidar points inside ROI:

        (-n_{k,j})^T p_k <= -( n_{k,j}^T o_{k,j} + rho_s )

    where n_{k,j} is computed from the fixed reference center c_k (= Xbar[k] position),
    NOT from the decision variable p_k.
    """
    M: int = 16
    pos_idx: tuple[int, int] = (0, 1)     # indices of (x,y) in state
    r_robot: float = 0.10
    r_buffer: float = 0.05
    roi: float = 1.5
    b_loose: float = 1e6                  # inactive constraint upper bound (OSQP)
    eps_norm: float = 1e-9                # avoid divide-by-zero

    @property
    def r_safe(self) -> float:
        return float(self.r_robot + self.r_buffer)


def compute_collision_halfspaces_horizon(
    obstacles_xy: np.ndarray | None,
    centers_xy: np.ndarray,      # (N,2) reference centers c_k (e.g., from shifted previous plan)
    *,
    M: int,
    rho: float,                  # rho_s (robot radius + margin)
    roi: float,
    b_loose: float,
    eps_norm: float = 1e-9,
    pick: str = "closest",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build per-stage halfspaces from obstacle (lidar) points.

    Constraint form returned:
        A_xy[k,j,:] @ p <= b[k,j]
    where p = [x,y] is the decision (robot center at that stage).

    We compute a unit normal:
        n = (c - o) / ||c - o||
    then enforce:
        n^T p >= n^T o + rho
    which becomes:
        (-n)^T p <= -(n^T o + rho)

    Returns
    -------
    A_xy : (N, M, 2)
        Row coefficients for p = [x,y].
        If inactive: [0,0].

    b : (N, M)
        Upper bounds. If inactive: b_loose.
    """
    centers_xy = np.asarray(centers_xy, dtype=float).reshape(-1, 2)
    N = centers_xy.shape[0]

    A_xy = np.zeros((N, M, 2), dtype=float)
    b = np.full((N, M), float(b_loose), dtype=float)

    if obstacles_xy is None:
        return A_xy, b

    obs = np.asarray(obstacles_xy, dtype=float).reshape(-1, 2)
    if obs.shape[0] == 0:
        return A_xy, b

    roi2 = float(roi) * float(roi)

    for k in range(N):
        c = centers_xy[k]                 # (2,)
        d = obs - c[None, :]              # (P,2)
        dist2 = d[:, 0] ** 2 + d[:, 1] ** 2
        mask = dist2 <= roi2
        pts = obs[mask]

        if pts.shape[0] == 0:
            continue

        # Choose which points to use (fixed M to preserve sparsity later)
        if pick == "closest":
            dd = pts - c[None, :]
            dd2 = dd[:, 0] ** 2 + dd[:, 1] ** 2
            order = np.argsort(dd2)
            pts_sel = pts[order[:M]]
        elif pick == "first":
            pts_sel = pts[:M]
        else:
            raise ValueError(f"Unknown pick='{pick}'. Use 'closest' or 'first'.")

        # Fill constraints
        for j in range(pts_sel.shape[0]):
            o = pts_sel[j]               # obstacle point
            v = c - o                    # points away from obstacle
            norm = float(np.sqrt(v[0] * v[0] + v[1] * v[1]))

            if norm < eps_norm:
                # Degenerate -> deactivate
                continue

            n0 = v[0] / norm
            n1 = v[1] / norm

            # OSQP row: (-n)^T p <= -(n^T o + rho)
            A_xy[k, j, 0] = -n0
            A_xy[k, j, 1] = -n1
            b[k, j] = -(n0 * o[0] + n1 * o[1] + float(rho))

        # remaining j are inactive by default

    return A_xy, b
