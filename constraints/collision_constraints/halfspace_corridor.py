# constraints/collision_constraints/halfspace_corridor.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass(frozen=True, slots=True)
class HalfspaceCorridorCollisionConfig:
    """
    Per stage k we build up to M halfspaces from lidar points inside ROI:

        (-n_{k,j})^T p_k <= -( n_{k,j}^T o_{k,j} + rho_s )

    where n_{k,j} is computed from the fixed reference center c_k (= centers_xy[k]),
    NOT from the decision variable p_k.
    """
    M: int = 500
    pos_idx: tuple[int, int] = (0, 1)     # indices of (x,y) in state
    r_robot: float = 0.10
    r_buffer: float = 0.05
    roi: float = 1.5
    b_loose: float = 1e6                  # inactive constraint upper bound (OSQP)
    eps_norm: float = 1e-9                # avoid divide-by-zero

    @property
    def r_safe(self) -> float:
        return float(self.r_robot + self.r_buffer)


@njit(cache=True)
def _compute_collision_halfspaces_horizon_inplace_numba(
    obs: np.ndarray,          # (P,2)
    centers_xy: np.ndarray,   # (N,2)
    A_xy_out: np.ndarray,     # (N,M,2) output
    b_out: np.ndarray,        # (N,M) output
    M: int,
    rho: float,
    roi: float,
    b_loose: float,
    eps_norm: float,
) -> None:
    """
    In-place kernel.

    For each stage k:
      - consider obstacle points within ROI of center c
      - select up to M closest ones
      - for each selected obstacle o:
           n = (c - o) / ||c - o||
           A = -n
           b = -(n^T o + rho)

    For inactive rows:
      A=[0,0], b=b_loose (already set by caller).
    """
    N = centers_xy.shape[0]
    P = obs.shape[0]
    roi2 = roi * roi

    # scratch arrays for each stage
    dist2 = np.empty(P, dtype=np.float64)
    sel_idx = np.empty(P, dtype=np.int64)   # indices of points inside ROI
    sel_d2 = np.empty(P, dtype=np.float64)  # their squared distances

    for k in range(N):
        cx = centers_xy[k, 0]
        cy = centers_xy[k, 1]

        # 1) collect points in ROI
        cnt = 0
        for p in range(P):
            dx = obs[p, 0] - cx
            dy = obs[p, 1] - cy
            d2 = dx * dx + dy * dy
            dist2[p] = d2
            if d2 <= roi2:
                sel_idx[cnt] = p
                sel_d2[cnt] = d2
                cnt += 1

        if cnt == 0:
            continue

        # 2) choose up to M closest among sel_idx[0:cnt]
        #    partial selection by repeated "min extraction" (O(cnt*M), ok for ~500)
        #    This avoids allocating/using np.argsort inside numba.
        m_use = M if cnt >= M else cnt

        for j in range(m_use):
            # find argmin sel_d2[j:cnt]
            best_pos = j
            best_val = sel_d2[j]
            for t in range(j + 1, cnt):
                if sel_d2[t] < best_val:
                    best_val = sel_d2[t]
                    best_pos = t

            # swap into position j
            if best_pos != j:
                tmpi = sel_idx[j]
                sel_idx[j] = sel_idx[best_pos]
                sel_idx[best_pos] = tmpi

                tmpd = sel_d2[j]
                sel_d2[j] = sel_d2[best_pos]
                sel_d2[best_pos] = tmpd

            p_idx = sel_idx[j]
            ox = obs[p_idx, 0]
            oy = obs[p_idx, 1]

            vx = cx - ox
            vy = cy - oy
            nrm = np.sqrt(vx * vx + vy * vy)

            if nrm < eps_norm:
                # keep inactive (already)
                continue

            n0 = vx / nrm
            n1 = vy / nrm

            # A = -n
            A_xy_out[k, j, 0] = -n0
            A_xy_out[k, j, 1] = -n1
            b_out[k, j] = -(n0 * ox + n1 * oy + rho)

        # remaining j are inactive by default


def compute_collision_halfspaces_horizon_inplace(
    obstacles_xy: np.ndarray | None,
    centers_xy: np.ndarray,     # (N,2)
    A_xy_out: np.ndarray,       # (N,M,2)
    b_out: np.ndarray,          # (N,M)
    *,
    M: int,
    rho: float,
    roi: float,
    b_loose: float,
    eps_norm: float = 1e-9,
    pick: str = "closest",
) -> None:
    """
    In-place API (no allocations).

    Output constraint form:
        A_xy_out[k,j,:] @ p <= b_out[k,j]
    where p = [x,y] is the decision (robot center at that stage).

    Notes
    -----
    - Currently supports pick="closest" only (fast + deterministic).
    - Caller should pass preallocated outputs with correct shapes:
        A_xy_out: (N,M,2)
        b_out:    (N,M)
    """
    if pick != "closest":
        raise ValueError("In-place implementation currently supports pick='closest' only.")

    centers_xy = np.asarray(centers_xy, dtype=float).reshape(-1, 2)
    N = centers_xy.shape[0]

    if A_xy_out.shape != (N, M, 2):
        raise ValueError(f"A_xy_out must have shape {(N, M, 2)}, got {A_xy_out.shape}.")
    if b_out.shape != (N, M):
        raise ValueError(f"b_out must have shape {(N, M)}, got {b_out.shape}.")

    # default inactive
    A_xy_out.fill(0.0)
    b_out.fill(float(b_loose))

    if obstacles_xy is None:
        return

    obs = np.asarray(obstacles_xy, dtype=float).reshape(-1, 2)
    if obs.shape[0] == 0:
        return

    _compute_collision_halfspaces_horizon_inplace_numba(
        obs=obs,
        centers_xy=centers_xy,
        A_xy_out=A_xy_out,
        b_out=b_out,
        M=int(M),
        rho=float(rho),
        roi=float(roi),
        b_loose=float(b_loose),
        eps_norm=float(eps_norm),
    )


def compute_collision_halfspaces_horizon(
    obstacles_xy: np.ndarray | None,
    centers_xy: np.ndarray,      # (N,2)
    *,
    M: int,
    rho: float,
    roi: float,
    b_loose: float,
    eps_norm: float = 1e-9,
    pick: str = "closest",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible allocating wrapper.

    Returns
    -------
    A_xy : (N, M, 2)
    b    : (N, M)
    """
    centers_xy = np.asarray(centers_xy, dtype=float).reshape(-1, 2)
    N = centers_xy.shape[0]

    A_xy = np.zeros((N, M, 2), dtype=float)
    b = np.full((N, M), float(b_loose), dtype=float)

    compute_collision_halfspaces_horizon_inplace(
        obstacles_xy=obstacles_xy,
        centers_xy=centers_xy,
        A_xy_out=A_xy,
        b_out=b,
        M=M,
        rho=rho,
        roi=roi,
        b_loose=b_loose,
        eps_norm=eps_norm,
        pick=pick,
    )
    return A_xy, b
