# constraints/collision_constraints/halfspace_corridor.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass(frozen=True, slots=True)
class HalfspaceCorridorCollisionConfig:
    """
    Collision constraints as halfspaces:

        A_xy[k,i,:] @ p <= b[k,i],     i = 0..M-1

    where p = (x,y) are the decision variables for the robot center at stage k (applied to x_{k+1} in your QP).

    Two selection modes are supported:
      - pick="closest":
          select up to M closest lidar points inside ROI (old baseline method)
          remaining constraints inactive with b=b_loose

      - pick="angular_bins":
          split 2pi into M equal slices (relative to heading psi_k) and pick
          the closest lidar point in each slice; if a slice has no point within ROI,
          create a virtual point on that slice centerline at distance roi.
          => always exactly M ACTIVE constraints per stage.
    """
    M: int = 16                           # number of halfspaces per stage
    pos_idx: tuple[int, int] = (0, 1)     # indices of (x,y) in state
    psi_idx: int = 2                      # heading index in state (needed for angular binning)
    r_robot: float = 0.10
    r_buffer: float = 0.05
    roi: float = 1.5
    b_loose: float = 1e6                  # inactive constraint upper bound (OSQP)
    eps_norm: float = 1e-9                # avoid divide-by-zero

    @property
    def r_safe(self) -> float:
        return float(self.r_robot + self.r_buffer)


# ==========================================================
# Helpers (Numba-friendly angle wrap and binning)
# ==========================================================

@njit(cache=True)
def _wrap_to_2pi(a: float) -> float:
    """Wrap angle to [0, 2*pi)."""
    two_pi = 2.0 * np.pi
    # Equivalent to: a % (2*pi) but more numba-stable
    a = a - two_pi * np.floor(a / two_pi)
    # Guard tiny numerical negatives
    if a < 0.0:
        a += two_pi
    return a


@njit(cache=True)
def _bin_index_centered(alpha_2pi: float, M: int) -> int:
    two_pi = 2.0 * np.pi
    Delta = two_pi / M
    i = int(np.floor((alpha_2pi + 0.5 * Delta) / Delta)) % M
    return i



# ==========================================================
# Numba kernels
# ==========================================================

@njit(cache=True)
def _compute_collision_halfspaces_horizon_inplace_closest_numba(
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
    Baseline method: pick up to M closest points in ROI per stage.
    Remaining rows are inactive (A=[0,0], b=b_loose).
    """
    N = centers_xy.shape[0]
    P = obs.shape[0]
    roi2 = roi * roi

    dist2 = np.empty(P, dtype=np.float64)
    sel_idx = np.empty(P, dtype=np.int64)
    sel_d2 = np.empty(P, dtype=np.float64)

    for k in range(N):
        cx = centers_xy[k, 0]
        cy = centers_xy[k, 1]

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

        m_use = M if cnt >= M else cnt

        for j in range(m_use):
            best_pos = j
            best_val = sel_d2[j]
            for t in range(j + 1, cnt):
                if sel_d2[t] < best_val:
                    best_val = sel_d2[t]
                    best_pos = t

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
                continue

            n0 = vx / nrm
            n1 = vy / nrm

            A_xy_out[k, j, 0] = -n0
            A_xy_out[k, j, 1] = -n1
            b_out[k, j] = -(n0 * ox + n1 * oy + rho)


@njit(cache=True)
def _compute_collision_halfspaces_horizon_inplace_angular_bins_numba(
    obs: np.ndarray,          # (P,2) lidar points in world
    centers_xy: np.ndarray,   # (N,2) cbar_k in world
    headings: np.ndarray,     # (N,)  psi_bar_k in world frame
    A_xy_out: np.ndarray,     # (N,M,2) output
    b_out: np.ndarray,        # (N,M) output
    M: int,
    rho: float,
    roi: float,
    eps_norm: float,
) -> None:
    """
    Angular binning method:

      - Split 2pi into M slices centered at i*Δ in robot frame (0 is forward).
      - For each stage k, map each obstacle to robot frame angle alpha in [0,2pi)
      - Compute bin index:
            i = floor((alpha + Δ/2)/Δ) mod M
      - Keep the closest obstacle (min distance) per bin.
      - If bin is empty, place a virtual obstacle at distance roi on the bin centerline.

    Produces exactly M active constraints per stage.
    """
    N = centers_xy.shape[0]
    P = obs.shape[0]
    roi2 = roi * roi

    two_pi = 2.0 * np.pi
    Delta = two_pi / M

    # scratch buffers (reused per stage)
    best_d2 = np.empty(M, dtype=np.float64)
    best_ox = np.empty(M, dtype=np.float64)
    best_oy = np.empty(M, dtype=np.float64)
    has = np.empty(M, dtype=np.int8)  # 0/1

    for k in range(N):
        cx = centers_xy[k, 0]
        cy = centers_xy[k, 1]
        psi = headings[k]

        # init bin winners
        for i in range(M):
            best_d2[i] = 1e300
            best_ox[i] = 0.0
            best_oy[i] = 0.0
            has[i] = 0

        # scan points, keep closest per bin
        for p in range(P):
            dx = obs[p, 0] - cx
            dy = obs[p, 1] - cy
            d2 = dx * dx + dy * dy
            if d2 > roi2:
                continue

            # alpha in robot frame, wrap to [0,2pi)
            alpha = np.arctan2(dy, dx) - psi
            alpha = _wrap_to_2pi(alpha)

            b = _bin_index_centered(alpha, M)

            if d2 < best_d2[b]:
                best_d2[b] = d2
                best_ox[b] = obs[p, 0]
                best_oy[b] = obs[p, 1]
                has[b] = 1

        # build constraints per bin
        for b in range(M):
            # obstacle point (real or virtual)
            if has[b] == 1:
                ox = best_ox[b]
                oy = best_oy[b]
            else:
                # virtual point at distance roi along bin centerline
                ang = psi + b * Delta
                ox = cx + roi * np.cos(ang)
                oy = cy + roi * np.sin(ang)

            vx = cx - ox
            vy = cy - oy
            nrm = np.sqrt(vx * vx + vy * vy)

            # n = (c - o)/||c - o||
            if nrm < eps_norm:
                # Extremely degenerate; make a harmless loose-ish constraint
                A_xy_out[k, b, 0] = 0.0
                A_xy_out[k, b, 1] = 0.0
                b_out[k, b] = 1e300
                continue

            n0 = vx / nrm
            n1 = vy / nrm

            # OSQP form: (-n)^T c <= -(n^T o + rho)
            A_xy_out[k, b, 0] = -n0
            A_xy_out[k, b, 1] = -n1
            b_out[k, b] = -(n0 * ox + n1 * oy + rho)


# ==========================================================
# Public APIs
# ==========================================================

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
    pick: str = "angular_bins",
    headings: np.ndarray | None = None,  # (N,) required for angular_bins
) -> None:
    """
    In-place API (no allocations).

    Output constraint form:
        A_xy_out[k,i,:] @ p <= b_out[k,i]
    where p = [x,y] is the decision variable (robot center at that stage).

    pick:
      - "closest"       : up to M constraints, rest inactive (b=b_loose)
      - "angular_bins"  : exactly M constraints (virtual points fill empty bins)
    """
    centers_xy = np.asarray(centers_xy, dtype=float).reshape(-1, 2)
    N = centers_xy.shape[0]

    if A_xy_out.shape != (N, M, 2):
        raise ValueError(f"A_xy_out must have shape {(N, M, 2)}, got {A_xy_out.shape}.")
    if b_out.shape != (N, M):
        raise ValueError(f"b_out must have shape {(N, M)}, got {b_out.shape}.")

    # defaults
    A_xy_out.fill(0.0)
    b_out.fill(float(b_loose))

    # normalize obstacles
    if obstacles_xy is None:
        obs = np.zeros((0, 2), dtype=np.float64)
    else:
        obs = np.asarray(obstacles_xy, dtype=float).reshape(-1, 2)

    if pick == "closest":
        if obs.shape[0] == 0:
            return
        _compute_collision_halfspaces_horizon_inplace_closest_numba(
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
        return

    if pick == "angular_bins":
        if headings is None:
            raise ValueError("pick='angular_bins' requires headings=(N,) array.")
        headings = np.asarray(headings, dtype=float).reshape(-1)
        if headings.shape[0] != N:
            raise ValueError(f"headings must have shape {(N,)}, got {headings.shape}.")

        _compute_collision_halfspaces_horizon_inplace_angular_bins_numba(
            obs=obs,
            centers_xy=centers_xy,
            headings=headings,
            A_xy_out=A_xy_out,
            b_out=b_out,
            M=int(M),
            rho=float(rho),
            roi=float(roi),
            eps_norm=float(eps_norm),
        )
        return

    raise ValueError(f"Unknown pick='{pick}'. Use 'closest' or 'angular_bins'.")


def compute_collision_halfspaces_horizon(
    obstacles_xy: np.ndarray | None,
    centers_xy: np.ndarray,      # (N,2)
    *,
    M: int,
    rho: float,
    roi: float,
    b_loose: float,
    eps_norm: float = 1e-9,
    pick: str = "angular_bins",
    headings: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Allocating wrapper for convenience / debugging.

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
        headings=headings,
    )
    return A_xy, b
