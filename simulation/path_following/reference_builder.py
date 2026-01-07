# nav_mpc/simulation/path_following/reference_builder.py

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Iterable


def _wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _blend_angle(a: float, b: float, w_b: float) -> float:
    """
    Blend angles on the circle.
    Returns angle corresponding to (1-w_b)*a + w_b*b in sin/cos space.
    """
    w_b = float(np.clip(w_b, 0.0, 1.0))
    sa, ca = np.sin(a), np.cos(a)
    sb, cb = np.sin(b), np.cos(b)
    s = (1.0 - w_b) * sa + w_b * sb
    c = (1.0 - w_b) * ca + w_b * cb
    return float(np.arctan2(s, c))


@dataclass
class PathReferenceBuilder:
    """
    Receding-horizon reference builder for 2D path following.

    Sets references for:
      - px,py from global path
      - phi from path tangent (blended to phi_goal near goal if provided)
      - v as cruise v_ref (blended to v_goal near goal if provided)

    Everything else remains "free" by default:
      - we keep it equal to the current state x at each call,
        so you don't accidentally drive it to zero.

    Goal behavior:
      - x_goal defines (px_goal, py_goal, phi_goal, v_goal, ...)
      - inside stop_radius, blend weight goes to goal
      - within stop_ramp, blend smoothly transitions
    """

    # indices in x
    pos_idx: tuple[int, int] = (0, 1)
    phi_idx: int | None = None
    v_idx: int | None = None

    # goal / cruise
    x_goal: np.ndarray | None = None     # full goal state (recommended)
    v_ref: float = 0.0                   # cruise speed
    stop_radius: float = 0.25
    stop_ramp: float = 0.50

    # path handling
    window: int = 40
    max_lookahead_points: int | None = None
    heading_lookahead: int = 3

    # optional: also blend additional indices to goal values near goal (e.g. yaw rate r)
    goal_indices: Iterable[int] | None = None

    # optional: force/fade (only if explicitly asked)
    zero_indices: Iterable[int] | None = None
    fade_indices: Iterable[int] | None = None

    _path_idx: int = 0

    def reset(self, path_idx: int = 0) -> None:
        self._path_idx = int(max(0, path_idx))

    def _blend_scale(self, dist_goal: float) -> float:
        """
        Returns s in [0,1]:
          s=1 far from goal (pure cruise/path-follow)
          s=0 at goal (pure goal-state)
        """
        if self.stop_ramp < 1e-9:
            return 0.0 if dist_goal <= self.stop_radius else 1.0

        if dist_goal <= self.stop_radius:
            return 0.0

        s = (dist_goal - self.stop_radius) / self.stop_ramp
        return float(np.clip(s, 0.0, 1.0))

    def __call__(self, global_path: np.ndarray, x: np.ndarray, N: int) -> np.ndarray:
        global_path = np.asarray(global_path, dtype=float)
        if global_path.ndim != 2 or global_path.shape[1] != 2:
            raise ValueError(f"global_path must be (M,2), got {global_path.shape}")
        if global_path.shape[0] < 1:
            raise ValueError("global_path is empty")

        x = np.asarray(x, dtype=float).reshape(-1)
        nx = x.size
        M = global_path.shape[0]

        px_i, py_i = self.pos_idx
        if not (0 <= px_i < nx and 0 <= py_i < nx):
            raise ValueError(f"pos_idx {self.pos_idx} out of bounds for nx={nx}")
        if self.phi_idx is not None and not (0 <= self.phi_idx < nx):
            raise ValueError(f"phi_idx {self.phi_idx} out of bounds for nx={nx}")
        if self.v_idx is not None and not (0 <= self.v_idx < nx):
            raise ValueError(f"v_idx {self.v_idx} out of bounds for nx={nx}")

        # goal state bookkeeping
        x_goal = None
        if self.x_goal is not None:
            x_goal = np.asarray(self.x_goal, dtype=float).reshape(-1)
            if x_goal.size != nx:
                raise ValueError(f"x_goal must have size nx={nx}, got {x_goal.size}")

        # goal position for blending scale
        if x_goal is not None:
            gx, gy = float(x_goal[px_i]), float(x_goal[py_i])
        else:
            gx, gy = float(global_path[-1, 0]), float(global_path[-1, 1])

        p = np.array([x[px_i], x[py_i]], dtype=float)
        dist_goal = float(np.hypot(p[0] - gx, p[1] - gy))
        s = self._blend_scale(dist_goal)          # 1 far, 0 near
        w_goal = 1.0 - s                          # 0 far, 1 near

        # clamp internal index
        self._path_idx = int(np.clip(self._path_idx, 0, M - 1))

        # forward search window for closest point
        i_start = self._path_idx
        i_end = min(M, self._path_idx + int(self.window) + 1)
        segment = global_path[i_start:i_end]

        if segment.shape[0] == 0:
            i0 = self._path_idx
        else:
            d2 = np.sum((segment - p[None, :]) ** 2, axis=1)
            i0 = i_start + int(np.argmin(d2))

        i0 = max(i0, self._path_idx)
        self._path_idx = i0

        if self.max_lookahead_points is None:
            i_max = M - 1
        else:
            i_max = min(M - 1, self._path_idx + int(self.max_lookahead_points))

        # default free behavior: copy current x
        Xref = np.tile(x.reshape(1, -1), (N + 1, 1))

        # positions from path
        for k in range(N + 1):
            idx = min(self._path_idx + k, i_max)
            Xref[k, px_i] = global_path[idx, 0]
            Xref[k, py_i] = global_path[idx, 1]

        # heading from tangent, then blend to phi_goal near goal (if available)
        if self.phi_idx is not None:
            lk = max(1, int(self.heading_lookahead))
            phi_path = np.empty(N + 1, dtype=float)

            for k in range(N + 1):
                idx0 = min(self._path_idx + k, i_max)
                idx1 = min(idx0 + lk, i_max)
                dx = global_path[idx1, 0] - global_path[idx0, 0]
                dy = global_path[idx1, 1] - global_path[idx0, 1]
                if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                    phi_path[k] = phi_path[k - 1] if k > 0 else float(x[self.phi_idx])
                else:
                    phi_path[k] = float(np.arctan2(dy, dx))

            if x_goal is not None:
                phi_goal = float(x_goal[self.phi_idx])
                for k in range(N + 1):
                    Xref[k, self.phi_idx] = _blend_angle(phi_path[k], phi_goal, w_goal)
            else:
                Xref[:, self.phi_idx] = phi_path

        # speed reference (if v is a STATE): blend v_ref -> v_goal
        if self.v_idx is not None:
            v_cruise = float(self.v_ref)
            if x_goal is not None:
                v_goal = float(x_goal[self.v_idx])
                v_ref = s * v_cruise + (1.0 - s) * v_goal
            else:
                # no goal speed provided -> optionally stop at goal by ramping to 0
                v_ref = s * v_cruise
            Xref[:, self.v_idx] = float(v_ref)

        # optionally blend additional indices to goal (e.g., yaw rate r -> 0)
        if x_goal is not None and self.goal_indices is not None:
            gi = list(self.goal_indices)
            for j in gi:
                if 0 <= j < nx:
                    Xref[:, j] = s * Xref[:, j] + (1.0 - s) * float(x_goal[j])

        # optional force/fade
        if self.zero_indices is not None:
            zi = list(self.zero_indices)
            if zi:
                Xref[:, zi] = 0.0

        if self.fade_indices is not None and s < 1.0:
            fi = list(self.fade_indices)
            if fi:
                Xref[:, fi] *= s

        return Xref


def make_reference_builder(
    *,
    pos_idx: tuple[int, int] = (0, 1),
    phi_idx: int | None = None,
    v_idx: int | None = None,
    x_goal: np.ndarray | None = None,
    v_ref: float = 0.0,
    stop_radius: float = 0.25,
    stop_ramp: float = 0.50,
    window: int = 40,
    max_lookahead_points: int | None = None,
    heading_lookahead: int = 3,
    goal_indices: Iterable[int] | None = None,
    zero_indices: Iterable[int] | None = None,
    fade_indices: Iterable[int] | None = None,
) -> PathReferenceBuilder:
    return PathReferenceBuilder(
        pos_idx=pos_idx,
        phi_idx=phi_idx,
        v_idx=v_idx,
        x_goal=x_goal,
        v_ref=v_ref,
        stop_radius=stop_radius,
        stop_ramp=stop_ramp,
        window=window,
        max_lookahead_points=max_lookahead_points,
        heading_lookahead=heading_lookahead,
        goal_indices=goal_indices,
        zero_indices=zero_indices,
        fade_indices=fade_indices,
    )
