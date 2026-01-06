# nav_mpc/simulation/path_following/reference_builder.py
# (or wherever your make_reference_builder currently lives)

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Iterable


@dataclass
class PathReferenceBuilder:
    """
    Generic receding-horizon reference builder from a 2D path.

    Path-following assumes you may want references for:
      - position: (px, py)
      - heading:  phi  (path tangent)
      - speed:    v    (cruise speed along the path)

    Everything else is left "free" by default: we simply keep it equal to the
    current state x so you don't accidentally drive it to zero.
    (This avoids NaNs, which would break your compiled SymPy/Autowrap kernels.)

    Optional:
      - If you explicitly provide `zero_indices` and/or `fade_indices`,
        then those indices can be forced/faded (useful for specific systems).
    """

    # --- indices into the system state vector x ---
    pos_idx: tuple[int, int] = (0, 1)          # (px, py)
    phi_idx: int | None = None                # heading index
    v_idx: int | None = None                  # longitudinal speed index (if it's a STATE)

    # --- path search / horizon handling ---
    window: int = 40
    max_lookahead_points: int | None = None

    # --- stop behavior near goal (mainly for v reference) ---
    stop_radius: float = 0.25
    stop_ramp: float = 0.50

    # --- speed reference ---
    v_ref: float = 0.0                        # desired cruise speed [m/s]
    stop_v: bool = True                       # ramp v_ref -> 0 near goal

    # Optional: explicitly control other states (ONLY if you set these)
    zero_indices: Iterable[int] | None = None
    fade_indices: Iterable[int] | None = None

    # heading from tangent: use a lookahead point for a more stable tangent
    heading_lookahead: int = 3

    _path_idx: int = 0

    def reset(self, path_idx: int = 0) -> None:
        self._path_idx = int(max(0, path_idx))

    def __call__(
        self,
        global_path: np.ndarray,
        x: np.ndarray,
        N: int,
        *,
        goal_xy: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Returns:
            Xref: (N+1, nx) reference sequence.
        """
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

        # Clamp internal index
        self._path_idx = int(np.clip(self._path_idx, 0, M - 1))

        # --- local forward search for closest path point ---
        i_start = self._path_idx
        i_end = min(M, self._path_idx + int(self.window) + 1)

        p = np.array([x[px_i], x[py_i]], dtype=float)
        segment = global_path[i_start:i_end]

        if segment.shape[0] == 0:
            i0 = self._path_idx
        else:
            d2 = np.sum((segment - p[None, :]) ** 2, axis=1)
            i0 = i_start + int(np.argmin(d2))

        # monotonic progress
        i0 = max(i0, self._path_idx)
        self._path_idx = i0

        # clamp how far ahead we reference along the path
        if self.max_lookahead_points is None:
            i_max = M - 1
        else:
            i_max = min(M - 1, self._path_idx + int(self.max_lookahead_points))

        # --- stop scaling (used for v_ref, and optional fade_indices) ---
        if goal_xy is None:
            gx, gy = float(global_path[-1, 0]), float(global_path[-1, 1])
        else:
            goal_xy = np.asarray(goal_xy, dtype=float).reshape(2)
            gx, gy = float(goal_xy[0]), float(goal_xy[1])

        dist_goal = float(np.hypot(p[0] - gx, p[1] - gy))

        if self.stop_ramp < 1e-9:
            scale = 0.0 if dist_goal <= self.stop_radius else 1.0
        else:
            if dist_goal <= self.stop_radius:
                scale = 0.0
            else:
                scale = (dist_goal - self.stop_radius) / self.stop_ramp
                scale = float(np.clip(scale, 0.0, 1.0))

        # --- build Xref ---
        # Default "free": keep everything equal to current x
        Xref = np.tile(x.reshape(1, -1), (N + 1, 1))

        # Fill positions from the path
        for k in range(N + 1):
            idx = min(self._path_idx + k, i_max)
            Xref[k, px_i] = global_path[idx, 0]
            Xref[k, py_i] = global_path[idx, 1]

        # Heading reference from path tangent
        if self.phi_idx is not None:
            lk = max(1, int(self.heading_lookahead))
            for k in range(N + 1):
                idx0 = min(self._path_idx + k, i_max)
                idx1 = min(idx0 + lk, i_max)

                dx = global_path[idx1, 0] - global_path[idx0, 0]
                dy = global_path[idx1, 1] - global_path[idx0, 1]

                if abs(dx) < 1e-12 and abs(dy) < 1e-12:
                    if k > 0:
                        Xref[k, self.phi_idx] = Xref[k - 1, self.phi_idx]
                    else:
                        Xref[k, self.phi_idx] = float(x[self.phi_idx])
                else:
                    Xref[k, self.phi_idx] = float(np.arctan2(dy, dx))

        # Speed reference (ONLY if v is a STATE in this system)
        if self.v_idx is not None:
            vref = float(self.v_ref)
            if self.stop_v:
                vref *= scale
            Xref[:, self.v_idx] = vref

        # Optional: force some indices to 0 (ONLY if explicitly requested)
        if self.zero_indices is not None:
            zi = list(self.zero_indices)
            if zi:
                Xref[:, zi] = 0.0

        # Optional: fade some indices near goal (ONLY if explicitly requested)
        if self.fade_indices is not None and scale < 1.0:
            fi = list(self.fade_indices)
            if fi:
                Xref[:, fi] *= scale

        return Xref


def make_reference_builder(
    *,
    pos_idx: tuple[int, int] = (0, 1),
    phi_idx: int | None = None,
    v_idx: int | None = None,
    v_ref: float = 0.0,
    stop_v: bool = True,
    window: int = 40,
    max_lookahead_points: int | None = None,
    stop_radius: float = 0.25,
    stop_ramp: float = 0.50,
    zero_indices: Iterable[int] | None = None,
    fade_indices: Iterable[int] | None = None,
    heading_lookahead: int = 3,
) -> PathReferenceBuilder:
    return PathReferenceBuilder(
        pos_idx=pos_idx,
        phi_idx=phi_idx,
        v_idx=v_idx,
        v_ref=v_ref,
        stop_v=stop_v,
        window=window,
        max_lookahead_points=max_lookahead_points,
        stop_radius=stop_radius,
        stop_ramp=stop_ramp,
        zero_indices=zero_indices,
        fade_indices=fade_indices,
        heading_lookahead=heading_lookahead,
    )
