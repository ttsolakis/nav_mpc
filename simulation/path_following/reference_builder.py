import numpy as np
from dataclasses import dataclass
from typing import Iterable, Callable


@dataclass
class PathReferenceBuilder:
    """
    Generic receding-horizon reference builder from a 2D path.

    Key ideas:
    - internal memory: self._path_idx (monotonic progress, no need to expose path_idx)
    - only requires: global_path (M,2), current state x, horizon N
    - user specifies which state indices correspond to position (px, py)
    - optional: phi_idx to set a heading reference from path tangent
    - all other state entries default to 0 (generic)
    - optional: stop logic that fades selected indices to 0 near goal (generic)
    """

    pos_idx: tuple[int, int] = (0, 1)          # which x entries are (px, py)
    phi_idx: int | None = None                # heading index in x, if any
    window: int = 40                          # forward search window
    max_lookahead_points: int | None = None   # clamp horizon in path indices
    stop_radius: float = 0.25                 # [m] distance where references become "stop"
    stop_ramp: float = 0.50                   # [m] ramp length
    zero_indices: Iterable[int] | None = None # which indices to force to 0 (default = all except pos_idx + phi_idx)
    fade_indices: Iterable[int] | None = None # which indices to fade to 0 near goal (default = zero_indices)

    _path_idx: int = 0                        # internal monotonic memory

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
            Xref: (N+1, nx)
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

        # Default: force everything except position (and phi if requested) to 0
        if self.zero_indices is None:
            keep = {px_i, py_i}
            if self.phi_idx is not None:
                keep.add(self.phi_idx)
            zero_indices = [j for j in range(nx) if j not in keep]
        else:
            zero_indices = list(self.zero_indices)

        # Default: fade same indices as we force to zero (no-op if they are already zero)
        if self.fade_indices is None:
            fade_indices = zero_indices
        else:
            fade_indices = list(self.fade_indices)

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

        # --- compute stop scaling (generic) ---
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
        Xref = np.zeros((N + 1, nx), dtype=float)

        # Fill positions from the path
        for k in range(N + 1):
            idx = min(self._path_idx + k, i_max)
            Xref[k, px_i] = global_path[idx, 0]
            Xref[k, py_i] = global_path[idx, 1]

        # Set heading from path tangent if requested
        if self.phi_idx is not None:
            for k in range(N):
                dx = Xref[k + 1, px_i] - Xref[k, px_i]
                dy = Xref[k + 1, py_i] - Xref[k, py_i]
                if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                    Xref[k, self.phi_idx] = Xref[k - 1, self.phi_idx] if k > 0 else float(x[self.phi_idx])
                else:
                    Xref[k, self.phi_idx] = float(np.arctan2(dy, dx))
            Xref[N, self.phi_idx] = Xref[N - 1, self.phi_idx] if N > 0 else float(x[self.phi_idx])

        # Force other indices to 0 (generic)
        if zero_indices:
            Xref[:, zero_indices] = 0.0

        # Fade selected indices to 0 near goal (generic)
        # Note: if they are already 0, this does nothing — still generic.
        if fade_indices and scale < 1.0:
            Xref[:, fade_indices] *= scale

        return Xref


def make_reference_builder(
    *,
    pos_idx: tuple[int, int] = (0, 1),
    phi_idx: int | None = None,
    window: int = 40,
    max_lookahead_points: int | None = None,
    stop_radius: float = 0.25,
    stop_ramp: float = 0.50,
    zero_indices: Iterable[int] | None = None,
    fade_indices: Iterable[int] | None = None,
) -> PathReferenceBuilder:
    return PathReferenceBuilder(
        pos_idx=pos_idx,
        phi_idx=phi_idx,
        window=window,
        max_lookahead_points=max_lookahead_points,
        stop_radius=stop_radius,
        stop_ramp=stop_ramp,
        zero_indices=zero_indices,
        fade_indices=fade_indices,
    )
