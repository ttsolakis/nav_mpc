# core/augmentations/mpcc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Any
import numpy as np


@dataclass
class MPCCConfig:
    enabled: bool = False

    # indices in the ORIGINAL state x (not augmented)
    pos_idx: Tuple[int, int] = (0, 1)       # (px, py)
    psi_idx: Optional[int] = None           # heading yaw (optional, for coupling)

    # progress and speed
    v_ref: float = 0.0                      # desired progress speed [m/s]
    v_s_bounds: Tuple[float, float] = (0.0, 1.0)  # bounds on virtual progress speed

    # weights
    w_contour: float = 100.0
    w_lag: float = 1.0
    w_vs: float = 10.0                      # (v_s - v_ref)^2
    w_coupling: float = 0.0                 # (v_s - t(s)^T p_dot)^2 (0 disables)

    # optional: if you want hard coupling later
    coupling_band: Optional[float] = None   # epsilon for |v_s - t^T p_dot| <= eps

    # user-provided mapping (generic)
    # returns world-frame planar velocity [vx, vy] given original x
    p_dot_world_fun: Optional[Callable[[np.ndarray], np.ndarray]] = None


def augment_problem_with_mpcc(
    *,
    system: Any,
    objective: Any,
    constraints: Any,
    mpcc: MPCCConfig,
    path: Any,
) -> tuple[Any, Any, Any]:
    """
    Return augmented (system, objective, constraints).

    This is intentionally a STUB in step-1:
      - it validates config and path
      - returns the original objects unchanged if mpcc.enabled is False
      - otherwise raises NotImplementedError (we'll implement in next steps)
    """
    if not mpcc.enabled:
        return system, objective, constraints

    # basic checks so errors are clear early
    if path is None:
        raise ValueError("MPCC enabled but 'path' is None")
    if mpcc.pos_idx is None or len(mpcc.pos_idx) != 2:
        raise ValueError("mpcc.pos_idx must be a tuple (px_idx, py_idx)")

    raise NotImplementedError(
        "MPCC augmentation is enabled but not implemented yet. "
        "Next steps will add: augmented state/input + MPCC objective + bounds."
    )
