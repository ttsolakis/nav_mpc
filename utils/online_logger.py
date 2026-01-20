from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class OnlineLogger:
    x_traj: List[np.ndarray] = field(default_factory=list)
    u_traj: List[np.ndarray] = field(default_factory=list)
    X_pred_traj: List[np.ndarray] = field(default_factory=list)
    X_ref_traj: List[np.ndarray] = field(default_factory=list)
    scans: List[np.ndarray] = field(default_factory=list)
    col_bounds_traj: List[np.ndarray | None] = field(default_factory=list)
    col_Axy_traj: List[np.ndarray | None] = field(default_factory=list)

    def log(
        self,
        *,
        x: np.ndarray,
        u0: np.ndarray,
        X: np.ndarray,
        Xref_seq: np.ndarray,
        scan: np.ndarray,
        A_xy: np.ndarray | None,
        b: np.ndarray | None,
    ) -> None:
        """Store one MPC step worth of data."""
        self.x_traj.append(x.copy())
        self.u_traj.append(u0.copy())
        self.X_pred_traj.append(X.copy())
        self.X_ref_traj.append(Xref_seq.copy())
        self.scans.append(scan)
        self.col_bounds_traj.append(None if b is None else b.copy())
        self.col_Axy_traj.append(None if A_xy is None else A_xy.copy())
