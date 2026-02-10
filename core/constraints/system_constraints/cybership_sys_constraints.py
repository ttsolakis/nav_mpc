# nav_mpc/constraints/system_constraints/cybership_sys_constraints.py

import numpy as np
import sympy as sp

from core.constraints.sys_constraints import SystemConstraints
from core.models.dynamics import SystemModel


class CybershipSystemConstraints(SystemConstraints):
    """
    Simple box constraints (lower/upper bounds) for CybershipModel.

    State:
        x = [px, py, psi, u, v, r]^T

    Input:
        u = [thrust_left, thrust_right, thrust_bow, azimuth_left, azimuth_right]^T
    """

    def __init__(
        self,
        system: SystemModel,
        *,
        # State bounds
        px_min: float = -np.inf,
        px_max: float = +np.inf,
        py_min: float = -np.inf,
        py_max: float = +np.inf,
        psi_min: float = -np.inf,
        psi_max: float = +np.inf,
        u_min: float = -np.inf,
        u_max: float = +np.inf,
        v_min: float = -np.inf,
        v_max: float = +np.inf,
        r_min: float = -np.inf,
        r_max: float = +np.inf,
        # Input bounds
        thrust_left_min: float = -7.5,
        thrust_left_max: float = +7.5,
        thrust_right_min: float = -7.5,
        thrust_right_max: float = +7.5,
        thrust_bow_min: float = -5.0,
        thrust_bow_max: float = +5.0,
        azimuth_left_min: float = -np.pi/2,
        azimuth_left_max: float = +np.pi/2,
        azimuth_right_min: float = -np.pi/2,
        azimuth_right_max: float = +np.pi/2,
    ) -> None:
        self.system = system

        # Defensive checks to catch wrong model wiring early
        if system.state_dim != 6:
            raise ValueError(f"Cybership constraints expect state_dim=6, got {system.state_dim}.")
        if system.input_dim != 5:
            raise ValueError(f"Cybership constraints expect input_dim=5, got {system.input_dim}.")

        super().__init__(system.state_dim, system.input_dim)

        # ---- Box bounds for x ----
        # x = [px, py, psi, u, v, r]
        self.x_min[:] = -np.inf
        self.x_max[:] = +np.inf

        self.x_min[0], self.x_max[0] = float(px_min), float(px_max)
        self.x_min[1], self.x_max[1] = float(py_min), float(py_max)
        self.x_min[2], self.x_max[2] = float(psi_min), float(psi_max)
        self.x_min[3], self.x_max[3] = float(u_min), float(u_max)
        self.x_min[4], self.x_max[4] = float(v_min), float(v_max)
        self.x_min[5], self.x_max[5] = float(r_min), float(r_max)

        # ---- Box bounds for u ----
        # u = [thrust_left, thrust_right, thrust_bow, azimuth_left, azimuth_right]
        self.u_min[:] = -np.inf
        self.u_max[:] = +np.inf

        self.u_min[0], self.u_max[0] = float(thrust_left_min), float(thrust_left_max)
        self.u_min[1], self.u_max[1] = float(thrust_right_min), float(thrust_right_max)
        self.u_min[2], self.u_max[2] = float(thrust_bow_min), float(thrust_bow_max)
        self.u_min[3], self.u_max[3] = float(azimuth_left_min), float(azimuth_left_max)
        self.u_min[4], self.u_max[4] = float(azimuth_right_min), float(azimuth_right_max)

    def build_system_constraints(self) -> sp.Matrix:
        x = self.x_sym
        u = self.u_sym

        g = []

        # state bounds: x_min <= x <= x_max
        for i in range(self.state_dim):
            xmin = float(self.x_min[i])
            xmax = float(self.x_max[i])

            if np.isfinite(xmax):
                g.append(x[i] - sp.Float(xmax))
            if np.isfinite(xmin):
                g.append(-x[i] + sp.Float(xmin))

        # input bounds: u_min <= u <= u_max
        for i in range(self.input_dim):
            umin = float(self.u_min[i])
            umax = float(self.u_max[i])

            if np.isfinite(umax):
                g.append(u[i] - sp.Float(umax))
            if np.isfinite(umin):
                g.append(-u[i] + sp.Float(umin))

        return sp.Matrix(g).reshape(len(g), 1)

