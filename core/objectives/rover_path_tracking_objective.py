# nav_mpc/objectives/rover_path_tracking_objective.py

import numpy as np
import sympy as sp

from core.models.dynamics import SystemModel
from core.objectives.objectives import Objective


class RoverPathTrackingObjective(Objective):
    def __init__(
        self,
        system: SystemModel,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        QN: np.ndarray | None = None,
        u_ref: np.ndarray | None = None,    # constant
        *,
        phi_goal_switch_dist: float = 0.01,
        phi_ramp_dist: float = 2.0,
        phi_weight_far: float = 0.0,
        phi_weight_near: float = 1.0,
        angle_eps: float = 1e-9,
    ) -> None:
        super().__init__(system)

        nx = system.state_dim
        nu = system.input_dim

        if Q is None:
            Q = np.diag([50.0, 50.0, 10.0, 1.0, 1.0])
        if QN is None:
            QN = np.diag([100.0, 100.0, 50.0, 0.1, 0.1])
        if R is None:
            R = np.diag([1.0, 1.0])
        if u_ref is None:
            u_ref = np.zeros(nu)

        self.Q = np.asarray(Q, dtype=float)
        self.QN = np.asarray(QN, dtype=float)
        self.R = np.asarray(R, dtype=float)

        # constant input reference (kept simple)
        self.u_ref = np.asarray(u_ref, dtype=float).reshape(nu)


        # heading knobs
        self.phi_goal_switch_dist = float(phi_goal_switch_dist)
        self.phi_ramp_dist = float(phi_ramp_dist)
        self.phi_weight_far = float(phi_weight_far)
        self.phi_weight_near = float(phi_weight_near)
        self.angle_eps = float(angle_eps)

        # NEW: symbolic time-varying reference r ∈ R^nx
        self.r_sym = sp.Matrix(sp.symbols(f"r0:{nx}"))

    def state_error_symbolic(self) -> sp.Matrix:
        return self.build_state_error()

    def input_error_symbolic(self) -> sp.Matrix:
        return self.build_input_error()

    @staticmethod
    def _wrap_to_pi(angle: sp.Expr) -> sp.Expr:
        return sp.atan2(sp.sin(angle), sp.cos(angle))

    def build_state_error(self) -> sp.Matrix:
        x = self.x_sym
        px, py, phi = x[0], x[1], x[2]
        omega_l, omega_r = x[3], x[4]

        r = self.r_sym
        px_g, py_g, phi_g, omega_l_g, omega_r_g = r[0], r[1], r[2], r[3], r[4]

        dx = px_g - px
        dy = py_g - py
        d  = sp.sqrt(dx**2 + dy**2 + self.angle_eps)

        phi_bearing = sp.atan2(dy, dx)

        switch_slope = 1.0
        w = 0.5 * (1 - sp.tanh((d - self.phi_goal_switch_dist) / switch_slope))

        s = (1 - w) * sp.sin(phi_bearing) + w * sp.sin(phi_g)
        c = (1 - w) * sp.cos(phi_bearing) + w * sp.cos(phi_g)
        phi_ref = sp.atan2(s, c)

        e_phi = self._wrap_to_pi(phi - phi_ref)

        if self.phi_ramp_dist > 0.0:
            ramp_slope = 1.0
            w01 = 0.5 * (1 - sp.tanh((d - self.phi_ramp_dist) / ramp_slope))
        else:
            w01 = sp.Integer(1)

        w_phi = self.phi_weight_far + (self.phi_weight_near - self.phi_weight_far) * w01
        e_phi = sp.sqrt(w_phi) * e_phi

        return sp.Matrix([
            px - px_g,
            py - py_g,
            e_phi,
            omega_l - omega_l_g,
            omega_r - omega_r_g,
        ])

    def build_input_error(self) -> sp.Matrix:
        # constant reference (same as your previous setup)
        u = self.u_sym
        return u - sp.Matrix(self.u_ref)
