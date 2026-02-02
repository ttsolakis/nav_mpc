# nav_mpc/objectives/cybership_path_tracking_objective.py

import numpy as np
import sympy as sp

from core.models.dynamics import SystemModel
from core.objectives.objectives import Objective


class CybershipPathTrackingObjective(Objective):
    """
    Path tracking objective for CybershipModel:

        x = [px, py, psi, ux, uy, r]
        u_in = [thrust_left, thrust_right, thrust_bow, azimuth_left, azimuth_right]

    Tracks:
      - receding reference for px, py, psi, and ux Xref_seq (r_sym)
      - heading reference uses the same "bearing far / psi_ref near" blend as the unicycle objective

    Additionally:
      - penalizes thruster commands (smooth control / effort)
        via constant u_ref interface expected by build_qp().

    Notes:
      - Like the unicycle objective, heading error is wrapped to [-pi, pi] and its weight is distance-dependent.
      - If in your main you only care about position/heading, just set Xref_seq[:,3:]=0.
    """

    def __init__(
        self,
        system: SystemModel,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        QN: np.ndarray | None = None,
        u_ref: np.ndarray | None = None,  # constant input reference, typically zeros
        *,
        # heading blend parameters
        psi_goal_switch_dist: float = 0.25,  # [m]
        psi_ramp_dist: float = 5.0,          # [m]
        psi_weight_far: float = 0.0,
        psi_weight_near: float = 1.0,
        angle_eps: float = 1e-9,
    ) -> None:
        super().__init__(system)

        nx = system.state_dim
        nu = system.input_dim

        if nx != 6:
            raise ValueError(f"CybershipPathTrackingObjective expects nx=6, got nx={nx}")
        if nu != 5:
            raise ValueError(f"CybershipPathTrackingObjective expects nu=5, got nu={nu}")

        # State weights: [px, py, psi, ux, uy, r]
        # Keep psi weight moderate because we also scale e_psi by sqrt(w_psi).
        if Q is None:
            Q = np.diag([50.0, 50.0, 5.0, 2.0, 2.0, 1.0])
        if QN is None:
            QN = np.diag([120.0, 120.0, 10.0, 2.0, 2.0, 1.0])

        # Input weights: penalize thrust + azimuth magnitudes (effort/smoothness)
        if R is None:
            # thrusts often deserve larger penalty than azimuth angles; tune as needed
            R = np.diag([0.2, 0.2, 0.2, 0.05, 0.05])

        self.Q = np.asarray(Q, dtype=float)
        self.QN = np.asarray(QN, dtype=float)
        self.R = np.asarray(R, dtype=float)

        if u_ref is None:
            u_ref = np.zeros(nu, dtype=float)
        self.u_ref = np.asarray(u_ref, dtype=float).reshape(nu)

        # heading knobs
        self.psi_goal_switch_dist = float(psi_goal_switch_dist)
        self.psi_ramp_dist = float(psi_ramp_dist)
        self.psi_weight_far = float(psi_weight_far)
        self.psi_weight_near = float(psi_weight_near)
        self.angle_eps = float(angle_eps)

        # symbolic time-varying state reference r ∈ R^nx (here nx=6)
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
        px, py, psi, ux, uy, r = x[0], x[1], x[2], x[3], x[4], x[5]

        rr = self.r_sym
        px_g, py_g, psi_g, ux_g, _, _ = rr[0], rr[1], rr[2], rr[3], rr[4], rr[5]

        dx = px_g - px
        dy = py_g - py
        d = sp.sqrt(dx**2 + dy**2 + self.angle_eps)

        # Far from goal: align to bearing toward reference point
        psi_bearing = sp.atan2(dy, dx)

        # Blend bearing vs. desired reference heading depending on distance
        switch_slope = 1.0
        w_goal = sp.Rational(1, 2) * (1 - sp.tanh((d - self.psi_goal_switch_dist) / switch_slope))

        s = (1 - w_goal) * sp.sin(psi_bearing) + w_goal * sp.sin(psi_g)
        c = (1 - w_goal) * sp.cos(psi_bearing) + w_goal * sp.cos(psi_g)
        psi_ref = sp.atan2(s, c)

        e_psi = self._wrap_to_pi(psi - psi_ref)

        # Ramp heading importance from far to near
        if self.psi_ramp_dist > 0.0:
            ramp_slope = 1.0
            w01 = sp.Rational(1, 2) * (1 - sp.tanh((d - self.psi_ramp_dist) / ramp_slope))
        else:
            w01 = sp.Integer(1)

        w_psi = self.psi_weight_far + (self.psi_weight_near - self.psi_weight_far) * w01
        e_psi = sp.sqrt(w_psi) * e_psi

        # Velocities:
        #  - track surge ux to reference
        #  - minimize sway uy to 0
        #  - do NOT track yaw-rate r (set error 0 so it contributes nothing)
        e_r = sp.Integer(0)

        # Track references in Xref_seq.
        return sp.Matrix([
            px - px_g,
            py - py_g,
            e_psi,
            ux - ux_g,
            uy,
            e_r,
        ])

    def build_input_error(self) -> sp.Matrix:
        # Penalize thruster commands (effort / smoothness)
        u = self.u_sym
        return u - sp.Matrix(self.u_ref)
