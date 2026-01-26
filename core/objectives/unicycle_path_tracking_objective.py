# nav_mpc/objectives/unicycle_path_tracking_objective.py

import numpy as np
import sympy as sp

from core.models.dynamics import SystemModel
from core.objectives.objectives import Objective


class UnicyclePathTrackingObjective(Objective):
    """
    Path tracking objective for UnicycleKinematicModel with wheel-accel inputs:

        x = [px, py, phi, v, r]
        u = [alpha_l, alpha_r]

    Tracks:
      - receding reference for px, py (from Xref_seq)
      - heading reference:
          far away: bearing to reference point
          near goal: phi reference from Xref_seq (typically path tangent or goal heading)
      - optional tracking for v,r via Xref_seq (you set Xref_seq[:,3]=v_ref, Xref_seq[:,4]=0 in main)

    Additionally:
      - penalizes wheel angular accelerations alpha_l, alpha_r (smooth control)
        via constant u_ref = [0, 0] interface expected by build_qp().
    """

    def __init__(
        self,
        system: SystemModel,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        QN: np.ndarray | None = None,
        u_ref: np.ndarray | None = None,  # constant accel reference, typically zeros
        *,
        # heading blend parameters (same spirit as your rover objective)
        phi_goal_switch_dist: float = 0.10,   # [m] below this, use phi_g more
        phi_ramp_dist: float = 2.0,           # [m] distance over which heading weight ramps
        phi_weight_far: float = 0.0,          # heading weight far away
        phi_weight_near: float = 1.0,         # heading weight near goal
        angle_eps: float = 1e-9,
    ) -> None:
        super().__init__(system)

        nx = system.state_dim
        nu = system.input_dim

        if nx != 5:
            raise ValueError(f"UnicyclePathTrackingObjective expects nx=5, got nx={nx}")
        if nu != 2:
            raise ValueError(f"UnicyclePathTrackingObjective expects nu=2, got nu={nu}")

        # State weights: [px, py, phi, v, r]
        # Note: phi weight is applied via scaled error (sqrt(w_phi)*e_phi) so keep Q[2,2] moderate.
        if Q is None:
            Q = np.diag([50.0, 50.0, 5.0, 2.0, 1.0])
        if QN is None:
            QN = np.diag([120.0, 120.0, 10.0, 2.0, 1.0])

        # Input weights: penalize accelerations (smoothness)
        if R is None:
            R = np.diag([0.5, 0.5])

        self.Q = np.asarray(Q, dtype=float)
        self.QN = np.asarray(QN, dtype=float)
        self.R = np.asarray(R, dtype=float)

        # Constant input reference expected by qp_offline.build_qp()
        if u_ref is None:
            u_ref = np.zeros(nu, dtype=float)
        self.u_ref = np.asarray(u_ref, dtype=float).reshape(nu)

        # heading knobs
        self.phi_goal_switch_dist = float(phi_goal_switch_dist)
        self.phi_ramp_dist = float(phi_ramp_dist)
        self.phi_weight_far = float(phi_weight_far)
        self.phi_weight_near = float(phi_weight_near)
        self.angle_eps = float(angle_eps)

        # symbolic time-varying state reference r ∈ R^nx (here nx=5)
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
        px, py, phi, v, r = x[0], x[1], x[2], x[3], x[4]

        rr = self.r_sym
        px_g, py_g, phi_g, v_g, r_g = rr[0], rr[1], rr[2], rr[3], rr[4]

        dx = px_g - px
        dy = py_g - py
        d = sp.sqrt(dx**2 + dy**2 + self.angle_eps)

        # Far from goal: align to bearing toward reference point
        phi_bearing = sp.atan2(dy, dx)

        # Blend bearing vs. desired goal/path heading depending on distance
        switch_slope = 1.0
        w_goal = 0.5 * (1 - sp.tanh((d - self.phi_goal_switch_dist) / switch_slope))

        s = (1 - w_goal) * sp.sin(phi_bearing) + w_goal * sp.sin(phi_g)
        c = (1 - w_goal) * sp.cos(phi_bearing) + w_goal * sp.cos(phi_g)
        phi_ref = sp.atan2(s, c)

        e_phi = self._wrap_to_pi(phi - phi_ref)

        # Ramp heading importance from far to near
        if self.phi_ramp_dist > 0.0:
            ramp_slope = 1.0
            w01 = 0.5 * (1 - sp.tanh((d - self.phi_ramp_dist) / ramp_slope))
        else:
            w01 = sp.Integer(1)

        w_phi = self.phi_weight_far + (self.phi_weight_near - self.phi_weight_far) * w01
        e_phi = sp.sqrt(w_phi) * e_phi

        # Track v and r directly to references in Xref_seq.
        # (In main, set Xref_seq[:,3]=v_ref and Xref_seq[:,4]=0.0)
        return sp.Matrix([
            px - px_g,
            py - py_g,
            e_phi,
            v - v_g,
            r - r_g,
        ])

    def build_input_error(self) -> sp.Matrix:
        # Keep wheel angular accelerations small (smoothness)
        u = self.u_sym
        return u - sp.Matrix(self.u_ref)
