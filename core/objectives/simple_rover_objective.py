import numpy as np
import sympy as sp

from core.models.dynamics import SystemModel
from core.objectives.objectives import Objective


class SimpleRoverObjective(Objective):
    """
    Quadratic (LQR-like) set-point objective for a simple rover.

    State (nx=5):
      x = [px, py, phi, omega_l, omega_r]^T

    Input (nu=2):
      u = [alpha_l, alpha_r]^T   (wheel accelerations)

    Key update:
      - phi_ref is not fixed: it's the bearing-to-goal atan2(y_g - y, x_g - x)
      - angle error is wrapped: wrapToPi(phi - phi_ref)
      - near the goal, we switch to phi_goal to avoid atan2(0,0) noise
      - optional distance-based ramp on phi tracking (weak far, stronger near)
    """

    def __init__(
        self,
        system: SystemModel,
        x_goal: np.ndarray | None = None,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        QN: np.ndarray | None = None,
        u_ref: np.ndarray | None = None,
        *,
        # --- new knobs for heading reference ---
        phi_goal_switch_dist: float = 0.01,   # [m] if closer than this -> use phi_goal
        phi_ramp_dist: float = 2.0,           # [m] ramp heading importance over this distance
        phi_weight_far: float = 0.0,          # multiplier on phi error far away
        phi_weight_near: float = 1.0,         # multiplier on phi error near goal
        angle_eps: float = 1e-9,              # numerical epsilon for distance
    ) -> None:
        super().__init__(system)

        # -------------------------
        # Defaults (tune as needed)
        # -------------------------
        if Q is None:
            # [px, py, phi, omega_l, omega_r]
            Q = np.diag([1.0, 1.0, 1e-7, 10, 10])

        if QN is None:
            QN = np.diag([1.0, 1.0, 1e-7, 10, 10])

        if R is None:
            # input effort (wheel accel)
            R = np.diag([1.0, 1.0])

        if x_goal is None:
            x_goal = np.array([2.0, 2.0, 0.0, 10.0, 10.0])

        if u_ref is None:
            u_ref = np.zeros(2)

        # -------------------------
        # Validate shapes
        # -------------------------
        x_goal = np.asarray(x_goal, dtype=float).reshape(-1)
        u_ref  = np.asarray(u_ref,  dtype=float).reshape(-1)
        Q  = np.asarray(Q,  dtype=float)
        QN = np.asarray(QN, dtype=float)
        R  = np.asarray(R,  dtype=float)

        self.Q  = Q
        self.QN = QN
        self.R  = R

        self.x_ref = x_goal
        self.u_ref = u_ref

        # New heading-reference parameters
        self.phi_goal_switch_dist = float(phi_goal_switch_dist)
        self.phi_ramp_dist = float(phi_ramp_dist)
        self.phi_weight_far = float(phi_weight_far)
        self.phi_weight_near = float(phi_weight_near)
        self.angle_eps = float(angle_eps)

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

        px_g = float(self.x_ref[0])
        py_g = float(self.x_ref[1])
        phi_g = float(self.x_ref[2])
        omega_l_g = float(self.x_ref[3])
        omega_r_g = float(self.x_ref[4])

        dx = px_g - px
        dy = py_g - py
        d  = sp.sqrt(dx**2 + dy**2 + self.angle_eps)

        # Bearing-to-goal (well-defined away from goal)
        phi_bearing = sp.atan2(dy, dx)

        # Smooth switch weight: ~0 far, ~1 near
        # switch_slope sets how "soft" the transition is (bigger = smoother)
        switch_slope = 1.0
        w = 0.5 * (1 - sp.tanh((d - self.phi_goal_switch_dist) / switch_slope))

        # Blend references on unit circle (avoids wrap issues)
        s = (1 - w) * sp.sin(phi_bearing) + w * sp.sin(phi_g)
        c = (1 - w) * sp.cos(phi_bearing) + w * sp.cos(phi_g)
        phi_ref = sp.atan2(s, c)

        e_phi = self._wrap_to_pi(phi - phi_ref)

        # Optional: ramp heading weight so far away it doesn't dominate
        if self.phi_ramp_dist > 0.0:
            # smooth 0..1 ramp (near goal -> 1)
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
        """
        e_u(u) = u - u_ref
        """
        u = self.u_sym
        u_ref_sym = sp.Matrix(self.u_ref)
        return u - u_ref_sym
