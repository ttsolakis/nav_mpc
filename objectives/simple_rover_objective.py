import numpy as np
import sympy as sp

from models.dynamics import SystemModel
from objectives.objectives import Objective


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
        phi_goal_switch_dist: float = 0.25,   # [m] if closer than this -> use phi_goal
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
            Q = np.diag([10.0, 10.0, 5.0, 1e-7, 1e-7])

        if QN is None:
            QN = np.diag([10.0, 10.0, 5.0, 1e-7, 1e-7])

        if R is None:
            # input effort (wheel accel)
            R = np.diag([2.0, 2.0])

        if x_goal is None:
            x_goal = np.array([0.0, 2.0, np.pi / 2, 0.0, 0.0])

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

    @staticmethod
    def _wrap_to_pi(angle: sp.Expr) -> sp.Expr:
        """
        Smooth-ish wrap using atan2(sin, cos):
          wrapToPi(a) = atan2(sin(a), cos(a))
        """
        return sp.atan2(sp.sin(angle), sp.cos(angle))

    def build_state_error(self) -> sp.Matrix:
        """
        e_x(x) = [px-px_g, py-py_g, wrapped(phi - phi_ref(x)), omega_l-omg_l_g, omega_r-omg_r_g]^T

        where phi_ref(x) is bearing-to-goal far away and phi_goal near the goal.
        """
        x = self.x_sym

        # symbols
        px, py, phi = x[0], x[1], x[2]
        omega_l, omega_r = x[3], x[4]

        # goal values
        px_g = float(self.x_ref[0])
        py_g = float(self.x_ref[1])
        phi_g = float(self.x_ref[2])
        omega_l_g = float(self.x_ref[3])
        omega_r_g = float(self.x_ref[4])

        # vector to goal
        dx = px_g - px
        dy = py_g - py

        # distance (with epsilon to avoid 0/0 nastiness)
        d = sp.sqrt(dx**2 + dy**2 + self.angle_eps)

        # bearing-to-goal
        phi_bearing = sp.atan2(dy, dx)
        phi_ref = sp.Piecewise(
            (phi_g, d <= self.phi_goal_switch_dist),
            (phi_bearing, True),
        )
        e_phi = self._wrap_to_pi(phi - phi_ref)

        # wrapped heading error
        e_phi = self._wrap_to_pi(phi - phi_ref)

        # distance-based ramp on heading importance
        # w(d) = 0 far away, 1 near the goal (over phi_ramp_dist)
        if self.phi_ramp_dist > 0.0:
            w01 = sp.Max(0, sp.Min(1, 1 - d / self.phi_ramp_dist))
        else:
            w01 = sp.Integer(1)

        # actual multiplier between far and near
        w_phi = self.phi_weight_far + (self.phi_weight_near - self.phi_weight_far) * w01

        # scale error so effective weight becomes Q[2,2] * w_phi
        e_phi_scaled = sp.sqrt(w_phi) * e_phi

        return sp.Matrix([
            px - px_g,
            py - py_g,
            e_phi_scaled,
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
