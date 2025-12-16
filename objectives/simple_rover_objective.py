# nav_mpc/objectives/simple_rover_objective.py

import numpy as np
import sympy as sp

from models.dynamics import SystemModel
from objectives.objectives import Objective


class SimpleRoverObjective(Objective):
    """
    Quadratic (LQR-like) set-point objective for a simple rover:

      stage:   J  = 0.5 * e_x^T Q  e_x  + 0.5 * e_u^T R e_u
      terminal JN = 0.5 * e_x^T QN e_x

    with:
      e_x(x) = x - x_ref
      e_u(u) = u - u_ref

    State:
      x = [px, py, phi]^T

    Input:
      u = [omega_l, omega_r]^T

    Notes:
    - "velocities to zero": the kinematic state has no velocities. Smoothness is
      typically enforced via R (input effort) and/or input rate penalties
      (can be added later as extra states or constraints).
    - If you want robust angle behavior, consider wrapping phi error later.
    """

    def __init__(
        self,
        system: SystemModel,
        x_goal: np.ndarray | None = None,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        QN: np.ndarray | None = None,
        u_ref: np.ndarray | None = None,
    ) -> None:
        super().__init__(system)

        # -------------------------
        # Defaults (tune as needed)
        # -------------------------
        if Q is None:
            # [px, py, phi, omega_l, omega_r]
            Q = np.diag([200.0, 200.0, 1e-7, 0.1, 0.1])

        if QN is None:
            QN = np.diag([800.0, 800.0, 1e-7, 50, 50])

        if R is None:
            # R now penalizes wheel acceleration => smoothness
            R = np.diag([5.0, 5.0])

        if x_goal is None:
            # goal wheel speeds usually 0 at the end
            x_goal = np.array([3.0, 3.0, np.pi/4, 0.0, 0.0])

        if u_ref is None:
            # reference accel = 0
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

    def build_state_error(self) -> sp.Matrix:
        """
        e_x(x) = x - x_ref
        """
        x = self.x_sym
        x_ref_sym = sp.Matrix(self.x_ref)
        return x - x_ref_sym

    def build_input_error(self) -> sp.Matrix:
        """
        e_u(u) = u - u_ref
        """
        u = self.u_sym
        u_ref_sym = sp.Matrix(self.u_ref)
        return u - u_ref_sym
