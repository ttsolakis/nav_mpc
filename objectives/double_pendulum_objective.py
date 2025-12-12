# nav_mpc/objectives/double_pendulum_objective.py

import numpy as np
import sympy as sp
from models.dynamics import SystemModel
from objectives.objectives import Objective


class DoublePendulumObjective(Objective):
    """
    Simple quadratic objective (LQR-like) for the double pendulum:

      J = 0.5 * e_x^T Q  e_x  + 0.5 * e_u^T R e_u
      JN = 0.5 * e_x^T QN e_x

    with e_x(x) = x - x_ref, e_u(u) = u - u_ref.

    State:
      x = [θ1, θ2, θ1_dot, θ2_dot]^T
    Input:
      u = [τ]
    """

    def __init__(self, system: SystemModel) -> None:
        super().__init__(system)

        # LQR-like weights (tune as you like)
        self.Q  = np.diag([100.0, 100.0, 1.0, 1.0])   # stage state cost
        self.QN = np.diag([100.0, 100.0, 1.0, 1.0])   # terminal state cost
        self.R  = np.diag([0.1])                      # stage input cost

        # Reference: upright for both links (pi, pi), zero velocities, zero torque
        self.x_ref = np.array([np.pi, np.pi, 0.0, 0.0])
        self.u_ref = np.array([0.0])

    def build_state_error(self) -> sp.Matrix:
        """
        e_x(x - x_ref) = x - x_ref.
        """
        x = self.x_sym
        x_ref_sym = sp.Matrix(self.x_ref)  # numpy -> sympy column
        return x - x_ref_sym

    def build_input_error(self) -> sp.Matrix:
        """
        e_u(u - u_ref) = u - u_ref.
        """
        u = self.u_sym
        u_ref_sym = sp.Matrix(self.u_ref)
        return u - u_ref_sym
