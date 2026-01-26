# nav_mpc/objectives/simple_pendulum_objective.py

import numpy as np
import sympy as sp
from core.models.dynamics import SystemModel
from core.objectives.objectives import Objective


class SimplePendulumObjective(Objective):
    """
    Simple quadratic objective for the pendulum:

      J = 0.5 * e_x^T Q  e_x  + 0.5 * e_u^T R e_u

    with e_x(x) = x - x_ref, e_u(u) = u - u_ref.
    """

    def __init__(self, system: SystemModel) -> None:
        super().__init__(system)

        # LQR-like weights
        self.Q  = np.diag([100.0, 1.0])   # stage state cost
        self.QN = np.diag([100.0, 1.0])   # terminal state cost
        self.R  = np.diag([0.1])         # stage input cost

        # Reference: upright (pi, 0), zero torque
        self.x_ref = np.array([np.pi, 0.0])
        self.u_ref = np.array([0.0])

    def build_state_error(self) -> sp.Matrix:
        """
        e_x(x - x_ref) = x - x_ref.

        Here x is symbolic, x_ref is numeric; we convert x_ref to a
        sympy constant vector so the whole expression is symbolic.
        """
        x = self.x_sym                         # sympy column (2, 1)
        x_ref_sym = sp.Matrix(self.x_ref)      # converts numpy -> sympy column

        e_x = x - x_ref_sym                    # still a sympy Matrix
        return e_x

    def build_input_error(self) -> sp.Matrix:
        """
        e_u(u - u_ref) = u - u_ref.
        """
        u = self.u_sym
        u_ref_sym = sp.Matrix(self.u_ref)      # sympy column (1, 1)

        e_u = u - u_ref_sym
        return e_u
