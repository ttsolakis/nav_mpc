# nav_mpc/constraints/system_constraints/double_pendulum_sys_constraints.py

import numpy as np
import sympy as sp

from constraints.sys_constraints import SystemConstraints
from models.dynamics import SystemModel


class DoublePendulumSystemConstraints(SystemConstraints):
    """
    Box bounds for DoublePendulumModel.

    For now:
      - no state constraints: x = [theta1, theta2, theta1_dot, theta2_dot] free
      - input torque bounded: u ∈ [u_min, u_max]

    Symbolic constraints:
      g(x,u) <= 0,  g: R^{nx} x R^{nu} -> R^{nc}

      Here, nc = 2:

        g1(x,u) = u - u_max <= 0
        g2(x,u) = -u + u_min <= 0
    """

    def __init__(self, system: SystemModel) -> None:
        self.system = system

        super().__init__(system.state_dim, system.input_dim)

        # State constraints: unconstrained by default
        self.x_min[:] = -np.inf
        self.x_max[:] = +np.inf

        # Input constraints (SISO)
        self.u_min[:] = -100.0
        self.u_max[:] = +100.0

    def build_system_constraints(self) -> sp.Matrix:
        """
        Return symbolic inequality constraints g(x,u) <= 0 as a column vector.
        """
        u_sym = self.u_sym  # (nu, 1)

        u_max = float(self.u_max[0])
        u_min = float(self.u_min[0])

        g1 = u_sym[0] - u_max   # u <= u_max  -> u - u_max <= 0
        g2 = -u_sym[0] + u_min  # u >= u_min  -> -u + u_min <= 0

        return sp.Matrix([g1, g2])
