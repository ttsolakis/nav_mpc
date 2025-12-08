# nav_mpc/constraints/system_constraints/simple_pendulum_sys_constraints.py

import numpy as np
import sympy as sp

from constraints.sys_constraints import SystemConstraints
from models.dynamics import SystemModel


class SimplePendulumSystemConstraints(SystemConstraints):
    """
    Box bounds for SimplePendulumModel.

    For now:
      - no state constraints: x = [theta, theta_dot] free
      - input torque bounded: u ∈ [u_min, u_max]

    Symbolic constraints:
      g(x,u) <= 0,  g: R^{nx} x R^{nu} -> R^{nc}

      Here, nc = 2:

        g1(x,u) = u - u_max <= 0
        g2(x,u) = -u + u_min <= 0
    """

    def __init__(self, system: SystemModel) -> None:
        self.system = system

        # Initialize base (creates x_sym, u_sym, bounds defaults)
        super().__init__(system.state_dim, system.input_dim)

        # State constraints
        self.x_min[:] = -np.inf
        self.x_max[:] = +np.inf

        # Input constraints (SISO; using scalars)
        self.u_min = -100.0
        self.u_max = +100.0

    def build_system_constraints(self) -> sp.Matrix:
        """
        Return symbolic inequality constraints g(x,u) <= 0 as a column vector.

        For this simple example:
          g1(x,u) = u - u_max <= 0
          g2(x,u) = -u + u_min <= 0
        """
        # You can either use the base x_sym/u_sym or the system ones.
        # Here we use the system's, to guarantee same symbols as in dynamics:
        # x_sym = self.system.state_symbolic()   # (nx, 1) – unused here
        # u_sym = self.system.input_symbolic()   # (nu, 1)

        # Or use the base class symbolic variables:
        x_sym = self.x_sym               # (nx, 1) – unused here
        u_sym = self.u_sym               # (nu, 1)

        g1 = u_sym[0] - self.u_max   # u <= u_max
        g2 = -u_sym[0] + self.u_min  # u >= u_min -> -u + u_min <= 0

        return sp.Matrix([g1, g2])
