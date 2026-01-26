import numpy as np
import sympy as sp

from core.constraints.sys_constraints import SystemConstraints
from core.models.dynamics import SystemModel


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

        # Initialize base (creates x_sym, u_sym, default bounds)
        super().__init__(system.state_dim, system.input_dim)

        # State constraints
        self.x_min[:] = -np.inf
        self.x_max[:] = +np.inf

        # Input constraints (SISO): fill arrays in the base class
        self.u_min[:] = -50.0
        self.u_max[:] = +50.0

    def build_system_constraints(self) -> sp.Matrix:
        """
        Return symbolic inequality constraints g(x,u) <= 0 as a column vector.

        For this simple example:
          g1(x,u) = u - u_max <= 0
          g2(x,u) = -u + u_min <= 0
        """
        # Use base class symbolic variables
        u_sym = self.u_sym  # (nu, 1)

        # Convert numpy bounds to plain Python floats for sympy:
        u_max = float(self.u_max[0])
        u_min = float(self.u_min[0])

        g1 = u_sym[0] - u_max   # u <= u_max  -> u - u_max <= 0
        g2 = -u_sym[0] + u_min  # u >= u_min  -> -u + u_min <= 0

        return sp.Matrix([g1, g2])
