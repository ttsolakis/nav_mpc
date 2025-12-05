# nav_mpc/constraints/system_constraints/simple_pendulum_sys_constraints.py

import numpy as np

from constraints.constraints import SystemConstraints
from models.dynamics import SystemModel


class SimplePendulumSystemConstraints(SystemConstraints):
    """
    Box bounds for SimplePendulumModel.

    For now:
      - no state constraints: x = [theta, theta_dot] free
      - input torque bounded: u ∈ [-max_torque, max_torque]
    """

    def __init__(self, system: SystemModel) -> None:
        self.system = system
        super().__init__(system.state_dim, system.input_dim)

    def build_bounds(self) -> None:

        # State constraints
        self.x_min[:] = -np.inf
        self.x_max[:] = +np.inf

        # Input constraints
        max_torque = 100.0  #Nm
        self.u_min = -max_torque
        self.u_max = max_torque