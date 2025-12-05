# nav_mpc/objectives/simple_pendulum_objective.py

import numpy as np

from objectives.objectives import Objective
from models.dynamics import SystemModel


class SimplePendulumObjective(Objective):
    """
    Quadratic MPC objective for the SimplePendulumModel:

        ℓ(x, u) = (x - x_ref)^T Q (x - x_ref) + u^T R u

    with tunable diagonal weights.
    """

    def __init__(self, system: SystemModel) -> None:
        self.system = system
        self._q_theta = 100.0
        self._q_theta_dot = 1.0
        self._r_torque = 0.1

        super().__init__(system.state_dim, system.input_dim)

    def build_weights(self) -> None:
        # State weights: Q = diag(q_theta, q_theta_dot)
        self.Q = np.diag([self._q_theta, self._q_theta_dot])

        # Terminal weight: for now, same as stage cost
        self.QN = self.Q.copy()

        # Input weight R (scalar for single input)
        self.R = np.array([[self._r_torque]], dtype=float)
