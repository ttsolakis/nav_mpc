# nav_mpc/constraints/system_constraints/simple_rover_sys_constraints.py

import numpy as np
import sympy as sp

from core.constraints.sys_constraints import SystemConstraints
from core.models.dynamics import SystemModel


class SimpleRoverSystemConstraints(SystemConstraints):
    def __init__(self, system: SystemModel) -> None:
        self.system = system
        super().__init__(system.state_dim, system.input_dim)

        # state: [px, py, phi, omega_l, omega_r]
        self.x_min[:] = -np.inf
        self.x_max[:] = +np.inf

        # wheel speed limits (same as before, but now for state)
        self.x_min[3] = -15.0
        self.x_max[3] = +15.0
        self.x_min[4] = -15.0
        self.x_max[4] = +15.0

        # input: wheel acceleration limits (tune these!)
        # Start conservative; too small => sluggish, too big => still jittery.
        self.u_min[:] = -2.0
        self.u_max[:] = +2.0

    def build_system_constraints(self) -> sp.Matrix:
        x = self.x_sym
        u = self.u_sym

        g = []

        # state bounds on omega_l, omega_r
        omega_l = x[3]
        omega_r = x[4]
        g += [omega_l - float(self.x_max[3])]
        g += [-omega_l + float(self.x_min[3])]
        g += [omega_r - float(self.x_max[4])]
        g += [-omega_r + float(self.x_min[4])]

        # input bounds on alpha_l, alpha_r
        alpha_l = u[0]
        alpha_r = u[1]
        g += [alpha_l - float(self.u_max[0])]
        g += [-alpha_l + float(self.u_min[0])]
        g += [alpha_r - float(self.u_max[1])]
        g += [-alpha_r + float(self.u_min[1])]

        return sp.Matrix(g)
