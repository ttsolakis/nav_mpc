# nav_mpc/constraints/system_constraints/unicycle_kinematic_sys_constraints.py

import numpy as np
import sympy as sp

from core.constraints.sys_constraints import SystemConstraints
from core.models.dynamics import SystemModel


class UnicycleKinematicSystemConstraints(SystemConstraints):
    """
    Constraints for UnicycleKinematicModel (x=[px,py,phi,v,r], u=[alpha_l,alpha_r]):

    - Enforce wheel speed limits via algebraic mapping:
        omega_r = (v + L*r)/R
        omega_l = (v - L*r)/R
      and constrain |omega_l|, |omega_r| <= omega_max.

    - Optionally bound wheel angular accelerations alpha_l, alpha_r.
    """

    def __init__(
        self,
        system: SystemModel,
        *,
        R: float = 0.040,
        L: float = 0.062,
        omega_max: float = 15.0,          # [rad/s]
        alpha_max: float | None = 30.0,   # [rad/s^2] set None to disable
    ) -> None:
        self.system = system
        self.R = float(R)
        self.L = float(L)
        self.omega_max = float(omega_max)
        self.alpha_max = None if alpha_max is None else float(alpha_max)

        super().__init__(system.state_dim, system.input_dim)

        # state: [px, py, phi, v, r]
        self.x_min[:] = -np.inf
        self.x_max[:] = +np.inf

        # input: [alpha_l, alpha_r]
        self.u_min[:] = -np.inf
        self.u_max[:] = +np.inf
        if self.alpha_max is not None:
            self.u_min[:] = -self.alpha_max
            self.u_max[:] = +self.alpha_max

    def build_system_constraints(self) -> sp.Matrix:
        x = self.x_sym
        u = self.u_sym

        v = x[3]
        r = x[4]

        R = sp.Float(self.R)
        L = sp.Float(self.L)
        wmax = sp.Float(self.omega_max)

        # wheel speeds implied by (v,r)
        omega_r = (v + L * r) / R
        omega_l = (v - L * r) / R

        g = []

        # |omega_l| <= wmax, |omega_r| <= wmax  -> g(x,u) <= 0 form
        g += [omega_l - wmax]
        g += [-omega_l - wmax]
        g += [omega_r - wmax]
        g += [-omega_r - wmax]

        # Optional accel bounds (if alpha_max was set)
        if self.alpha_max is not None:
            alpha_l = u[0]
            alpha_r = u[1]
            amax = sp.Float(self.alpha_max)

            g += [alpha_l - amax]
            g += [-alpha_l - amax]
            g += [alpha_r - amax]
            g += [-alpha_r - amax]

        return sp.Matrix(g)
