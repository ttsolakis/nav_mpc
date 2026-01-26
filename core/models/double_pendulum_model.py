# nav_mpc/models/double_pendulum_model.py

import sympy as sp
from core.models.dynamics import SystemModel


class DoublePendulumModel(SystemModel):
    """
    Double pendulum with point masses m1, m2 and massless rods l1, l2.
    Actuation: torque τ applied at the first joint (θ1).

    State:
        x = [θ1, θ2, θ1_dot, θ2_dot]^T
          x[0] = θ1      [rad]
          x[1] = θ2      [rad]
          x[2] = θ1_dot  [rad/s]
          x[3] = θ2_dot  [rad/s]

    Input:
        u = [τ]
          u[0] = τ       [Nm]

    Dynamics:
        x_dot = f(x, u)
    """

    def __init__(self) -> None:
        # Model dimensions
        state_dim = 4
        input_dim = 1

        # Model parameters
        self.g = 9.81
        self.l1 = 0.5
        self.l2 = 0.5
        self.m1 = 1.0
        self.m2 = 1.0

        # This will create x_sym, u_sym and call build_dynamics()
        super().__init__(state_dim, input_dim)

    def build_dynamics(self) -> None:
        x = self.x_sym
        u = self.u_sym

        theta1 = x[0]
        theta2 = x[1]
        theta1_dot = x[2]
        theta2_dot = x[3]
        tau = u[0]

        # Convenience
        Delta = theta2 - theta1
        c = sp.cos(Delta)
        s = sp.sin(Delta)

        m1, m2 = self.m1, self.m2
        l1, l2 = self.l1, self.l2
        g = self.g

        # RHS terms from the second-order EoMs (so that M * [ddθ1, ddθ2]^T = b)
        b1 = tau + m2 * l1 * l2 * (theta2_dot**2) * s - (m1 + m2) * g * l1 * sp.sin(theta1)
        b2 = -m2 * l1 * l2 * (theta1_dot**2) * s - m2 * g * l2 * sp.sin(theta2)

        # Mass matrix
        M11 = (m1 + m2) * l1**2
        M12 = m2 * l1 * l2 * c
        M21 = M12
        M22 = m2 * l2**2

        M = sp.Matrix([[M11, M12],
                       [M21, M22]])
        b = sp.Matrix([b1, b2])

        # Solve for accelerations
        dd = M.LUsolve(b)  # dd = [ddθ1, ddθ2]
        theta1_ddot = dd[0]
        theta2_ddot = dd[1]

        # First-order dynamics
        f0 = theta1_dot
        f1 = theta2_dot
        f2 = theta1_ddot
        f3 = theta2_ddot

        self.f_sym = sp.Matrix([f0, f1, f2, f3])
