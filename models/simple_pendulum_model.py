# nav_mpc/models/simple_pendulum_model.py

import sympy as sp
from models.dynamics import SystemModel


class SimplePendulumModel(SystemModel):
    """
    Simple pendulum model (symbolic) to test the pipeline.

    Dynamics:
        x_dot = f(x, u)

    State:
        x = [θ1, θ2]^T
          x[0] = θ1 : angular position [rad]
          x[1] = θ2 : angular velocity [rad/s]

    Input:
        u = [τ]
          u[0] = τ : torque input [Nm]

    Parameters
    ----------
    g : acceleration of gravity [m/s^2].
    l : length of the pendulum [m].
    """

    def __init__(self) -> None:

        # Model dimensions
        state_dim = 2  
        input_dim = 1

        # Model parameters
        self.g = 9.81
        self.l = 0.1

        # This will create x_sym, u_sym and call build_dynamics()
        super().__init__(state_dim, input_dim)

    def build_dynamics(self) -> None:
        x = self.x_sym  
        u = self.u_sym

        # Dynamics equations:
        f0 = x[1]                                   # θ1_dot = θ2
        f1 = u[0] - self.g / self.l * sp.sin(x[0])  # θ2_dot = τ - (g/l) sin(θ1)

        self.f_sym = sp.Matrix([f0, f1])
