# nav_mpc/models/simple_pendulum_model.py

import numpy as np
from .dynamics import SystemModel


class SimplePendulumModel(SystemModel):
    """
    Simple pendulum model to test pipeline.
    State:
        x = [Θ1 θ2]^T
          θ1  : angular position [rad]
          θ2  : angular velocity [rad/s]

    Input:
        u = Τ
          Τ : torque input [Nm]

    Parameters
    ----------
    g : float
        Acceleration of gravity [m/s^2].
    l : float
        Length of the pendulum [m].
    """

    def __init__(self) -> None:
        
        # Define state and input dimensions
        state_dim = 2  # 2 states: θ1, θ2
        input_dim = 1  # 1 input: Τ
        super().__init__(state_dim, input_dim)

        # Define model parameters
        self.g = float(9.81)  # Acceleration of gravity [m/s^2]
        self.l = float(0.1)   # Length of the pendulum [m]

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Continuous-time dynamics x_dot = f(x, u).

        Parameters
        ----------
        x : np.ndarray
            State vector [θ1, θ2], shape (2,).
        u : np.ndarray
            Input vector [Τ], shape (1,).

        Returns
        -------
        x_dot : np.ndarray
            Time derivative [θ1_dot, θ2_dot], shape (2,).
        """
        x = np.asarray(x).reshape(-1)
        u = np.asarray(u).reshape(-1)

        θ1, θ2 = x
        Τ = u[0]

        x_dot = np.empty(2, dtype=float)
        x_dot[0] = θ2
        x_dot[1] = (Τ - self.g / self.l * np.sin(θ1))

        return x_dot
