# nav_mpc/models/waveshare_rover_model.py

import numpy as np
from .dynamics import SystemModel


class WaveshareRoverModel(SystemModel):
    """
    Simple differential-drive (unicycle) kinematic model for the Waveshare rover.

    State:
        x = [px, py, phi]^T
          px  : position in world x-axis [m]
          py  : position in world y-axis [m]
          phi : heading angle [rad]

    Input:
        u = [omega_l, omega_r]^T
          omega_l : left wheel angular velocity  [rad/s]
          omega_r : right wheel angular velocity [rad/s]

    Parameters
    ----------
    wheel_radius : float
        Wheel radius R [m].
    wheel_base : float
        Distance L between the left and right wheel contact points [m].
    """

    def __init__(self) -> None:
        
        # Define state and input dimensions
        state_dim = 3  # 3 states: px, py, phi
        input_dim = 2  # 2 inputs: omega_l, omega_r
        super().__init__(state_dim, input_dim)

        # Define model parameters
        self.R = float(0.4)  # Wheel radius [m]
        self.L = float(0.6)  # Wheel base [m]

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Continuous-time dynamics x_dot = f(x, u).

        Parameters
        ----------
        x : np.ndarray
            State vector [px, py, phi], shape (3,).
        u : np.ndarray
            Input vector [omega_l, omega_r], shape (2,).

        Returns
        -------
        x_dot : np.ndarray
            Time derivative [px_dot, py_dot, phi_dot], shape (3,).
        """
        x = np.asarray(x).reshape(-1)
        u = np.asarray(u).reshape(-1)

        px, py, phi = x
        omega_l, omega_r = u

        v = 0.5 * self.R * (omega_r + omega_l)
        r = 0.5 * self.R / self.L * (omega_r - omega_l)

        x_dot = np.empty(3, dtype=float)
        x_dot[0] = v * np.cos(phi)   # px_dot
        x_dot[1] = v * np.sin(phi)   # py_dot
        x_dot[2] = r                 # phi_dot

        return x_dot
