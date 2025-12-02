# nav_mpc/models/dynamics.py

from abc import ABC, abstractmethod
import numpy as np

class SystemModel(ABC):
    """
    Abstract base class for continuous-time system models.

    Each concrete system must specify:
      - state_dim: dimension of the state vector x
      - input_dim: dimension of the input vector u
      - f(x, u): continuous-time dynamics, i.e. x_dot = f(x, u)

    Notes
    -----
    * x and u are assumed to be 1D numpy arrays of shapes
      (state_dim,) and (input_dim,) respectively.
    """

    def __init__(self, state_dim: int, input_dim: int) -> None:
        self.state_dim = state_dim
        self.input_dim = input_dim

    @abstractmethod
    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute the continuous-time state derivative x_dot = f(x, u).

        Parameters
        ----------
        x : np.ndarray
            State vector of shape (state_dim,).
        u : np.ndarray
            Input vector of shape (input_dim,).

        Returns
        -------
        x_dot : np.ndarray
            Time derivative of the state, shape (state_dim,).
        """
        raise NotImplementedError
